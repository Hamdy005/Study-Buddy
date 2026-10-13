"""
Comprehensive test suite for the CRAG pipeline.
Covers:
- Config validation
- Schema validation
- merge_pool selective carry-over and deduplication
- Score gating and conditional branching
- Retry count and loop termination
- End-to-end execution with mocks across all paths (high score, mid+correct, ambiguous retry, fallback)
- Verifying reranker receives original_q, never rewritten query.
"""

import asyncio
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import pytest
from pydantic import ValidationError

from src.rag.config import Settings
from src.rag.constants import (
    DEFAULT_CARRY_MAX,
    DEFAULT_FINAL_CONTEXT_N,
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_KEEP_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_MAX_RETRIES,
    DEFAULT_POOL_MAX,
)
from src.rag.crag_graph import build_graph, graph
from src.rag.crag_llm import MockLLM
from src.rag.crag import run_crag as run
from src.rag.crag_nodes import (
    create_merge_pool_node,
    score_gate,
    evaluator_gate,
    retry_check_gate,
)
from src.rag.reranker import MockReranker
from src.rag.schemas import (
    Chunk,
    EvaluatorOutput,
    FinalAnswer,
    GraphState,
    RouterOutput,
)


# ═════════════════════════════════════════════════════════════════
# 1. Config Validation Tests
# ═════════════════════════════════════════════════════════════════

def test_config_valid_defaults():
    cfg = Settings()
    assert cfg.MAX_RETRIES == DEFAULT_MAX_RETRIES
    assert cfg.HIGH_THRESHOLD == DEFAULT_HIGH_THRESHOLD
    assert cfg.LOW_THRESHOLD == DEFAULT_LOW_THRESHOLD
    assert cfg.KEEP_THRESHOLD == DEFAULT_KEEP_THRESHOLD


def test_config_max_retries_rejected():
    with pytest.raises(ValidationError):
        Settings(MAX_RETRIES=3)

    with pytest.raises(ValidationError):
        Settings(MAX_RETRIES=-1)


def test_config_low_greater_or_equal_high_rejected():
    # LOW_THRESHOLD >= HIGH_THRESHOLD
    with pytest.raises(ValidationError):
        Settings(LOW_THRESHOLD=0.85, HIGH_THRESHOLD=0.80)

    with pytest.raises(ValidationError):
        Settings(LOW_THRESHOLD=0.80, HIGH_THRESHOLD=0.80)


def test_config_keep_outside_range_rejected():
    # KEEP_THRESHOLD < LOW_THRESHOLD
    with pytest.raises(ValidationError):
        Settings(LOW_THRESHOLD=0.4, KEEP_THRESHOLD=0.3, HIGH_THRESHOLD=0.8)

    # KEEP_THRESHOLD > HIGH_THRESHOLD
    with pytest.raises(ValidationError):
        Settings(LOW_THRESHOLD=0.3, KEEP_THRESHOLD=0.9, HIGH_THRESHOLD=0.8)


def test_config_carry_max_context_n_pool_max_ordering():
    # CARRY_MAX > FINAL_CONTEXT_N
    with pytest.raises(ValidationError):
        Settings(CARRY_MAX=10, FINAL_CONTEXT_N=5)

    # FINAL_CONTEXT_N > POOL_MAX
    with pytest.raises(ValidationError):
        Settings(FINAL_CONTEXT_N=20, POOL_MAX=15)


def test_config_rerank_window_validations():
    cfg0 = Settings(RERANK_WINDOW=384)
    assert cfg0.RERANK_WINDOW == 384

    cfg1 = Settings(RERANK_WINDOW=512)
    assert cfg1.RERANK_WINDOW == 512

    cfg2 = Settings(RERANK_WINDOW=1024)
    assert cfg2.RERANK_WINDOW == 1024

    with pytest.raises(ValidationError):
        Settings(RERANK_WINDOW=256)


# ═════════════════════════════════════════════════════════════════
# 2. Schema Validation Tests
# ═════════════════════════════════════════════════════════════════

def test_schema_router_empty_sources_rejected():
    with pytest.raises(ValidationError):
        RouterOutput(sources=[], search_query="test query")


def test_schema_router_duplicate_sources_rejected():
    with pytest.raises(ValidationError):
        RouterOutput(sources=["supabase", "supabase"], search_query="test query")


def test_schema_router_query_length():
    with pytest.raises(ValidationError):
        RouterOutput(sources=["web"], search_query="hi")  # too short (<3)

    with pytest.raises(ValidationError):
        RouterOutput(sources=["web"], search_query="x" * 305)  # too long (>300)

    ro = RouterOutput(sources=["web"], search_query="  valid query  ")
    assert ro.search_query == "valid query"


def test_schema_evaluator_correct_with_non_none_cause_rejected():
    # verdict == correct but likely_cause != none
    with pytest.raises(ValidationError):
        EvaluatorOutput(verdict="correct", missing="", likely_cause="bad_query")


def test_schema_evaluator_incorrect_with_none_cause_rejected():
    # verdict != correct but likely_cause == none
    with pytest.raises(ValidationError):
        EvaluatorOutput(verdict="incorrect", missing="missing detail", likely_cause="none")


def test_schema_evaluator_ambiguous_with_empty_missing_rejected():
    # verdict != correct but missing is empty
    with pytest.raises(ValidationError):
        EvaluatorOutput(verdict="ambiguous", missing="", likely_cause="bad_query")

    with pytest.raises(ValidationError):
        EvaluatorOutput(verdict="ambiguous", missing="   ", likely_cause="bad_query")


def test_schema_evaluator_valid_instances():
    e1 = EvaluatorOutput(verdict="correct", missing="", likely_cause="none")
    assert e1.verdict == "correct"
    assert e1.likely_cause == "none"

    e2 = EvaluatorOutput(verdict="ambiguous", missing="Need more context", likely_cause="bad_query")
    assert e2.verdict == "ambiguous"

    e3 = EvaluatorOutput(verdict="incorrect", missing="Completely off topic", likely_cause="not_in_source")
    assert e3.verdict == "incorrect"


def test_schema_chunk_validation():
    # Empty text rejected
    with pytest.raises(ValidationError):
        Chunk(id="c1", text="", source="wikipedia")

    with pytest.raises(ValidationError):
        Chunk(id="c1", text="   ", source="wikipedia")

    # URL without http(s) rejected
    with pytest.raises(ValidationError):
        Chunk(id="c1", text="valid passage", source="web", url="ftp://example.com")

    # Valid chunk
    c = Chunk.create(source="wikipedia", text="Clean passage text", url="https://en.wikipedia.org")
    assert c.id is not None
    assert c.source == "wikipedia"
    assert c.url == "https://en.wikipedia.org"


def test_schema_chunk_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        Chunk(id="c1", text="test", source="web", unknown_field="invalid")  # type: ignore


# ═════════════════════════════════════════════════════════════════
# 3. merge_pool Tests
# ═════════════════════════════════════════════════════════════════

def test_merge_pool_first_pass():
    node = create_merge_pool_node()
    new_chunks = [
        Chunk.create("supabase", "chunk 1"),
        Chunk.create("supabase", "chunk 2"),
    ]
    state: GraphState = {
        "retries": 0,
        "pool": [],
        "_new_chunks": new_chunks,
        "verdict": None,
    }
    res = node(state)
    result = asyncio.run(res) if asyncio.iscoroutine(res) else res
    assert len(result["pool"]) == 2
    assert result["pool"] == new_chunks


def test_merge_pool_appends_and_preserves_old():
    cfg = Settings()
    node = create_merge_pool_node(settings=cfg)

    old_chunks = [
        Chunk.create("supabase", "old low", score=0.3),
        Chunk.create("supabase", "old high 1", score=0.8),
        Chunk.create("supabase", "old mid", score=0.6),
        Chunk.create("supabase", "old high 2", score=0.9),
    ]
    new_chunks = [
        Chunk.create("wikipedia", "new 1", score=None),
    ]

    state: GraphState = {
        "retries": 1,
        "pool": old_chunks,
        "_new_chunks": new_chunks,
        "verdict": "ambiguous",
    }
    res = node(state)
    result = asyncio.run(res) if asyncio.iscoroutine(res) else res
    pool = result["pool"]

    # All old chunks preserved plus new 1 appended
    carried_old_texts = [c.text for c in pool if c.source == "supabase"]
    assert len(carried_old_texts) == 4
    assert any(c.text == "new 1" for c in pool)
    # Scored chunks are ranked descendingly (0.9, 0.8, 0.6, 0.3)
    scored = [c for c in pool if c.score is not None]
    assert [c.score for c in scored] == [0.9, 0.8, 0.6, 0.3]


def test_merge_pool_incorrect_preserves_old():
    node = create_merge_pool_node()
    old_chunks = [
        Chunk.create("supabase", "old high", score=0.9),
    ]
    new_chunks = [
        Chunk.create("wikipedia", "new fresh", score=None),
    ]
    state: GraphState = {
        "retries": 1,
        "pool": old_chunks,
        "_new_chunks": new_chunks,
        "verdict": "incorrect",
    }
    res = node(state)
    result = asyncio.run(res) if asyncio.iscoroutine(res) else res
    pool = result["pool"]
    # Does not drop old chunks
    assert len(pool) == 2
    assert any(c.text == "old high" for c in pool)
    assert any(c.text == "new fresh" for c in pool)


def test_merge_pool_dedupe_and_pool_max_cap():
    cfg = Settings(POOL_MAX=10)
    node = create_merge_pool_node(settings=cfg)

    # 15 distinct new chunks
    new_chunks = [
        Chunk.create("web", f"unique chunk {i}", score=None)
        for i in range(15)
    ]
    # Add a duplicate chunk with score
    duplicate = Chunk.create("web", "unique chunk 0", score=0.95)
    new_chunks.append(duplicate)

    state: GraphState = {
        "retries": 0,
        "pool": [],
        "_new_chunks": new_chunks,
        "verdict": None,
    }
    res = node(state)
    result = asyncio.run(res) if asyncio.iscoroutine(res) else res
    pool = result["pool"]

    # Capped at POOL_MAX (10)
    assert len(pool) == 10
    # Duplicate resolved
    ids = [c.id for c in pool]
    assert len(ids) == len(set(ids))


# ═════════════════════════════════════════════════════════════════
# 4. Routing, Gating, and Termination Tests
# ═════════════════════════════════════════════════════════════════

def test_score_gate_branches():
    cfg = Settings(HIGH_THRESHOLD=0.8, LOW_THRESHOLD=0.3)

    # High path
    s_high: GraphState = {
        "ranked": [Chunk.create("supabase", "top text", score=0.85)]
    }
    assert score_gate(s_high, settings=cfg) == "generate"

    # Mid path
    s_mid: GraphState = {
        "ranked": [Chunk.create("supabase", "mid text", score=0.55)]
    }
    assert score_gate(s_mid, settings=cfg) == "evaluator"

    # Low path
    s_low: GraphState = {
        "ranked": [Chunk.create("supabase", "low text", score=0.25)]
    }
    assert score_gate(s_low, settings=cfg) == "retry_check"

    # Empty ranked path -> low -> retry_check
    s_empty: GraphState = {"ranked": []}
    assert score_gate(s_empty, settings=cfg) == "retry_check"


def test_evaluator_gate_branches():
    assert evaluator_gate({"verdict": "correct"}) == "generate"
    assert evaluator_gate({"verdict": "ambiguous"}) == "retry_check"
    assert evaluator_gate({"verdict": "incorrect"}) == "retry_check"


def test_retry_check_gate_and_termination():
    cfg = Settings(MAX_RETRIES=2)

    # First retry (retries=1 <= 2) -> router
    assert retry_check_gate({"retries": 1}, settings=cfg) == "router"
    # Second retry (retries=2 <= 2) -> router
    assert retry_check_gate({"retries": 2}, settings=cfg) == "router"
    # Exceeded max (retries=3 > 2) -> fallback
    assert retry_check_gate({"retries": 3}, settings=cfg) == "fallback"


# ═════════════════════════════════════════════════════════════════
# 5. End-to-End Pipeline Tests with Mocks
# ═════════════════════════════════════════════════════════════════

def test_e2e_path_a_high_score_to_generate():
    """Path A: Initial retrieval gives top score >= HIGH -> straight to generate."""
    import asyncio
    cfg = Settings(HIGH_THRESHOLD=0.8, LOW_THRESHOLD=0.3, MAX_RETRIES=2)

    mock_llm = MockLLM(text_response="Answer generated with high confidence.")
    mock_reranker = MockReranker(scores=[0.95, 0.85])

    async def mock_supabase(q: str, k: int):
        return [
            {"text": "Supabase doc passage 1", "source": "supabase", "url": "https://supabase.com/docs/1", "title": "Doc 1"},
            {"text": "Supabase doc passage 2", "source": "supabase", "url": "https://supabase.com/docs/2", "title": "Doc 2"},
        ]

    tools = {"supabase": mock_supabase}

    ans: FinalAnswer = asyncio.run(run(
        question="What is the capital of France?",
        settings=cfg,
        llm=mock_llm,
        reranker=mock_reranker,
        tools=tools,
    ))

    assert not ans.low_confidence
    assert ans.answer == "Answer generated with high confidence."
    assert len(ans.citations) > 0
    # Reranker called with original question
    assert mock_reranker.calls[0][0] == "What is the capital of France?"


def test_e2e_path_b_mid_score_correct_evaluator_to_generate():
    """Path B: Score is mid (LOW < top < HIGH), Evaluator says correct -> generate."""
    import asyncio
    cfg = Settings(HIGH_THRESHOLD=0.8, LOW_THRESHOLD=0.3, MAX_RETRIES=2)

    mock_llm = MockLLM(
        text_response="Mid-score answer generated correctly.",
        structured_responses=[
            EvaluatorOutput(verdict="correct", missing="", likely_cause="none"),
        ],
    )
    mock_reranker = MockReranker(scores=[0.60, 0.50])

    async def mock_supabase(q: str, k: int):
        return [{"text": "Supabase relevant passage", "source": "supabase", "url": "https://supabase.com", "title": "KB"}]

    tools = {"supabase": mock_supabase}

    ans: FinalAnswer = asyncio.run(run(
        question="Explain gradient descent.",
        settings=cfg,
        llm=mock_llm,
        reranker=mock_reranker,
        tools=tools,
    ))

    assert not ans.low_confidence
    assert ans.answer == "Mid-score answer generated correctly."


def test_e2e_path_c_ambiguous_then_correct_on_retry():
    """Path C: First pass ambiguous, router escalates, second pass correct -> generate."""
    import asyncio
    cfg = Settings(HIGH_THRESHOLD=0.8, LOW_THRESHOLD=0.3, KEEP_THRESHOLD=0.4, MAX_RETRIES=2)

    mock_llm = MockLLM(
        text_response="Answer generated after successful retry.",
        structured_responses=[
            # Evaluator 1: ambiguous
            EvaluatorOutput(verdict="ambiguous", missing="Details about Bellman operator", likely_cause="not_in_source"),
            # Router retry 1
            RouterOutput(sources=["arxiv"], search_query="Distributional RL Bellman operator"),
            # Evaluator 2: correct
            EvaluatorOutput(verdict="correct", missing="", likely_cause="none"),
        ],
    )

    # Reranker returns 0.50 on pass 1 (mid score), 0.65 on pass 2 (mid score)
    mock_reranker = MockReranker(scores=[0.50, 0.65, 0.55])

    async def mock_supabase(q: str, k: int):
        return [{"text": "Intro to RL", "source": "supabase", "url": "https://supabase.com", "title": "RL Intro"}]

    async def mock_arxiv(q: str, k: int):
        return [{"text": "A Distributional Perspective on RL (Bellemare et al.)", "source": "arxiv", "url": "https://arxiv.org/abs/1707.06887", "title": "Distributional RL"}]

    tools = {
        "supabase": mock_supabase,
        "arxiv": mock_arxiv,
    }

    ans: FinalAnswer = asyncio.run(run(
        question="Explain distributional reinforcement learning and the Bellman operator.",
        settings=cfg,
        llm=mock_llm,
        reranker=mock_reranker,
        tools=tools,
    ))

    assert not ans.low_confidence
    assert ans.answer == "Answer generated after successful retry."
    # Check that reranker was always called with original_q
    for call_query, _ in mock_reranker.calls:
        assert call_query == "Explain distributional reinforcement learning and the Bellman operator."


def test_e2e_path_d_always_incorrect_fallback_after_two_retries():
    """Path D: Both retries fail/incorrect -> fallback with low_confidence=True after exactly 2 retries."""
    import asyncio
    cfg = Settings(HIGH_THRESHOLD=0.8, LOW_THRESHOLD=0.3, MAX_RETRIES=2)

    mock_llm = MockLLM(
        text_response="Fallback answer summarizing partial knowledge.",
        structured_responses=[
            # Evaluator 1: incorrect
            EvaluatorOutput(verdict="incorrect", missing="Everything", likely_cause="not_in_source"),
            # Router 1
            RouterOutput(sources=["wikipedia"], search_query="Obscure Topic"),
            # Evaluator 2: incorrect
            EvaluatorOutput(verdict="incorrect", missing="Still missing", likely_cause="not_in_source"),
            # Router 2
            RouterOutput(sources=["web"], search_query="Obscure Topic Recent"),
            # Evaluator 3: incorrect
            EvaluatorOutput(verdict="incorrect", missing="Cannot find authoritative source", likely_cause="not_in_source"),
        ],
    )

    # Mid scores so it hits evaluator each time
    mock_reranker = MockReranker(scores=[0.45])

    async def dummy_retriever(q: str, k: int):
        return [{"text": "Fragmentary snippet", "source": "web", "url": "https://web.org", "title": "Snippet"}]

    tools = {
        "supabase": dummy_retriever,
        "wikipedia": dummy_retriever,
        "arxiv": dummy_retriever,
        "web": dummy_retriever,
    }

    ans: FinalAnswer = asyncio.run(run(
        question="What is the quantum state of non-existent entity 404?",
        settings=cfg,
        llm=mock_llm,
        reranker=mock_reranker,
        tools=tools,
    ))

    assert ans.low_confidence is True
    assert "Fallback answer" in ans.answer


def test_reranker_always_called_with_original_query():
    """Confirm that the reranker is NEVER called with a rewritten search query."""
    import asyncio
    cfg = Settings(HIGH_THRESHOLD=0.8, LOW_THRESHOLD=0.3, MAX_RETRIES=2)

    original_question = "What is the formulation of categorical DQN?"
    rewritten_query = "categorical DQN distribution update"

    mock_llm = MockLLM(
        text_response="Answer with citations",
        structured_responses=[
            EvaluatorOutput(verdict="ambiguous", missing="categorical projection", likely_cause="bad_query"),
            RouterOutput(sources=["supabase"], search_query=rewritten_query),
            EvaluatorOutput(verdict="correct", missing="", likely_cause="none"),
        ],
    )

    mock_reranker = MockReranker(scores=[0.5, 0.6])

    async def mock_supabase(q: str, k: int):
        return [{"text": "Categorical DQN doc", "source": "supabase", "url": "https://example.com", "title": "C51"}]

    tools = {"supabase": mock_supabase}

    asyncio.run(run(
        question=original_question,
        settings=cfg,
        llm=mock_llm,
        reranker=mock_reranker,
        tools=tools,
    ))

    # Verify all reranker calls used original_question and none used rewritten_query
    for query, _ in mock_reranker.calls:
        assert query == original_question
        assert query != rewritten_query


def test_mermaid_graph_edges():
    """Verify that graph.get_graph().draw_mermaid() reflects the required spec topology."""
    from src.rag.crag_graph import graph
    mermaid = graph.get_graph().draw_mermaid()

    # Core nodes present
    expected_nodes = [
        "triage", "reply_chitchat", "reply_refusal",
        "init_state", "router", "retrieve", "normalize", "merge_pool",
        "rerank", "evaluator", "retry_check", "generate", "fallback"
    ]
    for node in expected_nodes:
        assert node in mermaid, f"Node {node} missing from mermaid diagram"

    # Triage entry and branching
    assert "__start__ --> triage" in mermaid
    assert "triage -.-> reply_chitchat" in mermaid
    assert "triage -.-> reply_refusal" in mermaid
    assert "triage -.-> init_state" in mermaid
    assert "reply_chitchat --> __end__" in mermaid
    assert "reply_refusal --> __end__" in mermaid

    # Sequential edges
    assert "init_state --> router" in mermaid
    assert "router --> retrieve" in mermaid
    assert "retrieve --> normalize" in mermaid
    assert "normalize --> merge_pool" in mermaid
    assert "merge_pool --> rerank" in mermaid

    # Score gate branching
    assert "rerank -.-> generate" in mermaid
    assert "rerank -.-> evaluator" in mermaid
    assert "rerank -.-> retry_check" in mermaid

    # Evaluator branching
    assert "evaluator -.-> generate" in mermaid
    assert "evaluator -.-> retry_check" in mermaid

    # Retry check branching
    assert "retry_check -.-> router" in mermaid
    assert "retry_check -.-> fallback" in mermaid

    # Terminations
    assert "generate --> __end__" in mermaid
    assert "fallback --> __end__" in mermaid


def test_crag_tracing_observability():
    """Verify that every node in CRAG records latency, inputs/outputs, and LLM call spans."""
    from src.rag.crag import run_crag

    node_logs = []
    llm_logs = []
    rerank_logs = []
    retrieval_logs = []

    class MockTracer:
        def __init__(self):
            self.run_id = "mock-trace-run-456"

        def log_node(self, name, run_type="chain", inputs=None, outputs=None, latency_ms=None, error=None):
            node_logs.append({
                "name": name,
                "run_type": run_type,
                "inputs": inputs,
                "outputs": outputs,
                "latency_ms": latency_ms,
            })

        def log_llm_call(self, model, prompt_text, model_settings, usage=None, ttft_ms=None, is_fallback=False, role=None, latency_ms=None, input_chunks=None, output_text=None):
            llm_logs.append({
                "model": model,
                "role": role,
                "latency_ms": latency_ms,
                "input_chunks": input_chunks,
                "output_text": output_text,
            })

        def log_reranker(self, query, num_candidates, top_k, latency_ms, scores=None, model=None):
            rerank_logs.append({
                "query": query,
                "num_candidates": num_candidates,
                "top_k": top_k,
                "latency_ms": latency_ms,
                "scores": scores,
                "model": model,
            })

        def log_retrieval(self, query, chunks, scores, tool_chosen):
            retrieval_logs.append({
                "query": query,
                "chunks": chunks,
                "tool_chosen": tool_chosen,
            })

        def finish(self, answer, used_fallback=False):
            pass

        def finish_error(self, error):
            pass

    tracer = MockTracer()
    cfg = Settings(
        HIGH_THRESHOLD=0.7,
        LOW_THRESHOLD=0.3,
        FINAL_CONTEXT_N=4,
        CARRY_MAX=2,
    )
    mock_llm = MockLLM(text_response="Answer with LangSmith tracing verification.")
    mock_reranker = MockReranker(scores=[0.9, 0.8])

    async def mock_supabase(q: str, k: int):
        return [
            {"text": "Observed passage 1", "source": "supabase", "url": "https://supabase.com/1", "title": "Doc 1"},
            {"text": "Observed passage 2", "source": "supabase", "url": "https://supabase.com/2", "title": "Doc 2"},
        ]

    tools = {"supabase": mock_supabase}

    ans = asyncio.run(run_crag(
        question="Explain observability in CRAG pipelines",
        settings=cfg,
        llm=mock_llm,
        reranker=mock_reranker,
        tools=tools,
        tracer=tracer,  # type: ignore
    ))

    assert ans.run_id == "mock-trace-run-456"
    assert ans.answer == "Answer with LangSmith tracing verification."

    # 1. Verify every node logged its execution and latency
    logged_node_names = [log["name"] for log in node_logs]
    assert "node/init_state" in logged_node_names
    assert "node/router" in logged_node_names
    assert "node/retrieve" in logged_node_names
    assert "node/normalize" in logged_node_names
    assert "node/merge_pool" in logged_node_names
    assert "node/rerank" in logged_node_names
    assert "node/generate" in logged_node_names

    for log in node_logs:
        assert log["latency_ms"] is not None
        assert log["latency_ms"] >= 0.0

    # 2. Verify LLM call logged for generator with input chunks and output
    assert len(llm_logs) >= 1
    gen_llm = next(l for l in llm_logs if l["role"] == "primary")
    assert gen_llm["latency_ms"] is not None
    assert gen_llm["input_chunks"] is not None
    assert len(gen_llm["input_chunks"]) == 2
    assert "Observed passage 1" in gen_llm["input_chunks"][0].text
    assert gen_llm["output_text"] == "Answer with LangSmith tracing verification."

    # 3. Verify retrieval and reranker logged
    assert len(retrieval_logs) == 1
    assert len(retrieval_logs[0]["chunks"]) == 2
    assert len(rerank_logs) == 1
    assert rerank_logs[0]["num_candidates"] == 2
    assert rerank_logs[0]["top_k"] == 4
    assert rerank_logs[0]["scores"] == [0.9, 0.8]


def test_crag_tracing_evaluator_and_retry_nodes():
    """Verify that evaluator and retry nodes log spans, input chunks, and latency during retry."""
    from src.rag.crag import run_crag
    from src.rag.schemas import EvaluatorOutput, RouterOutput

    node_logs = []
    llm_logs = []

    class MockTracer:
        def __init__(self):
            self.run_id = "mock-trace-evaluator-789"

        def log_node(self, name, run_type="chain", inputs=None, outputs=None, latency_ms=None, error=None):
            node_logs.append({"name": name, "inputs": inputs, "outputs": outputs, "latency_ms": latency_ms})

        def log_llm_call(self, model, prompt_text, model_settings, usage=None, ttft_ms=None, is_fallback=False, role=None, latency_ms=None, input_chunks=None, output_text=None):
            llm_logs.append({"role": role, "latency_ms": latency_ms, "input_chunks": input_chunks, "output_text": output_text})

        def log_reranker(self, *args, **kwargs):
            pass

        def log_retrieval(self, *args, **kwargs):
            pass

        def finish(self, answer, used_fallback=False):
            pass

        def finish_error(self, error):
            pass

    tracer = MockTracer()
    cfg = Settings(
        HIGH_THRESHOLD=0.7,
        LOW_THRESHOLD=0.3,
        FINAL_CONTEXT_N=4,
        CARRY_MAX=2,
    )

    mock_llm = MockLLM(
        text_response="Synthesized answer after evaluator retry.",
        structured_responses=[
            # Pass 0: Evaluator says ambiguous
            EvaluatorOutput(verdict="ambiguous", missing="specific proof", likely_cause="not_in_source"),
            # Pass 1: Router routes to wikipedia
            RouterOutput(sources=["wikipedia"], search_query="proof query"),
            # Pass 1: Evaluator says correct
            EvaluatorOutput(verdict="correct", missing="", likely_cause="none"),
        ],
    )
    # Pass 0: mid score (0.5), triggers evaluator
    # Pass 1: mid score (0.6), triggers evaluator
    mock_reranker = MockReranker(scores=[0.5, 0.4, 0.6, 0.5])

    async def mock_supabase(q: str, k: int):
        return [{"text": "Passage 1", "source": "supabase"}]

    async def mock_wikipedia(q: str, k: int):
        return [{"text": "Wikipedia Proof Passage", "source": "wikipedia"}]

    tools = {"supabase": mock_supabase, "wikipedia": mock_wikipedia}

    ans = asyncio.run(run_crag(
        question="What is the proof of Bellman convergence?",
        settings=cfg,
        llm=mock_llm,
        reranker=mock_reranker,
        tools=tools,
        tracer=tracer,  # type: ignore
    ))

    assert ans.run_id == "mock-trace-evaluator-789"
    assert ans.answer == "Synthesized answer after evaluator retry."

    logged_node_names = [log["name"] for log in node_logs]
    assert "node/evaluator" in logged_node_names
    assert "node/retry_check" in logged_node_names

    eval_llm_spans = [l for l in llm_logs if l["role"] == "evaluator"]
    assert len(eval_llm_spans) >= 1
    for span in eval_llm_spans:
        assert span["latency_ms"] is not None
        assert span["input_chunks"] is not None
        assert span["output_text"] is not None


def test_crag_streaming_tokens():
    """Verify that run_crag_stream streams response tokens."""
    import asyncio
    from src.rag.crag import run_crag_stream

    cfg = Settings(HIGH_THRESHOLD=0.8, LOW_THRESHOLD=0.3, MAX_RETRIES=2)
    mock_llm = MockLLM(text_response="Streamed token response.")
    mock_reranker = MockReranker(scores=[0.95, 0.85])

    async def mock_supabase(q: str, k: int):
        return [{"text": "Stream test document passage", "source": "supabase"}]

    tools = {"supabase": mock_supabase}

    async def collect_stream():
        collected = []
        async for token in run_crag_stream(
            question="What is the streaming response test?",
            settings=cfg,
            llm=mock_llm,
            reranker=mock_reranker,
            tools=tools,
        ):
            collected.append(token)
        return collected

    tokens = asyncio.run(collect_stream())
    assert len(tokens) >= 1
    assert "".join(tokens) == "Streamed token response."


def test_multi_tool_selection_pass_0():
    """Verify router can select multiple tools concurrently on pass 0 with tool_queries."""
    import asyncio
    from src.rag.crag_nodes import create_router_node, create_retrieve_node
    from src.rag.schemas import GraphState

    # Mock tool selector returning multiple tools
    called_tools = {}

    async def mock_supabase(q: str, k: int):
        called_tools["supabase"] = q
        return [{"text": "Supabase doc", "source": "supabase"}]

    async def mock_arxiv(q: str, k: int):
        called_tools["arxiv"] = q
        return [{"text": "ArXiv paper", "source": "arxiv"}]

    async def mock_web(q: str, k: int):
        called_tools["web"] = q
        return [{"text": "Web snippet", "source": "web"}]

    tools = {
        "supabase": mock_supabase,
        "arxiv": mock_arxiv,
        "web": mock_web,
    }

    retrieve_node = create_retrieve_node(tools=tools)

    state: GraphState = {
        "original_q": "What is the distributional Bellman operator?",
        "sources_to_use": ["supabase", "arxiv", "web"],
        "search_query": "What is the distributional Bellman operator?",
        "tool_queries": {
            "supabase": "What is the distributional Bellman operator?",
            "arxiv": "Distributional Bellman operator Wasserstein",
            "web": "Distributional Bellman operator QR-DQN",
        },
        "retries": 0,
        "tried_sources": [],
    }

    res = asyncio.run(retrieve_node(state))
    assert res["all_retrievers_failed"] is False
    assert len(res["raw_results"]) == 3
    # Check that each tool was called with its specific query
    assert called_tools["supabase"] == "What is the distributional Bellman operator?"
    assert called_tools["arxiv"] == "Distributional Bellman operator Wasserstein"
    assert called_tools["web"] == "Distributional Bellman operator QR-DQN"


def test_search_tool_robustness_with_fallbacks(monkeypatch):
    """Verify retrieve_wikipedia and retrieve_arxiv fall back cleanly when APIs fail."""
    import asyncio
    from src.rag.tools.wikipedia import retrieve_wikipedia
    from src.rag.tools.arxiv import retrieve_arxiv

    # Mock direct_ddg_search to return valid fallback content
    import src.rag.rag as rag_mod
    monkeypatch.setattr(rag_mod, "direct_ddg_search", lambda q, max_chars=2000: f"Fallback content for: {q}")

    wiki_res = asyncio.run(retrieve_wikipedia(r"Distributional RL $\mathcal{T}^\pi$"))
    assert len(wiki_res) >= 1
    assert wiki_res[0]["source"] == "wikipedia"

    arxiv_res = asyncio.run(retrieve_arxiv(r"Distributional Bellman Wasserstein $\mathcal{T}^\pi$"))
    assert len(arxiv_res) >= 1
    assert arxiv_res[0]["source"] == "arxiv"


def test_fallback_node_prompt_selection_by_score():
    """Verify fallback node uses normal generation prompt when top_score >= LOW_THRESHOLD, else fallback prompt."""
    import asyncio
    from src.rag.crag_nodes import create_fallback_node
    from src.rag.schemas import Chunk

    class RecordingLLM:
        def __init__(self):
            self.prompts: list[str] = []

        async def ainvoke(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return "Test LLM response"

    cfg = Settings(LOW_THRESHOLD=0.2, HIGH_THRESHOLD=0.7)

    # 1. Score is 0.45 (between LOW_THRESHOLD and HIGH_THRESHOLD) -> normal prompt
    llm1 = RecordingLLM()
    fallback_node1 = create_fallback_node(llm=llm1, settings=cfg)
    res1 = asyncio.run(fallback_node1({
        "original_q": "What is distributional RL?",
        "ranked": [Chunk(id="c1", text="Sample text", source="supabase", score=0.45)],
    }))
    assert len(llm1.prompts) == 1
    assert "<hybrid_grounding_rules>" in llm1.prompts[0]
    assert "<fallback_rules>" not in llm1.prompts[0]
    assert res1["low_confidence"] is True

    # 2. Score is 0.10 (< LOW_THRESHOLD) -> fallback prompt
    llm2 = RecordingLLM()
    fallback_node2 = create_fallback_node(llm=llm2, settings=cfg)
    res2 = asyncio.run(fallback_node2({
        "original_q": "What is distributional RL?",
        "ranked": [Chunk(id="c2", text="Low score snippet", source="web", score=0.10)],
    }))
    assert len(llm2.prompts) == 1
    assert "<fallback_rules>" in llm2.prompts[0]
    assert res2["low_confidence"] is True

    # 3. Happy path generate node uses strict GENERATE_SYSTEM_PROMPT
    from src.rag.crag_nodes import create_generate_node
    llm3 = RecordingLLM()
    gen_node = create_generate_node(llm=llm3)
    res3 = asyncio.run(gen_node({
        "original_q": "What is distributional RL?",
        "ranked": [Chunk(id="c3", text="High score snippet", source="arxiv", score=0.88)],
    }))
    assert len(llm3.prompts) == 1
    assert "<grounding_rules>" in llm3.prompts[0]
    assert "<hybrid_grounding_rules>" not in llm3.prompts[0]
    assert res3["low_confidence"] is False


def test_rerank_preserves_top_2_per_tool_chunks():
    """Verify top 2 chunks of each tool are preserved in context rather than favoring only Supabase."""
    import asyncio
    from src.rag.crag_nodes import create_rerank_node

    cfg = Settings(FINAL_CONTEXT_N=5)

    # 4 Supabase chunks with modest scores
    supabase_chunks = [
        Chunk(id=f"sub_{i}", text=f"material chunk {i}", source="supabase", score=0.30 + i * 0.05)
        for i in range(4)
    ]
    # 6 Wikipedia chunks with higher scores
    wiki_chunks = [
        Chunk(id=f"wiki_{i}", text=f"wiki chunk {i}", source="wikipedia", score=0.70 + i * 0.02)
        for i in range(6)
    ]
    pool = supabase_chunks + wiki_chunks

    rerank_node = create_rerank_node(settings=cfg)
    state: GraphState = {"original_q": "test query", "pool": pool}
    res = asyncio.run(rerank_node(state))
    ranked = res["ranked"]

    assert len(ranked) == 5
    supabase_in_ranked = [c for c in ranked if c.source == "supabase"]
    assert len(supabase_in_ranked) == 2
    # The top 2 Supabase chunks (sub_3: 0.45, sub_2: 0.40) are preserved
    assert {c.id for c in supabase_in_ranked} == {"sub_3", "sub_2"}


def test_score_gate_routes_0_800_rounded_to_generate():
    """Verify that scores rounding to 0.800 (such as 0.7997) route directly to generate."""
    from src.rag.crag_nodes import score_gate

    cfg = Settings(HIGH_THRESHOLD=0.8, LOW_THRESHOLD=0.2)
    # 0.7997 formats as 0.800
    state = {
        "ranked": [Chunk(id="c1", text="text", source="supabase", score=0.7997)],
    }
    decision = score_gate(state, settings=cfg)
    assert decision == "generate"


# ─────────────────────────────────────────────────────────────
# Phase 1: Triage Node Tests
# ─────────────────────────────────────────────────────────────

def test_tier0_classification():
    from src.rag.crag_nodes import tier0

    # Chitchat
    assert tier0("hey") == "chitchat"
    assert tier0("Hello!") == "chitchat"
    assert tier0("thanks") == "chitchat"
    assert tier0("مرحبا") == "chitchat"
    assert tier0("شكرا") == "chitchat"
    assert tier0("") == "chitchat"
    assert tier0("a") == "chitchat"

    # Gibberish
    assert tier0("asdfgh") == "unsafe_gibberish"
    assert tier0("qwerty") == "unsafe_gibberish"
    assert tier0("aaaaaaa") == "unsafe_gibberish"

    # CS terms / questions -> None
    assert tier0("SQL") is None
    assert tier0("RNN") is None
    assert tier0("nltk") is None
    assert tier0("what is a pointer") is None


def test_triage_output_validators():
    import pytest
    from pydantic import ValidationError
    from src.rag.schemas import TriageOutput

    # 1. Unsafe with unsafe_type "none" rejected
    with pytest.raises(ValidationError):
        TriageOutput(intent="unsafe", unsafe_type="none", sub_questions=[])

    # 2. Complex with empty sub_questions rejected
    with pytest.raises(ValidationError):
        TriageOutput(intent="complex", unsafe_type="none", sub_questions=[])

    # 3. Simple with non-empty sub_questions rejected
    with pytest.raises(ValidationError):
        TriageOutput(intent="simple", unsafe_type="none", sub_questions=["what is a pointer"])

    # 4. 4 sub_questions rejected
    with pytest.raises(ValidationError):
        TriageOutput(intent="complex", unsafe_type="none", sub_questions=["sub 1", "sub 2", "sub 3", "sub 4"])

    # Valid instances
    t_simple = TriageOutput(intent="simple", unsafe_type="none", sub_questions=[])
    assert t_simple.intent == "simple"
    t_complex = TriageOutput(intent="complex", unsafe_type="none", sub_questions=["sub query 1", "sub query 2"])
    assert len(t_complex.sub_questions) == 2
    t_unsafe = TriageOutput(intent="unsafe", unsafe_type="nsfw", sub_questions=[])
    assert t_unsafe.unsafe_type == "nsfw"


def test_triage_tier1_failure_handling():
    import asyncio
    from src.rag.crag_nodes import create_triage_node
    from src.rag.config import Settings

    cfg = Settings(TRIAGE_TIMEOUT_S=1.0)

    class FailingLLM:
        def __init__(self):
            self.calls = 0

        async def ainvoke_structured(self, prompt: str, schema):
            self.calls += 1
            raise ValueError("Parse failed")

    # Two parse failures -> complex with [original_q]
    failing_llm = FailingLLM()
    triage_node = create_triage_node(llm=failing_llm, settings=cfg)
    res = asyncio.run(triage_node({"original_q": "what is a pointer"}))
    assert res["intent"] == "complex"
    assert res["sub_questions"] == ["what is a pointer"]
    assert failing_llm.calls == 2

    # Timeout -> same
    class TimeoutLLM:
        async def ainvoke_structured(self, prompt: str, schema):
            await asyncio.sleep(1.2)
            raise RuntimeError("Should have timed out")

    timeout_llm = TimeoutLLM()
    triage_node_timeout = create_triage_node(llm=timeout_llm, settings=cfg)
    res_timeout = asyncio.run(triage_node_timeout({"original_q": "what is a pointer"}))
    assert res_timeout["intent"] == "complex"
    assert res_timeout["sub_questions"] == ["what is a pointer"]


def test_triage_routing_chitchat_and_unsafe_zero_downstream_calls():
    import asyncio
    from src.rag.crag import run_crag

    calls = {"retrieve": 0, "llm": 0}

    def mock_retriever(q, k):
        calls["retrieve"] += 1
        return []

    tools = {
        "supabase": mock_retriever,
        "wikipedia": mock_retriever,
        "arxiv": mock_retriever,
        "web": mock_retriever,
    }

    class MockLLMForTest:
        async def ainvoke(self, prompt: str):
            calls["llm"] += 1
            return "Answer"

        async def ainvoke_structured(self, prompt: str, schema):
            calls["llm"] += 1
            raise RuntimeError("Should not be called for tier0")

    # 1. "hey" -> chitchat canned reply, 0 tool/downstream calls
    res_chitchat = asyncio.run(run_crag("hey", tools=tools, llm=MockLLMForTest()))
    assert res_chitchat.low_confidence is False
    assert "Study Buddy" in res_chitchat.answer
    assert calls["retrieve"] == 0
    assert calls["llm"] == 0

    # 2. "asdfgh" -> unsafe canned reply, 0 tool/downstream calls
    res_unsafe = asyncio.run(run_crag("asdfgh", tools=tools, llm=MockLLMForTest()))
    assert res_unsafe.low_confidence is False
    assert res_unsafe.answer == "I can't respond on a gibberish topic."
    assert calls["retrieve"] == 0
    assert calls["llm"] == 0


def test_canned_replies_arabic_and_exact_unsafe_strings():
    import asyncio
    from src.rag.crag import run_crag
    from src.rag.constants import UNSAFE_REPLIES

    # Arabic greeting gets Arabic reply
    res_ar = asyncio.run(run_crag("مرحبا"))
    assert "أهلاً!" in res_ar.answer or "Study Buddy" in res_ar.answer

    # Exact unsafe strings match
    assert UNSAFE_REPLIES["gibberish"] == "I can't respond on a gibberish topic."
    assert UNSAFE_REPLIES["nsfw"] == "I can't respond on a NSFW topic."
    assert UNSAFE_REPLIES["political"] == "I can't respond on a political topic."
    assert UNSAFE_REPLIES["religious"] == "I can't respond on a religious topic."


def test_end_to_end_hey_completes_instantly():
    import asyncio
    import time
    from src.rag.crag import run_crag

    t0 = time.perf_counter()
    res = asyncio.run(run_crag("hey"))
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0
    assert res.citations == []
    assert res.low_confidence is False
    assert "Study Buddy" in res.answer


def test_parse_triage_lines_key_value_format():
    from src.rag.crag_llm import LangChainLLM
    from src.rag.schemas import TriageOutput

    # 1. Complex example matching user request:
    text_complex = """
    intent: Complex
    unsafe_type: None
    sub_questions:
    - what is Wasserstein distance
    - why does Bellman operator contract
    """
    parsed = LangChainLLM._parse_triage_lines(text_complex)
    assert parsed is not None
    assert parsed["intent"] == "complex"
    assert parsed["unsafe_type"] == "none"
    assert parsed["sub_questions"] == [
        "what is Wasserstein distance",
        "why does Bellman operator contract",
    ]
    out = TriageOutput.model_validate(parsed)
    assert out.intent == "complex"

    # 2. Simple example:
    text_simple = """
    intent: Simple
    unsafe_type: None
    sub_questions: none
    """
    parsed_s = LangChainLLM._parse_triage_lines(text_simple)
    assert parsed_s is not None
    assert parsed_s["intent"] == "simple"
    assert parsed_s["unsafe_type"] == "none"
    assert parsed_s["sub_questions"] == []
    out_s = TriageOutput.model_validate(parsed_s)
    assert out_s.intent == "simple"

    # 3. Unsafe example:
    text_unsafe = """
    intent: Unsafe
    unsafe_type: Political
    """
    parsed_u = LangChainLLM._parse_triage_lines(text_unsafe)
    assert parsed_u is not None
    assert parsed_u["intent"] == "unsafe"
    assert parsed_u["unsafe_type"] == "political"
    assert parsed_u["sub_questions"] == []
    out_u = TriageOutput.model_validate(parsed_u)
    assert out_u.intent == "unsafe"

    # 4. Complex example without sub_questions (fallback):
    text_complex_nosub = """
    intent: Complex
    unsafe_type: None
    """
    parsed_cn = LangChainLLM._parse_triage_lines(text_complex_nosub)
    assert parsed_cn is not None
    assert parsed_cn["intent"] == "complex"
    out_cn = TriageOutput.model_validate(parsed_cn)
    assert out_cn.intent == "complex"


def test_langchain_llm_triage_line_based_parsing():
    import asyncio
    from src.rag.crag_llm import LangChainLLM
    from src.rag.schemas import TriageOutput

    class LineModel:
        def invoke(self, prompt: str):
            return "intent: Complex\nunsafe_type: None\nsub_questions:\n- query 1\n- query 2"

        async def ainvoke(self, prompt: str):
            class Resp:
                content = "intent: Complex\nunsafe_type: None\nsub_questions:\n- query 1\n- query 2"
            return Resp()

    llm = LangChainLLM(chat_model=LineModel())
    res = asyncio.run(llm.ainvoke_structured("prompt", TriageOutput))
    assert isinstance(res, TriageOutput)
    assert res.intent == "complex"
    assert res.unsafe_type == "none"
    assert res.sub_questions == ["query 1", "query 2"]


def test_retrieval_timeout_dynamic_by_intent():
    """Verify tool timeout dynamically switches: 7s for simple queries, 13s for complex queries."""
    from src.rag.crag_nodes import create_retrieve_node
    from src.rag.config import Settings

    cfg = Settings(TOOL_TIMEOUT_SIMPLE_S=7.0, TOOL_TIMEOUT_COMPLEX_S=13.0)

    async def mock_retriever(q: str, k: int):
        return [{"text": "result", "source": "web"}]

    node = create_retrieve_node(tools={"web": mock_retriever}, settings=cfg)

    # 1. Simple query intent
    simple_state: GraphState = {
        "original_q": "What is Python?",
        "sources_to_use": ["web"],
        "intent": "simple",
    }
    res_simple = asyncio.run(node(simple_state))
    assert res_simple["all_retrievers_failed"] is False
    assert len(res_simple["raw_results"]) == 1

    # 2. Complex query intent
    complex_state: GraphState = {
        "original_q": "Explain QR-DQN vs C51 Bellman contractions",
        "sources_to_use": ["web"],
        "intent": "complex",
    }
    res_complex = asyncio.run(node(complex_state))
    assert res_complex["all_retrievers_failed"] is False
    assert len(res_complex["raw_results"]) == 1


def test_simple_query_score_gate_and_retry_all_tools():
    """
    Verify simple query rules:
    1. Between low & high threshold (and above high threshold) -> generate (bypasses evaluator).
    2. Below low threshold -> retry_check.
    3. On retry -> router uses ALL tools and produces keyword-style search query.
    """
    from src.rag.crag_nodes import score_gate, create_router_node
    from src.rag.schemas import Chunk
    from src.rag.config import Settings

    cfg = Settings(HIGH_THRESHOLD=0.8, LOW_THRESHOLD=0.3)

    # 1. Simple query: mid score (0.55) -> generate directly! (Bypasses evaluator)
    state_simple_mid: GraphState = {
        "intent": "simple",
        "ranked": [Chunk.create("supabase", "mid relevance", score=0.55)],
    }
    assert score_gate(state_simple_mid, settings=cfg) == "generate"

    # 2. Simple query: high score (0.85) -> generate
    state_simple_high: GraphState = {
        "intent": "simple",
        "ranked": [Chunk.create("supabase", "high relevance", score=0.85)],
    }
    assert score_gate(state_simple_high, settings=cfg) == "generate"

    # 3. Simple query: low score (0.20) -> retry_check
    state_simple_low: GraphState = {
        "intent": "simple",
        "ranked": [Chunk.create("supabase", "low relevance", score=0.20)],
    }
    assert score_gate(state_simple_low, settings=cfg) == "retry_check"

    # 4. Complex query: mid score (0.55) must still go to evaluator!
    state_complex_mid: GraphState = {
        "intent": "complex",
        "ranked": [Chunk.create("supabase", "mid relevance", score=0.55)],
    }
    assert score_gate(state_complex_mid, settings=cfg) == "evaluator"

    # 5. Simple query router on retry (pass 1):
    # Must use ALL tools ["supabase", "arxiv", "wikipedia", "web"]
    class MockLLM:
        async def ainvoke_structured(self, prompt, schema):
            from src.rag.schemas import RouterOutput
            return RouterOutput(search_query="photosynthesis light dependent reactions")

    router_node = create_router_node(llm=MockLLM(), settings=cfg)
    retry_state: GraphState = {
        "original_q": "Can you explain how photosynthesis works in plants?",
        "intent": "simple",
        "retries": 1,
        "tried_sources": ["supabase"],
        "sources_to_use": ["supabase"],
        "missing": "Re-rank score was below threshold",
    }
    retry_res = asyncio.run(router_node(retry_state))
    assert retry_res["sources_to_use"] == ["supabase", "arxiv", "wikipedia", "web"]
    assert retry_res["search_query"] == "photosynthesis light dependent reactions"
    assert all(src in retry_res["tool_queries"] for src in ["supabase", "arxiv", "wikipedia", "web"])






