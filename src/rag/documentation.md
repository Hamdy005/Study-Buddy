# Corrective RAG (CRAG) Pipeline Documentation

This document outlines the architecture, execution workflow, node contracts, and core configuration settings for the Corrective RAG (`CRAG`) engine in Study Buddy.

---

## 1. Pipeline Graph

<div align="center">
  <a href="crag_graph.png" target="_blank" title="Click to view in full resolution">
    <img width="2717" height="4096" alt="crag_graph" src="https://github.com/user-attachments/assets/b938e601-bba1-4618-9b10-248157df44d2" />
  </a>
  <br>
  <sub><em>(Click graph to open full resolution)</em></sub>
</div>

<details>
<summary>View raw Mermaid specification</summary>

```mermaid
graph TD;
	__start__([__start__])
	triage(triage)
	reply_chitchat(reply_chitchat)
	reply_refusal(reply_refusal)
	init_state(init_state)
	router(router)
	retrieve(retrieve)
	normalize(normalize)
	merge_pool(merge_pool)
	rerank(rerank)
	evaluator(evaluator)
	retry_check(retry_check)
	generate(generate)
	fallback(fallback)
	__end__([__end__])

	__start__ --> triage;
	triage -.->|chitchat| reply_chitchat;
	triage -.->|unsafe| reply_refusal;
	triage -.->|study query| init_state;
	reply_chitchat --> __end__;
	reply_refusal --> __end__;
	init_state --> router;
	router --> retrieve;
	retrieve --> normalize;
	normalize --> merge_pool;
	merge_pool --> rerank;
	rerank -.->|top_score >= 0.65| generate;
	rerank -.->|0.30 < score < 0.65| evaluator;
	rerank -.->|top_score <= 0.30| retry_check;
	evaluator -.->|correct| generate;
	evaluator -.->|ambiguous / incorrect| retry_check;
	retry_check -.->|retries <= 2| router;
	retry_check -.->|retries > 2| fallback;
	generate --> __end__;
	fallback --> __end__;
```

</details>

---

## 2. End-to-End Workflow

The pipeline processes user queries through three distinct pathways:

1. **Zero-Retrieval Fast Paths (Bypass)**:
   - Evaluated at `triage` via Tier 0 (regex/rules, `<1ms`) and Tier 1 (fast LLM classification, line-based output).
   - If `chitchat`: routes directly to `reply_chitchat` with localized canned greetings/thanks and exits.
   - If `unsafe`: routes directly to `reply_refusal` with policy refusals (gibberish, NSFW, politics, religion) and exits.

2. **Standard RAG Pipeline (Happy Path)**:
   - Initialized at `init_state` (passes `intent` and sub-queries, sets `retries=0`).
   - `router` selects search sources and formulates query plans (defaults to curated Supabase materials on pass 0).
   - `retrieve` executes external tool queries (Supabase vector store, Wikipedia, ArXiv, Web).
   - `normalize` cleans text into structured `Chunk` objects and computes SHA-256 content IDs.
   - `merge_pool` deduplicates passages and merges them into the cumulative pool.
   - If top reranker score $\ge 0.65$ (`HIGH_THRESHOLD`) on simple queries, `score_gate` short-circuits directly to `generate` (skipping the evaluator LLM call).
   - `generate` drafts an educational answer strictly cited with chunk numbers `[n]`.

3. **Evaluation & Corrective Feedback Loop**:
   - For simple queries with score between $0.30$ and $0.65$, or complex queries with score $> 0.30$, `evaluator` inspects relevance and coverage (`correct`, `ambiguous`, `incorrect`).
   - If rerank score is $\le 0.30$ (`LOW_THRESHOLD`), `score_gate` routes directly to `retry_check`.
   - If evaluated as `correct`, proceeds to `generate`.
   - If `ambiguous` or `incorrect`, `retry_check` tracks attempt count.
   - If `retries <= MAX_RETRIES (2)`, loops back to `router` to re-query (`bad_query` rewrites query targeting `missing`; sources remain fixed with no tool escalation).
   - If retries are exhausted, routes to `fallback` (synthesizes hybrid response or general-knowledge answer with `low_confidence=True`).

---

## 3. Node Specifications & Outputs

### 1. `triage`
- **Purpose**: Classifies user input into intent (`chitchat`, `unsafe`, `simple`, `complex`) and decomposes complex queries into 1–3 sub-queries. Runs Tier 0 regex heuristics first, falling back to a structured line-based LLM call.
- **Output fields**:
  - `intent`: `"chitchat" | "unsafe" | "simple" | "complex"`
  - `sub_questions`: `list[str]` (non-empty only for complex)
  - `reply`: canned string for chitchat/unsafe, otherwise `None`

### 2. `reply_chitchat`
- **Purpose**: Emits immediate conversational response in English or Arabic without touching vector stores or tools.
- **Output fields**:
  - `answer`: localized greeting/thanks message
  - `citations`: `[]`
  - `low_confidence`: `False`

### 3. `reply_refusal`
- **Purpose**: Emits standardized safety refusal for out-of-scope queries (gibberish, adult, politics, religion).
- **Output fields**:
  - `answer`: standardized refusal string
  - `citations`: `[]`
  - `low_confidence`: `False`

### 4. `init_state`
- **Purpose**: Initializes graph state variables for legitimate study queries.
- **Output fields**:
  - `original_q`: validated question string
  - `sources_to_use`: `["supabase"]` (default first pass)
  - `retries`: `0`, `pool`: `[]`, `ranked`: `[]`, `verdict`: `None`

### 5. `router`
- **Purpose**: Selects target retrieval sources and constructs retrieval queries.
  - **Pass 0**: If triage `intent == "complex"`, queries **all tools concurrently** (`supabase`, `arxiv`, `wikipedia`, `web`) with zero LLM delay. If `intent == "simple"`, runs **automatic tool calling** via LangChain agentic tool selection.
  - **Retries**: Keeps existing sources fixed (no adding tools / no tool escalation); strictly performs **requerying** by rewriting queries targeting evaluator `missing` feedback.
- **Output fields**:
  - `sources_to_use`: list of sources (`["supabase", "arxiv", "wikipedia", "web"]`)
  - `search_query`: optimized search keywords
  - `tool_queries`: mapping of source to specific search query

### 6. `retrieve`
- **Purpose**: Calls registered tool functions concurrently across designated sources with intent- and retry-aware dynamic timeouts:
  - **Base timeout**: `7.0s` (`TOOL_TIMEOUT_SIMPLE_S`) for simple queries, `13.0s` (`TOOL_TIMEOUT_COMPLEX_S`) for complex queries.
  - **Progressive timeout escalation for empty context (`context == []`)**:
    - `retries == 1`: tool timeout escalates to `16.0s` (`TOOL_TIMEOUT_RETRY1_EMPTY_S`).
    - `retries == 2`: tool timeout escalates to `25.0s` (`TOOL_TIMEOUT_RETRY2_EMPTY_S`).
- **Output fields**:
  - `raw_results`: raw documents/payloads returned by tool APIs
  - `all_retrievers_failed`: boolean indicating whether all tool retrievers failed/timed out

### 7. `normalize`
- **Purpose**: Parses varied tool response formats, splits text into target chunk lengths, and computes deterministic content hashes.
- **Output fields**:
  - `new_chunks`: list of validated `Chunk` instances
  - `tried_sources`: updated cumulative list of queried sources

### 8. `merge_pool`
- **Purpose**: Accumulates incoming chunks into the existing pool, preserves scores, and ensures top-tier document slots are retained without unbounded growth.
- **Output fields**:
  - `pool`: deduped, bounded chunk list (capped at `POOL_MAX`)

### 9. `rerank`
- **Purpose**: Scores chunks against `original_q` using a cross-encoder model. Retains top Supabase chunks to preserve course material grounding.
- **Output fields**:
  - `ranked`: top candidate chunks sorted by descending score (length `FINAL_CONTEXT_N`)

### 10. `evaluator`
- **Purpose**: Fast LLM judge checking context relevance and completeness against the question. Emits line-based key-values (`VERDICT`, `CAUSE`, `MISSING`). The only cause for retrying is `bad_query` (requerying).
- **Output fields**:
  - `verdict`: `"correct" | "ambiguous" | "incorrect"`
  - `likely_cause`: `"none" | "bad_query"`
  - `missing`: description of missing factual concepts

### 11. `retry_check`
- **Purpose**: Increments retry counter and records diagnostic logs.
- **Output fields**:
  - `retries`: updated integer counter (`retries + 1`)

### 12. `generate`
- **Purpose**: Synthesizes the final educational answer strictly grounded in ranked context chunks cleanly without bracketed numbers (`[1]`, `[2]`).
- **Output fields**:
  - `answer`: synthesized text
  - `citations`: list of cited source titles/URLs
  - `low_confidence`: `False`

### 13. `fallback`
- **Purpose**: Failsafe generator triggered when retries are exhausted or retrieval fails. Generates hybrid context-assisted or knowledge-based explanations.
- **Output fields**:
  - `answer`: fallback educational text
  - `citations`: list of citations (if partial context was used)
  - `low_confidence`: `True`

---

## 4. Conditional Edge Gates

- **`triage_gate`**:
  - `intent == "chitchat"` $\rightarrow$ `reply_chitchat`
  - `intent == "unsafe"` $\rightarrow$ `reply_refusal`
  - default $\rightarrow$ `init_state`

- **`score_gate`**:
  - **For Simple Queries (`intent == "simple"`)**:
    - `top_score >= HIGH_THRESHOLD (0.65)` $\rightarrow$ `generate` (high confidence single-concept shortcut, skips evaluator).
    - `LOW_THRESHOLD (0.30) < top_score < HIGH_THRESHOLD (0.65)` $\rightarrow$ `evaluator` (ambiguous zone, LLM verifies relevance).
    - `top_score <= LOW_THRESHOLD (0.30)` $\rightarrow$ `retry_check` $\rightarrow$ retries with **all tools** (`supabase`, `arxiv`, `wikipedia`, `web`) using keyword-style search queries.
  - **For Complex Queries (`intent == "complex"`)**:
    - `top_score <= LOW_THRESHOLD (0.30)` $\rightarrow$ `retry_check` (immediate retry, context too weak).
    - `top_score > LOW_THRESHOLD (0.30)` $\rightarrow$ `evaluator` (ALWAYS verifies that ALL sub-parts are satisfied).

- **`evaluator_gate`**:
  - `verdict == "correct"` $\rightarrow$ `generate`
  - `verdict in {"ambiguous", "incorrect"}` $\rightarrow$ `retry_check`

- **`retry_check_gate`**:
  - `retries <= MAX_RETRIES (2)` $\rightarrow$ `router` (retry requerying missing facts)
  - `retries > MAX_RETRIES (2)` $\rightarrow$ `fallback` (exhausted retries)

---

## 5. Model Architecture & Tier Assignment

CRAG employs a high-efficiency 8B / 14B split tailored to the specific latency and reasoning requirements of each graph node, powered directly by Mistral AI:

| Node / Role | Model Tier | Why |
| :--- | :--- | :--- |
| **triage, reply_chitchat, reply_refusal, router** (re-query formatting) | `ministral-8b-latest` | Fast, lightweight classification calls where low latency (<1s) matters more than heavy parameter capacity. |
| **tool_selector** (agentic tool routing in router/retrieve), **evaluator**, **generate**, **fallback** | `ministral-14b-latest` | 14-billion parameter reasoning model with native tool-calling, Pydantic structured output, and high-fidelity cited generation. |

---

## 6. Key Configuration & Constants

All runtime thresholds and parameters are defined in `src/rag/constants.py` and validated via Pydantic in `src/rag/config.py`:

| Parameter | Default | Purpose / Behavior |
| :--- | :--- | :--- |
| `HIGH_THRESHOLD` | `0.65` | Quality cutoff. If top rerank score $\ge 0.65$ on simple queries, skips evaluator and answers immediately. |
| `LOW_THRESHOLD` | `0.30` | Minimum viability floor. If top score $\le 0.30$, skips evaluator and triggers corrective search. |
| `KEEP_THRESHOLD` | `0.5` | Threshold for carrying over candidate chunks into subsequent retry pools. |
| `MAX_RETRIES` | `2` | Maximum corrective search iterations before falling back to graceful synthesis. |
| `TOP_K` | `5` | Base number of chunks retrieved per tool call. |
| `RETRIEVE_MULTIPLIER`| `3` | Over-retrieval factor (retrieves `TOP_K * MULTIPLIER` candidates for reranker pool). |
| `FINAL_CONTEXT_N` | `8` | Maximum number of top reranked chunks passed into generation prompt context. |
| `POOL_MAX` | `50` | Hard cap on total deduped candidate chunks retained in pool across passes. |
| `CARRY_MAX` | `4` | Maximum high-scoring chunks carried forward across retry iterations. |
| `TRIAGE_TIMEOUT_S` | `15.0` | Max timeout allowed for Tier 1 triage classification before failing to complex path. |
| `TOOL_TIMEOUT_SIMPLE_S` | `7.0` | Default timeout for tool retrieval calls on simple queries. |
| `TOOL_TIMEOUT_COMPLEX_S` | `13.0` | Default timeout for tool retrieval calls on complex queries. |
| `TOOL_TIMEOUT_RETRY1_EMPTY_S` | `16.0` | Escalated tool timeout on retry 1 when context pool is empty. |
| `TOOL_TIMEOUT_RETRY2_EMPTY_S` | `25.0` | Escalated tool timeout on retry 2 when context pool remains empty. |
| `MAX_SUBQUERIES` | `3` | Upper bound on sub-queries generated for complex multi-part questions. |
| `TRIAGE_MODEL` | `ministral-8b-latest` | Fast classifier for intent triage and canned chitchat/refusal. |
| `ROUTER_MODEL` | `ministral-8b-latest` | Fast model for router re-query keyword formulation. |
| `TOOL_SELECTOR_MODEL`| `ministral-14b-latest` | Mistral 14B for high-fidelity agentic tool selection. |
| `EVALUATOR_MODEL` | `ministral-14b-latest` | Mistral 14B for structured context evaluation. |
| `GENERATOR_MODEL` | `ministral-14b-latest` | Mistral 14B for rigorous educational answer generation. |
| `FALLBACK_MODEL` | `ministral-14b-latest` | Mistral 14B for graceful fallback answer generation. |
| `RERANKER_MODEL` | `jina-reranker-v2-base-multilingual` | Via Jina API Key (Tier 1 cascade). |

---

## 7. RAG Evaluation & Benchmarks (RAGAS)

The Corrective RAG (CRAG) pipeline was benchmarked and optimized using **RAGAS** across 32 golden evaluation test queries spanning Factoid, Reasoning, Multi-hop, and Hybrid Fallback question types. Detailed evaluation datasets and runner scripts are located in [`evals/`](../../evals).

### 7.1 Overall Pipeline Average Scores (32 Queries)

Across all 32 benchmark questions evaluated end-to-end:

| Metric | Score (0 - 1) | Percentage | Evaluation Focus |
| :--- | :---: | :---: | :--- |
| **Context Precision** | **0.8612** | **86.1%** | Ground-truth relevant chunks ranked at the top of retrieved context |
| **Context Recall** | **0.7572** | **75.7%** | All necessary ground-truth information present in retrieved chunks |
| **Answer Relevancy** | **0.9488** | **94.9%** | Response pertinence and alignment with user question |
| **Faithfulness** | **0.7741** | **77.4%** | Factual grounding strictly derived from retrieved context (Strict RAG) |
| **Answer Correctness** | **0.5923** | **59.2%** | Semantic similarity and factual overlap against reference ground truth |

*Note on Faithfulness:* In accordance with the evaluation design in `evals/run_ragas_eval.py`, faithfulness is evaluated on the Strict RAG path (27 queries). Hybrid Fallback intentionally synthesizes external web knowledge to bridge local context gaps.

### 7.2 Breakdown by Execution Path

| Metric | Strict RAG (27 queries / 84.4%) | Hybrid Fallback (5 queries / 15.6%) |
| :--- | :---: | :---: |
| **Context Precision** | **0.8911** (89.1%) | **0.7000** (70.0%) |
| **Context Recall** | **0.8070** (80.7%) | **0.4881** (48.8%) |
| **Faithfulness** | **0.7741** (77.4%) | *N/A (Intentionally excluded)* |
| **Answer Relevancy** | **0.9513** (95.1%) | **0.9357** (93.6%) |
| **Answer Correctness** | **0.6156** (61.6%) | **0.4668** (46.7%) |

### 7.3 Observability & LangSmith Tracking

All benchmark experiments, evaluation runs, and per-query feedback scores are tracked in **LangSmith** under the `Study-Buddy CRAG Benchmark` dataset.




