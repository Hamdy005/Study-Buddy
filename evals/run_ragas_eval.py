"""
RAGAS Evaluation Runner for Study Buddy CRAG Pipeline.

Fixes & Features:
- Compiles the CRAG LangGraph ONCE for all evaluations.
- Dynamically routes Supabase retrieval by material_id per query.
- Extracts real chunk text from Chunk.text (.get bug fixed).
- Determines actual_pipeline_path from final_state (low_confidence).
- Filters out failed/error queries before RAGAS scoring.
- Wraps judge LLM and embeddings using LangchainLLMWrapper / LangchainEmbeddingsWrapper.
- Uses RunConfig(max_workers=2, timeout=90) for evaluate().
- Tracks metrics & experiments in LangSmith with per-question feedback.
"""

import os
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import json
import argparse
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("config.env")

# 1. RAGAS Imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
    )
    from ragas.run_config import RunConfig
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
except ImportError:
    evaluate = None
    RunConfig = None
    LangchainLLMWrapper = None
    LangchainEmbeddingsWrapper = None

from src.rag.crag_graph import build_graph
from src.rag.tools import TOOL_REGISTRY
from src.rag.tools.supabase import retrieve_supabase
from src.rag.constants import TOP_K_CHUNKS, GRAPH_RECURSION_LIMIT
from evals.models_cascade import build_judge_llm_cascade, get_judge_embeddings
from evals.langsmith_logger import LangSmithEvalMonitor, is_rate_limit_error


async def generate_pipeline_responses(dataset_items: list[dict]) -> list[dict]:
    """Runs the CRAG pipeline once against each test question and records actual outputs."""
    current_material_id = None

    def _dynamic_supabase_retriever(query: str, k: int = TOP_K_CHUNKS):
        return retrieve_supabase(query, k, material_id=current_material_id)

    active_tools = dict(TOOL_REGISTRY)
    active_tools["supabase"] = _dynamic_supabase_retriever

    # Compile the CRAG StateGraph once for all questions
    print("[CRAG] Compiling CRAG StateGraph once for all evaluations...")
    app = build_graph(tools=active_tools)

    enriched = []
    total = len(dataset_items)

    for i, item in enumerate(dataset_items, 1):
        q = item["user_input"]
        mat_id = item["material_id"]
        title = item["material_title"]
        print(f"\n[{i}/{total}] Running CRAG on: '{q[:60]}...' (Material: {title})")

        # Set material_id for this question
        current_material_id = mat_id

        try:
            initial_state = {
                "original_q": q.strip(),
                "material_id": mat_id,
            }
            final_state = await app.ainvoke(initial_state, config={"recursion_limit": GRAPH_RECURSION_LIMIT})

            answer = final_state.get("answer") or "No answer could be generated."
            ranked_chunks = final_state.get("ranked", []) or final_state.get("pool", [])

            # Extract passage text from Chunk.text (or dict/str)
            retrieved_contexts = [
                c.text if hasattr(c, "text") else (c.get("text", "") if isinstance(c, dict) else str(c))
                for c in ranked_chunks
            ]

            # Determine ACTUAL pipeline path taken by the graph
            low_conf = bool(final_state.get("low_confidence", False))
            actual_path = "hybrid_fallback" if low_conf else "strict_rag"

            enriched.append({
                **item,
                "response": answer,
                "retrieved_contexts": retrieved_contexts if retrieved_contexts else ["No sources retrieved."],
                "pipeline_low_confidence": low_conf,
                "actual_pipeline_path": actual_path,
            })
            print(f"    -> Answer generated ({len(answer)} chars) | Chunks: {len(retrieved_contexts)} | Actual Path: {actual_path}")
        except Exception as e:
            print(f"    [!] Error running CRAG for query: {e}")
            enriched.append({
                **item,
                "response": f"ERROR: {e}",
                "retrieved_contexts": [],
                "pipeline_low_confidence": True,
                "actual_pipeline_path": "failed",
            })

        # Inter-query throttling delay to prevent burst rate limits on Mistral / LLM APIs
        if i < total:
            await asyncio.sleep(3.0)

    return enriched


def run_evaluation(enriched_data: list[dict], out_dir: Path):
    # Always save the latest raw pipeline outputs (answers, retrieved chunks, actual paths)
    out_file = out_dir / "crag_pipeline_outputs.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)
    print(f"[✓] Latest pipeline responses and retrieved chunks saved to: {out_file}")

    if evaluate is None:
        print("\n[!] 'ragas' is not installed in the environment.")
        print("    Please install it using: python -m pip install ragas")
        return

    # 1. Filter out failed runs so they do NOT drag down Ragas averages
    failed_items = [d for d in enriched_data if d["response"].startswith("ERROR:") or not d["retrieved_contexts"]]
    valid_items = [d for d in enriched_data if not d["response"].startswith("ERROR:") and d["retrieved_contexts"]]

    if failed_items:
        print(f"\n[⚠️] Warning: {len(failed_items)} queries encountered pipeline errors and will be EXCLUDED from scoring:")
        for f_item in failed_items:
            print(f"     - [{f_item['id']}] {f_item['user_input'][:60]}... -> {f_item['response'][:50]}")

    if not valid_items:
        print("\n[!] No valid pipeline outputs to evaluate.")
        return

    # 2. Setup wrapped Ragas judge models and RunConfig
    raw_llm = build_judge_llm_cascade()
    raw_embeddings = get_judge_embeddings()

    judge_llm = LangchainLLMWrapper(raw_llm) if LangchainLLMWrapper else raw_llm
    judge_embeddings = LangchainEmbeddingsWrapper(raw_embeddings) if LangchainEmbeddingsWrapper else raw_embeddings
    run_config = RunConfig(max_workers=2, timeout=180) if RunConfig else None

    # 3. SPLIT BY WHAT THE PIPELINE ACTUALLY DID (actual_pipeline_path)
    strict_items = [d for d in valid_items if d.get("actual_pipeline_path") == "strict_rag"]
    hybrid_items = [d for d in valid_items if d.get("actual_pipeline_path") == "hybrid_fallback"]

    print(f"\n[CRAG Path Breakdown]")
    print(f"   Strict RAG (actual):      {len(strict_items)} samples")
    print(f"   Hybrid Fallback (actual):  {len(hybrid_items)} samples")

    results_dfs = []

    # ─────────────────────────────────────────────────────────────
    # Suite 1: Strict RAG (5 metrics including Faithfulness)
    # ─────────────────────────────────────────────────────────────
    if strict_items:
        print(f"\n{'=' * 60}")
        print(f"RUNNING SUITE 1: STRICT RAG ({len(strict_items)} samples)")
        print(f"Metrics: Context Precision, Context Recall, Faithfulness, Relevance, Correctness")
        print(f"{'=' * 60}")

        strict_dataset = Dataset.from_dict({
            "user_input": [d["user_input"] for d in strict_items],
            "response": [d["response"] for d in strict_items],
            "reference": [d["reference"] for d in strict_items],
            "retrieved_contexts": [d["retrieved_contexts"] for d in strict_items],
        })

        strict_metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness,
        ]

        eval_kwargs = {
            "dataset": strict_dataset,
            "metrics": strict_metrics,
            "llm": judge_llm,
            "embeddings": judge_embeddings,
        }
        if run_config:
            eval_kwargs["run_config"] = run_config

        strict_scores = evaluate(**eval_kwargs)

        strict_df = strict_scores.to_pandas()
        for col in ["id", "material_title", "question_type", "expected_pipeline_path", "actual_pipeline_path"]:
            strict_df[col] = [d[col] for d in strict_items]
        results_dfs.append(strict_df)

        print("\n[✓] Strict RAG Evaluation Scores (Target Faithfulness > 0.90):")
        strict_metric_cols = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]
        print(strict_df[[c for c in strict_metric_cols if c in strict_df.columns]].mean().round(3).to_string())

    # ─────────────────────────────────────────────────────────────
    # Suite 2: Hybrid Fallback (4 metrics, excludes naive context Faithfulness)
    # ─────────────────────────────────────────────────────────────
    if hybrid_items:
        print(f"\n{'=' * 60}")
        print(f"RUNNING SUITE 2: HYBRID FALLBACK ({len(hybrid_items)} samples)")
        print(f"Metrics: Context Precision, Context Recall, Relevance, Correctness")
        print(f"Note: Standard Faithfulness is excluded here because GENERATE_HYBRID_PROMPT")
        print(f"intentionally injects domain knowledge to fill context gaps.")
        print(f"{'=' * 60}")

        hybrid_dataset = Dataset.from_dict({
            "user_input": [d["user_input"] for d in hybrid_items],
            "response": [d["response"] for d in hybrid_items],
            "reference": [d["reference"] for d in hybrid_items],
            "retrieved_contexts": [d["retrieved_contexts"] for d in hybrid_items],
        })

        hybrid_metrics = [
            context_precision,
            context_recall,
            answer_relevancy,
            answer_correctness,
        ]

        eval_kwargs = {
            "dataset": hybrid_dataset,
            "metrics": hybrid_metrics,
            "llm": judge_llm,
            "embeddings": judge_embeddings,
        }
        if run_config:
            eval_kwargs["run_config"] = run_config

        hybrid_scores = evaluate(**eval_kwargs)

        hybrid_df = hybrid_scores.to_pandas()
        for col in ["id", "material_title", "question_type", "expected_pipeline_path", "actual_pipeline_path"]:
            hybrid_df[col] = [d[col] for d in hybrid_items]
        hybrid_df["faithfulness"] = float("nan")
        results_dfs.append(hybrid_df)

        print("\n[✓] Hybrid Fallback Evaluation Scores:")
        hybrid_metric_cols = ["context_precision", "context_recall", "answer_relevancy", "answer_correctness"]
        print(hybrid_df[[c for c in hybrid_metric_cols if c in hybrid_df.columns]].mean().round(3).to_string())

    # Combine and save full report
    import pandas as pd
    combined_df = pd.concat(results_dfs, ignore_index=True)
    report_csv = out_dir / "ragas_evaluation_report.csv"
    combined_df.to_csv(report_csv, index=False)
    print(f"\n[✓] All evaluations complete! Report saved to: {report_csv}")

    # Generate and save summary reports (JSON & Markdown)
    metric_cols = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]
    summary_data = {
        "total_evaluated_queries": len(combined_df),
        "overall_averages": combined_df[metric_cols].mean().round(4).to_dict(),
        "by_actual_pipeline_path": {},
        "by_expected_pipeline_path": {},
        "by_material": {},
        "by_question_type": {},
    }
    for col_key, group_dict in [
        ("actual_pipeline_path", summary_data["by_actual_pipeline_path"]),
        ("expected_pipeline_path", summary_data["by_expected_pipeline_path"]),
        ("material_title", summary_data["by_material"]),
        ("question_type", summary_data["by_question_type"]),
    ]:
        if col_key in combined_df.columns:
            for grp_name, grp_df in combined_df.groupby(col_key):
                metrics = grp_df[metric_cols].mean().round(4).to_dict()
                # Clean NaNs to None for valid JSON serialization
                metrics = {k: (None if pd.isna(v) else v) for k, v in metrics.items()}
                group_dict[str(grp_name)] = {
                    "count": len(grp_df),
                    "metrics": metrics,
                }

    summary_json_path = out_dir / "evaluation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"[✓] Evaluation summary scores saved to: {summary_json_path}")

    # Save failed queries report if any occurred
    if failed_items:
        failed_file = out_dir / "ragas_failed_queries.json"
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(failed_items, f, indent=2, ensure_ascii=False)
        print(f"[!] Saved {len(failed_items)} excluded error queries to: {failed_file}")

    # LangSmith Observability & Tracking in Datasets & Experiments
    try:
        monitor = LangSmithEvalMonitor()  # Uses unified 'study-buddy' project from config.env
        monitor.sync_dataset(dataset_name="Study-Buddy CRAG Benchmark", items=valid_items)
        monitor.log_evaluation_results(
            combined_df,
            dataset_name="Study-Buddy CRAG Benchmark",
            suite_name="crag_ragas",
        )
    except Exception as e:
        if not is_rate_limit_error(e):
            print(f"[!] Warning: LangSmith monitoring failed: {e}")


async def main():
    parser = argparse.ArgumentParser(description="RAGAS Evaluation for Study Buddy")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples to run")
    parser.add_argument("--material", type=str, default=None, help="Filter to a specific material title")
    parser.add_argument(
        "--path",
        type=str,
        default="all",
        choices=["all", "strict_rag", "hybrid_fallback"],
        help="Run only 'strict_rag', only 'hybrid_fallback', or 'all'",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="evals/datasets/ragas_eval_dataset.json",
        help="Path to dataset JSON (defaults to evals/datasets/ragas_eval_dataset.json)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}.")

    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.material:
        data = [d for d in data if args.material.lower() in d["material_title"].lower()]
    if args.path != "all":
        data = [d for d in data if d.get("expected_pipeline_path") == args.path]
    if args.limit:
        data = data[:args.limit]

    out_dir = Path("evals/runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(data)} evaluation questions (Filter: {args.path}).")

    enriched = await generate_pipeline_responses(data)
    run_evaluation(enriched, out_dir=out_dir)


if __name__ == "__main__":
    asyncio.run(main())
