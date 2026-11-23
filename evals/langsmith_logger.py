"""
LangSmith Metrics & Experiment Monitoring for RAGAS Evaluations.

Syncs benchmark datasets and logs evaluation experiments directly to
LangSmith under 'Datasets & Experiments' for the unified 'study-buddy' project.
"""

import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv("config.env")

try:
    from langsmith import Client
except ImportError:
    Client = None


def is_rate_limit_error(exc: Optional[BaseException]) -> bool:
    """Check whether an exception represents a LangSmith 429 / Rate Limit error."""
    if exc is None:
        return False
    try:
        from langsmith.utils import LangSmithRateLimitError
        if isinstance(exc, LangSmithRateLimitError):
            return True
    except Exception:
        pass

    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "ratelimit" in msg or "too many requests" in msg:
        return True

    cause = getattr(exc, "__cause__", None)
    if cause and is_rate_limit_error(cause):
        return True
    context = getattr(exc, "__context__", None)
    if context and is_rate_limit_error(context):
        return True

    return False


class _LangSmithRateLimitFilter(logging.Filter):
    """Filter out LangSmith 429 rate limit errors from logger records."""
    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info and record.exc_info[1] and is_rate_limit_error(record.exc_info[1]):
            return False
        msg = record.getMessage().lower()
        if "429" in msg or "rate limit" in msg or "ratelimit" in msg or "too many requests" in msg:
            return False
        return True


for _logger_name in ("langsmith", "langsmith.client", "langsmith.run_trees"):
    logging.getLogger(_logger_name).addFilter(_LangSmithRateLimitFilter())


class LangSmithEvalMonitor:
    def __init__(self, project_name: Optional[str] = None):
        self.api_key = os.environ.get("LANGSMITH_API_KEY", "")
        self.endpoint = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        self.project_name = project_name or os.environ.get("LANGSMITH_PROJECT", "study-buddy")
        self.rate_limited = False
        
        self.client: Optional[Client] = None
        if self.api_key and Client is not None:
            try:
                def _on_error(err: Exception) -> None:
                    if is_rate_limit_error(err):
                        self.rate_limited = True

                self.client = Client(
                    api_key=self.api_key,
                    api_url=self.endpoint,
                    tracing_error_callback=_on_error,
                )
            except Exception as e:
                if is_rate_limit_error(e):
                    self.rate_limited = True
                    self.client = None
                else:
                    print(f"[!] Warning: Could not initialize LangSmith client: {e}")

    def sync_dataset(self, dataset_name: str, items: list[dict]) -> Optional[Any]:
        """Creates or updates a benchmark dataset in LangSmith."""
        if self.rate_limited or not self.client:
            return None

        try:
            if not self.client.has_dataset(dataset_name=dataset_name):
                ds = self.client.create_dataset(
                    dataset_name=dataset_name,
                    description="Study-Buddy CRAG Pipeline Evaluation Benchmark (Python Celery, OS - Lecture 7, Reinforcement Learning)",
                )
                print(f"[LangSmith] Created dataset '{dataset_name}' (ID: {ds.id})")
                
                inputs = [{"question": d["user_input"]} for d in items]
                outputs = [{"reference": d["reference"], "reference_contexts": d["reference_contexts"]} for d in items]
                metadata = [
                    {
                        "id": d["id"],
                        "material_id": d["material_id"],
                        "material_title": d["material_title"],
                        "question_type": d["question_type"],
                        "expected_pipeline_path": d.get("expected_pipeline_path", "strict_rag"),
                    }
                    for d in items
                ]
                
                self.client.create_examples(
                    inputs=inputs,
                    outputs=outputs,
                    metadata=metadata,
                    dataset_id=ds.id,
                )
                print(f"[LangSmith] Synced {len(items)} examples to LangSmith dataset.")
                return ds
            else:
                return self.client.read_dataset(dataset_name=dataset_name)
        except Exception as e:
            if is_rate_limit_error(e):
                self.rate_limited = True
                return None
            print(f"[!] Warning: Could not sync dataset to LangSmith: {e}")
            return None

    def log_evaluation_results(
        self,
        results_df: Any,
        dataset_name: str = "Study-Buddy CRAG Benchmark",
        run_name: Optional[str] = None,
        suite_name: str = "all",
    ):
        """
        Logs Ragas evaluation results as an Experiment in LangSmith under 'Datasets & Experiments'.
        Links each evaluation run to its corresponding dataset example and logs metric feedbacks.
        """
        if self.rate_limited or not self.client:
            return

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        experiment_name = run_name or f"crag-{suite_name}-{timestamp_str}"

        metric_names = [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
        ]

        print(f"\n[LangSmith] Linking evaluation experiment to dataset '{dataset_name}' in project '{self.project_name}'...")

        dataset = None
        example_map = {}
        try:
            if self.client.has_dataset(dataset_name=dataset_name):
                dataset = self.client.read_dataset(dataset_name=dataset_name)
                examples = list(self.client.list_examples(dataset_id=dataset.id))
                for ex in examples:
                    if ex.inputs and "question" in ex.inputs:
                        example_map[ex.inputs["question"].strip()] = ex.id
                    if ex.metadata and "id" in ex.metadata:
                        example_map[ex.metadata["id"]] = ex.id
                print(f"[LangSmith] Found dataset '{dataset_name}' (ID: {dataset.id}) with {len(examples)} examples.")
        except Exception as e:
            if is_rate_limit_error(e):
                self.rate_limited = True
                return
            print(f"[!] Warning reading dataset: {e}")

        # Create the Experiment project linked to the reference dataset
        experiment_project = None
        target_project_name = experiment_name
        if dataset:
            try:
                experiment_project = self.client.create_project(
                    project_name=experiment_name,
                    reference_dataset_id=dataset.id,
                    description=f"RAGAS Benchmark Evaluation for Study-Buddy CRAG ({suite_name})",
                    metadata={
                        "parent_project": self.project_name,
                        "suite": suite_name,
                        "framework": "ragas",
                        "evaluated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                print(f"[LangSmith] Created Experiment '{experiment_name}' under 'Datasets & Experiments' (ID: {experiment_project.id})")
            except Exception as e:
                if is_rate_limit_error(e):
                    self.rate_limited = True
                    return
                print(f"[!] Warning: Could not create experiment project linked to dataset: {e}")
                target_project_name = self.project_name
        else:
            target_project_name = self.project_name

        try:
            logged_count = 0
            for _, row in results_df.iterrows():
                if self.rate_limited:
                    return
                example_run_id = uuid.uuid4()
                q_id = row.get("id", "q")
                q_text = row.get("user_input", "")
                ans_text = row.get("response", "")
                actual_path = row.get("actual_pipeline_path", row.get("expected_pipeline_path", "strict_rag"))
                material_title = row.get("material_title", "")

                ref_example_id = example_map.get(q_text.strip()) or example_map.get(q_id)

                now_utc = datetime.now(timezone.utc)
                run_kwargs = {
                    "id": example_run_id,
                    "name": f"eval-{q_id}",
                    "run_type": "chain",
                    "project_name": target_project_name,
                    "inputs": {"question": q_text, "material_title": material_title},
                    "outputs": {
                        "response": ans_text,
                        "retrieved_contexts": row.get("retrieved_contexts", []),
                    },
                    "tags": [actual_path, material_title],
                    "start_time": now_utc,
                    "end_time": now_utc,
                }
                if ref_example_id:
                    run_kwargs["reference_example_id"] = ref_example_id

                self.client.create_run(**run_kwargs)

                # Record metric feedback on the run
                for m in metric_names:
                    if m in row and row[m] is not None:
                        val = row[m]
                        # check if not NaN
                        if isinstance(val, (int, float)) and val == val:
                            fb_kwargs = {
                                "run_id": example_run_id,
                                "key": m,
                                "score": round(float(val), 4),
                                "comment": f"RAGAS {m} ({actual_path})",
                            }
                            if experiment_project:
                                fb_kwargs["session_id"] = experiment_project.id
                            self.client.create_feedback(**fb_kwargs)

                logged_count += 1

            # Compute and attach final aggregate scores
            mean_scores = {}
            for m in metric_names:
                if m in results_df.columns:
                    valid_vals = results_df[m].dropna()
                    if not valid_vals.empty:
                        mean_scores[m] = round(float(valid_vals.mean()), 4)

            if experiment_project and mean_scores:
                try:
                    existing_meta = getattr(experiment_project, "metadata", {}) or {}
                    self.client.update_project(
                        experiment_project.id,
                        metadata={
                            **existing_meta,
                            "final_scores": mean_scores,
                            "sample_count": logged_count,
                        },
                    )
                    print(f"[LangSmith] Saved final aggregate scores to Experiment metadata: {mean_scores}")
                except Exception as meta_err:
                    if is_rate_limit_error(meta_err):
                        self.rate_limited = True
                        return
                    print(f"[!] Warning: Could not update experiment metadata with final scores: {meta_err}")

            # Also log an explicit summary run with the overall scores
            if mean_scores:
                try:
                    summary_run_id = uuid.uuid4()
                    now_utc = datetime.now(timezone.utc)
                    self.client.create_run(
                        id=summary_run_id,
                        name="FINAL_EXPERIMENT_SUMMARY",
                        run_type="chain",
                        project_name=target_project_name,
                        inputs={"suite": suite_name, "sample_count": logged_count},
                        outputs={"final_mean_scores": mean_scores},
                        tags=["summary", suite_name],
                        start_time=now_utc,
                        end_time=now_utc,
                    )
                    for m, val in mean_scores.items():
                        fb_kwargs = {
                            "run_id": summary_run_id,
                            "key": f"mean_{m}",
                            "score": val,
                            "comment": f"Overall Mean {m} across {logged_count} samples",
                        }
                        if experiment_project:
                            fb_kwargs["session_id"] = experiment_project.id
                        self.client.create_feedback(**fb_kwargs)
                except Exception as sum_err:
                    if is_rate_limit_error(sum_err):
                        self.rate_limited = True
                        return
                    print(f"[!] Warning: Could not log summary run: {sum_err}")

            print(f"[✓] LangSmith tracking successfully logged {logged_count} evaluations.")
            if dataset:
                ds_url = getattr(dataset, "url", None) or f"https://smith.langchain.com/o/492a9209-76c6-42b6-b4a7-e1b56869a645/datasets/{dataset.id}"
                if experiment_project and getattr(experiment_project, "url", None):
                    print(f"[🔗] View Experiment Runs:")
                    print(f"     {experiment_project.url}")
                print(f"[🔗] View Dataset & Experiments:")
                print(f"     {ds_url}?tab=1" if "?tab=" not in ds_url else f"     {ds_url}")
            else:
                proj_url = getattr(experiment_project, "url", None) if experiment_project else f"https://smith.langchain.com/o/492a9209-76c6-42b6-b4a7-e1b56869a645/projects/p/{self.project_name}"
                print(f"[🔗] View LangSmith Project Traces:")
                print(f"     {proj_url}")

        except Exception as e:
            if is_rate_limit_error(e):
                self.rate_limited = True
                return
            print(f"[!] Warning: Could not log evaluation results to LangSmith: {e}")
