# RAGAS Evaluation Suite for Study Buddy

This directory contains the golden evaluation benchmark dataset and runner scripts for evaluating Study Buddy's Corrective RAG (CRAG) pipeline using **RAGAS**.

---

## 📁 Directory Structure

```text
evals/
├── datasets/
│   ├── ragas_eval_dataset.json    # Complete JSON dataset (exact RAGAS format)
│   └── ragas_eval_dataset.csv     # CSV summary table for quick viewing
├── runs/                          # Output directory for evaluation reports
│   ├── evaluation_summary.md      # Human-readable summary of average scores & benchmarks
│   ├── evaluation_summary.json    # Machine-readable average metrics across paths & materials
│   ├── ragas_evaluation_report.csv# Detailed per-query RAGAS scores
│   └── crag_pipeline_outputs.json # Raw pipeline responses and contexts
├── create_eval_dataset.py         # Script that pulled chunks & created the dataset
├── run_ragas_eval.py              # Main evaluation runner script
└── README.md                      # Documentation & metrics guidance
```

---

## 🎯 Target Materials (User: `48e0fa0e-85c8-40c4-a5a3-c51c5eb53b5e`)

| Material | Material ID | Total Chunks in DB | Eval Questions Generated | Question Types |
| :--- | :--- | :---: | :---: | :--- |
| **Python Celery** | `08f811e4-a8ea-48a4-857d-84d17b415ffd` | 4 | **10** | Factoid, Reasoning, Multi-hop, Hybrid Fallback |
| **OS - Lecture 7** | `d7c4958b-7972-49e6-aeb0-934190a0172c` | 6 | **10** | Factoid, Reasoning, Multi-hop, Hybrid Fallback |
| **Reinforcement Learning** | `b687f0f3-b56c-4c4d-a5fe-eb911f134753` | 163 | **12** | Factoid, Reasoning, Multi-hop, Hybrid Fallback |
| **Total** | | **173 chunks** | **32 test cases** | 20 Factoids, 3 Multi-hop, 5 Reasoning, 4 Hybrid |

---

## ⚖️ The 5 Metrics & How to Interpret Them

1. **Context Precision**: Evaluates whether the ground-truth relevant chunks are ranked at the top of retrieved context.
2. **Context Recall**: Evaluates whether all information in the reference answer is present in the retrieved chunks.
3. **Answer Relevancy**: Evaluates how pertinent the model's response is to the user's question.
4. **Answer Correctness**: Evaluates accuracy against the reference ground truth (semantic similarity + factual overlap).
5. **Faithfulness**:
   - **On `strict_rag` path** (26 questions): Evaluates whether the answer is strictly grounded in retrieved chunks (Target: `> 0.90`).
   - **On `hybrid_fallback` path** (6 questions): Intentionally tests questions where context has gaps and `GENERATE_HYBRID_PROMPT` supplements answers with domain knowledge. Standard RAGAS faithfulness will drop here as expected—focus on **Answer Correctness** and **Answer Relevancy** for these cases.

---

## 🤖 Judge Models Cascade & Embeddings

The evaluation runner uses a robust, automatic fallback cascade to avoid rate limits or service timeouts:

### LLM Cascade:
1. **Tier 1**: `gemini-3.5-flash-lite` (request_timeout: 25s)
2. **Tier 2**: `gemini-3.1-flash-lite` (request_timeout: 25s)

### Embeddings:
* **`gemini-embedding-2`** (`models/gemini-embedding-2`, dimension = 3072) via Google Generative AI.

---

## 📊 LangSmith Observability & Tracking

All evaluation runs and dataset benchmarks are automatically tracked in **LangSmith**:
* **Dataset Sync**: The 32 benchmark questions and ground-truth contexts are automatically registered under dataset `Study-Buddy CRAG Benchmark`.
* **Experiment Runs**: Each execution creates a benchmark Experiment linked to the dataset under **Datasets & Experiments** in your unified `study-buddy` workspace.
* **Metric Feedback**: Each question run is stamped with individual feedback scores for `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`, and `answer_correctness`.
* **Dashboard Tab**: Directly accessible via the **Datasets & Experiments** tab on the left navigation bar in LangSmith.

---

## 🚀 How to Run the Evaluation

### Step 1: Install `ragas`
```bash
uv add ragas
# or
.venv/bin/python3 -m pip install ragas
```

### Step 2: Run the Benchmark
To run on all 32 questions:
```bash
.venv/bin/python3 evals/run_ragas_eval.py
```

To run only the Strict RAG suite (with Faithfulness):
```bash
.venv/bin/python3 evals/run_ragas_eval.py --path strict_rag
```

To run only the Hybrid Fallback suite (excluding naive Faithfulness):
```bash
.venv/bin/python3 evals/run_ragas_eval.py --path hybrid_fallback
```

To run a quick test on a subset (e.g. 3 questions):
```bash
.venv/bin/python3 evals/run_ragas_eval.py --limit 3
```

To run on a specific material:
```bash
.venv/bin/python3 evals/run_ragas_eval.py --material "Python Celery"
```


