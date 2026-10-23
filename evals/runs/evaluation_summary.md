# Study Buddy RAGAS Evaluation Summary Report

**Run Date:** October 2026  
**Dataset:** `evals/datasets/ragas_eval_dataset.json` (32 golden test queries)  
**Detailed Report:** [`ragas_evaluation_report.csv`](./ragas_evaluation_report.csv)  
**Raw Outputs:** [`crag_pipeline_outputs.json`](./crag_pipeline_outputs.json)

---

## 1. Overall Pipeline Average Scores (32 Queries)

Across all 32 benchmark questions evaluated end-to-end:

| Metric | Score (0 - 1) | Percentage |
| :--- | :---: | :---: |
| **Context Precision** | **0.8612** | **86.1%** |
| **Context Recall** | **0.7572** | **75.7%** |
| **Answer Relevancy** | **0.9488** | **94.9%** |
| **Faithfulness** | **0.7741** | **77.4%** |
| **Answer Correctness** | **0.5923** | **59.2%** |

*Note on Faithfulness:* In accordance with the evaluation design in `run_ragas_eval.py`, faithfulness is evaluated on the Strict RAG path (27 queries). Hybrid Fallback intentionally synthesizes external web knowledge to bridge local context gaps.

---

## 2. Scores by Actual Pipeline Path (`actual_pipeline_path`)

The Corrective RAG (CRAG) system dynamically routes low-confidence retrieval queries to web-search hybrid fallback:

| Metric | Strict RAG (27 queries / 84.4%) | Hybrid Fallback (5 queries / 15.6%) |
| :--- | :---: | :---: |
| **Context Precision** | **0.8911** (89.1%) | **0.7000** (70.0%) |
| **Context Recall** | **0.8070** (80.7%) | **0.4881** (48.8%) |
| **Faithfulness** | **0.7741** (77.4%) | *N/A (Intentionally excluded)* |
| **Answer Relevancy** | **0.9513** (95.1%) | **0.9357** (93.6%) |
| **Answer Correctness** | **0.6156** (61.6%) | **0.4668** (46.7%) |
