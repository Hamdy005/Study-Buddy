"""
Reranker module for Study-Buddy RAG pipeline.

Model: jinaai/jina-reranker-v2-base-multilingual (CC-BY-NC-4.0)
Source: https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual

This is a non-commercial personal project — use is compliant with CC-BY-NC-4.0.
Attribution: Jina AI GmbH (https://jina.ai)
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Optional, Union

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from src.rag.constants import (
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL,
    RERANKER_OVER_RETRIEVAL_FACTOR,
    RERANKER_QUANTIZED_MODEL,
)

_RERANKER_MODEL = RERANKER_MODEL


def _patch_transformers_xlm_roberta():
    """Patch transformers.models.xlm_roberta.modeling_xlm_roberta for compatibility
    with Jina reranker models on newer transformers versions (>=4.45) where
    create_position_ids_from_input_ids was refactored or removed.
    """
    try:
        import transformers.models.xlm_roberta.modeling_xlm_roberta as xlm_module

        if not hasattr(xlm_module, "create_position_ids_from_input_ids"):
            import torch

            def create_position_ids_from_input_ids(
                input_ids, padding_idx=1, past_key_values_length=0
            ):
                """Replace non-padding symbols with their position numbers.
                Position numbers begin at padding_idx+1. Padding symbols are ignored.
                """
                if padding_idx is None:
                    padding_idx = 1
                mask = input_ids.ne(padding_idx).int()
                incremental_indices = (
                    torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
                ) * mask
                return incremental_indices.long() + padding_idx

            setattr(xlm_module, "create_position_ids_from_input_ids", create_position_ids_from_input_ids)
            try:
                import transformers.models.xlm_roberta as xlm_pkg
                setattr(xlm_pkg, "create_position_ids_from_input_ids", create_position_ids_from_input_ids)
            except Exception:
                pass
            logger.info("Applied compatibility patch: xlm_roberta.create_position_ids_from_input_ids")
    except Exception as exc:
        logger.debug(f"Transformers XLM-RoBERTa patch skipped: {exc}")


_patch_transformers_xlm_roberta()


def _configure_cpu_threads():
    """Explicitly assign all available CPU cores to PyTorch for parallel matrix operations."""
    try:
        import torch

        cpu_cores = os.cpu_count() or 4
        torch.set_num_threads(cpu_cores)
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass  # Inter-op threads can only be configured before parallel work starts
        logger.info(f"PyTorch CPU threads configured: {cpu_cores} intra-op threads")
    except Exception as exc:
        logger.debug(f"Failed to configure PyTorch CPU threads: {exc}")


_configure_cpu_threads()


class ONNXReranker:
    """Wrapper around ONNX Runtime session and AutoTokenizer to provide
    the exact same interface as Jina's PyTorch model.compute_score().
    """

    def __init__(self, session, tokenizer):
        self.session = session
        self.tokenizer = tokenizer

    def eval(self):
        """No-op for parity with PyTorch model."""
        return self

    def compute_score(
        self, pairs: list[list[str]], max_length: int = RERANKER_MAX_LENGTH
    ) -> list[float]:
        if not pairs:
            return []
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        outputs = self.session.run(None, onnx_inputs)
        scores = outputs[0].flatten().tolist()
        return [float(s) for s in scores]


def _load_onnx_reranker(model_id_or_path: str) -> ONNXReranker:
    """Load an INT8 quantized ONNX reranker model either from a local directory
    or by downloading from Hugging Face Hub.
    """
    # pyrefly: ignore [missing-import]
    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_file = None
    # 1. Local path check
    if os.path.exists(model_id_or_path):
        if os.path.isdir(model_id_or_path):
            for fname in ("model_quantized.onnx", "model.onnx"):
                candidate = os.path.join(model_id_or_path, fname)
                if os.path.exists(candidate):
                    model_file = candidate
                    break
            if not model_file:
                raise FileNotFoundError(
                    f"No ONNX model file found in local directory '{model_id_or_path}'"
                )
        else:
            model_file = model_id_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    else:
        # 2. Remote Hugging Face Hub download
        clean_repo_id = model_id_or_path
        if "huggingface.co/" in clean_repo_id:
            clean_repo_id = clean_repo_id.split("huggingface.co/")[-1].strip("/")

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None
        from huggingface_hub import hf_hub_download

        try:
            model_file = hf_hub_download(
                repo_id=clean_repo_id,
                filename="model_quantized.onnx",
                token=hf_token,
            )
        except Exception:
            model_file = hf_hub_download(
                repo_id=clean_repo_id,
                filename="model.onnx",
                token=hf_token,
            )

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                clean_repo_id, trust_remote_code=True, token=hf_token
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                RERANKER_MODEL, trust_remote_code=True, token=hf_token
            )

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 4
    session = ort.InferenceSession(model_file, sess_options=opts, providers=["CPUExecutionProvider"])
    logger.info(f"Loaded ONNX INT8 reranker from '{model_id_or_path}' ({model_file})")
    return ONNXReranker(session, tokenizer)


@lru_cache(maxsize=1)
def get_reranker():
    """Lazily load and cache the reranker model on first call.

    Checks RERANKER_QUANTIZED_MODEL first (Hugging Face repo or local path).
    If configured and valid, returns the fast INT8 ONNX reranker.
    If not configured or if loading fails, gracefully falls back to the original
    FP32 PyTorch model (RERANKER_MODEL).
    """
    # 1. Attempt loading quantized ONNX model if specified
    if RERANKER_QUANTIZED_MODEL:
        try:
            logger.info(f"Attempting to load INT8 ONNX reranker from '{RERANKER_QUANTIZED_MODEL}' ...")
            onnx_model = _load_onnx_reranker(RERANKER_QUANTIZED_MODEL)
            logger.info("INT8 ONNX reranker initialized and ready.")
            return onnx_model
        except Exception as exc:
            logger.warning(
                f"Failed to load INT8 ONNX reranker '{RERANKER_QUANTIZED_MODEL}': {exc}. "
                f"Gracefully falling back to base PyTorch model '{RERANKER_MODEL}'."
            )

    # 2. Base PyTorch model fallback
    _patch_transformers_xlm_roberta()
    _configure_cpu_threads()
    from transformers import AutoModelForSequenceClassification

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None
    logger.info(f"Loading base reranker model '{RERANKER_MODEL}' from HuggingFace Hub ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL,
        num_labels=1,
        trust_remote_code=True,
        token=hf_token,
    )
    model.eval()
    logger.info("Base PyTorch reranker model loaded and ready.")
    return model


_get_reranker = get_reranker


def warmup_reranker():
    """Warmup forward pass for Jina reranker to prime PyTorch/OpenMP thread pools."""
    try:
        model = get_reranker()
        import torch

        with torch.inference_mode():
            model.compute_score([["warmup query", "warmup document"]], max_length=64)
        logger.info("Reranker warmup completed successfully.")
    except Exception as exc:
        logger.warning(f"Reranker warmup failed (non-fatal): {exc}")


def _score_pairs_batch(pairs: list[list[str]]) -> list[float]:
    """Batch score arbitrary (query, text) pairs in a single forward pass."""
    if not pairs:
        return []

    import torch

    model = get_reranker()
    with torch.inference_mode():
        scores = model.compute_score(pairs, max_length=RERANKER_MAX_LENGTH)

    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    elif isinstance(scores, (float, int)):
        scores = [float(scores)]

    return [float(s) for s in scores]


score_pairs_batch = _score_pairs_batch


def _compute_scores(query: str, texts: list[str]) -> Optional[list[float]]:
    """Compute relevance scores for (query, text) pairs using the reranker model.

    Returns a list of float scores or None if inference fails.
    """
    if not texts:
        return []

    try:
        pairs = [[query, text] for text in texts]
        return _score_pairs_batch(pairs)
    except Exception as exc:
        logger.warning(
            f"Reranker inference failed ({type(exc).__name__}: {exc}). "
            f"Falling back to original vector-similarity ordering."
        )
        return None


def rerank(
    query: str,
    chunks: list[Union[str, dict[str, Any]]],
    top_k: int,
    tracer: Optional[Any] = None,
) -> list[Union[str, dict[str, Any]]]:
    """Score *chunks* against *query* and return the *top_k* highest-scoring ones.

    Accepts either a list of string passages or a list of chunk dicts.

    Args:
        query:  The user's question / search query.
        chunks: Candidate passages retrieved by vector similarity search.
                Expected to be ``top_k * RERANKER_OVER_RETRIEVAL_FACTOR`` items.
        top_k:  Number of chunks to keep after reranking.
        tracer: Optional ChatTracer instance to log reranker metrics to LangSmith.

    Returns:
        A list of at most *top_k* chunks, ordered by relevance
        (highest reranker score first).
    """
    if not chunks:
        return []

    if isinstance(chunks[0], dict):
        return rerank_dicts(query, chunks, top_k=top_k, tracer=tracer)  # type: ignore[arg-type]

    top_k = min(top_k, len(chunks))
    if len(chunks) <= 1:
        return chunks[:top_k]

    t0 = time.perf_counter()
    str_chunks: list[str] = [str(c) for c in chunks]
    scores = _compute_scores(query, str_chunks)
    latency_ms = (time.perf_counter() - t0) * 1000

    if scores is None or len(scores) != len(str_chunks):
        logger.warning(
            f"Reranker failed after {latency_ms:.1f}ms; using original ordering."
        )
        return chunks[:top_k]

    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:top_k]]

    logger.info(
        f"⏱️ Reranker took {latency_ms:.1f}ms ({latency_ms/1000:.2f}s) for {len(chunks)} passages "
        f"| top score={scored[0][0]:.4f} | bottom kept score={scored[top_k - 1][0]:.4f}"
    )

    if tracer and hasattr(tracer, "log_reranker"):
        tracer.log_reranker(
            query=query,
            num_candidates=len(chunks),
            top_k=top_k,
            latency_ms=latency_ms,
            scores=[float(s) for s, _ in scored[:top_k]],
        )

    return top_chunks


def rerank_dicts(
    query: str,
    records: list[dict[str, Any]],
    top_k: int,
    content_key: str = "content",
    tracer: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Score a list of chunk dictionaries and return the *top_k* highest-scoring ones.

    Each returned dictionary is updated with a ``rerank_score`` key.

    Args:
        query:       The user's question / search query.
        records:     List of candidate dicts (e.g. from match_material_chunks).
        top_k:       Number of records to keep.
        content_key: Dictionary key containing the text content to score.
        tracer:      Optional ChatTracer instance to log reranker metrics to LangSmith.

    Returns:
        The top_k dicts, sorted descending by rerank score.
    """
    if not records:
        return []

    top_k = min(top_k, len(records))
    if len(records) <= 1:
        for r in records[:top_k]:
            if "rerank_score" not in r:
                r["rerank_score"] = float(r.get("similarity", 0.0))
        return records[:top_k]

    t0 = time.perf_counter()
    texts = [str(r.get(content_key, "")) for r in records]
    scores = _compute_scores(query, texts)
    latency_ms = (time.perf_counter() - t0) * 1000

    if scores is None or len(scores) != len(records):
        logger.warning(
            f"Reranker failed after {latency_ms:.1f}ms; using fallback similarity scores."
        )
        for r in records[:top_k]:
            if "rerank_score" not in r:
                r["rerank_score"] = float(r.get("similarity", 0.0))
        return records[:top_k]

    scored_records: list[tuple[float, dict[str, Any]]] = []
    for score, rec in zip(scores, records):
        r_copy = dict(rec)
        r_copy["rerank_score"] = float(score)
        scored_records.append((float(score), r_copy))

    scored_records.sort(key=lambda x: x[0], reverse=True)
    top_records = [rec for _, rec in scored_records[:top_k]]

    logger.info(
        f"⏱️ Reranker took {latency_ms:.1f}ms ({latency_ms/1000:.2f}s) for {len(records)} candidates "
        f"| top score={scored_records[0][0]:.4f} | bottom kept score={scored_records[top_k - 1][0]:.4f}"
    )

    if tracer and hasattr(tracer, "log_reranker"):
        tracer.log_reranker(
            query=query,
            num_candidates=len(records),
            top_k=top_k,
            latency_ms=latency_ms,
            scores=[float(score) for score, _ in scored_records[:top_k]],
        )

    return top_records
