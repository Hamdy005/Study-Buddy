"""
Reranker module for Study-Buddy RAG pipeline.

Model: jinaai/jina-reranker-v2-base-multilingual (CC-BY-NC-4.0)
Source: https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual

This is a non-commercial personal project — use is compliant with CC-BY-NC-4.0.
Attribution: Jina AI GmbH (https://jina.ai)
"""

from __future__ import annotations

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
)

_RERANKER_MODEL = RERANKER_MODEL


@lru_cache(maxsize=1)
def _get_reranker():
    """Lazily load and cache the Jina reranker model on first call.

    Uses AutoModelForSequenceClassification with trust_remote_code=True,
    as required by the Jina v2 reranker architecture.
    Runs on CPU; the model is 278 M params but inference on short chunks
    is fast enough for synchronous per-turn use.
    """
    from transformers import AutoModelForSequenceClassification
    import torch

    logger.info(f"Loading reranker model '{RERANKER_MODEL}' from HuggingFace Hub ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL,
        num_labels=1,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Reranker model loaded and ready.")
    return model


def _compute_scores(query: str, texts: list[str]) -> Optional[list[float]]:
    """Compute relevance scores for (query, text) pairs using the reranker model.

    Returns a list of float scores or None if inference fails.
    """
    if not texts:
        return []

    try:
        import torch

        model = _get_reranker()
        pairs = [[query, text] for text in texts]

        with torch.no_grad():
            scores = model.compute_score(pairs, max_length=RERANKER_MAX_LENGTH)

        # Handle tensor, numpy array, list, or scalar
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        elif isinstance(scores, (float, int)):
            scores = [float(scores)]

        return [float(s) for s in scores]

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
) -> list[Union[str, dict[str, Any]]]:
    """Score *chunks* against *query* and return the *top_k* highest-scoring ones.

    Accepts either a list of string passages or a list of chunk dicts.

    Args:
        query:  The user's question / search query.
        chunks: Candidate passages retrieved by vector similarity search.
                Expected to be ``top_k * RERANKER_OVER_RETRIEVAL_FACTOR`` items.
        top_k:  Number of chunks to keep after reranking.

    Returns:
        A list of at most *top_k* chunks, ordered by relevance
        (highest reranker score first).
    """
    if not chunks:
        return []

    if isinstance(chunks[0], dict):
        return rerank_dicts(query, chunks, top_k=top_k)  # type: ignore[arg-type]

    top_k = min(top_k, len(chunks))
    if len(chunks) <= 1:
        return chunks[:top_k]

    str_chunks: list[str] = [str(c) for c in chunks]
    scores = _compute_scores(query, str_chunks)
    if scores is None or len(scores) != len(str_chunks):
        return chunks[:top_k]

    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:top_k]]

    logger.debug(
        f"Reranker: kept {len(top_chunks)}/{len(chunks)} chunks "
        f"| top score={scored[0][0]:.4f} | bottom kept score={scored[top_k - 1][0]:.4f}"
    )
    return top_chunks


def rerank_dicts(
    query: str,
    records: list[dict[str, Any]],
    top_k: int,
    content_key: str = "content",
) -> list[dict[str, Any]]:
    """Score a list of chunk dictionaries and return the *top_k* highest-scoring ones.

    Each returned dictionary is updated with a ``rerank_score`` key.

    Args:
        query:       The user's question / search query.
        records:     List of candidate dicts (e.g. from match_material_chunks).
        top_k:       Number of records to keep.
        content_key: Dictionary key containing the text content to score.

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

    texts = [str(r.get(content_key, "")) for r in records]
    scores = _compute_scores(query, texts)

    if scores is None or len(scores) != len(records):
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

    logger.debug(
        f"Reranker: kept {len(top_records)}/{len(records)} dicts "
        f"| top score={scored_records[0][0]:.4f} | bottom kept score={scored_records[top_k - 1][0]:.4f}"
    )
    return top_records
