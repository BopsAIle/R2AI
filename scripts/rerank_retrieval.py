"""Cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder

MAX_RERANK_LENGTH = 2304


def load_reranker(model_path: str | Path, device: str = "cuda") -> CrossEncoder:
    return CrossEncoder(str(model_path), max_length=MAX_RERANK_LENGTH, device=device)


def dense_hits_to_chunks(hits) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        chunks.append(
            {
                "rank": rank,
                "score": float(hit.score),
                "rrf_score": None,
                "dense_score": float(hit.score),
                "bm25_score": None,
                "rerank_score": None,
                "law_type": payload.get("law_type", ""),
                "law_code": payload.get("law_code", ""),
                "law_title": payload.get("law_title", ""),
                "file_name": payload.get("file_name", ""),
                "article_number": payload.get("article_number", ""),
                "text": payload.get("text", ""),
            }
        )
    return chunks


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    reranker: CrossEncoder,
    *,
    top_k: int,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    pairs = [(query, str(c.get("text", ""))) for c in chunks]
    scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    ranked = sorted(zip(chunks, scores), key=lambda x: float(x[1]), reverse=True)[:top_k]

    result: list[dict[str, Any]] = []
    for rank, (chunk, score) in enumerate(ranked, start=1):
        out = dict(chunk)
        if out.get("rrf_score") is None and out.get("bm25_score") is not None:
            out["rrf_score"] = out.get("score")
        elif out.get("rrf_score") is None and out.get("dense_score") is not None and out.get("bm25_score") is None:
            out["rrf_score"] = None
        else:
            out["rrf_score"] = out.get("rrf_score", out.get("score"))
        out["rerank_score"] = float(score)
        out["score"] = float(score)
        out["rank"] = rank
        result.append(out)
    return result
