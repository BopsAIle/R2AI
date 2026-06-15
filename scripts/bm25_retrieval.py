"""BM25 sparse retrieval and hybrid fusion with dense (vector) search."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if len(t) > 1]


def chunk_key(payload: dict[str, Any]) -> str:
    doc_id = str(payload.get("doc_id", ""))
    chunk_id = str(payload.get("chunk_id", ""))
    if doc_id or chunk_id:
        return f"{doc_id}:{chunk_id}"
    return str(payload.get("point_id", ""))


def payload_from_qdrant(point) -> dict[str, Any]:
    payload = dict(point.payload or {})
    payload["point_id"] = str(point.id)
    return payload


def scroll_all_chunks(client: QdrantClient, collection: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = payload_from_qdrant(point)
            text = str(payload.get("text") or "").strip()
            if text:
                chunks.append(payload)
        if offset is None:
            break
    return chunks


def load_or_build_bm25_index(
    client: QdrantClient,
    collection: str,
    cache_path: Path | None,
) -> tuple[BM25Okapi, list[dict[str, Any]]]:
    if cache_path and cache_path.is_file():
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        if cached.get("collection") == collection:
            print(f"  BM25 cache: {cache_path} ({len(cached['corpus'])} chunks)")
            return cached["bm25"], cached["corpus"]

    print(f"  Building BM25 index from Qdrant collection '{collection}'...")
    corpus = scroll_all_chunks(client, collection)
    if not corpus:
        raise RuntimeError(f"Không có chunk trong collection '{collection}'")

    tokenized = [tokenize(c.get("text", "")) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    print(f"  BM25 index: {len(corpus)} documents")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump({"collection": collection, "corpus": corpus, "bm25": bm25}, f)
        print(f"  Saved BM25 cache: {cache_path}")

    return bm25, corpus


def bm25_top_k(
    bm25: BM25Okapi,
    corpus: list[dict[str, Any]],
    query: str,
    top_k: int,
) -> list[tuple[dict[str, Any], float]]:
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(corpus[i], float(score)) for i, score in ranked if score > 0]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float | None]]],
    *,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists keyed by chunk_key. Each item: (key, optional raw score)."""
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (key, _) in enumerate(ranked, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def chunk_record(
    payload: dict[str, Any],
    *,
    rank: int,
    score: float,
    dense_score: float | None = None,
    bm25_score: float | None = None,
    rerank_score: float | None = None,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": score,
        "rrf_score": score,
        "dense_score": dense_score,
        "bm25_score": bm25_score,
        "rerank_score": rerank_score,
        "law_type": payload.get("law_type", ""),
        "law_code": payload.get("law_code", ""),
        "law_title": payload.get("law_title", ""),
        "file_name": payload.get("file_name", ""),
        "article_number": payload.get("article_number", ""),
        "text": payload.get("text", ""),
    }


def hybrid_retrieve_one(
    query: str,
    *,
    dense_hits: list,
    bm25: BM25Okapi,
    corpus: list[dict[str, Any]],
    top_k: int,
    pool_size: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    dense_ranked: list[tuple[str, float | None]] = []
    payload_by_key: dict[str, dict[str, Any]] = {}
    dense_score_by_key: dict[str, float] = {}

    for hit in dense_hits[:pool_size]:
        payload = dict(hit.payload or {})
        payload["point_id"] = str(hit.id)
        key = chunk_key(payload)
        payload_by_key[key] = payload
        dense_score_by_key[key] = float(hit.score)
        dense_ranked.append((key, hit.score))

    bm25_hits = bm25_top_k(bm25, corpus, query, pool_size)
    bm25_ranked: list[tuple[str, float | None]] = []
    bm25_score_by_key: dict[str, float] = {}
    for payload, score in bm25_hits:
        key = chunk_key(payload)
        payload_by_key[key] = payload
        bm25_score_by_key[key] = score
        bm25_ranked.append((key, score))

    fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked], rrf_k=rrf_k)[:top_k]

    chunks: list[dict[str, Any]] = []
    for rank, (key, fused_score) in enumerate(fused, start=1):
        payload = payload_by_key[key]
        chunks.append(
            chunk_record(
                payload,
                rank=rank,
                score=fused_score,
                dense_score=dense_score_by_key.get(key),
                bm25_score=bm25_score_by_key.get(key),
            )
        )
    return chunks
