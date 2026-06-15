#!/usr/bin/env python3
"""RAG pipeline: retrieve chunks from Qdrant and answer R2AIStage1 questions."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bm25_retrieval import hybrid_retrieve_one, load_or_build_bm25_index
from rerank_retrieval import dense_hits_to_chunks, load_reranker, rerank_chunks

DEFAULT_QUESTIONS = ROOT / "test" / "R2AIStage1DATA.json"
DEFAULT_EMBED_MODEL = ROOT / "models" / "Vietnamese_Embedding_v2"
DEFAULT_LLM_MODEL = ROOT / "models" / "Qwen3-4B-VietNamese-Legal-Chat"
DEFAULT_RERANK_MODEL = ROOT / "models" / "Vietnamese_Reranker"
DEFAULT_OUTPUT = ROOT / "test" / "R2AIStage1_answers.json"
DEFAULT_RETRIEVED = ROOT / "test" / "R2AIStage1_retrieved.json"
DEFAULT_BM25_CACHE = ROOT / "output" / "bm25_corpus.pkl"

_ARTICLE_RE = re.compile(r"Điều\s+(\d+[a-zA-Z]?)", re.IGNORECASE)
_CODE_RE = re.compile(
    r"\b(\d{1,3}/\d{4}/(?:QH\d+|NĐ-CP|TT-[A-Z]+|QĐ-[A-Z]+|NQ-HĐND))\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--retrieved-cache", type=Path, default=DEFAULT_RETRIEVED)
    p.add_argument("--embed-model", type=Path, default=DEFAULT_EMBED_MODEL)
    p.add_argument("--llm-model", type=Path, default=DEFAULT_LLM_MODEL)
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--collection", default="legal_documents")
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Final chunks after rerank",
    )
    p.add_argument(
        "--llm-top-k",
        type=int,
        default=10,
        help="Chunks for LLM context and relevant_articles submission (~6 điều/câu, tối ưu F2)",
    )
    p.add_argument("--retrieve-batch", type=int, default=4)
    p.add_argument("--gen-batch", type=int, default=2, help="LLM generation batch size")
    p.add_argument("--max-context-chars", type=int, default=3500)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-id", type=int, default=1)
    p.add_argument("--skip-retrieve", action="store_true")
    p.add_argument("--skip-answered", action="store_true", help="Skip question ids already in output file")
    p.add_argument("--retrieve-only", action="store_true")
    p.add_argument(
        "--citations-only",
        action="store_true",
        help="Chỉ retrieve và xuất id, relevant_docs, relevant_articles (không dùng LLM)",
    )
    p.add_argument("--device-embed", default="cuda")
    p.add_argument("--device-llm", default="cuda")
    p.add_argument(
        "--include-chunks",
        action="store_true",
        help="Include retrieved_chunks in output JSON (debug)",
    )
    p.add_argument(
        "--use-bm25",
        action="store_true",
        help="Hybrid retrieval: dense (Qdrant) + BM25, fused by RRF",
    )
    p.add_argument(
        "--bm25-cache",
        type=Path,
        default=DEFAULT_BM25_CACHE,
        help="Cache file for BM25 corpus index",
    )
    p.add_argument(
        "--retrieve-pool",
        type=int,
        default=20,
        help="Candidate pool per retrieval method (ANN / BM25) before fusion",
    )
    p.add_argument(
        "--rrf-top-k",
        type=int,
        default=15,
        help="Chunks kept after RRF fusion, before rerank (hybrid only)",
    )
    p.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant k for hybrid score fusion",
    )
    p.add_argument(
        "--rerank-model",
        type=Path,
        default=DEFAULT_RERANK_MODEL,
        help="Cross-encoder reranker model path",
    )
    p.add_argument("--device-rerank", default="cuda")
    p.add_argument("--rerank-batch", type=int, default=8)
    p.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking",
    )
    return p.parse_args()


def _looks_bad_title(title: str) -> bool:
    if not title or len(title) > 180:
        return True
    stripped = title.strip()
    if not stripped or all(c in "-_ " for c in stripped):
        return True
    lowered = stripped.lower()
    return lowered.startswith("căn cứ") or lowered.startswith("theo ")


def _title_from_file_name(file_name: str, law_code: str) -> str:
    if not file_name:
        return ""
    stem = Path(file_name).stem
    code_token = law_code.replace("/", "_") if law_code else ""
    if code_token and code_token in stem:
        rest = stem.split(code_token, 1)[-1].lstrip("_")
        if rest:
            return rest.replace("_", " ").strip()
    parts = stem.split("_", 2)
    if len(parts) >= 3:
        return parts[2].replace("_", " ").strip()
    return stem.replace("_", " ").strip()


def _title_from_text(text: str, law_code: str) -> str:
    if not text:
        return ""
    head = text[:500]
    if law_code:
        pattern = re.compile(
            rf"{re.escape(law_code)}\s+(.+?)(?:\n|$)",
            re.IGNORECASE,
        )
        match = pattern.search(head)
        if match:
            return match.group(1).strip(" -")
    first_line = head.split("\n", 1)[0].strip()
    if law_code and law_code in first_line:
        return first_line.split(law_code, 1)[-1].strip(" -")
    return first_line


def resolve_law_code(chunk: dict) -> str:
    code = str(chunk.get("law_code") or "").strip()
    if code:
        return code
    text = chunk.get("text") or ""
    match = _CODE_RE.search(text[:400])
    return match.group(1) if match else ""


def resolve_law_title(chunk: dict) -> str:
    title = str(chunk.get("law_title") or "").strip()
    law_code = resolve_law_code(chunk)
    if not _looks_bad_title(title):
        return title
    title = _title_from_file_name(chunk.get("file_name") or "", law_code)
    if not _looks_bad_title(title):
        return title
    return _title_from_text(chunk.get("text") or "", law_code)


def is_valid_chunk(chunk: dict) -> bool:
    law_code = resolve_law_code(chunk)
    law_title = resolve_law_title(chunk)
    return bool(law_code) and bool(law_title) and not _looks_bad_title(law_title)


def filter_valid_chunks(chunks: list[dict], limit: int | None = None) -> list[dict]:
    valid = [c for c in chunks if is_valid_chunk(c)]
    if limit is not None:
        valid = valid[:limit]
    for rank, chunk in enumerate(valid, start=1):
        chunk["rank"] = rank
    return valid


def resolve_article_label(chunk: dict) -> str | None:
    article = str(chunk.get("article_number") or "").strip()
    if article:
        return f"Điều {article}" if not article.lower().startswith("điều") else article
    text = chunk.get("text") or ""
    match = _ARTICLE_RE.search(text[:800])
    if match:
        return f"Điều {match.group(1)}"
    return None


def chunk_to_doc_ref(chunk: dict) -> str | None:
    if not is_valid_chunk(chunk):
        return None
    law_code = resolve_law_code(chunk)
    law_title = resolve_law_title(chunk)
    return f"{law_code}|{law_title}"


def chunk_to_article_ref(chunk: dict) -> str | None:
    if not is_valid_chunk(chunk):
        return None
    law_code = resolve_law_code(chunk)
    law_title = resolve_law_title(chunk)
    article = resolve_article_label(chunk)
    if not article:
        return None
    return f"{law_code}|{law_title}|{article}"


def extract_citations(chunks: list[dict]) -> tuple[list[str], list[str]]:
    relevant_docs: list[str] = []
    relevant_articles: list[str] = []
    seen_docs: set[str] = set()
    seen_articles: set[str] = set()
    for chunk in filter_valid_chunks(chunks):
        doc_ref = chunk_to_doc_ref(chunk)
        if doc_ref and doc_ref not in seen_docs:
            seen_docs.add(doc_ref)
            relevant_docs.append(doc_ref)
        article_ref = chunk_to_article_ref(chunk)
        if article_ref and article_ref not in seen_articles:
            seen_articles.add(article_ref)
            relevant_articles.append(article_ref)
    return relevant_docs, relevant_articles


def build_citation_results(items: list[dict], llm_top_k: int) -> list[dict]:
    results: list[dict] = []
    for item in items:
        relevant_docs, relevant_articles = extract_citations(item["chunks"][:llm_top_k])
        results.append({
            "id": item["id"],
            "question": item.get("question", ""),
            "answer": "",
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
        })
    return results


def load_questions(path: Path, start_id: int, limit: int | None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = [q for q in data if q.get("id", 0) >= start_id]
    if limit:
        items = items[:limit]
    return items


class RetrievalEngine:
    """Embedding + BM25 + rerank retrieval; supports GPU offload between batches."""

    def __init__(
        self,
        *,
        embed_model_path: Path,
        qdrant_url: str,
        collection: str,
        top_k: int,
        device: str,
        use_bm25: bool = False,
        bm25_cache: Path | None = None,
        retrieve_pool: int = 50,
        rrf_top_k: int = 50,
        rrf_k: int = 60,
        use_rerank: bool = True,
        rerank_model_path: Path | None = None,
        device_rerank: str = "cuda",
        rerank_batch: int = 32,
    ) -> None:
        self.embed_model_path = embed_model_path
        self.qdrant_url = qdrant_url
        self.collection = collection
        self.top_k = top_k
        self.device = device
        self.use_bm25 = use_bm25
        self.bm25_cache = bm25_cache
        self.pool_size = retrieve_pool
        self.rrf_top_k = rrf_top_k
        self.rrf_k = rrf_k
        self.use_rerank = use_rerank
        self.rerank_model_path = rerank_model_path
        self.device_rerank = device_rerank
        self.rerank_batch = rerank_batch

        self.fusion_top_k = rrf_top_k if (use_bm25 and use_rerank) else top_k
        self.ann_limit = retrieve_pool if (use_bm25 or use_rerank) else top_k

        self.client: QdrantClient | None = None
        self.embed_model: SentenceTransformer | None = None
        self.reranker = None
        self.bm25 = None
        self.corpus = None
        self._on_gpu = False

    def load(self) -> None:
        if self.client is not None:
            return
        self.client = QdrantClient(url=self.qdrant_url)
        if self.use_bm25:
            self.bm25, self.corpus = load_or_build_bm25_index(
                self.client, self.collection, self.bm25_cache
            )
            if self.use_rerank:
                print(
                    f"  Hybrid retrieve: dense + BM25 + rerank "
                    f"(pool={self.pool_size}, rrf_top_k={self.rrf_top_k}, "
                    f"top_k={self.top_k}, rrf_k={self.rrf_k})"
                )
            else:
                print(
                    f"  Hybrid retrieve: dense + BM25 "
                    f"(pool={self.pool_size}, top_k={self.top_k}, rrf_k={self.rrf_k})"
                )
        elif self.use_rerank:
            print(f"  Dense retrieve + rerank (pool={self.pool_size}, top_k={self.top_k})")

        self.ensure_on_gpu()

    def ensure_on_gpu(self) -> None:
        if self.embed_model is None:
            self.embed_model = SentenceTransformer(str(self.embed_model_path), device=self.device)
            self.embed_model.max_seq_length = 2048
        elif not self._on_gpu and self.device == "cuda":
            self.embed_model.to(self.device)

        if self.use_rerank and self.reranker is None:
            self.reranker = load_reranker(
                self.rerank_model_path or DEFAULT_RERANK_MODEL,
                self.device_rerank,
            )
        elif (
            self.use_rerank
            and self.reranker is not None
            and not self._on_gpu
            and self.device_rerank == "cuda"
        ):
            self.reranker.model.to(self.device_rerank)

        self._on_gpu = self.device == "cuda" or self.device_rerank == "cuda"

    def offload_gpu(self) -> None:
        if self.embed_model is not None and self.device == "cuda":
            self.embed_model.to("cpu")
        if self.reranker is not None and self.device_rerank == "cuda":
            self.reranker.model.to("cpu")
        self._on_gpu = False
        if self.device == "cuda" or self.device_rerank == "cuda":
            torch.cuda.empty_cache()

    def unload(self) -> None:
        self.offload_gpu()
        self.embed_model = None
        self.reranker = None
        self.client = None
        self.bm25 = None
        self.corpus = None

    def retrieve_batch(self, questions: list[dict]) -> list[dict]:
        if not questions:
            return []
        self.load()
        self.ensure_on_gpu()
        assert self.client is not None and self.embed_model is not None

        results: list[dict] = []
        texts = [q["question"] for q in questions]
        vectors = self.embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

        for q_item, vec in zip(questions, vectors):
            hits = self.client.query_points(
                collection_name=self.collection,
                query=vec.tolist(),
                limit=self.ann_limit,
            )
            if self.use_bm25:
                chunks = hybrid_retrieve_one(
                    q_item["question"],
                    dense_hits=hits.points,
                    bm25=self.bm25,
                    corpus=self.corpus,
                    top_k=self.fusion_top_k,
                    pool_size=self.pool_size,
                    rrf_k=self.rrf_k,
                )
            else:
                chunks = dense_hits_to_chunks(hits.points)

            if self.use_rerank and self.reranker is not None:
                chunks = rerank_chunks(
                    q_item["question"],
                    chunks,
                    self.reranker,
                    top_k=len(chunks),
                    batch_size=self.rerank_batch,
                )
            chunks = filter_valid_chunks(chunks, limit=self.top_k)

            results.append({
                "id": q_item["id"],
                "question": q_item["question"],
                "chunks": chunks,
            })
        return results


def build_context(chunks: list[dict], max_chars: int, max_chunks: int | None = None) -> str:
    chunks = filter_valid_chunks(chunks, limit=max_chunks)
    parts = []
    total = 0
    for c in chunks:
        law_code = resolve_law_code(c)
        law_title = resolve_law_title(c)
        article = resolve_article_label(c) or ""
        header = f"[{law_code}|{law_title}|{article}]".rstrip("|")
        block = f"{header}\n{c.get('text', '')}"
        if total + len(block) > max_chars:
            remain = max_chars - total
            if remain > 200:
                parts.append(block[:remain])
            break
        parts.append(block)
        total += len(block) + 4
    return "\n\n---\n\n".join(parts)


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    system = (
        "Bạn là chuyên gia pháp luật Việt Nam. Hãy trả lời câu hỏi pháp luật dựa trên các đoạn văn bản được cung cấp.\n"
        "- Chỉ dựa vào ngữ cảnh, không bịa thêm.\n"
        "- Viết một đoạn văn liền mạch, trả lời trực tiếp câu hỏi, nêu đủ điều kiện, mức hỗ trợ, thời hạn, trách nhiệm nếu có.\n"
        "- Không thêm tiêu đề, không thêm mục Kết luận/Phân tích, không liệt kê bullet."
    )
    user = f"Ngữ cảnh pháp luật:\n{context}\n\nCâu hỏi: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_prompt(tokenizer, question: str, context: str) -> str:
    messages = build_messages(question, context)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def clean_answer(text: str) -> str:
    text = text.strip()
    think_close = "</" + "redacted_thinking>"
    think_open = "<" + "redacted_thinking>"
    if think_close in text:
        text = text.split(think_close, 1)[-1].strip()
    text = re.sub(rf"{re.escape(think_open)}.*?{re.escape(think_close)}", "", text, flags=re.DOTALL).strip()
    for marker in ("Câu hỏi:", "assistant", "Ngữ cảnh pháp luật:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    text = re.sub(r"\s+", " ", text).strip()
    lines = [line.strip() for line in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if line in {"Có", "Không"} and cleaned and cleaned[-1] == line:
            continue
        cleaned.append(line)
    return " ".join(cleaned).strip()


def generate_batch_answers(
    model,
    tokenizer,
    batch_items: list[dict],
    *,
    max_context_chars: int,
    max_new_tokens: int,
    eos_ids: list[int],
    llm_top_k: int,
) -> list[dict]:
    prompts = [
        build_prompt(
            tokenizer,
            item["question"],
            build_context(item["chunks"], max_context_chars, max_chunks=llm_top_k),
        )
        for item in batch_items
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=6144,
    ).to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    results: list[dict] = []
    for item, seq in zip(batch_items, generated):
        answer = clean_answer(
            tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        )
        relevant_docs, relevant_articles = extract_citations(item["chunks"][:llm_top_k])
        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "relevant_docs": relevant_docs,
            "relevant_articles": relevant_articles,
        })
    return results


@dataclass
class AdaptiveGenBatch:
    """Gen batch size; tự giảm một nửa khi OOM (8→4→2→1)."""

    size: int

    def try_reduce(self) -> bool:
        if self.size <= 1:
            return False
        new_size = max(1, self.size // 2)
        print(f"  OOM gen_batch {self.size} → giảm xuống {new_size}")
        self.size = new_size
        return True


def generate_single_answer_safe(
    model,
    tokenizer,
    item: dict,
    *,
    max_context_chars: int,
    max_new_tokens: int,
    eos_ids: list[int],
    llm_top_k: int,
    device: str,
) -> dict:
    batch_items = [item]
    try:
        return generate_batch_answers(
            model,
            tokenizer,
            batch_items,
            max_context_chars=max_context_chars,
            max_new_tokens=max_new_tokens,
            eos_ids=eos_ids,
            llm_top_k=llm_top_k,
        )[0]
    except torch.cuda.OutOfMemoryError:
        if device == "cuda":
            torch.cuda.empty_cache()
        if llm_top_k > 5:
            reduced_k = max(5, llm_top_k // 2)
            reduced_chars = max(1500, max_context_chars // 2)
            print(
                f"  OOM câu id={item['id']} → "
                f"llm_top_k={reduced_k}, max_context_chars={reduced_chars}"
            )
            return generate_single_answer_safe(
                model,
                tokenizer,
                item,
                max_context_chars=reduced_chars,
                max_new_tokens=max_new_tokens,
                eos_ids=eos_ids,
                llm_top_k=reduced_k,
                device=device,
            )
        if max_new_tokens > 128:
            reduced_tokens = max(128, max_new_tokens // 2)
            print(f"  OOM câu id={item['id']} → max_new_tokens={reduced_tokens}")
            return generate_single_answer_safe(
                model,
                tokenizer,
                item,
                max_context_chars=max_context_chars,
                max_new_tokens=reduced_tokens,
                eos_ids=eos_ids,
                llm_top_k=llm_top_k,
                device=device,
            )
        raise


def generate_items_adaptive(
    model,
    tokenizer,
    items: list[dict],
    *,
    gen_batch: AdaptiveGenBatch,
    max_context_chars: int,
    max_new_tokens: int,
    eos_ids: list[int],
    llm_top_k: int,
    device: str,
) -> list[dict]:
    """Generate answers; giảm gen_batch khi OOM và giữ size mới cho các batch sau."""
    results: list[dict] = []
    idx = 0
    while idx < len(items):
        chunk = items[idx : idx + gen_batch.size]
        try:
            batch_results = generate_batch_answers(
                model,
                tokenizer,
                chunk,
                max_context_chars=max_context_chars,
                max_new_tokens=max_new_tokens,
                eos_ids=eos_ids,
                llm_top_k=llm_top_k,
            )
            results.extend(batch_results)
            idx += len(chunk)
        except torch.cuda.OutOfMemoryError:
            if device == "cuda":
                torch.cuda.empty_cache()
            if len(chunk) > 1 and gen_batch.try_reduce():
                continue
            for item in chunk:
                results.append(
                    generate_single_answer_safe(
                        model,
                        tokenizer,
                        item,
                        max_context_chars=max_context_chars,
                        max_new_tokens=max_new_tokens,
                        eos_ids=eos_ids,
                        llm_top_k=llm_top_k,
                        device=device,
                    )
                )
            idx += len(chunk)
    return results


def load_llm(llm_model_path: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(str(llm_model_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(llm_model_path),
        dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)

    eos_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_ids.append(im_end_id)
    return model, tokenizer, eos_ids


def offload_llm(model, device: str) -> None:
    """Move LLM off GPU without reloading from disk."""
    if device == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()


def unload_llm(model, device: str) -> None:
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def ensure_llm_on_device(model, device: str) -> None:
    if device == "cuda" and next(model.parameters()).device.type != "cuda":
        model.to(device)


def append_retrieved_cache(cache_path: Path, new_items: list[dict]) -> None:
    if not new_items:
        return
    save_json(cache_path, merge_existing(cache_path, new_items))


def run_citations_pipeline(
    questions: list[dict],
    *,
    args: argparse.Namespace,
) -> int:
    """Retrieve chunks and export citations only (no LLM)."""
    total = len(questions)
    pipeline_batch = args.retrieve_batch
    engine = RetrievalEngine(
        embed_model_path=args.embed_model,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        top_k=args.top_k,
        device=args.device_embed,
        use_bm25=args.use_bm25,
        bm25_cache=args.bm25_cache,
        retrieve_pool=args.retrieve_pool,
        rrf_top_k=args.rrf_top_k,
        rrf_k=args.rrf_k,
        use_rerank=not args.no_rerank,
        rerank_model_path=args.rerank_model,
        device_rerank=args.device_rerank,
        rerank_batch=args.rerank_batch,
    )

    citation_count = 0
    print(
        f"=== Citations only: retrieve + extract (batch={pipeline_batch}, "
        f"llm_top_k={args.llm_top_k}) ==="
    )
    if (
        args.output is not None
        and not args.skip_answered
        and args.start_id <= 1
        and args.limit is None
    ):
        save_json(args.output, [])
    t0 = time.time()

    try:
        for start in range(0, total, pipeline_batch):
            batch_qs = questions[start : start + pipeline_batch]
            retrieved_batch = engine.retrieve_batch(batch_qs)
            done_retrieve = min(start + pipeline_batch, total)
            print(f"  Retrieved {done_retrieve}/{total}")

            append_retrieved_cache(args.retrieved_cache, retrieved_batch)

            batch_results = build_citation_results(retrieved_batch, args.llm_top_k)
            citation_count += len(batch_results)

            if args.output is not None and batch_results:
                save_json(args.output, merge_existing(args.output, batch_results))

            done = min(start + pipeline_batch, total)
            print(f"  Done {done}/{total} (retrieve + citations)")
            del retrieved_batch
    finally:
        engine.unload()

    print(f"Citations xong: {citation_count} câu ({time.time() - t0:.1f}s)")
    return citation_count


def run_interleaved_pipeline(
    questions: list[dict],
    *,
    args: argparse.Namespace,
) -> int:
    """Retrieve and generate per batch; only one pipeline batch of chunks in RAM."""
    total = len(questions)
    pipeline_batch = args.retrieve_batch
    engine = RetrievalEngine(
        embed_model_path=args.embed_model,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        top_k=args.top_k,
        device=args.device_embed,
        use_bm25=args.use_bm25,
        bm25_cache=args.bm25_cache,
        retrieve_pool=args.retrieve_pool,
        rrf_top_k=args.rrf_top_k,
        rrf_k=args.rrf_k,
        use_rerank=not args.no_rerank,
        rerank_model_path=args.rerank_model,
        device_rerank=args.device_rerank,
        rerank_batch=args.rerank_batch,
    )

    llm_model = None
    tokenizer = None
    eos_ids: list[int] = []
    answer_count = 0
    gen_batch = AdaptiveGenBatch(args.gen_batch)

    print(
        f"=== Pipeline: retrieve + generate (batch={pipeline_batch}, "
        f"gen_batch={gen_batch.size}) ==="
    )
    if args.output is not None and not args.skip_answered:
        save_json(args.output, [])
    t0 = time.time()

    try:
        for start in range(0, total, pipeline_batch):
            batch_qs = questions[start : start + pipeline_batch]
            retrieved_batch = engine.retrieve_batch(batch_qs)
            done_retrieve = min(start + pipeline_batch, total)
            print(f"  Retrieved {done_retrieve}/{total}")

            append_retrieved_cache(args.retrieved_cache, retrieved_batch)

            engine.offload_gpu()
            if llm_model is None:
                llm_model, tokenizer, eos_ids = load_llm(args.llm_model, args.device_llm)
            else:
                ensure_llm_on_device(llm_model, args.device_llm)

            gen_start = 0
            while gen_start < len(retrieved_batch):
                gen_items = retrieved_batch[gen_start : gen_start + gen_batch.size]
                batch_results = generate_items_adaptive(
                    llm_model,
                    tokenizer,
                    gen_items,
                    gen_batch=gen_batch,
                    max_context_chars=args.max_context_chars,
                    max_new_tokens=args.max_new_tokens,
                    eos_ids=eos_ids,
                    llm_top_k=args.llm_top_k,
                    device=args.device_llm,
                )
                if args.device_llm == "cuda":
                    torch.cuda.empty_cache()

                for result in batch_results:
                    if args.include_chunks:
                        chunks = next(
                            x["chunks"] for x in gen_items if x["id"] == result["id"]
                        )
                        result["retrieved_chunks"] = chunks
                    answer_count += 1

                if args.output is not None and batch_results:
                    save_json(args.output, merge_existing(args.output, batch_results))
                gen_start += len(gen_items)

            done = min(start + pipeline_batch, total)
            print(f"  Done {done}/{total} (retrieve + generate)")
            del retrieved_batch

            if llm_model is not None:
                offload_llm(llm_model, args.device_llm)
    finally:
        engine.unload()
        if llm_model is not None:
            unload_llm(llm_model, args.device_llm)

    print(f"Pipeline xong: {answer_count} câu ({time.time() - t0:.1f}s)")
    return answer_count


def generate_answers(
    retrieved: list[dict],
    *,
    llm_model_path: Path,
    max_context_chars: int,
    max_new_tokens: int,
    llm_top_k: int,
    device: str,
    gen_batch: int,
    output_path: Path | None = None,
    include_chunks: bool = False,
    fresh_output: bool = False,
) -> list[dict]:
    model, tokenizer, eos_ids = load_llm(llm_model_path, device)

    outputs: list[dict] = []
    total = len(retrieved)
    adaptive_batch = AdaptiveGenBatch(gen_batch)
    if fresh_output and output_path is not None:
        save_json(output_path, [])

    start = 0
    while start < total:
        batch_items = retrieved[start : start + adaptive_batch.size]
        batch_results = generate_items_adaptive(
            model,
            tokenizer,
            batch_items,
            gen_batch=adaptive_batch,
            max_context_chars=max_context_chars,
            max_new_tokens=max_new_tokens,
            eos_ids=eos_ids,
            llm_top_k=llm_top_k,
            device=device,
        )
        if device == "cuda":
            torch.cuda.empty_cache()
        for result in batch_results:
            if include_chunks:
                chunks = next(x["chunks"] for x in batch_items if x["id"] == result["id"])
                result["retrieved_chunks"] = chunks
            outputs.append(result)

        done = min(start + len(batch_items), total)
        print(f"  Generated {done}/{total}")
        if output_path is not None:
            if fresh_output:
                existing = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else []
                existing.extend(batch_results)
                save_json(output_path, existing)
            else:
                save_json(output_path, merge_existing(output_path, outputs))
        start += len(batch_items)

    unload_llm(model, device)
    return outputs


def save_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_existing(output_path: Path, new_items: list[dict]) -> list[dict]:
    if not output_path.exists():
        return new_items
    existing = json.loads(output_path.read_text(encoding="utf-8"))
    by_id = {x["id"]: x for x in existing}
    for item in new_items:
        by_id[item["id"]] = item
    return [by_id[k] for k in sorted(by_id)]


def main() -> int:
    args = parse_args()

    if not args.questions.exists():
        print(f"Không tìm thấy: {args.questions}", file=sys.stderr)
        return 1

    questions = load_questions(args.questions, args.start_id, args.limit)
    if args.skip_answered and args.output.exists():
        answered_ids = {
            item["id"] for item in json.loads(args.output.read_text(encoding="utf-8"))
        }
        before = len(questions)
        questions = [q for q in questions if q["id"] not in answered_ids]
        print(f"Bỏ qua {before - len(questions)} câu đã trả lời, còn {len(questions)}")
    print(f"Câu hỏi: {len(questions)} (start_id={args.start_id})")
    if not questions:
        print("Không còn câu hỏi cần xử lý.")
        return 0

    if args.citations_only and args.skip_retrieve and args.retrieved_cache.exists():
        retrieved = json.loads(args.retrieved_cache.read_text(encoding="utf-8"))
        ids = {q["id"] for q in questions}
        retrieved = [r for r in retrieved if r["id"] in ids]
        print(f"Dùng cache retrieve: {len(retrieved)} câu")
        print("=== Extract citations (from cache, no LLM) ===")
        t1 = time.time()
        citations = build_citation_results(retrieved, args.llm_top_k)
        del retrieved
        if args.skip_answered:
            merged = merge_existing(args.output, citations)
            save_json(args.output, merged)
            print(f"Đã lưu: {args.output} ({len(merged)} câu, {time.time()-t1:.1f}s)")
        else:
            save_json(args.output, citations)
            print(f"Đã lưu: {args.output} ({len(citations)} câu, {time.time()-t1:.1f}s)")
        return 0

    if args.citations_only:
        citation_count = run_citations_pipeline(questions, args=args)
        print(f"Đã lưu: {args.output} ({citation_count} câu)")
        return 0

    if args.retrieve_only:
        print("=== Retrieve only (incremental cache) ===")
        t0 = time.time()
        engine = RetrievalEngine(
            embed_model_path=args.embed_model,
            qdrant_url=args.qdrant_url,
            collection=args.collection,
            top_k=args.top_k,
            device=args.device_embed,
            use_bm25=args.use_bm25,
            bm25_cache=args.bm25_cache,
            retrieve_pool=args.retrieve_pool,
            rrf_top_k=args.rrf_top_k,
            rrf_k=args.rrf_k,
            use_rerank=not args.no_rerank,
            rerank_model_path=args.rerank_model,
            device_rerank=args.device_rerank,
            rerank_batch=args.rerank_batch,
        )
        try:
            total = len(questions)
            for start in range(0, total, args.retrieve_batch):
                batch_qs = questions[start : start + args.retrieve_batch]
                retrieved_batch = engine.retrieve_batch(batch_qs)
                append_retrieved_cache(args.retrieved_cache, retrieved_batch)
                done = min(start + args.retrieve_batch, total)
                print(f"  Retrieved {done}/{total}")
                del retrieved_batch
        finally:
            engine.unload()
        print(f"Đã lưu retrieve cache: {args.retrieved_cache} ({time.time()-t0:.1f}s)")
        return 0

    if args.skip_retrieve and args.retrieved_cache.exists():
        retrieved = json.loads(args.retrieved_cache.read_text(encoding="utf-8"))
        ids = {q["id"] for q in questions}
        retrieved = [r for r in retrieved if r["id"] in ids]
        print(f"Dùng cache retrieve: {len(retrieved)} câu")
        print("=== Generate answers (from cache) ===")
        t1 = time.time()
        answers = generate_answers(
            retrieved,
            llm_model_path=args.llm_model,
            max_context_chars=args.max_context_chars,
            max_new_tokens=args.max_new_tokens,
            llm_top_k=args.llm_top_k,
            device=args.device_llm,
            gen_batch=args.gen_batch,
            output_path=args.output,
            include_chunks=args.include_chunks,
            fresh_output=not args.skip_answered,
        )
        del retrieved
        if args.skip_answered:
            merged = merge_existing(args.output, answers)
            save_json(args.output, merged)
            print(f"Đã lưu: {args.output} ({len(merged)} câu, {time.time()-t1:.1f}s)")
        else:
            print(f"Đã lưu: {args.output} ({len(answers)} câu, {time.time()-t1:.1f}s)")
        return 0

    answer_count = run_interleaved_pipeline(questions, args=args)
    print(f"Đã lưu: {args.output} ({answer_count} câu)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
