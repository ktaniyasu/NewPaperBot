from __future__ import annotations

import argparse
import csv
import itertools
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

from ..ingestion.pdf_reader import extract as extract_pdf_text
from ..ingestion.chunking import splitText
from ..rag.embedding import EmbeddingModel
from ..rag.vector_store import FaissVectorStore
from ..utils.config import settings


@dataclass
class EvalResult:
    pdfPath: str
    nChunksTotal: int
    minSimilarity: float
    topK: int
    usedTopK: int
    filteredCount: int
    selectedCount: int
    selectedChars: int
    avgScore: float
    buildMs: int
    searchMs: int
    fallbackUsed: int


def estimate_tokens(s: str) -> int:
    try:
        return max(1, len(s) // 4)
    except Exception:
        return max(1, len(str(s)) // 4)


def fits_context(text: str, prompt: str) -> bool:
    input_tokens = estimate_tokens(text) + estimate_tokens(prompt)
    reserve = settings.LLM_MAX_TOKENS + settings.LLM_THINKING_BUDGET
    budget = max(1024, settings.LLM_CONTEXT_WINDOW_TOKENS - reserve)
    return input_tokens <= budget


def greedy_pack(results: List[Tuple[str, float]], prompt: str) -> Tuple[List[Tuple[str, float]], int]:
    buf: list[Tuple[str, float]] = []
    for chunk, score in results:
        candidate_text = "\n\n".join([c for c, _ in buf] + [chunk])
        if fits_context(candidate_text, prompt):
            buf.append((chunk, score))
        else:
            break
    total_chars = sum(len(c) for c, _ in buf)
    return buf, total_chars


def evaluate_single_pdf(pdf_path: str, queries: Iterable[str], min_sims: Iterable[float], top_ks: Iterable[int]) -> List[EvalResult]:
    raw_text = asyncio_run_extract(pdf_path)
    chunks = splitText(raw_text)
    if not chunks:
        return []

    # Build index once per PDF
    t0 = time.perf_counter()
    embedder = EmbeddingModel.from_pretrained(settings.RAG_EMBED_MODEL)
    vectors = embedder.embed(chunks)
    store = FaissVectorStore(int(vectors.shape[1]))
    store.add(vectors, chunks)
    build_ms = int((time.perf_counter() - t0) * 1000)

    results: list[EvalResult] = []
    for query, min_sim, initial_top_k in itertools.product(queries, min_sims, top_ks):
        t1 = time.perf_counter()
        qvec = embedder.embed([query])[0]

        # PaperAnalyzer と同等: top_k 段階縮小 + しきい値フィルタ + 貪欲パッキング + 最上位1件フォールバック
        top_k = int(initial_top_k)
        used_top_k = top_k
        filtered_count = 0
        packed: List[Tuple[str, float]] = []
        total_chars = 0
        fallback_used = 0
        results_cache: List[Tuple[str, float, int]] | None = None

        while top_k >= 1:
            hits = store.search(qvec, top_k=top_k)
            results_cache = hits
            filtered_pairs: List[Tuple[str, float]] = [(p, s) for (p, s, _i) in hits if s >= float(min_sim)]
            filtered_count = len(filtered_pairs)
            if not filtered_pairs:
                next_top_k = max(1, top_k // 2)
                if next_top_k == top_k:
                    break
                top_k = next_top_k
                continue
            candidate, chars = greedy_pack(filtered_pairs, query)
            if candidate:
                packed = candidate
                total_chars = chars
                used_top_k = top_k
                break
            next_top_k = max(1, top_k // 2)
            if next_top_k == top_k:
                break
            top_k = next_top_k

        if not packed:
            # 何も入らなければ最上位1件でフォールバック
            if results_cache:
                top1_text, top1_score, _ = results_cache[0]
                packed = [(top1_text, top1_score)]
                total_chars = len(top1_text)
                fallback_used = 1
                used_top_k = max(1, top_k)

        search_ms = int((time.perf_counter() - t1) * 1000)
        avg_score = sum(s for _c, s in packed) / len(packed) if packed else 0.0

        results.append(
            EvalResult(
                pdfPath=str(pdf_path),
                nChunksTotal=len(chunks),
                minSimilarity=float(min_sim),
                topK=int(initial_top_k),
                usedTopK=int(used_top_k),
                filteredCount=int(filtered_count),
                selectedCount=len(packed),
                selectedChars=total_chars,
                avgScore=float(avg_score),
                buildMs=build_ms,
                searchMs=search_ms,
                fallbackUsed=int(fallback_used),
            )
        )
    return results


def asyncio_run_extract(pdf_path: str) -> str:
    # Local helper to run async extract synchronously (tests are sync-only policy)
    import asyncio

    return asyncio.run(extract_pdf_text(pdf_path))


def find_pdfs(paths: List[str]) -> List[str]:
    out: list[str] = []
    for p in paths:
        pp = Path(p)
        if pp.is_file() and pp.suffix.lower() == ".pdf":
            out.append(str(pp))
        elif pp.is_dir():
            out.extend([str(x) for x in pp.rglob("*.pdf")])
    return sorted(out)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate RAG retrieval (offline) by sweeping parameters.")
    ap.add_argument("--pdf", nargs="*", default=[], help="PDF file paths to evaluate")
    ap.add_argument("--pdf-dir", default=None, help="Directory containing PDFs (recursively)")
    ap.add_argument("--query", default=None, help="Query text for retrieval. If omitted, uses first chunk of each PDF.")
    ap.add_argument("--min-sim", nargs="*", type=float, default=[0.15, 0.2, 0.25, 0.3], help="Similarity thresholds to sweep")
    ap.add_argument("--top-k", nargs="*", type=int, default=[4, 6, 8, 12], help="Top-K values to sweep")
    ap.add_argument("--out", default=None, help="Output CSV path (default: logs/eval/rag_eval_YYYYmmdd_HHMMSS.csv)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    pdfs: list[str] = []
    if args.pdf:
        pdfs.extend(args.pdf)
    if args.pdf_dir:
        pdfs.extend(find_pdfs([args.pdf_dir]))
    pdfs = find_pdfs(pdfs)
    if not pdfs:
        print("No PDFs found. Specify --pdf paths or --pdf-dir.")
        return

    # Prepare output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else Path("logs/eval") / f"rag_eval_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[EvalResult] = []
    for pdf in pdfs:
        # default query: first chunk
        default_query = None
        try:
            # lightweight: read first chunk only
            raw_text = asyncio_run_extract(pdf)
            chunks = splitText(raw_text)
            default_query = chunks[0] if chunks else ""
        except Exception:
            default_query = ""
        queries = [args.query] if args.query else [default_query]
        rows = evaluate_single_pdf(pdf, queries, args.min_sim, args.top_k)
        all_rows.extend(rows)

    # Write CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pdf_path",
            "n_chunks_total",
            "min_similarity",
            "top_k",
            "used_top_k",
            "filtered_count",
            "selected_count",
            "selected_chars",
            "avg_score",
            "build_ms",
            "search_ms",
            "fallback_used",
        ])
        for r in all_rows:
            w.writerow([
                r.pdfPath,
                r.nChunksTotal,
                f"{r.minSimilarity:.3f}",
                r.topK,
                r.usedTopK,
                r.filteredCount,
                r.selectedCount,
                r.selectedChars,
                f"{r.avgScore:.4f}",
                r.buildMs,
                r.searchMs,
                r.fallbackUsed,
            ])

    print(f"Saved evaluation results: {out_path}")


if __name__ == "__main__":
    main()
