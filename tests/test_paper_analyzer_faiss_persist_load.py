import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest

from src.llm.paper_analyzer import PaperAnalyzer
from src.llm.providers import ProviderFactory
from src.utils.config import settings


class _FakeProvider:
    def __init__(self):
        self.calls: list[str] = []
        self.last_text: str | None = None

    def supportsFileUpload(self) -> bool:
        return False

    async def generateStructuredFromText(self, prompt: str, text: str, schema: Any, **kwargs):
        self.calls.append("text")
        self.last_text = text
        return {
            "summary": f"S:{text[:5]}",
            "novelty": "N",
            "methodology": "M",
            "results": "R",
            "future_work": "F",
            "research_themes": ["A", "B", "C"],
        }


class _FakeEmbedder:
    def embed(self, texts: list[str]):
        import numpy as np

        return np.zeros((len(texts), 3), dtype=np.float32)


class _LoadedStore:
    def __init__(self, payloads: list[str]):
        self._payloads = payloads

    def search(self, query, top_k: int = 8):
        # deterministic scores, already sorted
        scores = {"c1": 0.9, "c2": 0.7, "c3": 0.4, "c4": 0.1}
        ranked = sorted(((p, scores.get(p, 0.0), i) for i, p in enumerate(self._payloads)), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


async def _fake_extract(_: str) -> str:
    # 長文で単発経路に入らないようにする
    return "X" * 50000


def _fake_split(_: str) -> list[str]:
    return ["c1", "c2", "c3", "c4"]


@pytest.fixture()
def restore_settings():
    orig = {
        "RAG_USE_FAISS": settings.RAG_USE_FAISS,
        "RAG_TOP_K": settings.RAG_TOP_K,
        "RAG_MIN_SIMILARITY": settings.RAG_MIN_SIMILARITY,
        "RAG_INDEX_PERSIST": settings.RAG_INDEX_PERSIST,
        "RAG_INDEX_DIR": settings.RAG_INDEX_DIR,
        "LLM_CONTEXT_WINDOW_TOKENS": settings.LLM_CONTEXT_WINDOW_TOKENS,
        "LLM_MAX_TOKENS": settings.LLM_MAX_TOKENS,
        "LLM_THINKING_BUDGET": settings.LLM_THINKING_BUDGET,
        "RAG_EMBED_MODEL": settings.RAG_EMBED_MODEL,
    }
    try:
        yield
    finally:
        settings.RAG_USE_FAISS = orig["RAG_USE_FAISS"]
        settings.RAG_TOP_K = orig["RAG_TOP_K"]
        settings.RAG_MIN_SIMILARITY = orig["RAG_MIN_SIMILARITY"]
        settings.RAG_INDEX_PERSIST = orig["RAG_INDEX_PERSIST"]
        settings.RAG_INDEX_DIR = orig["RAG_INDEX_DIR"]
        settings.LLM_CONTEXT_WINDOW_TOKENS = orig["LLM_CONTEXT_WINDOW_TOKENS"]
        settings.LLM_MAX_TOKENS = orig["LLM_MAX_TOKENS"]
        settings.LLM_THINKING_BUDGET = orig["LLM_THINKING_BUDGET"]
        settings.RAG_EMBED_MODEL = orig["RAG_EMBED_MODEL"]


def test_faiss_persistence_load_path(tmp_path: Path, monkeypatch, restore_settings):
    # Arrange: 永続化インデックスが存在し、load が呼ばれることを検証
    settings.RAG_USE_FAISS = True
    settings.RAG_INDEX_PERSIST = True
    settings.RAG_INDEX_DIR = str(tmp_path)
    settings.RAG_TOP_K = 4
    settings.RAG_MIN_SIMILARITY = 0.6

    fake = _FakeProvider()
    monkeypatch.setattr(ProviderFactory, "fromSettings", lambda cfg=None: fake)

    # I/O & embeddings
    monkeypatch.setattr("src.llm.paper_analyzer.extract_pdf_text", _fake_extract)
    monkeypatch.setattr("src.llm.paper_analyzer.splitText", lambda t: _fake_split(t))
    monkeypatch.setattr(
        "src.llm.paper_analyzer.EmbeddingModel",
        type("_E", (), {"from_pretrained": staticmethod(lambda name: _FakeEmbedder())}),
    )

    # Compute expected persistence file paths
    text = asyncio.run(_fake_extract("/tmp/dummy.pdf"))
    chunks = _fake_split(text)
    content_key = hashlib.sha1(
        f"{settings.RAG_EMBED_MODEL}\x1f{len(chunks)}\x1f{hashlib.sha1(text.encode('utf-8')).hexdigest()}".encode(
            "utf-8"
        )
    ).hexdigest()
    base = Path(settings.RAG_INDEX_DIR) / content_key
    idx_path = base.with_suffix(".faiss")
    pl_path = base.with_suffix(".payloads.json")

    # Create dummy files to satisfy existence checks
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_bytes(b"")
    pl_path.write_text("[\"c1\", \"c2\", \"c3\", \"c4\"]", encoding="utf-8")

    # Spy: replace FaissVectorStore.load to capture args and return loaded store
    calls: dict[str, Path] = {}

    class _FakeFaiss:
        @classmethod
        def load(cls, ip: str | Path, pp: str | Path):
            calls["idx_path"] = Path(ip)
            calls["pl_path"] = Path(pp)
            return _LoadedStore(["c1", "c2", "c3", "c4"])  # loaded payloads

    monkeypatch.setattr("src.llm.paper_analyzer.FaissVectorStore", _FakeFaiss)

    # Keep context fitting for small chunks only
    monkeypatch.setattr(PaperAnalyzer, "_fits_context", lambda self, text, prompt: len(text) < 100)

    analyzer = PaperAnalyzer()
    out = asyncio.run(analyzer._analyze_via_text_pipeline("/tmp/dummy.pdf", "prompt"))

    # Assert provider called and selected text from retrieved chunks
    assert isinstance(out, dict)
    assert fake.calls.count("text") >= 1
    assert fake.last_text == "c1\n\nc2"  # threshold 0.6 includes c1(0.9), c2(0.7)

    # Assert load was invoked with our persisted paths
    assert calls["idx_path"] == idx_path
    assert calls["pl_path"] == pl_path
