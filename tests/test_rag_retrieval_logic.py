import asyncio
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

        # 形だけのゼロベクトル（cosine 前提で正規化不要）
        return np.zeros((len(texts), 3), dtype=np.float32)


class _FakeVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._payloads: list[str] = []

    def add(self, vectors, payloads):
        self._payloads = list(payloads)

    def search(self, query, top_k: int = 8):
        # スコアを固定（c1>c2>c3>c4）
        scores = {"c1": 0.90, "c2": 0.70, "c3": 0.40, "c4": 0.10}
        ranked = sorted(((p, scores.get(p, 0.0), i) for i, p in enumerate(self._payloads)), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


@pytest.fixture()
def restore_settings():
    orig = {
        "RAG_USE_FAISS": settings.RAG_USE_FAISS,
        "RAG_TOP_K": settings.RAG_TOP_K,
        "RAG_MIN_SIMILARITY": settings.RAG_MIN_SIMILARITY,
        "LLM_CONTEXT_WINDOW_TOKENS": settings.LLM_CONTEXT_WINDOW_TOKENS,
        "LLM_MAX_TOKENS": settings.LLM_MAX_TOKENS,
        "LLM_THINKING_BUDGET": settings.LLM_THINKING_BUDGET,
    }
    try:
        yield
    finally:
        settings.RAG_USE_FAISS = orig["RAG_USE_FAISS"]
        settings.RAG_TOP_K = orig["RAG_TOP_K"]
        settings.RAG_MIN_SIMILARITY = orig["RAG_MIN_SIMILARITY"]
        settings.LLM_CONTEXT_WINDOW_TOKENS = orig["LLM_CONTEXT_WINDOW_TOKENS"]
        settings.LLM_MAX_TOKENS = orig["LLM_MAX_TOKENS"]
        settings.LLM_THINKING_BUDGET = orig["LLM_THINKING_BUDGET"]


async def _fake_extract(_: str) -> str:
    # 十分に長いテキストを返して単発では context に入らないようにする（FAISS 経路に誘導）
    return "X" * 50000


def _fake_split(_: str) -> list[str]:
    # 固定チャンク順
    return ["c1", "c2", "c3", "c4"]


def test_rag_threshold_and_packing(monkeypatch, restore_settings):
    # Arrange
    settings.RAG_USE_FAISS = True
    settings.RAG_TOP_K = 4
    settings.RAG_MIN_SIMILARITY = 0.6  # -> c1, c2 が通過
    # 単発経路に入らないようにコンテキスト予算を小さくする
    settings.LLM_CONTEXT_WINDOW_TOKENS = 2000

    fake = _FakeProvider()
    monkeypatch.setattr(ProviderFactory, "fromSettings", lambda cfg=None: fake)

    # I/O・RAG をすべてフェイクに
    monkeypatch.setattr("src.llm.paper_analyzer.extract_pdf_text", _fake_extract)
    monkeypatch.setattr("src.llm.paper_analyzer.splitText", _fake_split)
    monkeypatch.setattr("src.llm.paper_analyzer.EmbeddingModel", type("_E", (), {"from_pretrained": staticmethod(lambda name: _FakeEmbedder())}))
    monkeypatch.setattr("src.llm.paper_analyzer.FaissVectorStore", _FakeVectorStore)
    # context 判定を単純化（長文は False、小さいチャンクは True）
    monkeypatch.setattr(PaperAnalyzer, "_fits_context", lambda self, text, prompt: len(text) < 100)

    analyzer = PaperAnalyzer()
    out = asyncio.run(analyzer._analyze_via_text_pipeline("/tmp/dummy.pdf", "prompt"))

    assert isinstance(out, dict)
    # しきい値フィルタ + 貪欲パックで c1 と c2 が選択される
    assert fake.last_text == "c1\n\nc2"
    assert fake.calls.count("text") >= 1


def test_rag_fallback_top1_when_no_filtered(monkeypatch, restore_settings):
    # Arrange
    settings.RAG_USE_FAISS = True
    settings.RAG_TOP_K = 4
    settings.RAG_MIN_SIMILARITY = 0.95  # -> フィルタ通過なし -> 最上位1件フォールバック
    # 単発経路に入らないようにコンテキスト予算を小さくする
    settings.LLM_CONTEXT_WINDOW_TOKENS = 2000

    fake = _FakeProvider()
    monkeypatch.setattr(ProviderFactory, "fromSettings", lambda cfg=None: fake)

    monkeypatch.setattr("src.llm.paper_analyzer.extract_pdf_text", _fake_extract)
    monkeypatch.setattr("src.llm.paper_analyzer.splitText", _fake_split)
    monkeypatch.setattr("src.llm.paper_analyzer.EmbeddingModel", type("_E", (), {"from_pretrained": staticmethod(lambda name: _FakeEmbedder())}))
    monkeypatch.setattr("src.llm.paper_analyzer.FaissVectorStore", _FakeVectorStore)

    analyzer = PaperAnalyzer()
    out = asyncio.run(analyzer._analyze_via_text_pipeline("/tmp/dummy.pdf", "prompt"))

    assert isinstance(out, dict)
    # しきい値通過が無い場合、トップ1件(c1)でフォールバック
    assert fake.last_text == "c1"
    assert fake.calls.count("text") >= 1


def test_topk_reduction_and_fallback_when_packing_fails(monkeypatch, restore_settings):
    # Arrange: フィルタは通過するが、どのチャンクも context に入らずパッキング失敗 -> 段階縮小の後にトップ1フォールバック
    settings.RAG_USE_FAISS = True
    settings.RAG_TOP_K = 4
    settings.RAG_MIN_SIMILARITY = 0.0  # 全件通過

    fake = _FakeProvider()
    monkeypatch.setattr(ProviderFactory, "fromSettings", lambda cfg=None: fake)

    monkeypatch.setattr("src.llm.paper_analyzer.extract_pdf_text", _fake_extract)
    monkeypatch.setattr("src.llm.paper_analyzer.splitText", _fake_split)
    monkeypatch.setattr("src.llm.paper_analyzer.EmbeddingModel", type("_E", (), {"from_pretrained": staticmethod(lambda name: _FakeEmbedder())}))
    monkeypatch.setattr("src.llm.paper_analyzer.FaissVectorStore", _FakeVectorStore)

    # どの候補テキストでも context に入らないようにする
    monkeypatch.setattr(PaperAnalyzer, "_fits_context", lambda self, text, prompt: False)

    analyzer = PaperAnalyzer()
    out = asyncio.run(analyzer._analyze_via_text_pipeline("/tmp/dummy.pdf", "prompt"))

    assert isinstance(out, dict)
    # パッキング不能のため最上位1件フォールバック
    assert fake.last_text == "c1"
    assert fake.calls.count("text") >= 1


def test_threshold_boundary_inclusive(monkeypatch, restore_settings):
    # Arrange: スコア==しきい値の要素を包含する（>= 判定）
    settings.RAG_USE_FAISS = True
    settings.RAG_TOP_K = 4
    settings.RAG_MIN_SIMILARITY = 0.70  # c2 がちょうど通過
    settings.LLM_CONTEXT_WINDOW_TOKENS = 2000

    fake = _FakeProvider()
    monkeypatch.setattr(ProviderFactory, "fromSettings", lambda cfg=None: fake)

    monkeypatch.setattr("src.llm.paper_analyzer.extract_pdf_text", _fake_extract)
    monkeypatch.setattr("src.llm.paper_analyzer.splitText", _fake_split)
    monkeypatch.setattr("src.llm.paper_analyzer.EmbeddingModel", type("_E", (), {"from_pretrained": staticmethod(lambda name: _FakeEmbedder())}))
    monkeypatch.setattr("src.llm.paper_analyzer.FaissVectorStore", _FakeVectorStore)
    monkeypatch.setattr(PaperAnalyzer, "_fits_context", lambda self, text, prompt: len(text) < 100)

    analyzer = PaperAnalyzer()
    out = asyncio.run(analyzer._analyze_via_text_pipeline("/tmp/dummy.pdf", "prompt"))

    assert isinstance(out, dict)
    # c1(0.90) と c2(0.70==threshold) が選ばれる
    assert fake.last_text == "c1\n\nc2"
    assert fake.calls.count("text") >= 1
