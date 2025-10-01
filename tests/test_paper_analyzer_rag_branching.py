import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.llm.paper_analyzer import PaperAnalyzer
from src.schemas.paper_analysis import PaperAnalysisSchema
from src.utils.config import settings
from src.llm.providers import ProviderFactory


class _FakeProvider:
    def __init__(self, *, support_upload: bool, file_should_fail: bool = False):
        self._support_upload = support_upload
        self._file_should_fail = file_should_fail
        self.calls: list[str] = []

    def supportsFileUpload(self) -> bool:
        return self._support_upload

    async def generateStructuredFromFile(self, prompt: str, pdfPath: str, schema: Any, **kwargs):
        self.calls.append("file")
        if self._file_should_fail:
            raise RuntimeError("upload failed")
        return {
            "summary": "s",
            "novelty": "n",
            "methodology": "m",
            "results": "r",
            "future_work": "f",
            "research_themes": ["t1", "t2", "t3"],
        }

    async def generateStructuredFromText(self, prompt: str, text: str, schema: Any, **kwargs):
        self.calls.append("text")
        # return a valid minimal dict
        return {
            "summary": f"S:{text[:5]}",
            "novelty": "N",
            "methodology": "M",
            "results": "R",
            "future_work": "F",
            "research_themes": ["A", "B", "C"],
        }

    async def translate(self, text: str, **kwargs) -> str:  # not used here
        return text


@pytest.fixture()
def restore_settings():
    orig = {
        "RAG_USE_FAISS": settings.RAG_USE_FAISS,
        "LLM_CONTEXT_WINDOW_TOKENS": settings.LLM_CONTEXT_WINDOW_TOKENS,
        "PDF_CHUNK_SIZE": settings.PDF_CHUNK_SIZE,
        "PDF_CHUNK_OVERLAP": settings.PDF_CHUNK_OVERLAP,
    }
    try:
        yield
    finally:
        settings.RAG_USE_FAISS = orig["RAG_USE_FAISS"]
        settings.LLM_CONTEXT_WINDOW_TOKENS = orig["LLM_CONTEXT_WINDOW_TOKENS"]
        settings.PDF_CHUNK_SIZE = orig["PDF_CHUNK_SIZE"]
        settings.PDF_CHUNK_OVERLAP = orig["PDF_CHUNK_OVERLAP"]


@pytest.fixture()
def tmp_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "dummy.pdf"
    p.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\nstartxref\n0\n%%EOF\n")
    return p


async def _fake_extract(_: str) -> str:
    return "X" * 1000


def _result_is_valid(schema: PaperAnalysisSchema) -> bool:
    return (
        isinstance(schema, PaperAnalysisSchema)
        and len(schema.research_themes) == 3
    )


def test_generate_analysis_file_upload_success(monkeypatch, tmp_pdf: Path, restore_settings):
    # Arrange: provider supports upload and succeeds
    fake = _FakeProvider(support_upload=True, file_should_fail=False)
    monkeypatch.setattr(ProviderFactory, "fromSettings", lambda cfg=None: fake)

    # Ensure text path would raise if called (to detect unintended calls)
    async def _fail_text(*args, **kwargs):
        raise AssertionError("text path should not be called when upload succeeds")

    monkeypatch.setattr(_FakeProvider, "generateStructuredFromText", _fail_text, raising=True)

    analyzer = PaperAnalyzer()
    out = asyncio.run(analyzer._generate_analysis("prompt", str(tmp_pdf)))

    assert _result_is_valid(out)
    assert fake.calls == ["file"]


def test_generate_analysis_file_upload_fallback_to_text(monkeypatch, tmp_pdf: Path, restore_settings):
    # Arrange: provider upload path fails -> fallback to text pipeline
    fake = _FakeProvider(support_upload=True, file_should_fail=True)
    monkeypatch.setattr(ProviderFactory, "fromSettings", lambda cfg=None: fake)
    # Patch text extractor to avoid real PDF reading
    monkeypatch.setattr("src.llm.paper_analyzer.extract_pdf_text", _fake_extract)

    analyzer = PaperAnalyzer()
    out = asyncio.run(analyzer._generate_analysis("prompt", str(tmp_pdf)))

    assert _result_is_valid(out)
    # Both file and text should be attempted (file path may retry before fallback)
    assert "file" in fake.calls
    assert "text" in fake.calls
    # Ensure fallback occurred after at least one file attempt
    assert fake.calls[-1] == "text"
    assert fake.calls.index("text") > fake.calls.index("file")


def test_text_pipeline_single_shot_vs_chunk_fallback(monkeypatch, tmp_pdf: Path, restore_settings):
    # Use fake provider without upload to force text pipeline
    fake = _FakeProvider(support_upload=False)
    monkeypatch.setattr(ProviderFactory, "fromSettings", lambda cfg=None: fake)
    monkeypatch.setattr("src.llm.paper_analyzer.extract_pdf_text", _fake_extract)

    analyzer = PaperAnalyzer()

    # Case 1: fits context -> single shot
    settings.RAG_USE_FAISS = False
    settings.LLM_CONTEXT_WINDOW_TOKENS = 10_000  # big enough for 1000 chars + prompt
    d = asyncio.run(analyzer._analyze_via_text_pipeline(str(tmp_pdf), "short-prompt"))
    assert isinstance(d, dict) and d.get("summary", "").startswith("S:")
    assert fake.calls[-1] == "text"

    # Case 2: does not fit -> chunking
    settings.LLM_CONTEXT_WINDOW_TOKENS = 100  # very small to force chunking
    # make chunk size small to get multiple calls
    settings.PDF_CHUNK_SIZE = 100
    settings.PDF_CHUNK_OVERLAP = 0

    d2 = asyncio.run(analyzer._analyze_via_text_pipeline(str(tmp_pdf), "very-long-prompt" * 50))
    # combine() should merge multiple partials; at least a dict should be returned
    assert isinstance(d2, dict)
    # ensure multiple text calls happened
    assert fake.calls.count("text") >= 2
