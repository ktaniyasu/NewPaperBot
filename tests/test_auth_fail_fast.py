import asyncio
import tempfile
from datetime import datetime

import pytest

from src.models.paper import Author, Paper, PaperMetadata
import src.llm.paper_analyzer as analyzer_mod


def make_paper(arxiv_id: str = "2401.00002v1") -> Paper:
    meta = PaperMetadata(
        arxiv_id=arxiv_id,
        title="Sample Paper",
        authors=[Author(name="Doe, J.")],
        abstract="This is an abstract.",
        pdf_url="https://example.com/sample.pdf",
        published_date=datetime.utcnow(),
        categories=["cs.AI"],
        last_updated=datetime.utcnow(),
    )
    return Paper(metadata=meta)


class AuthFailTextProvider:
    def __init__(self):
        self.calls_text = 0

    def supportsFileUpload(self) -> bool:
        return False

    async def generateStructuredFromText(self, prompt, text, schema):
        self.calls_text += 1
        # Simulate OpenRouter auth error signature
        raise RuntimeError("401 Unauthorized: No auth credentials found")


def test_auth_error_in_text_path_does_not_retry_or_fallback(monkeypatch):
    from src.utils.config import settings

    # Force HTML/text path and keep it lightweight
    monkeypatch.setattr(settings, "USE_AR5IV_HTML", True, raising=False)
    monkeypatch.setattr(settings, "RAG_USE_FAISS", False, raising=False)

    # Provide small body text so it goes to single-shot text path
    async def _fake_extract_html(_aid: str) -> str:
        return "Short body text for testing."

    monkeypatch.setattr(analyzer_mod, "extractTextByArxivId", _fake_extract_html, raising=False)

    # Inject provider that always raises 401 on text path
    provider = AuthFailTextProvider()
    monkeypatch.setattr(analyzer_mod.ProviderFactory, "fromSettings", lambda _s: provider, raising=False)

    paper = make_paper()
    analyzer = analyzer_mod.PaperAnalyzer()

    async def run():
        res = await analyzer.analyze_paper(paper)
        # PaperAnalyzer catches at top-level and returns error placeholder
        assert res.analysis is not None
        assert res.analysis.summary == "解析エラー発生"
        assert "401" in (res.error_log or "") or "unauthorized" in (res.error_log or "").lower()
        # Crucially, only a single provider call should have occurred; no retries, no chunk fallback
        assert provider.calls_text == 1

    asyncio.run(run())


class AuthFailFileProvider:
    def __init__(self):
        self.calls_file = 0
        self.calls_text = 0

    def supportsFileUpload(self) -> bool:
        return True

    async def generateStructuredFromFile(self, prompt, pdf_path, schema):
        self.calls_file += 1
        # Simulate 403 auth/forbidden
        raise RuntimeError("403 Forbidden: Invalid API key")

    async def generateStructuredFromText(self, prompt, text, schema):
        self.calls_text += 1
        return {"summary": "should not be called"}


def test_auth_error_in_file_path_does_not_fallback_to_text(monkeypatch):
    from src.utils.config import settings

    # Force PDF path
    monkeypatch.setattr(settings, "USE_AR5IV_HTML", False, raising=False)
    monkeypatch.setattr(settings, "RAG_USE_FAISS", False, raising=False)

    # Avoid real download; create a temp PDF path
    async def _fake_download(self, url: str) -> str:
        f = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")
        finally:
            f.close()
        return f.name

    monkeypatch.setattr(analyzer_mod.PaperAnalyzer, "_download_pdf", _fake_download, raising=False)

    provider = AuthFailFileProvider()
    monkeypatch.setattr(analyzer_mod.ProviderFactory, "fromSettings", lambda _s: provider, raising=False)

    paper = make_paper("2401.00003v1")
    analyzer = analyzer_mod.PaperAnalyzer()

    async def run():
        res = await analyzer.analyze_paper(paper)
        assert res.analysis is not None
        assert res.analysis.summary == "解析エラー発生"
        assert "403" in (res.error_log or "") or "forbidden" in (res.error_log or "").lower()
        # File path tried exactly once; no fallback to text path should happen on auth error
        assert provider.calls_file == 1
        assert provider.calls_text == 0

    asyncio.run(run())
