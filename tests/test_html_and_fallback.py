import asyncio
import tempfile
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.models.paper import Author, Paper, PaperMetadata
import src.ingestion.html_reader as html_reader
import src.llm.paper_analyzer as analyzer_mod


class MockProvider:
    def __init__(self, *, text_response: dict | None = None, file_response: dict | None = None, fail_file: bool = False):
        self._text_response = text_response or {}
        self._file_response = file_response or {}
        self._fail_file = fail_file
        self.calls = SimpleNamespace(text=0, file=0)

    def supportsFileUpload(self) -> bool:
        return True

    async def generateStructuredFromText(self, prompt, text, schema):
        self.calls.text += 1
        return self._text_response

    async def generateStructuredFromFile(self, prompt, pdf_path, schema):
        self.calls.file += 1
        if self._fail_file:
            raise RuntimeError("file path forced failure")
        return self._file_response


def make_paper(arxiv_id: str = "2401.00001v1") -> Paper:
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


def test_extract_text_from_ar5iv_html_article():
    html = """
    <html>
      <head><title>T</title><script>var x=1;</script></head>
      <body>
        <nav>menu</nav>
        <article>
            <h1>Title</h1>
            <p>Paragraph 1</p>
            <p>Paragraph 2</p>
        </article>
        <footer>footer</footer>
      </body>
    </html>
    """
    text = html_reader.extractTextFromAr5ivHtml(html)
    assert "Paragraph 1" in text and "Paragraph 2" in text
    assert "menu" not in text and "var x=1" not in text and "footer" not in text


def test_analyze_paper_prefers_html_path(monkeypatch):
    # Ensure HTML path is enabled and RAG is off to keep test light
    from src.utils.config import settings

    monkeypatch.setattr(settings, "USE_AR5IV_HTML", True, raising=False)
    monkeypatch.setattr(settings, "RAG_USE_FAISS", False, raising=False)

    # HTML ingestion returns body text (patch the alias used inside paper_analyzer)
    async def _fake_extract_html(_aid: str) -> str:
        return "This is HTML body text."
    monkeypatch.setattr(analyzer_mod, "extractTextByArxivId", _fake_extract_html, raising=False)

    # Provider mocked; file path should not be used
    mock_provider = MockProvider(
        text_response={
            "1_要約": "要約",
            "新規性": "新規性",
            "手法": "手法",
            "結果": "結果",
            "今後の課題": "課題",
            "研究テーマ": ["A", "B", "C"],
        }
    )

    # Patch ProviderFactory to return our mock
    monkeypatch.setattr(analyzer_mod.ProviderFactory, "fromSettings", lambda _s: mock_provider, raising=False)

    # Guard to ensure PDF path is not called (async to be awaitable)
    async def _should_not_call(self, url: str):
        raise AssertionError("PDF path should not be used")
    monkeypatch.setattr(analyzer_mod.PaperAnalyzer, "_download_pdf", _should_not_call, raising=False)

    paper = make_paper()
    analyzer = analyzer_mod.PaperAnalyzer()

    async def run():
        res = await analyzer.analyze_paper(paper)
        assert res.analysis is not None
        # Normalized English keys should be present in Pydantic model
        assert res.analysis.summary == "要約"
        assert res.analysis.novelty == "新規性"
        assert res.analysis.methodology == "手法"
        assert res.analysis.results == "結果"
        assert res.analysis.future_work == "課題"
        assert res.analysis.research_themes == ["A", "B", "C"]
        assert mock_provider.calls.text == 1
        assert mock_provider.calls.file == 0

    asyncio.run(run())


def test_analyze_paper_fallbacks_to_pdf_when_html_fails(monkeypatch):
    from src.utils.config import settings

    monkeypatch.setattr(settings, "USE_AR5IV_HTML", True, raising=False)
    monkeypatch.setattr(settings, "RAG_USE_FAISS", False, raising=False)

    # HTML path fails (patch the alias used inside paper_analyzer)
    async def _fail_extract_html(_aid: str) -> str:
        raise RuntimeError("HTML path failed")
    monkeypatch.setattr(analyzer_mod, "extractTextByArxivId", _fail_extract_html, raising=False)

    # Avoid real download; create a real temporary PDF and return its path (async to be awaitable)
    async def _fake_download(self, url: str) -> str:
        f = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")
        finally:
            f.close()
        return f.name
    monkeypatch.setattr(analyzer_mod.PaperAnalyzer, "_download_pdf", _fake_download, raising=False)

    # Avoid real PDF parsing; return short text (patch the alias used inside paper_analyzer)
    async def _fake_extract_pdf(_p: str) -> str:
        return "PDF TEXT"
    monkeypatch.setattr(analyzer_mod, "extract_pdf_text", _fake_extract_pdf, raising=False)

    # Mock provider to reject file path and use text path
    mock_provider = MockProvider(
        text_response={
            "summary": "S",
            "novelty": "N",
            "methodology": "M",
            "results": "R",
            "futurework": "F",  # will be normalized to future_work
            "research_themes": "T1; T2 / T3",
        },
        fail_file=True,
    )
    monkeypatch.setattr(analyzer_mod.ProviderFactory, "fromSettings", lambda _s: mock_provider, raising=False)

    paper = make_paper("2401.99999v1")
    analyzer = analyzer_mod.PaperAnalyzer()

    async def run():
        res = await analyzer.analyze_paper(paper)
        assert res.analysis is not None
        # futurework -> future_work, research_themes string -> list of 3
        assert res.analysis.future_work == "F"
        assert res.analysis.research_themes == ["T1", "T2", "T3"]
        # HTML path failed, so text path used at least once
        assert mock_provider.calls.text >= 1

    asyncio.run(run())
