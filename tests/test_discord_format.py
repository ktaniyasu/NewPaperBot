import datetime

from src.api.discord_format import buildPaperEmbed, collectEmbedMetrics, formatSection, truncateText
from src.models.paper import AnalysisResult, Author, Paper, PaperMetadata


def _make_paper(long_summary: bool = False):
    authors = [Author(name=f"Author {i}") for i in range(1, 5)]  # 4名で"..."を誘発
    meta = PaperMetadata(
        arxiv_id="1234.5678",
        title="Test Paper",
        authors=authors,
        abstract="Abstract text",
        pdf_url="https://example.org/paper.pdf",
        published_date=datetime.datetime(2024, 1, 2),
        categories=["cs.AI", "cs.CL"],
        last_updated=datetime.datetime(2024, 1, 3),
    )
    base_text = "** Intro\n\n** * bullet A\n** * bullet B\n"
    summary_text = ("X" * 1200 + "end") if long_summary else base_text
    analysis = AnalysisResult(
        summary=summary_text,
        novelty=base_text,
        methodology=base_text,
        results=base_text,
        future_work=base_text,
        research_themes=["theme1", "theme2"],
    )
    return Paper(metadata=meta, analysis=analysis)


def test_build_paper_embed_fields_and_formatting():
    paper = _make_paper(long_summary=False)
    embed = buildPaperEmbed(paper)

    # タイトルとURL
    assert embed.title == paper.metadata.title
    assert embed.url == paper.metadata.pdf_url

    # フィールド数: Authors, Categories, 5セクション, Tips = 8
    assert len(embed.fields) == 8

    # Authors整形（3名+...）
    authors_value = embed.fields[0].value
    assert ", ..." in authors_value or authors_value.endswith(", ...")

    # 箇条整形（"** * " -> "* ")
    any_section_value = embed.fields[2].value  # 要約
    assert "* bullet A" in any_section_value
    assert "** * bullet" not in any_section_value


def test_collect_embed_metrics_detects_truncation_with_trailing_newline():
    # 要約のみ長文で切り詰めを発生させる（末尾に\nが付与される）
    paper = _make_paper(long_summary=True)
    embed = buildPaperEmbed(paper)

    # 要約は truncateText で "..." 付与 + divider("\n")
    # 改行付きでも省略検知できることを確認
    metrics = collectEmbedMetrics(embed)
    assert metrics["fieldsCount"] >= 6
    assert metrics["truncatedFieldsCount"] >= 1

    # 念のため末尾が改行を含んでも"..."を含む
    summary_val = embed.fields[2].value  # 要約
    assert summary_val.rstrip().endswith("...")


def test_format_section_and_truncate_text_units():
    s = "** Example\n\n** * item1\n** * item2\n"
    formatted = formatSection(s)
    assert formatted.split("\n")[0] == "Example"
    assert "* item1" in formatted and "* item2" in formatted

    t = truncateText("A" * 1005, maxLength=1000)
    assert t.endswith("...") and len(t) == 1000
