import argparse
import asyncio
import json
from datetime import datetime
from typing import Optional

from src.api.arxiv_client import ArxivClient
from src.llm.paper_analyzer import PaperAnalyzer
from src.models.paper import Paper, PaperMetadata
from src.utils.logger import log


def build_paper_from_pdf_url(pdf_url: str) -> Paper:
    """最小限のメタデータでPDF直指定の Paper を生成"""
    metadata = PaperMetadata(
        arxiv_id="manual",
        title="Manual PDF",
        authors=[],
        abstract="N/A",
        pdf_url=pdf_url,
        published_date=datetime.utcnow(),
        categories=[],
        last_updated=datetime.utcnow(),
    )
    return Paper(metadata=metadata)


async def run(arxiv_id: Optional[str], pdf_url: Optional[str], output: Optional[str]) -> int:
    analyzer = PaperAnalyzer()
    client = ArxivClient()

    if bool(arxiv_id) == bool(pdf_url):
        log.error("--arxiv-id か --pdf-url のどちらか一方を指定してください。")
        return 2

    if arxiv_id:
        paper = await client.fetch_paper_by_id(arxiv_id)
    else:
        paper = build_paper_from_pdf_url(pdf_url)

    paper = await analyzer.analyze_paper(paper)

    # 結果を表示
    result = {
        "arxiv_id": paper.metadata.arxiv_id,
        "title": paper.metadata.title,
        "summary": paper.analysis.summary if paper.analysis else None,
        "novelty": paper.analysis.novelty if paper.analysis else None,
        "methodology": paper.analysis.methodology if paper.analysis else None,
        "results": paper.analysis.results if paper.analysis else None,
        "future_work": paper.analysis.future_work if paper.analysis else None,
        "research_themes": paper.analysis.research_themes if paper.analysis else None,
        "error_log": paper.error_log,
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    if output:
        try:
            with open(output, "w", encoding="utf-8") as f:
                f.write(text)
            log.info(f"結果を保存しました: {output}")
        except Exception as e:
            log.error(f"結果の保存に失敗しました: {str(e)}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="単発の論文解析を実行します（Discord不要）。"
    )
    parser.add_argument("--arxiv-id", "-i", type=str, help="解析する arXiv ID")
    parser.add_argument("--pdf-url", "-u", type=str, help="直接PDFのURLを指定")
    parser.add_argument(
        "--output", "-o", type=str, help="結果JSONの保存先パス（省略可）"
    )
    args = parser.parse_args()

    # 実行
    code = asyncio.run(run(args.arxiv_id, args.pdf_url, args.output))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
