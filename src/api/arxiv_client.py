import arxiv
import datetime
from typing import List
from src.models.paper import Paper, PaperMetadata, Author
from src.utils.logger import log
from src.utils.config import settings

class ArxivClient:
    def __init__(self):
        """ArXiv APIクライアントの初期化"""
        self.client = arxiv.Client()

    async def fetch_recent_papers(self, category: str, days_back: int = 1) -> List[Paper]:
        """
        指定されたカテゴリと日数前から現在までの論文を取得する。
        論文が見つからない場合は、最大14日前まで遡って検索を試みる。
        
        Args:
            category (str): カテゴリ名
            days_back (int): 検索を開始する日数（デフォルト: 1日前）
            
        Returns:
            List[Paper]: 取得した論文のリスト
            
        Raises:
            ValueError: 14日以内に論文が見つからなかった場合
        """
        papers = []
        current_days_back = days_back
        while True:
            try:
                query = self._build_query(category, current_days_back)
                search = arxiv.Search(
                    query=query,
                    max_results=settings.ARXIV_MAX_RESULTS,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )

                results = list(self.client.results(search))
                
                if not results:
                    current_days_back += 1
                    if current_days_back > 14:
                        log.error(f"カテゴリ {category} の過去14日分の論文が見つかりませんでした")
                        break
                    continue

                day = (datetime.date.today() - datetime.timedelta(days=current_days_back)).strftime('%Y%m%d')
                log.info(f"{category} の {day}日付の論文を取得しました")
                
                for result in results:
                    paper = self._convert_to_paper(result)
                    papers.append(paper)
                
                break  # Found papers for this category
                
            except Exception as e:
                log.error(f"{category} の論文取得中にエラーが発生しました: {str(e)}")
                break  # Error occurred, stop processing this category

        if not papers:
            raise ValueError("過去14日の検索で論文が見つかりませんでした。検索条件を確認してください。")

        return papers

    def _build_query(self, category: str, days_back: int) -> str:
        """検索クエリを構築する"""
        day = (datetime.date.today() - datetime.timedelta(days=days_back))
        start = day.strftime('%Y%m%d') + "0000"
        end = day.strftime('%Y%m%d') + "2359"
        return f"cat:{category} AND submittedDate:[{start} TO {end}]"

    def _convert_to_paper(self, arxiv_result: arxiv.Result) -> Paper:
        """ArXiv APIの結果をPaperモデルに変換する"""
        authors = [
            Author(
                name=author.name,
                affiliations=getattr(author, 'affiliations', [])
            )
            for author in arxiv_result.authors
        ]

        metadata = PaperMetadata(
            arxiv_id=arxiv_result.entry_id.split('/')[-1],
            title=arxiv_result.title,
            authors=authors,
            abstract=arxiv_result.summary,
            pdf_url=arxiv_result.pdf_url,
            published_date=arxiv_result.published,
            categories=[cat.lower() for cat in arxiv_result.categories],
            last_updated=arxiv_result.updated
        )

        return Paper(metadata=metadata)
