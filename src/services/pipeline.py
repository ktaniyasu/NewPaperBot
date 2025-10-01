import datetime
from typing import Optional

from ..api.arxiv_client import ArxivClient
from ..api.discord_bot import ArxivBot
from ..llm.paper_analyzer import PaperAnalyzer
from ..utils.config import settings
from ..utils.logger import log


class Pipeline:
    """
    取得→解析→通知のパイプラインを司るサービス層。
    - Discord トークンが無い場合は通知をスキップ（ログのみ）。
    - 週末スキップなどの判断は基本スケジューラ側で行うが、安全のため冪等にチェック。
    """

    def __init__(
        self,
        arxivClient: ArxivClient,
        paperAnalyzer: PaperAnalyzer,
        bot: Optional[ArxivBot] = None,
    ) -> None:
        self.arxivClient = arxivClient
        self.paperAnalyzer = paperAnalyzer
        self.bot = bot

    async def process_all_categories_once(self) -> None:
        """設定された全カテゴリを1サイクル処理する。"""
        # 週末スキップ（冪等チェック）
        current_weekday = datetime.datetime.now().weekday()
        if current_weekday >= 5:
            log.info("Weekend detected. Skipping pipeline run.")
            return

        targets = settings.target_channels or []
        log.info(f"Processing {len(targets)} categories...")
        processed_dates: set[tuple[int, datetime.date]] = set()

        for target in targets:
            category = target.get("category")
            channel_id_raw = target.get("channel_id")
            if not category or not channel_id_raw:
                log.error(f"Invalid target entry: {target}")
                continue

            try:
                channel_id = int(channel_id_raw)
            except ValueError:
                log.error(
                    f"Invalid channel ID format for category {category}: {channel_id_raw}"
                )
                continue

            log.info(f"Checking category '{category}'")
            try:
                papers = await self.arxivClient.fetch_recent_papers(category=category)

                # ヘッダー送信（チャンネル×日付で一度のみ）
                if papers and self.bot:
                    post_date = papers[0].metadata.published_date.date()
                    header_key = (channel_id, post_date)
                    if header_key not in processed_dates:
                        await self.bot.send_daily_header(channel_id, post_date)
                        processed_dates.add(header_key)

                for paper in papers:
                    try:
                        paper = await self.paperAnalyzer.analyze_paper(paper)
                        if paper.analysis:
                            if self.bot:
                                await self.bot.send_paper_analysis(channel_id, paper)
                            else:
                                log.info(
                                    f"[Headless] Analysis ready for {paper.metadata.arxiv_id}"
                                )
                        else:
                            log.warning(
                                f"Skipping paper {paper.metadata.arxiv_id} due to analysis failure"
                            )
                    except Exception as e:
                        log.error(
                            f"Error processing paper {paper.metadata.arxiv_id} for category {category}: {str(e)}"
                        )
                        continue

            except Exception as e:
                log.error(
                    f"Error fetching or processing papers for category {category}: {str(e)}"
                )
                continue

        log.info("Pipeline cycle finished.")
