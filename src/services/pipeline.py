import asyncio
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
        if not targets:
            log.info("No target channels configured; nothing to process.")
            return

        category_limit = max(1, settings.PIPELINE_CATEGORY_CONCURRENCY)
        paper_limit = max(1, settings.PIPELINE_PAPER_CONCURRENCY)
        category_sem = asyncio.Semaphore(category_limit)
        paper_sem = asyncio.Semaphore(paper_limit)
        processed_dates: set[tuple[int, datetime.date]] = set()
        processed_lock = asyncio.Lock()

        async def process_category(target: dict[str, str]) -> None:
            async with category_sem:
                category = target.get("category")
                channel_id_raw = target.get("channel_id")
                if not category or not channel_id_raw:
                    log.error(f"Invalid target entry: {target}")
                    return

                try:
                    channel_id = int(channel_id_raw)
                except ValueError:
                    log.error(
                        f"Invalid channel ID format for category {category}: {channel_id_raw}"
                    )
                    return

                log.info(f"Checking category '{category}'")
                try:
                    papers = await self.arxivClient.fetch_recent_papers(category=category)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.error(
                        f"Error fetching papers for category {category}: {str(e)}"
                    )
                    return

                if papers and self.bot:
                    post_date = papers[0].metadata.published_date.date()
                    header_key = (channel_id, post_date)
                    should_send_header = False
                    async with processed_lock:
                        if header_key not in processed_dates:
                            processed_dates.add(header_key)
                            should_send_header = True
                    if should_send_header:
                        try:
                            await self.bot.send_daily_header(channel_id, post_date)
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            log.error(
                                f"Failed to send header for category {category}: {str(e)}"
                            )

                if not papers:
                    return

                try:
                    async with asyncio.TaskGroup() as paper_group:
                        for paper in papers:
                            paper_group.create_task(
                                self._process_paper(channel_id, category, paper, paper_sem)
                            )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log.error(
                        f"Error while processing papers for category {category}: {str(e)}"
                    )

        try:
            async with asyncio.TaskGroup() as category_group:
                for target in targets:
                    category_group.create_task(process_category(target))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"Unexpected error while processing categories: {str(e)}")

        log.info("Pipeline cycle finished.")

    async def _process_paper(
        self,
        channel_id: int,
        category: str,
        paper,
        paper_sem: asyncio.Semaphore,
    ) -> None:
        async with paper_sem:
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
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(
                    f"Error processing paper {paper.metadata.arxiv_id} for category {category}: {str(e)}"
                )
