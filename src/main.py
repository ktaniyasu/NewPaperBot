import asyncio

from src.api.arxiv_client import ArxivClient
from src.api.discord_bot import ArxivBot
from src.llm.paper_analyzer import PaperAnalyzer
from src.scheduler.runner import SchedulerRunner
from src.services.pipeline import Pipeline
from src.utils.config import settings
from src.utils.logger import log
from pydantic import SkipValidation


async def _run() -> None:
    log.info("Starting Arxiv Analyzer (scheduler + pipeline)...")
    arxiv_client = ArxivClient()
    paper_analyzer = PaperAnalyzer()
    bot = ArxivBot() if settings.DISCORD_BOT_TOKEN else None
    pipeline = Pipeline(arxiv_client, paper_analyzer, bot)
    runner = SchedulerRunner(pipeline, bot)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(_run())
