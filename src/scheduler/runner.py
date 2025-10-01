import asyncio
import contextlib
import datetime
import signal
from typing import Optional

from ..api.discord_bot import ArxivBot
from ..services.pipeline import Pipeline
from ..utils.config import settings
from ..utils.logger import log


class SchedulerRunner:
    """
    平日 8:30 実行（DEBUG_RUN_IMMEDIATELY で即時実行）を司るスケジューラ。
    Discord ボット有無を吸収し、Pipeline を所定スケジュールで駆動します。
    """

    def __init__(self, pipeline: Pipeline, bot: Optional[ArxivBot] = None) -> None:
        self.pipeline = pipeline
        self.bot = bot
        self.running = True
        self._periodic_task: Optional[asyncio.Task] = None

    async def run(self) -> None:
        """スケジューラのメインループを開始する"""
        # シグナルハンドリング
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(Exception):
                signal.signal(sig, self._handle_shutdown)

        if self.bot and settings.DISCORD_BOT_TOKEN:
            log.info("Starting Discord bot and scheduler loop...")
            await asyncio.gather(
                self._start_bot(settings.DISCORD_BOT_TOKEN),
                self._run_loop(),
            )
        else:
            log.info("Starting headless scheduler loop (no Discord bot)...")
            await self._run_loop()

    async def _start_bot(self, token: str) -> None:
        assert self.bot is not None
        await self.bot.start(token)

    async def _run_loop(self) -> None:
        # Discord ボットの準備完了を待つ（存在する場合）
        if self.bot:
            await self.bot.wait_until_ready()
            log.info("Discord Bot is ready.")

        # 起動直後の待機ロジック
        if not settings.DEBUG_RUN_IMMEDIATELY:
            await self.sleep_until_next_weekday_830()

        while self.running:
            try:
                await self.pipeline.process_all_categories_once()
            except Exception as e:
                log.error(f"Pipeline cycle failed: {str(e)}")

            if not self.running:
                break
            await self.sleep_until_next_weekday_830()

    async def sleep_until_next_weekday_830(self) -> None:
        now = datetime.datetime.now()
        next_run = now.replace(hour=8, minute=30, second=0, microsecond=0)
        if now >= next_run:
            next_run += datetime.timedelta(days=1)
        while next_run.weekday() >= 5:
            next_run += datetime.timedelta(days=1)
        wait_seconds = max(0.0, (next_run - now).total_seconds())
        log.info(
            f"次回の実行まで {wait_seconds/3600:.2f} 時間待機します（実行予定: {next_run}）"
        )
        await asyncio.sleep(wait_seconds)

    def _handle_shutdown(self, signum, frame) -> None:  # type: ignore[no-untyped-def]
        log.info(f"Received signal {signum}. Shutting down...")
        self.running = False
        # Bot が居ればクローズをトリガ
        if self.bot:
            asyncio.create_task(self._close_bot())

    async def _close_bot(self) -> None:
        try:
            assert self.bot is not None
            await self.bot.close()
        except Exception as e:
            log.warning(f"Error while closing Discord bot: {str(e)}")
