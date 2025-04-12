import asyncio
from src.api.arxiv_client import ArxivClient
from src.api.discord_bot import ArxivBot
from src.llm.paper_analyzer import PaperAnalyzer
from src.utils.config import settings
from src.utils.logger import log
import signal
import sys
import datetime

class ArxivAnalyzer:
    def __init__(self):
        self.arxiv_client = ArxivClient()
        self.bot = ArxivBot()
        self.paper_analyzer = PaperAnalyzer()
        self.running = True

    async def start(self):
        """サービスを開始する"""
        try:
            log.info("Starting Arxiv Analyzer service...")
            # シグナルハンドラの設定
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, self.handle_shutdown)

            # ボットの準備完了を待つ
            async def wait_for_bot_ready():
                await self.bot.wait_until_ready()
                log.info("Discord Bot is ready.")
                # ボット準備完了後に定期チェックタスクを開始
                self.periodic_task = asyncio.create_task(self.periodic_paper_check())
            
            # ボット起動と準備完了待機タスクを開始
            await asyncio.gather(
                self.bot.start(settings.DISCORD_BOT_TOKEN),
                wait_for_bot_ready() 
            )
            
            # Keep the service running until stopped
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            log.error(f"Error starting service: {str(e)}")
            await self.shutdown()

    async def periodic_paper_check(self):
        """定期的に新しい論文をチェックする"""
        while self.running:
            log.info("Starting periodic paper check cycle...")
            log.info(f"Processing {len(settings.target_channels)} categories...")
            processed_dates = set() # Track dates for which headers have been sent per channel
            
            # 設定ファイルから取得したチャンネルとカテゴリのペアをループ
            for target in settings.target_channels:
                category = target["category"]
                try:
                    channel_id = int(target["channel_id"])
                except ValueError:
                    log.error(f"Invalid channel ID format for category {category}: {target['channel_id']}")
                    continue # Skip this target if channel ID is invalid

                log.info(f"Checking category '{category}'")
                try:
                    # 特定のカテゴリの新しい論文を取得
                    papers = await self.arxiv_client.fetch_recent_papers(category=category)
                    
                    # ヘッダーメッセージを送信 (チャンネルごとに日付が新しい場合のみ)
                    if papers: 
                        post_date = papers[0].metadata.published_date.date() # Use date part only
                        header_key = (channel_id, post_date)
                        if header_key not in processed_dates:
                            await self.bot.send_daily_header(channel_id, post_date)
                            processed_dates.add(header_key)

                    # 各論文の処理
                    for paper in papers:
                        try:
                            # PDFダウンロードとLLM解析を実行
                            paper = await self.paper_analyzer.analyze_paper(paper)
                            
                            # Discord通知 (指定されたチャンネルに送信)
                            if paper.analysis:
                                await self.bot.send_paper_analysis(channel_id, paper)
                            else:
                                log.warning(f"Skipping paper {paper.metadata.arxiv_id} due to analysis failure")
                        except Exception as e:
                            log.error(f"Error processing paper {paper.metadata.arxiv_id} for category {category}: {str(e)}")
                            continue # Continue with the next paper in the same category
                            
                except Exception as e:
                    log.error(f"Error fetching or processing papers for category {category}: {str(e)}")
                    continue # Continue with the next category
                log.info(f"Finished checking category '{category}'")

            # 次の実行まで待機（24時間）
            wait_time = 24 * 60 * 60
            log.info(f"Periodic check cycle finished. Waiting for {wait_time / 3600:.1f} hours...")
            await asyncio.sleep(wait_time)

    def handle_shutdown(self, signum, frame):
        """シャットダウンハンドラ"""
        self.running = False
        asyncio.create_task(self.shutdown())

    async def shutdown(self):
        """サービスを終了する"""
        log.info("Shutting down...")
        self.running = False
        await self.bot.close()
        sys.exit(0)

if __name__ == "__main__":
    analyzer = ArxivAnalyzer()
    asyncio.run(analyzer.start())
