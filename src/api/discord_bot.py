import datetime

import discord
from discord.ext import commands

from ..models.paper import Paper
from ..utils.config import settings
from ..utils.logger import log
from .discord_format import buildPaperEmbed, collectEmbedMetrics


class ArxivBot(commands.Bot):
    def __init__(self):
        # 必要最小限のintentsを設定
        intents = discord.Intents.default()
        intents.message_content = True  # メッセージ内容の読み取りに必要
        intents.guilds = True  # サーバー情報の読み取りに必要
        super().__init__(command_prefix="!", intents=intents)

        # デフォルトのhelpコマンドを削除
        self.remove_command("help")

    async def setup_hook(self):
        """Botの初期設定"""
        # コマンドの登録
        await self.add_cog(ArxivCommands(self))

    async def on_ready(self):
        """Bot起動時の処理"""
        log.info(f"{self.user} has connected to Discord!")

    async def send_daily_header(self, channel_id: int, postDate: datetime.date):
        """指定されたチャンネルに日付のヘッダーメッセージを送信する"""
        channel = self.get_channel(channel_id)
        if not channel:
            log.error(f"Channel {channel_id} not found")
            return

        headerMessage = f"**📅 {postDate.strftime('%Y-%m-%d')} の新着論文**"
        await channel.send(headerMessage)

    async def send_paper_analysis(self, channel_id: int, paper: Paper):
        """指定されたチャンネルに論文の解析結果を送信する"""
        channel = self.get_channel(channel_id)
        if not channel:
            log.error(f"Channel {channel_id} not found")
            return
        embed = buildPaperEmbed(paper)
        embedMetrics = collectEmbedMetrics(embed)
        log.info(
            "Discord embed metrics | fields={fields} value_chars={vchars} truncated={t} title_chars={tchars}",
            fields=embedMetrics["fieldsCount"],
            vchars=embedMetrics["valueChars"],
            t=embedMetrics["truncatedFieldsCount"],
            tchars=embedMetrics["titleChars"],
        )
        await channel.send(embed=embed)


class ArxivCommands(commands.Cog):
    def __init__(self, bot: ArxivBot):
        self.bot = bot

    @commands.command(name="help")
    async def help_command(self, ctx):
        """ヘルプメッセージを表示"""
        help_text = """
        **ArXiv Paper Analyzer Bot**
        Available commands:
        - `!help`: このヘルプメッセージを表示
        - `!status`: Botの状態を確認
        - `!categories`: 監視中のカテゴリを表示
        """
        await ctx.send(help_text)

    @commands.command(name="status")
    async def status_command(self, ctx):
        """Botの状態を確認"""
        await ctx.send("✅ Bot is running and monitoring ArXiv papers!")

    @commands.command(name="categories")
    async def categories_command(self, ctx):
        """監視中のカテゴリと対応チャンネルを表示"""
        if not settings.target_channels:
            await ctx.send("監視中のカテゴリはありません。")
            return

        message = "**📚 Monitoring Categories and Channels:**\n"
        for target in settings.target_channels:
            message += f"- Category: `{target['category']}`, Channel ID: `{target['channel_id']}`\n"

        await ctx.send(message)
