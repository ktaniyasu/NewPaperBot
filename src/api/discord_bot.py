import discord
from discord.ext import commands
from ..models.paper import Paper, AnalysisResult
from ..utils.logger import log
from ..utils.config import settings
import datetime

class ArxivBot(commands.Bot):
    def __init__(self):
        # 必要最小限のintentsを設定
        intents = discord.Intents.default()
        intents.message_content = True  # メッセージ内容の読み取りに必要
        intents.guilds = True          # サーバー情報の読み取りに必要
        super().__init__(command_prefix='!', intents=intents)
        
        # デフォルトのhelpコマンドを削除
        self.remove_command('help')
        
    async def setup_hook(self):
        """Botの初期設定"""
        # コマンドの登録
        await self.add_cog(ArxivCommands(self))
        
    async def on_ready(self):
        """Bot起動時の処理"""
        log.info(f'{self.user} has connected to Discord!')
        
    async def send_daily_header(self, channel_id: int, post_date: datetime.date):
        """指定されたチャンネルに日付のヘッダーメッセージを送信する"""
        channel = self.get_channel(channel_id)
        if not channel:
            log.error(f"Channel {channel_id} not found")
            return
        
        header_message = f"**📅 {post_date.strftime('%Y-%m-%d')} の新着論文**"
        await channel.send(header_message)

    async def send_paper_analysis(self, channel_id: int, paper: Paper):
        """指定されたチャンネルに論文の解析結果を送信する"""
        channel = self.get_channel(channel_id)
        if not channel:
            log.error(f"Channel {channel_id} not found")
            return

        embed = self._create_paper_embed(paper)
        await channel.send(embed=embed)

    def _create_paper_embed(self, paper: Paper) -> discord.Embed:
        """論文情報をDiscord Embedに変換する"""
        metadata = paper.metadata
        analysis = paper.analysis

        embed = discord.Embed(
            title=metadata.title,
            url=metadata.pdf_url,
            color=discord.Color.dark_blue()  # より濃い青色に変更
        )

        # メタデータの追加
        authors = [a.name for a in metadata.authors[:3]]
        if len(metadata.authors) > 3:
            authors.append("...")
        embed.add_field(
            name="**👤 Authors**",
            value=", ".join(authors),
            inline=False
        )
        
        # カテゴリの追加
        embed.add_field(
            name="**📂 Categories**",
            value=", ".join(metadata.categories),
            inline=True
        )

        # 分析結果の追加（存在する場合）
        if analysis:
            divider = "\n"  # セクション間の区切り
            
            def truncate_text(text: str, max_length: int = 1000) -> str:
                """テキストを指定された長さに切り詰める（余裕を持って1000文字に制限）"""
                if not text:
                    return "N/A"
                if len(text) <= max_length:
                    return text
                return text[:max_length-3] + "..."

            def format_section(text: str) -> str:
                cleaned_text = text.strip()
                # テキスト全体の先頭にある '** ' を削除
                if cleaned_text.startswith('** '):
                    cleaned_text = cleaned_text[len('** '):]
                
                # Markdownを維持しつつ、不要な空白や連続改行を整理
                lines = [line.strip() for line in cleaned_text.split('\n')]
                # 空行を削除し、意味のある行だけを残し、先頭のマーカーを置換
                formatted_lines = []
                for line in lines:
                    if line:
                        # 先頭の "** * " を "* " に置換
                        if line.startswith('** * '):
                            formatted_lines.append('* ' + line[len('** * '):])
                        else:
                            formatted_lines.append(line)
                # 改行で結合
                return '\n'.join(formatted_lines)

            summary = format_section(analysis.summary)
            novelty = format_section(analysis.novelty)
            methodology = format_section(analysis.methodology)
            results = format_section(analysis.results)
            future_work = format_section(analysis.future_work)

            embed.add_field(name="**📝 要約**", value=truncate_text(summary) + divider, inline=False)
            embed.add_field(name="**💡 新規性**", value=truncate_text(novelty) + divider, inline=False)
            embed.add_field(name="**🔍 手法**", value=truncate_text(methodology) + divider, inline=False)
            embed.add_field(name="**📊 結果**", value=truncate_text(results) + divider, inline=False)
            embed.add_field(name="**🔮 Future Work**", value=truncate_text(future_work), inline=False)
            # 研究テーマ Tips を追加
            if analysis.research_themes:
                tips_text = "\n".join(f"{i+1}. {theme}" for i, theme in enumerate(analysis.research_themes))
                embed.add_field(name="**🧩 研究テーマのTips**", value=truncate_text(tips_text), inline=False)

        embed.set_footer(text=f"Published: {metadata.published_date.strftime('%Y-%m-%d')}")
        
        return embed

class ArxivCommands(commands.Cog):
    def __init__(self, bot: ArxivBot):
        self.bot = bot

    @commands.command(name='help')
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

    @commands.command(name='status')
    async def status_command(self, ctx):
        """Botの状態を確認"""
        await ctx.send("✅ Bot is running and monitoring ArXiv papers!")

    @commands.command(name='categories')
    async def categories_command(self, ctx):
        """監視中のカテゴリと対応チャンネルを表示"""
        if not settings.target_channels:
            await ctx.send("監視中のカテゴリはありません。")
            return

        message = "**📚 Monitoring Categories and Channels:**\n"
        for target in settings.target_channels:
            message += f"- Category: `{target['category']}`, Channel ID: `{target['channel_id']}`\n"
        
        await ctx.send(message)
