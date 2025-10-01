from typing import Dict

import discord

from ..models.paper import Paper


def truncateText(text: str, maxLength: int = 1000) -> str:
    """テキストを指定長に切り詰める。空やNoneは"N/A"。
    末尾には省略記号を付与します。
    """
    if not text:
        return "N/A"
    if len(text) <= maxLength:
        return text
    return text[: maxLength - 3] + "..."


def formatSection(text: str) -> str:
    """Discord表示向けに整形。
    - 先頭の "** " を除去
    - 行頭が "** * " の箇条を "* " に置換
    - 余分な空白/空行を削除
    """
    cleaned = (text or "").strip()
    if cleaned.startswith("** "):
        cleaned = cleaned[len("** ") :]

    lines = [line.strip() for line in cleaned.split("\n")]
    formatted: list[str] = []
    for line in lines:
        if not line:
            continue
        if line.startswith("** * "):
            formatted.append("* " + line[len("** * ") :])
        else:
            formatted.append(line)
    return "\n".join(formatted)


def buildPaperEmbed(paper: Paper) -> discord.Embed:
    """論文情報をDiscord Embedに変換する共通関数"""
    metadata = paper.metadata
    analysis = paper.analysis

    embed = discord.Embed(
        title=metadata.title,
        url=metadata.pdf_url,
        color=discord.Color.dark_blue(),
    )

    # Authors
    authors = [a.name for a in metadata.authors[:3]]
    if len(metadata.authors) > 3:
        authors.append("...")
    embed.add_field(name="**👤 Authors**", value=", ".join(authors), inline=False)

    # Categories
    embed.add_field(name="**📂 Categories**", value=", ".join(metadata.categories), inline=True)

    # Analysis sections
    if analysis:
        divider = "\n"

        summary = formatSection(analysis.summary)
        novelty = formatSection(analysis.novelty)
        methodology = formatSection(analysis.methodology)
        results = formatSection(analysis.results)
        future_work = formatSection(analysis.future_work)

        embed.add_field(name="**📝 要約**", value=truncateText(summary) + divider, inline=False)
        embed.add_field(name="**💡 新規性**", value=truncateText(novelty) + divider, inline=False)
        embed.add_field(name="**🔍 手法**", value=truncateText(methodology) + divider, inline=False)
        embed.add_field(name="**📊 結果**", value=truncateText(results) + divider, inline=False)
        embed.add_field(name="**🔮 Future Work**", value=truncateText(future_work), inline=False)

        if analysis.research_themes:
            tips_text = "\n".join(f"{i+1}. {theme}" for i, theme in enumerate(analysis.research_themes))
            embed.add_field(name="**🧩 研究テーマのTips**", value=truncateText(tips_text), inline=False)

    embed.set_footer(text=f"Published: {metadata.published_date.strftime('%Y-%m-%d')}")
    return embed


def collectEmbedMetrics(embed: discord.Embed) -> Dict[str, int]:
    """Embedの単純なメトリクスを収集（フィールド数、文字数など）"""
    fieldsCount = len(embed.fields)
    valueChars = 0
    truncatedFieldsCount = 0
    for f in embed.fields:
        v = f.value
        if isinstance(v, str):
            valueChars += len(v)
            if v.rstrip().endswith("..."):
                truncatedFieldsCount += 1
    titleChars = len(embed.title or "")
    return {
        "fieldsCount": fieldsCount,
        "valueChars": valueChars,
        "truncatedFieldsCount": truncatedFieldsCount,
        "titleChars": titleChars,
    }
