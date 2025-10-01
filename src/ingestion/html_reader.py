from __future__ import annotations

from bs4 import BeautifulSoup

from ..utils.config import settings
from ..utils.http import fetchText
from ..utils.logger import log


async def fetchAr5ivHtml(arxivId: str) -> str:
    """ar5iv から HTML を取得する。
    例: https://ar5iv.org/html/2401.00001v1
    """
    base = settings.AR5IV_BASE_URL.rstrip("/")
    url = f"{base}/{arxivId}"
    html = await fetchText(url)
    return html or ""


def extractTextFromAr5ivHtml(html: str) -> str:
    """ar5iv の HTML から本文テキストを抽出する。
    主に <article> 要素を優先し、なければ #content, body をフォールバックに使う。
    不要な script/style/nav を除去。
    """
    if not html:
        return ""
    # パーサーは lxml 優先、未インストール環境では標準の html.parser にフォールバック
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # 不要要素を除去
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        try:
            tag.decompose()
        except Exception:
            pass

    # 優先: article
    article = soup.find("article")
    root = article or soup.select_one("#content") or soup.body or soup

    # 数式や参考文献識別用の軽い後処理（ラベル類除去）
    text = root.get_text("\n", strip=True) if root else soup.get_text("\n", strip=True)

    # 連続改行を軽く正規化
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = "\n".join([ln for ln in lines if ln])
    return cleaned


async def extractTextByArxivId(arxivId: str) -> str:
    """arXiv ID を受け取り、ar5iv HTML を取得して本文テキストを返す。"""
    try:
        html = await fetchAr5ivHtml(arxivId)
        return extractTextFromAr5ivHtml(html)
    except Exception as e:
        log.warning(f"Failed to fetch/parse ar5iv HTML for {arxivId}: {e}")
        return ""
