import asyncio
from pathlib import Path

import httpx
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from .config import settings
from .logger import log
from .rate_limit import AsyncRateLimiter

# グローバルな HTTP クライアント（接続の再利用）
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()

# レートリミッタ（QPS/並列度）
httpLimiter = AsyncRateLimiter(qps=settings.HTTP_QPS, concurrency=settings.HTTP_CONCURRENCY)


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:
                _client = httpx.AsyncClient(
                    timeout=settings.HTTP_REQUEST_TIMEOUT,
                    follow_redirects=True,
                    http2=True,
                )
    return _client


async def downloadToFile(url: str, destPath: str) -> None:
    """URL からファイルをストリームでダウンロードし、destPath に保存する。
    リトライ・タイムアウト・レートリミットを統一的に適用する。
    """
    # http を https に正規化
    url = url.replace("http://", "https://")
    Path(destPath).parent.mkdir(parents=True, exist_ok=True)

    async with httpLimiter:
        client = await _get_client()
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(settings.HTTP_RETRY_ATTEMPTS),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        ):
            with attempt:
                log.debug(f"HTTP GET {url} -> {destPath}")
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(destPath, "wb") as f:
                        async for chunk in resp.aiter_bytes():
                            f.write(chunk)


async def fetchText(url: str) -> str:
    """URL からテキストレスポンスを取得するヘルパー。
    リトライ・タイムアウト・レートリミットを統一的に適用する。
    """
    # http を https に正規化
    url = url.replace("http://", "https://")
    async with httpLimiter:
        client = await _get_client()
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(settings.HTTP_RETRY_ATTEMPTS),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        ):
            with attempt:
                log.debug(f"HTTP GET {url}")
                resp = await client.get(url, headers={"User-Agent": "arxiv-analyzer/1.0"})
                resp.raise_for_status()
                # encoding は httpx が推定する。明示すべき場合は resp.encoding を調整。
                return resp.text
