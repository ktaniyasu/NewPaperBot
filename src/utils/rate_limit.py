import asyncio
import time
import contextlib
from contextlib import AbstractAsyncContextManager


class AsyncRateLimiter(AbstractAsyncContextManager):
    """
    非同期のレートリミッタ（QPS と並列度セマフォ）。
    - QPS: 1 秒あたりの最大実行回数を制御（min-interval ベース）
    - concurrency: 同時実行数を制限（asyncio.Semaphore）

    使い方:
        limiter = AsyncRateLimiter(qps=2.0, concurrency=2)
        async with limiter:
            await do_something()
    """

    def __init__(self, qps: float = 0.0, concurrency: int = 0) -> None:
        self.qps = max(0.0, float(qps))
        self.min_interval = (1.0 / self.qps) if self.qps > 0.0 else 0.0
        # concurrency <= 0 の場合は実質無制限に近い大きな値を用いる
        self._sem = asyncio.Semaphore(concurrency if concurrency and concurrency > 0 else 1_000_000)
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0  # monotonic time

    async def __aenter__(self):  # type: ignore[override]
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):  # type: ignore[override]
        self.release()

    async def acquire(self) -> None:
        await self._sem.acquire()
        if self.min_interval <= 0:
            return
        # min-interval に基づくスロットリング
        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            # 直列化して呼び出し間隔を空ける
            self._next_allowed = max(self._next_allowed, now) + self.min_interval

    def release(self) -> None:
        with contextlib.suppress(ValueError):
            # 二重解放等の安全ガード（通常発生しない）
            self._sem.release()
