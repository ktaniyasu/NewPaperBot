import asyncio
import contextlib
import time
from collections import deque
from contextlib import AbstractAsyncContextManager
from typing import Deque


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


class RateLimitError(RuntimeError):
    """明示的なレート制限超過を表す例外"""

    pass


class AsyncSlidingWindowRateLimiter(AbstractAsyncContextManager):
    """
    任意の期間内の呼び出し回数を制限するスライディングウィンドウ型のレートリミッタ。

    max_calls が 0 以下の場合は無効化される。
    raise_on_exceed が True の場合、上限到達時に RateLimitError を送出する。
    """

    def __init__(
        self,
        max_calls: int,
        period_seconds: float,
        *,
        raise_on_exceed: bool = False,
    ) -> None:
        self.max_calls = int(max_calls)
        self.period_seconds = float(period_seconds)
        self.raise_on_exceed = raise_on_exceed
        self._timestamps: Deque[float] = deque()
        self._lock = asyncio.Lock()

    async def __aenter__(self):  # type: ignore[override]
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):  # type: ignore[override]
        return False

    async def acquire(self) -> None:
        if self.max_calls <= 0 or self.period_seconds <= 0:
            return

        while True:
            async with self._lock:
                now = time.monotonic()
                # 期間外の呼び出しを除去
                cutoff = now - self.period_seconds
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return

                wait_time = self.period_seconds - (now - self._timestamps[0])
                if wait_time <= 0:
                    continue
                if self.raise_on_exceed:
                    raise RateLimitError(
                        f"Rate limit exceeded: {self.max_calls} calls per {self.period_seconds} seconds"
                    )

            await asyncio.sleep(wait_time)
