from ..utils.config import settings


def splitText(text: str, chunkSize: int | None = None, overlap: int | None = None) -> list[str]:
    """重なりを持つ固定長チャンクに分割"""
    size = chunkSize or settings.PDF_CHUNK_SIZE
    ov = overlap or settings.PDF_CHUNK_OVERLAP
    if size <= 0:
        return [text]
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        out.append(text[i : i + size])
        i += max(1, size - ov)
    return out
