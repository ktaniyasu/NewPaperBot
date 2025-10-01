import asyncio
from pathlib import Path

from pypdf import PdfReader


async def extract(pdfPath: str) -> str:
    """PDF からテキストを抽出（同期I/Oはスレッドオフロード）"""
    path = Path(pdfPath)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdfPath}")

    def _read() -> str:
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")
        return "\n".join(texts)

    return await asyncio.to_thread(_read)
