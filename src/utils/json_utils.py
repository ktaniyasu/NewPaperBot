from __future__ import annotations

import json
import re
from typing import Any


def extractJsonBlocks(text: str) -> list[str]:
    """
    テキストから JSON ブロック候補を抽出する。
    優先度:
      1) ```json ... ``` フェンスブロック
      2) 最初と最後の波括弧で囲まれた範囲
    """
    if not text:
        return []

    blocks: list[str] = []

    # ```json ... ``` の抽出
    pattern = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
    for m in pattern.finditer(text):
        blocks.append(m.group(1))

    if blocks:
        return blocks

    # 最初と最後の { ... } を抽出
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        blocks.append(text[start : end + 1])

    return blocks


def parseJsonFromText(text: str) -> dict[str, Any]:
    """
    テキストから最もそれらしい JSON をパースして辞書を返す。
    失敗時は空 dict を返す。
    """
    for block in extractJsonBlocks(text):
        try:
            return json.loads(block)
        except Exception:
            continue
    return {}
