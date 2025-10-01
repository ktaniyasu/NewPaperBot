from typing import Any, Optional

from google import genai

from ...utils.config import settings
from ...utils.json_utils import parseJsonFromText


class GeminiProvider:
    """
    Google Gemini 用の Provider ラッパー。
    - ファイルアップロード + 構造化出力（response_schema）
    - 手動 JSON 抽出フォールバック
    """

    def __init__(self, model: str, apiKey: Optional[str] = None):
        self.model = model
        self.apiKey = apiKey  # None の場合は環境変数 GOOGLE_API_KEY を利用
        self.client = genai.Client(api_key=self.apiKey)

    def supportsFileUpload(self) -> bool:
        return True

    async def generateStructuredFromFile(
        self, prompt: str, pdfPath: str, schema: Any, **kwargs
    ) -> dict[str, Any]:
        file = await self.client.aio.files.upload(file=pdfPath)
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=[file, prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                    "temperature": settings.LLM_TEMPERATURE,
                    "max_output_tokens": settings.LLM_MAX_TOKENS,
                },
            )
            return self._parse_or_extract(response)
        finally:
            # Files API にアップロードした一時ファイルは必ず削除
            try:
                await self.client.aio.files.delete(name=getattr(file, "name", None))
            except Exception:
                # ここでの失敗は本処理に影響させない
                pass

    async def generateStructuredFromText(
        self, prompt: str, text: str, schema: Any, **kwargs
    ) -> dict[str, Any]:
        contents = f"{prompt}\n\n本文:\n{text}"
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": schema,
                "temperature": settings.LLM_TEMPERATURE,
                "max_output_tokens": settings.LLM_MAX_TOKENS,
            },
        )
        return self._parse_or_extract(response)

    async def translate(self, text: str, **kwargs) -> str:
        prompt = f"次の英文を日本語に翻訳してください。\n\n{text}"
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": 0.3},
        )
        return (getattr(response, "text", None) or "").strip() or "翻訳エラー発生"

    def _parse_or_extract(self, response: Any) -> dict[str, Any]:
        # 1) SDK の自動パース
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            # Pydantic モデルであれば model_dump、辞書ならそのまま
            if hasattr(parsed, "model_dump"):
                return parsed.model_dump()
            if isinstance(parsed, dict):
                return parsed
        # 2) テキストから JSON 抽出（ユーティリティ）
        raw_text = getattr(response, "text", None) or ""
        return parseJsonFromText(raw_text)
