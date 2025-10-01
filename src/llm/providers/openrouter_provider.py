import json
from typing import Any, Optional, Union, Dict

from openai import AsyncOpenAI

from ...utils.config import settings
from ...utils.logger import log
from ...utils.json_utils import parseJsonFromText


class OpenRouterProvider:
    """
    OpenRouter (OpenAI 互換 API) 用の Provider 実装。
    - Chat Completions エンドポイントを使用
    - 構造化出力フォールバック: json_schema → json_object → 手動抽出
    - ファイルアップロードは非対応
    """

    def __init__(
        self,
        model: str,
        apiKey: Optional[str],
        baseUrl: str,
        providerOptions: Optional[Union[str, Dict[str, Any]]] = None,
        reasoningOptions: Optional[Union[str, Dict[str, Any]]] = None,
    ):
        self.model = model
        self.apiKey = apiKey
        self.baseUrl = baseUrl
        # OpenAI SDK を OpenRouter エンドポイントに向ける
        self.client = AsyncOpenAI(
            api_key=self.apiKey,
            base_url=self.baseUrl,
            timeout=float(settings.LLM_REQUEST_TIMEOUT),
        )
        # Sanity checks for API key (do not print the key itself)
        try:
            sk = (self.apiKey or "").strip()
            if not sk:
                log.error("OpenRouter API key is missing or empty; requests will fail with 401.")
            elif not sk.startswith("sk-or-"):
                log.warning("OpenRouter API key does not look like 'sk-or-' prefixed key.")
        except Exception:
            pass
        # OpenRouter 拡張オプション
        self._provider = self._parse_json(providerOptions)
        self._reasoning = self._parse_json(reasoningOptions)
        try:
            log.info(
                "OpenRouter initialized: model={}, provider={}, reasoning={}",
                self.model,
                self._provider,
                self._reasoning,
            )
        except Exception:
            pass

    def _parse_json(self, v: Optional[Union[str, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        try:
            return json.loads(v)
        except Exception:
            try:
                log.warning("Failed to parse OpenRouter options JSON: {}", v)
            except Exception:
                pass
            return None

    def _extra_body(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if self._provider:
            body["provider"] = self._provider
        if self._reasoning:
            body["reasoning"] = self._reasoning
        try:
            log.debug("OpenRouter extra_body={}", body)
        except Exception:
            pass
        return body

    def supportsFileUpload(self) -> bool:
        return False

    async def generateStructuredFromFile(
        self, prompt: str, pdfPath: str, schema: Any, **kwargs
    ) -> dict[str, Any]:
        raise NotImplementedError("OpenRouterProvider: file upload is not supported")

    async def generateStructuredFromText(
        self, prompt: str, text: str, schema: Any, **kwargs
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON."},
            {
                "role": "user",
                "content": f"{prompt}\n\n本文:\n{text}",
            },
        ]

        # 1) json_schema（対応モデルのみ、失敗時はフォールバック）
        if settings.LLM_STRICT_JSON and hasattr(schema, "model_json_schema"):
            try:
                schema_dict = schema.model_json_schema()
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "paper_analysis",
                            "schema": schema_dict,
                            "strict": True,
                        },
                    },
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    timeout=float(settings.LLM_REQUEST_TIMEOUT),
                    extra_body=self._extra_body(),
                )
                return self._as_dict(resp)
            except Exception:
                # 次にフォールバック
                pass

        # 2) json_object
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                timeout=float(settings.LLM_REQUEST_TIMEOUT),
                extra_body=self._extra_body(),
            )
            return self._as_dict(resp)
        except Exception:
            pass

        # 3) 手動抽出
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=float(settings.LLM_REQUEST_TIMEOUT),
            extra_body=self._extra_body(),
        )
        return self._manual_extract(resp)

    async def translate(self, text: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": "You are a translator from English to Japanese."},
            {"role": "user", "content": f"次の英文を日本語に翻訳してください。\n\n{text}"},
        ]
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            timeout=float(settings.LLM_REQUEST_TIMEOUT),
            extra_body=self._extra_body(),
        )
        content = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        return content or "翻訳エラー発生"

    def _as_dict(self, resp: Any) -> dict[str, Any]:
        try:
            content = resp.choices[0].message.content if resp.choices else None
        except Exception:
            content = None
        if not content:
            return {}
        try:
            return json.loads(content)
        except Exception:
            # フェンス/波括弧から抽出
            parsed = parseJsonFromText(content)
            if not isinstance(parsed, dict) or not parsed:
                try:
                    head = (content or "")[:512].replace("\n", " ")
                    log.debug("JSON parse failed; head={} ...", head)
                except Exception:
                    pass
            return parsed

    def _manual_extract(self, resp: Any) -> dict[str, Any]:
        try:
            text = resp.choices[0].message.content or ""
        except Exception:
            return {}
        parsed = parseJsonFromText(text)
        if not isinstance(parsed, dict) or not parsed:
            try:
                head = (text or "")[:512].replace("\n", " ")
                log.debug("Manual extract failed; head={} ...", head)
            except Exception:
                pass
        return parsed
