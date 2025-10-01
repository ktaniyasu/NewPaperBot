from typing import Optional

from ...utils.config import settings
from .base import BaseLLMProvider
from .gemini_provider import GeminiProvider
from .openrouter_provider import OpenRouterProvider


class ProviderFactory:
    @staticmethod
    def fromSettings(cfg: Optional[object] = None) -> BaseLLMProvider:
        cfg = cfg or settings
        provider = cfg.LLM_PROVIDER.lower()
        if provider == "gemini":
            return GeminiProvider(model=cfg.LLM_MODEL, apiKey=cfg.GOOGLE_API_KEY)
        if provider == "openrouter":
            return OpenRouterProvider(
                model=cfg.LLM_MODEL,
                apiKey=cfg.OPENROUTER_API_KEY,
                baseUrl=cfg.OPENROUTER_BASE_URL,
                providerOptions=cfg.OPENROUTER_PROVIDER_OPTIONS,
                reasoningOptions=cfg.OPENROUTER_REASONING_OPTIONS,
            )
        raise ValueError(f"Unsupported LLM_PROVIDER: {cfg.LLM_PROVIDER}")
