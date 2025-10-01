import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv, find_dotenv


class Settings(BaseSettings):
    # API Keys
    GOOGLE_API_KEY: str
    DISCORD_BOT_TOKEN: Optional[str] = None
    # OpenRouter
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    # OpenRouter vendor-specific options (JSON string pass-through)
    # Example:
    # OPENROUTER_PROVIDER_OPTIONS='{"only":["chutes"],"allow_fallbacks":false,"data_collection":"deny"}'
    # OPENROUTER_REASONING_OPTIONS='{"enabled": false}'
    OPENROUTER_PROVIDER_OPTIONS: Optional[str] = None
    OPENROUTER_REASONING_OPTIONS: Optional[str] = None

    # Discord Settings
    DISCORD_GUILD_ID: Optional[str] = None

    # ArXiv API Settings
    ARXIV_MAX_RESULTS: int = 100
    ARXIV_RETRY_ATTEMPTS: int = 3
    ARXIV_TIMEOUT: int = 30
    ARXIV_QPS: float = 1.0
    ARXIV_CONCURRENCY: int = 1

    # ar5iv HTML ingestion
    USE_AR5IV_HTML: bool = True
    AR5IV_BASE_URL: str = "https://ar5iv.org/html"

    # PDF Processing
    PDF_CHUNK_SIZE: int = 1000
    PDF_CHUNK_OVERLAP: int = 200

    # RAG Settings
    RAG_USE_FAISS: bool = False
    RAG_EMBED_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    RAG_TOP_K: int = 6
    RAG_MIN_SIMILARITY: float = 0.2
    # RAG cache/persistence
    RAG_EMBED_CACHE: bool = True
    RAG_EMBED_CACHE_DIR: Path = Path("cache/embeddings")
    RAG_INDEX_PERSIST: bool = True
    RAG_INDEX_DIR: Path = Path("cache/faiss")

    # Database Settings
    DATABASE_URL: str = "sqlite:///data/arxiv_papers.db"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = Path("logs/arxiv_analyzer.log")
    LOG_JSON: bool = False

    # LLM Settings
    LLM_PROVIDER: str = "gemini"  # gemini | openrouter | openai (将来互換)
    LLM_MODEL: Optional[str] = None  # Optional custom model name
    # Token budget for Gemini thinking phase
    LLM_THINKING_BUDGET: int = 1024
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 8192
    # 入力側のコンテキストウィンドウの想定トークン数（概算）
    # 各モデル固有の実値は異なるため、必要に応じて .env で調整してください。
    LLM_CONTEXT_WINDOW_TOKENS: int = 200_000
    LLM_REQUEST_TIMEOUT: int = 60
    # Structured output/tool flags
    LLM_STRICT_JSON: bool = True
    LLM_USE_TOOLS: bool = True
    LLM_CONCURRENCY: int = 3

    # HTTP (PDF ダウンロード等) の共通設定
    HTTP_REQUEST_TIMEOUT: int = 60
    HTTP_RETRY_ATTEMPTS: int = 3
    HTTP_QPS: float = 4.0
    HTTP_CONCURRENCY: int = 4

    # Debug Settings
    DEBUG_RUN_IMMEDIATELY: bool = False

    # Target channels for ArXiv categories
    target_channels: list[dict[str, str]] = []

    # Pydantic v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Normalize API keys (trim whitespace) before validations
        if self.OPENROUTER_API_KEY is not None:
            self.OPENROUTER_API_KEY = self.OPENROUTER_API_KEY.strip()

        # Override LLM_MODEL based on provider
        provider = self.LLM_PROVIDER.lower()
        if provider == "gemini":
            # Use custom model if set, otherwise default to 2.5 flash preview
            if not self.LLM_MODEL:
                self.LLM_MODEL = "models/gemini-2.5-flash-preview-05-20"
            # Google GenAI の低レベルAPIでは models/ プレフィックスが必要
            if not self.LLM_MODEL.startswith("models/"):
                self.LLM_MODEL = f"models/{self.LLM_MODEL}"
        elif provider == "openrouter":
            # OpenRouter ではモデルIDの指定が必須（例: openai/gpt-4o-mini）
            if not self.LLM_MODEL:
                raise ValueError(
                    "LLM_MODEL is required when LLM_PROVIDER is 'openrouter' (e.g., 'openai/gpt-4o-mini')."
                )
            if not self.OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY is required for LLM_PROVIDER 'openrouter'")
            # (Optional) heuristic warning for suspicious key format
            try:
                if not str(self.OPENROUTER_API_KEY).startswith("sk-or-"):
                    print("Warning: OPENROUTER_API_KEY does not look like an OpenRouter key (expected prefix 'sk-or-').")
            except Exception:
                pass
        elif provider == "openai":
            self.LLM_MODEL = "o4-mini"
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.LLM_PROVIDER}")

        # Load ArXiv category and Discord channel pairs dynamically
        i = 1
        self.target_channels = []  # Reset the list to ensure we start fresh
        while True:
            category_key = f"ARXIV_CATEGORY_{i}"
            channel_id_key = f"DISCORD_CHANNEL_ID_{i}"

            # Try getting values from both kwargs and environment
            category = os.getenv(category_key)
            channel_id = os.getenv(channel_id_key)

            if not category or not channel_id:
                break

            # Basic sanitization
            category = category.strip()
            channel_id = channel_id.strip()
            if not category:
                raise ValueError(f"{category_key} must not be empty")
            if not channel_id.isdigit():
                raise ValueError(f"{channel_id_key} must be a numeric string, got: {channel_id!r}")

            self.target_channels.append({"category": category, "channel_id": channel_id})
            i += 1

        if not self.target_channels:
            print("Warning: No ARXIV_CATEGORY_N/DISCORD_CHANNEL_ID_N pairs found in .env file.")

        # target_channels の簡易バリデーション
        for ch in self.target_channels:
            if not isinstance(ch, dict) or "category" not in ch or "channel_id" not in ch:
                raise ValueError(
                    "Invalid target_channels entry. Must include 'category' and 'channel_id'."
                )

# Ensure .env variables are loaded regardless of current working directory.
# find_dotenv() searches upward to locate the nearest .env.
_DOTENV_PATH = find_dotenv()
if _DOTENV_PATH:
    load_dotenv(dotenv_path=_DOTENV_PATH, override=True)
else:
    # Fallback to local .env path if discovery failed
    load_dotenv(dotenv_path=".env", override=True)
settings = Settings()
