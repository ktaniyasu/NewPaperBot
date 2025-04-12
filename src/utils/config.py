from pydantic_settings import BaseSettings
from typing import Optional, List, Dict
from pathlib import Path
import os
from dotenv import load_dotenv

# .envファイルを明示的に読み込む
load_dotenv(override=True)

class Settings(BaseSettings):
    # API Keys
    GOOGLE_API_KEY: str
    DISCORD_BOT_TOKEN: str

    # Discord Settings
    DISCORD_GUILD_ID: str

    # ArXiv API Settings
    ARXIV_MAX_RESULTS: int = 100
    ARXIV_RETRY_ATTEMPTS: int = 3
    ARXIV_TIMEOUT: int = 30

    # PDF Processing
    PDF_CHUNK_SIZE: int = 1000
    PDF_CHUNK_OVERLAP: int = 200

    # Database Settings
    DATABASE_URL: str = "sqlite:///data/arxiv_papers.db"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = Path("logs/arxiv_analyzer.log")

    # LLM Settings
    LLM_MODEL: str = "gemini-2.5-pro-exp-03-25"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048
    LLM_REQUEST_TIMEOUT: int = 60

    # Target channels for ArXiv categories
    target_channels: List[Dict[str, str]] = []

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # Allow extra fields in env file

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
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
                
            self.target_channels.append({
                "category": category,
                "channel_id": channel_id
            })
            i += 1

        if not self.target_channels:
            print("Warning: No ARXIV_CATEGORY_N/DISCORD_CHANNEL_ID_N pairs found in .env file.")


settings = Settings()
