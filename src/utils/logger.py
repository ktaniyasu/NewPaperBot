import sys
from pathlib import Path

from loguru import logger

from .config import settings


def setup_logger():
    """ロギングの設定を行う"""
    # ログファイルのディレクトリが存在しない場合は作成
    log_dir = Path(settings.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # 既存のハンドラをクリア
    logger.remove()

    # フォーマット/シリアライズを選択
    if getattr(settings, "LOG_JSON", False):
        # JSON で構造化ログを出力（stdout とファイル）
        logger.add(
            sys.stdout,
            level=settings.LOG_LEVEL,
            serialize=True,
        )
        logger.add(
            settings.LOG_FILE,
            level=settings.LOG_LEVEL,
            serialize=True,
            rotation="1 day",
            retention="1 month",
        )
    else:
        # 人が読みやすいテキストフォーマット
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=settings.LOG_LEVEL,
        )
        logger.add(
            settings.LOG_FILE,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=settings.LOG_LEVEL,
            rotation="1 day",
            retention="1 month",
        )

    return logger


# グローバルなロガーインスタンスを作成
log = setup_logger()
