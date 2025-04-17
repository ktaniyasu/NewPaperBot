import sys
from loguru import logger
from pathlib import Path
from .config import settings

def setup_logger():
    """ロギングの設定を行う"""
    # ログファイルのディレクトリが存在しない場合は作成
    log_dir = Path(settings.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # ログファイルを削除（新規実行時にクリア）
    log_file = Path(settings.LOG_FILE)
    if log_file.exists():
        log_file.unlink()

    def error_handler(message):
        """エラーが発生した場合にプログラムを停止する"""
        print(f"\nERROR: {message}")
        sys.exit(1)

    # ロガーの設定
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                "level": settings.LOG_LEVEL,
            },
            {
                "sink": settings.LOG_FILE,
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                "level": settings.LOG_LEVEL,
                "rotation": "1 day",
                "retention": "1 month",
            },
        ],
    }

    # 既存のハンドラをクリア
    logger.remove()

    # 新しい設定を適用
    for handler in config["handlers"]:
        logger.add(**handler)

    # クリティカルなエラーのみプログラム停止用ハンドラとして登録
    logger.add(error_handler, level="CRITICAL")

    return logger

# グローバルなロガーインスタンスを作成
log = setup_logger()
