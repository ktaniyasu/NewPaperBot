<<<<<<< HEAD
# PaperAutoSummarizer
AI-Driven Summarizes for the Latest Research on ArXiv
=======
# Arxiv Analyzer

arXivから最新の論文情報を取得し、指定されたカテゴリの論文を要約してDiscordに通知するツールです。

## 機能

- 指定したarXivカテゴリの最新論文を定期的にチェック
- 取得した論文のPDFをダウンロードし、内容を要約
- 要約結果を指定したDiscordチャンネルに通知

## 必要な環境

- Python 3.8 以上

## 環境構築

1. リポジトリをクローンします。
   ```bash
   git clone <リポジトリURL>
   cd arxiv_analyzer
   ```

2. 必要なライブラリをインストールします。
   ```bash
   pip install -r requirements.txt
   ```

3. `.env`ファイルを作成し、以下の環境変数を設定します。
   ```env
   # Discord Bot Token
   DISCORD_BOT_TOKEN="YOUR_DISCORD_BOT_TOKEN"

   # OpenAI API Key (or other LLM provider key)
   OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

   # Target Discord channels and arXiv categories (JSON format)
   # 例:
   # TARGET_CHANNELS='[{"channel_id": "123456789012345678", "category": "cs.AI"}, {"channel_id": "987654321098765432", "category": "cs.CL"}]'
   TARGET_CHANNELS='YOUR_TARGET_CHANNELS_JSON'

   # Optional: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   # LOG_LEVEL="INFO"
   ```
   - `DISCORD_BOT_TOKEN`: Discordボットのトークン。
   - `OPENAI_API_KEY`: 論文要約に使用するLLMのAPIキー。
   - `TARGET_CHANNELS`: 通知先のDiscordチャンネルIDと対象のarXivカテゴリをJSON形式で指定します。

## 実行方法

以下のコマンドでアプリケーションを起動します。

```bash
python src/main.py
```

アプリケーションは起動後、`TARGET_CHANNELS`で指定されたカテゴリの論文を定期的にチェックし、新しい論文が見つかると要約して該当するDiscordチャンネルに通知します。

## 停止方法

Ctrl+C でアプリケーションを安全に停止できます。
>>>>>>> master
