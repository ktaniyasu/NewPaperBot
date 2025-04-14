# PaperAutoSummarizer

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
   git clone https://github.com/ktaniyasu/PaperAutoSummarizer
   cd arxiv_analyzer
   ```
2. 必要なライブラリをインストールします。

   ```bash
   pip install -r requirements.txt
   ```
3. `.env`ファイルを `.env.template`から作成し、以下の環境変数を設定します。

   ```env
   # API Keys
   GOOGLE_API_KEY=your_google_api_key_here //Gemini API KEY
   DISCORD_BOT_TOKEN=your_discord_bot_token_here //Discord Botのトークン

   # Discord Settings
   DISCORD_GUILD_ID=your_guild_id_here //DiscordのサーバーID
   DISCORD_CHANNEL_ID=your_channel_id_here //内容を投稿するチャンネルID

   # ArXiv API Settings
   ARXIV_CATEGORIES=cs.AI,cs.LG,cs.CL //ArXivから取得したいカテゴリ
   ```

## 実行方法

以下のコマンドでアプリケーションを起動します。

```bash
python -m src.main
```

アプリケーションは起動後、`TARGET_CHANNELS`で指定されたカテゴリの論文を定期的にチェックし、新しい論文が見つかると要約して該当するDiscordチャンネルに通知します。

現在の設定では平日のみ作動し、

## 停止方法

バックグラウンドで24時間毎に実行する設定になっているので、ctrl+cで停止させてください。
