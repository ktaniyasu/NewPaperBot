# NewPaperBot

arXivから最新の論文情報を取得し、指定されたカテゴリの論文を要約してDiscordに通知するツールです。


## 機能

- 指定したarXivカテゴリの最新論文を定期的にチェック
- 取得した論文のPDFをダウンロードし、内容を要約
- 要約結果を指定したDiscordチャンネルに通知

## 必要な環境

- Python 3.10 以上

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

アプリケーションは起動後、`TARGET_CHANNELS`で指定されたカテゴリの論文を定期的にチェックし、新しい論文が見つかると要約して該当するDiscordチャンネルに通知します。複数カテゴリを指定する場合は

```markdown
ARXIV_CATEGORY_1="astro-ph.CO"
DISCORD_CHANNEL_ID_1="hogehoge"

ARXIV_CATEGORY_2="astro-ph.EP"
DISCORD_CHANNEL_ID_2="hogehoge"

ARXIV_CATEGORY_3="astro-ph.GA"
DISCORD_CHANNEL_ID_3="hogehoge"

ARXIV_CATEGORY_4="astro-ph.HE"
DISCORD_CHANNEL_ID_4="hogehoge"

ARXIV_CATEGORY_5="astro-ph.IM"
DISCORD_CHANNEL_ID_5="hogehgoe"

ARXIV_CATEGORY_6="astro-ph.SR"
DISCORD_CHANNEL_ID_6="hogehoge"
```

の形で指定をしてください。

現在の設定では平日のみ作動するようになっています。

## 停止方法

バックグラウンドで24時間毎に実行する設定になっているので、ctrl+cで停止させてください。

## 現状起こっている問題

* [ ] 出力そのものが安定しない (LLM)
* [ ] 50MB以上・1000ページ以上の論文は処理できない(LLM)
* [ ] 出力のparsing(markdownの整形)
