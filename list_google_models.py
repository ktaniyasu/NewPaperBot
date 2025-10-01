import os

from dotenv import load_dotenv
from google import genai

# .envファイルから環境変数を読み込む
# プロジェクトルートに .env があることを想定
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("エラー: .envファイルに GOOGLE_API_KEY が見つかりません。")
    print(f".envファイルのパス: {dotenv_path}")
    # .envファイルの内容を部分的に表示（デバッグ用、APIキー自体は表示しない）
    try:
        with open(dotenv_path) as f:
            print(".envの内容（一部）:")
            for i, line in enumerate(f):
                if i < 5:  # 最初の5行のみ表示
                    if "GOOGLE_API_KEY" in line:
                        print("GOOGLE_API_KEY=**** (found)")
                    else:
                        print(line.strip())
                else:
                    break
    except FileNotFoundError:
        print(".envファイルが見つかりませんでした。")

else:
    try:
        client = genai.Client(api_key=api_key)
        print("利用可能なモデル (generateContentをサポート):")
        found_models = False
        for model in client.models.list():
            # 'supported_actions' 属性が存在し、かつ 'generateContent' が含まれているか確認
            if hasattr(model, "supported_actions") and "generateContent" in model.supported_actions:
                print(f"  モデル名: {model.name}, 表示名: {model.display_name}")
                # 必要であれば、他の属性も表示
                found_models = True

        if not found_models:
            print("  generateContent をサポートするモデルが見つかりませんでした。")
            print("  利用可能なモデルとサポートされるアクション:")
            for model in client.models.list():
                actions = getattr(model, "supported_actions", "N/A")
                print(f"    - {model.name} ({model.display_name}): {actions}")
    except Exception as e:
        print(f"モデルリスト取得中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()

print("\nスクリプトの実行が完了しました。")
