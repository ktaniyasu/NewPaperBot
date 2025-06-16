import httpx
from pathlib import Path
import tempfile
from typing import Optional
import re
import datetime
import json
from google import genai
from google.genai import types as gtypes
from ..models.paper import Paper, AnalysisResult
from ..schemas.paper_analysis import PaperAnalysisSchema
from ..utils.config import settings
from ..utils.logger import log

class PaperAnalyzer:
    def __init__(self):
        """論文解析クラスの初期化"""
        log.info(
            f"LLMモデル: {settings.LLM_MODEL}, 温度: {settings.LLM_TEMPERATURE}, 最大トークン数: {settings.LLM_MAX_TOKENS}, タイムアウト: {settings.LLM_REQUEST_TIMEOUT}"
        )
        # google-genai SDK では Client を用いる
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.model_name = settings.LLM_MODEL
        self.MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024  # 50MB

    async def analyze_paper(self, paper: Paper) -> Paper:
        """論文を解析する"""
        pdf_path = None
        try:
            # PDFのダウンロード
            pdf_path = await self._download_pdf(paper.metadata.pdf_url)
            if not pdf_path:
                raise ValueError("Failed to download PDF")

            # ファイルサイズチェック
            pdf_file_size = Path(pdf_path).stat().st_size
            if pdf_file_size > self.MAX_PDF_SIZE_BYTES:
                # 50MB超の場合はsummaryのみ翻訳して返す
                translated_summary = await self._translate_text(paper.metadata.abstract)
                paper.analysis = AnalysisResult(
                    summary=translated_summary,
                    novelty="PDFファイルが大きすぎて解析できませんでした",
                    methodology="PDFファイルが大きすぎて解析できませんでした",
                    results="PDFファイルが大きすぎて解析できませんでした",
                    future_work="PDFファイルが大きすぎて解析できませんでした",
                    research_themes=[],
                )
                paper.error_log = f"PDFサイズが50MBを超えているため解析不可"
                return paper

            # プロンプトの構築
            prompt = self._build_prompt(
                paper.metadata.title, paper.metadata.abstract
            )

            # Gemini による解析（構造化出力）
            analysis_schema: PaperAnalysisSchema = await self._generate_analysis(
                prompt, pdf_path
            )

            # 生の JSON を保存
            try:
                log_dir = Path("logs/llm_raw_responses")
                log_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_arxiv_id = paper.metadata.arxiv_id.replace("/", "_")
                log_filename = log_dir / f"{safe_arxiv_id}_{timestamp}.json"
                with open(log_filename, "w", encoding="utf-8") as f:
                    f.write(analysis_schema.model_dump_json(indent=2))
                log.info(
                    f"Saved structured LLM response for {paper.metadata.arxiv_id} to {log_filename}"
                )
            except Exception as log_save_error:
                log.error(
                    f"Failed to save structured LLM response for {paper.metadata.arxiv_id}: {str(log_save_error)}"
                )

            # Paper オブジェクトへ反映
            paper.analysis = AnalysisResult(**analysis_schema.model_dump())

            return paper

        except Exception as e:
            log.error(f"Error analyzing paper {paper.metadata.arxiv_id}: {str(e)}")
            paper.error_log = str(e)
            paper.analysis = AnalysisResult(
                summary="解析エラー発生",
                novelty="解析エラー発生",
                methodology="解析エラー発生",
                results="解析エラー発生",
                future_work="解析エラー発生",
                research_themes=[],
            )
            return paper
        finally:
            if pdf_path and Path(pdf_path).exists():
                try:
                    Path(pdf_path).unlink()
                except Exception as unlink_error:
                    log.error(f"Error deleting temporary PDF file {pdf_path}: {str(unlink_error)}")

    async def _download_pdf(self, url: str) -> Optional[str]:
        """PDFをダウンロードして一時ファイルとして保存する"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()

                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_file.write(response.content)
                    return tmp_file.name

        except Exception as e:
            log.error(f"Error downloading PDF from {url}: {str(e)}")
            return None

    def _build_prompt(self, title: str, abstract: str) -> str:
        """解析用のプロンプトを構築する"""
        # NOTE: Structured output is enforced via `response_schema` in the API call.
        # Therefore, avoid giving the model contradictory instructions such as
        # "JSON output is forbidden". Simply describe the analysis task.
        return (
            f"""以下の論文（PDFファイル）を読み、図表も含めて総合的に分析してください。

論文タイトル: {title}

アブストラクト:
{abstract}

次の 6 項目について日本語で簡潔にまとめてください:
1. 要約 (summary)
2. 新規性 (novelty)
3. 手法 (methodology)
4. 結果 (results)
5. 今後の課題 (future_work)
6. 新しい論文を考える際のTipsとしての研究テーマ3件 (research_themes)
"""
        )

    async def _generate_analysis(
        self, prompt: str, pdf_path: str
    ) -> PaperAnalysisSchema:
        """Gemini 2.5 Flash で解析（構造化 JSON 出力）"""
        try:
            # PDF をアップロード（SDK 0.8 系では `file=` 引数が正式）
            file = await self.client.aio.files.upload(file=pdf_path)

            log.debug(
                f"Sending request to Gemini. Prompt length: {len(prompt)}, PDF: {pdf_path}"
            )

            # ファイルを先頭、テキストを後ろに配置するのが推奨
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=[file, prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": PaperAnalysisSchema,
                    "temperature": settings.LLM_TEMPERATURE,
                    "max_output_tokens": settings.LLM_MAX_TOKENS,
                },
            )

            # 1) SDK の自動パース結果を試す
            parsed = getattr(response, "parsed", None)

            # 2) 自動パースに失敗した場合は text を手動で検証
            if parsed is None:
                raw_text = getattr(response, "text", None)
                if not raw_text:
                    log.error(f"LLM response text is empty. Response: {str(response)[:400]}")
                    raise ValueError("LLM returned empty response.")

                try:
                    # 文字列から JSON 部分だけを抽出して再パースを試みる
                    # ```json ... ``` やコードブロックが含まれる場合を考慮
                    json_pattern = r"```json\s*(\{[\s\S]*?\})\s*```"
                    match = re.search(json_pattern, raw_text)
                    if match:
                        json_str = match.group(1)
                    else:
                        # 最初の { から最後の } までを強引に抜き出す
                        start = raw_text.find('{')
                        end = raw_text.rfind('}')
                        json_str = raw_text[start:end+1] if start != -1 and end != -1 else raw_text

                    parsed = PaperAnalysisSchema.model_validate_json(json_str)
                    log.debug("Extracted and validated JSON from raw response.")
                except Exception as ve:
                    log.error(
                        f"Failed manual JSON extraction. Raw: {raw_text[:500]}, Error: {str(ve)}"
                    )
                    raise ValueError("LLM returned invalid JSON format.") from ve

            # 応答がまったく返ってこない（安全ブロック等）場合
            if parsed is None:
                # safety 情報を確認
                safety_info = getattr(response, "prompt_feedback", None)
                log.error(f"LLM response blocked. Safety feedback: {safety_info}")
                raise ValueError("LLM response was blocked by safety settings.")

            return parsed  # PaperAnalysisSchema インスタンス

        except Exception as e:
            log.error(f"Error generating structured content with Gemini: {str(e)}")
            raise

    async def _translate_text(self, text: str) -> str:
        """LLMで英文を日本語訳する"""
        try:
            prompt = f"次の英文を日本語に翻訳してください。\n\n{text}"
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={"temperature": 0.3},
            )
            return response.text.strip()
        except Exception as e:
            log.error(f"Error translating text: {str(e)}")
            return "翻訳エラー発生"