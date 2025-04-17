import google.generativeai as genai
import httpx
from pathlib import Path
import tempfile
from typing import Optional, Tuple
import re
import datetime
import asyncio
from ..models.paper import Paper, AnalysisResult
from ..utils.config import settings
from ..utils.logger import log

class PaperAnalyzer:
    def __init__(self):
        """論文解析クラスの初期化"""
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=settings.LLM_MODEL,
            generation_config={
                "temperature": settings.LLM_TEMPERATURE,
                "max_output_tokens": settings.LLM_MAX_TOKENS,
            }
        )
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
                    novelty="PDFサイズが50MBを超えているため解析不可",
                    methodology="PDFサイズが50MBを超えているため解析不可",
                    results="PDFサイズが50MBを超えているため解析不可",
                    future_work="PDFサイズが50MBを超えているため解析不可"
                )
                paper.error_log = f"PDFサイズが50MBを超えているため解析不可"
                return paper

            # プロンプトの構築とPDFの読み込み
            prompt = self._build_prompt(paper.metadata.title, paper.metadata.abstract)

            # Geminiによる解析
            response = await self._generate_analysis(prompt, pdf_path)

            try:
                # LLMの生の応答をファイルに保存
                log_dir = Path("logs/llm_raw_responses")
                log_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_arxiv_id = paper.metadata.arxiv_id.replace('/', '_')
                log_filename = log_dir / f"{safe_arxiv_id}_{timestamp}.txt"
                with open(log_filename, "w", encoding="utf-8") as f:
                    f.write(response)
                log.info(f"Saved raw LLM response for {paper.metadata.arxiv_id} to {log_filename}")
            except Exception as log_save_error:
                log.error(f"Failed to save raw LLM response for {paper.metadata.arxiv_id}: {str(log_save_error)}")

            try:
                # 解析結果をパースしてPaperオブジェクトを更新
                paper.analysis = self._parse_response(response)
            except Exception as parse_error:
                log.error(f"Failed to parse LLM response: {str(parse_error)}")
                paper.error_log = f"Parse error: {str(parse_error)}\n\nRaw LLM response:\n{response}"
                paper.analysis = AnalysisResult(
                    summary="解析失敗",
                    novelty="解析失敗",
                    methodology="解析失敗",
                    results="解析失敗",
                    future_work="解析失敗"
                )

            return paper

        except Exception as e:
            log.error(f"Error analyzing paper {paper.metadata.arxiv_id}: {str(e)}")
            paper.error_log = str(e)
            paper.analysis = AnalysisResult(
                summary="解析エラー発生",
                novelty="解析エラー発生",
                methodology="解析エラー発生",
                results="解析エラー発生",
                future_work="解析エラー発生"
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
        return f"""以下の論文（PDFファイル）を解析し、5つの観点から日本語で要約してください。
                    図表も含めて総合的に分析してください。

                    論文タイトル: {title}

                    アブストラクト:
                    {abstract}

                    以下の5つの観点【出力フォーマット】から分析してください:
                    1. 要約: 論文の主要なポイントを簡潔に
                    2. 新規性: この研究の新しい点、独創的な点（図表から読み取れる新しい発見も含めて）
                    3. 手法: 採用されている手法や実験方法の概要（図表から読み取れる実験設定なども含めて）
                    4. 結果: 主要な実験結果や成果（図表の内容も含めた具体的な数値や比較結果）
                    5. Future Work: 今後の課題や展望
                    
                    上記【出力フォーマット】を必ずこの形式で出力してください。
                    Jsonによる出力は禁止されています。
                    """

    async def _generate_analysis(self, prompt: str, pdf_path: str) -> str:
        """Geminiを使用して解析を実行する"""
        try:
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()

            pdf_size_kb = len(pdf_data) / 1024
            log.debug(f"Sending request to Gemini. Prompt length: {len(prompt)}, PDF path: {pdf_path}, PDF size: {pdf_size_kb:.2f} KB")

            # Geminiリクエストにタイムアウトを設定
            response = await asyncio.wait_for(
                self.model.generate_content_async([
                    prompt,
                    {
                        "mime_type": "application/pdf",
                        "data": pdf_data
                    }
                ]),
                timeout=settings.LLM_REQUEST_TIMEOUT
            )

            if not response.candidates:
                raise ValueError("No response received from Gemini")
            
            if not response.candidates[0].content.parts:
                raise ValueError("No content parts in Gemini response")
            
            return response.candidates[0].content.parts[0].text

        except Exception as e:
            log.error(f"Error generating content with Gemini: {str(e)}")
            raise

    def _parse_response(self, response: str) -> AnalysisResult:
        try:            
            sections = response.split('\n\n')
            
            analysis = {
                'summary': '解析失敗',
                'novelty': '解析失敗',
                'methodology': '解析失敗',
                'results': '解析失敗',
                'future_work': '解析失敗'
            }
            
            current_section = None
            current_text = []
            
            for section in sections:
                section_text = section.strip()
                if not section_text:
                    continue
                
                cleaned_text = re.sub(r"^\s*[\*#]+\s*|\d+\.\s*", "", section_text).strip()
                cleaned_text = re.sub(r"^\*+|\*+$", "", cleaned_text).strip()
                cleaned_text = re.sub(r"^_|_$", "", cleaned_text).strip()

                try:
                    if '要約:' in cleaned_text or '要約：' in cleaned_text:
                        current_section = 'summary'
                        split_text = section_text.replace('：', ':').split(':', 1)
                        current_text = [split_text[1].strip()] if len(split_text) > 1 else []
                    elif '新規性:' in cleaned_text or '新規性：' in cleaned_text:
                        current_section = 'novelty'
                        split_text = section_text.replace('：', ':').split(':', 1)
                        current_text = [split_text[1].strip()] if len(split_text) > 1 else []
                    elif '手法:' in cleaned_text or '手法：' in cleaned_text:
                        current_section = 'methodology'
                        split_text = section_text.replace('：', ':').split(':', 1)
                        current_text = [split_text[1].strip()] if len(split_text) > 1 else []
                    elif '結果:' in cleaned_text or '結果：' in cleaned_text:
                        current_section = 'results'
                        split_text = section_text.replace('：', ':').split(':', 1)
                        current_text = [split_text[1].strip()] if len(split_text) > 1 else []
                    elif 'Future Work:' in cleaned_text or 'Future Work：' in cleaned_text:
                        current_section = 'future_work'
                        split_text = section_text.replace('：', ':').split(':', 1)
                        current_text = [split_text[1].strip()] if len(split_text) > 1 else []
                    elif current_section:
                         current_text.append(section_text)

                    if current_section and current_text:
                        analysis[current_section] = ' '.join(current_text)

                except Exception as e:
                    log.error(f"Error parsing section '{section_text}': {str(e)}")
                    continue
            
            failed_sections = [section for section in analysis if analysis[section] == '解析失敗']
            if failed_sections:
                log.debug(f"Failed to parse sections: {failed_sections}")
                log.debug("Raw LLM response:")
                log.debug("-" * 50)
                log.debug(response)
                log.debug("-" * 50)

            return AnalysisResult(**analysis)

        except Exception as e:
            log.error(f"Error in _parse_response: {str(e)}")
            raise

    async def _translate_text(self, text: str) -> str:
        """LLMで英文を日本語訳する"""
        try:
            prompt = f"次の英文を日本語に翻訳してください。\n\n{text} 翻訳文そのもの以外の出力は禁止されています。"
            response = await self.model.generate_content_async([prompt])
            if not response.candidates or not response.candidates[0].content.parts:
                raise ValueError("No response from LLM for translation")
            return response.candidates[0].content.parts[0].text.strip()
        except Exception as e:
            log.error(f"Error translating text: {str(e)}")
            return "翻訳エラー発生"