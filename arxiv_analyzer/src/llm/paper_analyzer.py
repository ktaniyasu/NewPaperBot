import google.generativeai as genai
import httpx
from pathlib import Path
import tempfile
from typing import Optional, Tuple
import re
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

    async def analyze_paper(self, paper: Paper) -> Paper:
        """論文を解析する"""
        try:
            # PDFのダウンロード
            pdf_path = await self._download_pdf(paper.metadata.pdf_url)
            if not pdf_path:
                raise ValueError("Failed to download PDF")

            # プロンプトの構築とPDFの読み込み
            prompt = self._build_prompt(paper.metadata.title, paper.metadata.abstract)

            # Geminiによる解析
            response = await self._generate_analysis(prompt, pdf_path)

            # 一時ファイルの削除
            Path(pdf_path).unlink()

            try:
                # 解析結果をパースしてPaperオブジェクトを更新
                paper.analysis = self._parse_response(response)
            except Exception as parse_error:
                # パース失敗時にはLLMの生の出力を保存
                log.error(f"Failed to parse LLM response: {str(parse_error)}")
                paper.error_log = f"Parse error: {str(parse_error)}\n\nRaw LLM response:\n{response}"
                # デフォルトの解析失敗結果を設定
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
            return paper

    async def _download_pdf(self, url: str) -> Optional[str]:
        """PDFをダウンロードして一時ファイルとして保存する"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()

                # 一時ファイルにPDFを保存
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

                    """

    async def _generate_analysis(self, prompt: str, pdf_path: str) -> str:
        """Geminiを使用して解析を実行する"""
        try:
            # PDFファイルを読み込む
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()

            # PDFをmime_typeとともにモデルに渡す
            response = await self.model.generate_content_async([
                prompt,
                {
                    "mime_type": "application/pdf",
                    "data": pdf_data
                }
            ])

            # レスポンスの検証
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
            # 応答テキストを各セクションに分割
            sections = response.split('\n\n')
            
            # デフォルト値を設定
            analysis = {
                'summary': '解析失敗',
                'novelty': '解析失敗',
                'methodology': '解析失敗',
                'results': '解析失敗',
                'future_work': '解析失敗'
            }
            
            # 各セクションを解析
            current_section = None
            current_text = []
            
            for section in sections:
                section_text = section.strip()
                if not section_text:
                    continue
                
                # Remove leading markdown-like formatting (**, *, #, 1. etc.) and surrounding emphasis
                cleaned_text = re.sub(r"^\s*[\*#]+\s*|\d+\.\s*", "", section_text).strip()
                cleaned_text = re.sub(r"^\*+|\*+$", "", cleaned_text).strip() # Handle asterisks
                cleaned_text = re.sub(r"^_|_$", "", cleaned_text).strip() # Handle underscores

                # セクションヘッダーの検出
                try:
                    # Use cleaned_text for header checking
                    if '要約:' in cleaned_text or '要約：' in cleaned_text:
                        current_section = 'summary'
                        # Extract text after the header from the original section_text
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
                    # Append the original, uncleaned text if it's part of a section
                    elif current_section:
                         current_text.append(section_text) # Append original text

                    # Assign the joined original text
                    if current_section and current_text:
                        analysis[current_section] = ' '.join(current_text)

                except Exception as e:
                    log.error(f"Error parsing section '{section_text}': {str(e)}")
                    continue
            
            # 解析結果の確認
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