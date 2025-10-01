import asyncio
import datetime
import os  # tmpfs 判定用
import shutil  # gs の存在確認
import subprocess  # Ghostscript 呼び出し用
import tempfile
import hashlib
from pathlib import Path
from typing import Any, Optional

from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from ..ingestion.chunking import splitText
from ..ingestion.pdf_reader import extract as extract_pdf_text
from ..ingestion.summarizer import combine
from ..ingestion.html_reader import extractTextByArxivId
from ..llm.providers import ProviderFactory
from ..models.paper import AnalysisResult, Paper
from ..rag.embedding import EmbeddingModel
from ..rag.vector_store import FaissVectorStore
from ..schemas.paper_analysis import PaperAnalysisSchema
from ..utils.config import settings
from ..utils.http import downloadToFile
from ..utils.json_utils import parseJsonFromText
from ..utils.logger import log


class PaperAnalyzer:
    def __init__(self):
        """論文解析クラスの初期化"""
        log.info(
            f"LLMモデル: {settings.LLM_MODEL}, 温度: {settings.LLM_TEMPERATURE}, 最大トークン数: {settings.LLM_MAX_TOKENS}, タイムアウト: {settings.LLM_REQUEST_TIMEOUT}"
        )
        # Provider ファサード
        self.provider = ProviderFactory.fromSettings(settings)
        self.model_name = settings.LLM_MODEL
        self.MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024  # 50MB
        # LLM 同時実行制御（フェーズ3）
        self._llm_sem = asyncio.Semaphore(settings.LLM_CONCURRENCY)

    async def _call_with_retry(self, func, *args, **kwargs):
        """tenacity による簡易リトライ（指数バックオフ）"""
        async with self._llm_sem:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=8),
                # 非再試行系（コンテキスト超過/認証系）は即時中断
                retry=retry_if_exception(lambda e: not self._is_non_retryable_error(e)),
                reraise=True,
            ):
                with attempt:
                    return await func(*args, **kwargs)

    def _is_context_overflow_error(self, e: Exception) -> bool:
        """LLM/Provider からのエラーがコンテキスト超過/入力長超過に該当するかの簡易判定。
        SDKやバックエンドに依存せず文字列ベースで安全に検知する。
        """
        try:
            msg = str(e).lower()
        except Exception:
            return False
        keywords = [
            "context window",
            "context length",
            "max context",
            "maximum context",
            "too many tokens",
            "token limit",
            "exceeds the context",
            "exceeds maximum input",
            "input too long",
            "prompt is too long",
            "invalidargument: 400",
        ]
        return any(k in msg for k in keywords)

    def _is_auth_error(self, e: Exception) -> bool:
        """OpenRouter/SDK 由来の認証エラー(401/403)を簡易検出。"""
        try:
            msg = str(e).lower()
        except Exception:
            return False
        auth_tokens = [
            "401",
            "403",
            "unauthorized",
            "forbidden",
            "no auth credentials found",
            "invalid api key",
        ]
        return any(t in msg for t in auth_tokens)

    def _is_non_retryable_error(self, e: Exception) -> bool:
        return self._is_context_overflow_error(e) or self._is_auth_error(e)

    def _truncate(self, s: str, length: int) -> str:
        return s if len(s) <= length else s[:length]

    def _clamp_to_schema_limits(self, raw: dict[str, Any]) -> dict[str, Any]:
        """PaperAnalysisSchema の制約に合わせて値を安全に短縮/整形する"""
        if not isinstance(raw, dict):
            return {}
        max_len = 1500
        keys = ["summary", "novelty", "methodology", "results", "future_work"]
        out: dict[str, Any] = dict(raw)
        for k in keys:
            if k in out and out[k] is not None:
                out[k] = self._truncate(str(out[k]), max_len)
        # research_themes: 最大3件、各要素を適度に短縮
        if isinstance(out.get("research_themes"), list):
            themes: list[Any] = out["research_themes"]
            clamped = [self._truncate(str(t), 300) for t in themes]
            out["research_themes"] = clamped[:3]
        return out

    def _normalize_keys(self, raw: dict[str, Any]) -> dict[str, Any]:
        """LLM出力のキーをスキーマ英語キーに正規化する。
        - 先頭の番号/記号プレフィックスを除去（例: "1_要約", "1. summary"）
        - 日本語キー→英語キーへマップ
        - 英語の揺れ（futurework→future_work, method→methodology 等）を吸収
        """
        if not isinstance(raw, dict):
            return {}
        import re

        def strip_prefix(k: str) -> str:
            k = k.strip()
            k = re.sub(r"^[\s\d０-９_＿\.-]+", "", k)
            return k.strip().lower()

        mapping = {
            # Japanese
            "要約": "summary",
            "新規性": "novelty",
            "手法": "methodology",
            "方法": "methodology",
            "結果": "results",
            "今後の課題": "future_work",
            "将来の課題": "future_work",
            "研究テーマ": "research_themes",
            # English variations
            "summary": "summary",
            "novelty": "novelty",
            "method": "methodology",
            "methods": "methodology",
            "methodology": "methodology",
            "result": "results",
            "results": "results",
            "future work": "future_work",
            "future_work": "future_work",
            "futurework": "future_work",
            "research theme": "research_themes",
            "research themes": "research_themes",
            "research_themes": "research_themes",
        }

        normalized: dict[str, Any] = {}
        for k, v in raw.items():
            key = strip_prefix(str(k))
            key = mapping.get(key, key)
            normalized[key] = v

        # research_themes を配列化
        if "research_themes" in normalized and not isinstance(normalized["research_themes"], list):
            val = normalized["research_themes"]
            if isinstance(val, str):
                # 区切りで分割（;、/、改行、・など）
                parts = re.split(r"[\n;,/・・]+", val)
                normalized["research_themes"] = [p.strip() for p in parts if p.strip()]
            else:
                normalized["research_themes"] = [str(val)]
        return normalized

    def _estimate_tokens(self, s: str) -> int:
        """非常に粗いトークン数推定 (約 1 token ~= 4 chars)。
        依存を増やさずに context-window 判定の分岐にだけ用いる。
        """
        try:
            return max(1, len(s) // 4)
        except Exception:
            return max(1, len(str(s)) // 4)

    def _fits_context(self, text: str, prompt: str) -> bool:
        """入力テキストが context-window に収まるかの概算判定。
        予備として出力用トークンと思考用バジェットを差し引く。
        """
        input_tokens = self._estimate_tokens(text) + self._estimate_tokens(prompt)
        reserve = settings.LLM_MAX_TOKENS + settings.LLM_THINKING_BUDGET
        budget = max(1024, settings.LLM_CONTEXT_WINDOW_TOKENS - reserve)
        return input_tokens <= budget

    async def analyze_paper(self, paper: Paper) -> Paper:
        """論文を解析する"""
        pdf_path = None
        try:
            # 1) ar5iv HTML（テキスト）経路（有効時）
            if settings.USE_AR5IV_HTML:
                try:
                    text = await extractTextByArxivId(paper.metadata.arxiv_id)
                    if text:
                        paper.full_text = text
                        prompt = self._build_prompt(paper.metadata.title, paper.metadata.abstract, input_modality="text")
                        analysis_schema: PaperAnalysisSchema = await self._generate_analysis_from_text(prompt, text)
                        # 保存
                        try:
                            log_dir = Path("logs/llm_raw_responses")
                            log_dir.mkdir(parents=True, exist_ok=True)
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            safe_arxiv_id = paper.metadata.arxiv_id.replace("/", "_")
                            log_filename = log_dir / f"{safe_arxiv_id}_{timestamp}.json"
                            with open(log_filename, "w", encoding="utf-8") as f:
                                f.write(analysis_schema.model_dump_json(indent=2))
                            log.info(
                                f"Saved structured LLM response (HTML path) for {paper.metadata.arxiv_id} to {log_filename}"
                            )
                        except Exception as log_save_error:
                            log.error(
                                f"Failed to save structured LLM response for {paper.metadata.arxiv_id}: {str(log_save_error)}"
                            )
                        paper.analysis = AnalysisResult(**analysis_schema.model_dump())
                        return paper
                except Exception as he:
                    # 認証エラーは PDF 経路へフォールバックせず即時伝播
                    if self._is_auth_error(he):
                        raise
                    log.warning(f"HTML ingestion path failed; falling back to PDF. Reason: {str(he)}")

            # PDFのダウンロード
            pdf_path = await self._download_pdf(paper.metadata.pdf_url)
            if not pdf_path:
                raise ValueError("Failed to download PDF")

            # ファイルサイズチェック
            pdf_file_size = Path(pdf_path).stat().st_size
            if pdf_file_size > self.MAX_PDF_SIZE_BYTES:
                # 50MB超でも、まずはアップロード経由の解析を試行し、失敗時はテキスト抽出+RAGにフォールバック
                log.info(
                    "PDFサイズが50MBを超えています。まずはアップロードによる解析を試み、失敗時はテキスト抽出+RAGにフォールバックします。"
                )

            # プロンプトの構築
            prompt = self._build_prompt(paper.metadata.title, paper.metadata.abstract, input_modality="pdf")

            # Provider による解析（構造化出力）
            analysis_schema: PaperAnalysisSchema = await self._generate_analysis(prompt, pdf_path)

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
        """PDFをダウンロードし、50MB超なら Ghostscript で圧縮したパスを返す。
        一時ファイルは /dev/shm (tmpfs) が存在すればそこに置き、プロセス終了後に削除されるため
        物理ディスクに残りません。"""
        try:
            # tmpfs があれば優先的に使用（RAM 上なので痕跡が残りにくい）
            tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()

            # URL を https に正規化
            url = url.replace("http://", "https://")

            # ---- 1) ストリームでダウンロード（共通ユーティリティ経由） ----
            tmp_in = tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".pdf", delete=False)
            tmp_in.close()
            await downloadToFile(url, tmp_in.name)

            file_size = Path(tmp_in.name).stat().st_size
            # 100MB 超の大容量PDFを検出（情報ログ）
            if file_size > 100 * 1024 * 1024:
                log.info(f"大容量 PDF (>=100MB) を検出: {file_size/1_048_576:.2f} MB")

            if file_size <= self.MAX_PDF_SIZE_BYTES:
                return tmp_in.name  # 圧縮不要

            # ---- 2) 50MB超の場合は Ghostscript で圧縮 ----
            if shutil.which("gs") is None:
                log.warning("Ghostscript(gs) が見つかりませんでした。圧縮をスキップします。")
                # analyze_paper 側のサイズ判定でフォールバック処理に委譲
                return tmp_in.name

            tmp_out = tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".pdf", delete=False)
            tmp_out.close()  # パスのみ利用

            cmd = [
                "gs",
                "-sDEVICE=pdfwrite",
                "-dDownsampleColorImages=true",
                "-dColorImageResolution=150",
                "-dNOPAUSE",
                "-dBATCH",
                f"-sOutputFile={tmp_out.name}",
                tmp_in.name,
            ]
            try:
                subprocess.run(cmd, check=True)
                # 元の大容量 PDF を削除して痕跡を残さない
                Path(tmp_in.name).unlink(missing_ok=True)
                return tmp_out.name
            except Exception as ce:
                log.warning(f"Ghostscript による圧縮に失敗しました。未圧縮で続行します: {str(ce)}")
                return tmp_in.name

        except Exception as e:
            log.error(f"Error downloading/compressing PDF from {url}: {str(e)}")
            return None

    def _build_prompt(self, title: str, abstract: str, input_modality: str = "pdf") -> str:
        """解析用のプロンプトを構築する。
        input_modality: "pdf" | "text" で先頭行の表現のみ変える。
        LLM には英語キーの厳密な JSON を要求する。
        """
        modality = "PDFファイル" if input_modality == "pdf" else "本文テキスト"
        return f"""以下の論文（{modality}）を読み、図表や本文を踏まえて総合的に分析してください。

論文タイトル: {title}

アブストラクト:
{abstract}

出力は厳密な JSON オブジェクトのみ、キーは英語で以下の6つに固定してください。
- summary: 要約（日本語）
- novelty: 新規性（日本語）
- methodology: 手法（日本語）
- results: 結果（日本語）
- future_work: 今後の課題（日本語）
- research_themes: 3件の研究テーマ（日本語の配列、ちょうど3要素）

注意:
- JSON 以外のテキストや番号付きの見出しは出力しない。
- キー名は必ず上記の英語名を使用し、日本語や番号プレフィックスは付けない。
"""

    async def _generate_analysis(self, prompt: str, pdf_path: str) -> PaperAnalysisSchema:
        """Provider 経由で解析（構造化 JSON 出力）。
        方針: 可能ならファイルアップロード→失敗時にテキスト抽出+チャンク推論（RAG風）にフォールバック。
        """
        try:
            log.debug(f"Sending request to provider. Prompt length: {len(prompt)}, PDF: {pdf_path}")
            used_text_pipeline = False
            raw: dict[str, Any] | str | Any
            if self.provider.supportsFileUpload():
                try:
                    raw = await self._call_with_retry(
                        self.provider.generateStructuredFromFile,
                        prompt,
                        pdf_path,
                        PaperAnalysisSchema,
                    )
                except Exception as fe:
                    if self._is_auth_error(fe):
                        # 認証エラーは即時打ち切り
                        raise
                    log.warning(
                        f"File-upload path failed; falling back to text pipeline. Reason: {str(fe)}"
                    )
                    raw = await self._analyze_via_text_pipeline(pdf_path, prompt)
                    used_text_pipeline = True
            else:
                raw = await self._analyze_via_text_pipeline(pdf_path, prompt)
                used_text_pipeline = True

            # フォールバック1: 非辞書/空応答なら JSON 抽出を試みる
            if not isinstance(raw, dict) or not raw:
                if isinstance(raw, str):
                    parsed = parseJsonFromText(raw)
                    if isinstance(parsed, dict) and parsed:
                        raw = parsed
                # 抽出しても不正なら、まだ未実施であればテキストパイプラインに切替
                if (not isinstance(raw, dict) or not raw) and not used_text_pipeline:
                    log.warning("Structured result invalid; retrying via text pipeline fallback.")
                    raw = await self._analyze_via_text_pipeline(pdf_path, prompt)
                    used_text_pipeline = True
                # 依然として不正ならエラー
                if not isinstance(raw, dict) or not raw:
                    raise ValueError("LLM returned empty/invalid structured result.")
            # キー正規化 → スキーマ制約に合わせて安全に短縮
            normalized = self._normalize_keys(raw)
            clamped = self._clamp_to_schema_limits(normalized)
            return PaperAnalysisSchema.model_validate(clamped)
        except Exception as e:
            log.error(f"Error generating structured content with provider: {str(e)}")
            raise

    async def _generate_analysis_from_text(self, prompt: str, text: str) -> PaperAnalysisSchema:
        """テキストを直接用いた構造化解析。単発→RAG/チャンクへフォールバック。"""
        try:
            raw: dict[str, Any] | str | Any
            used_text_pipeline = False
            fits = self._fits_context(text, prompt)
            # 単発で入るならまず単発を試行
            if fits:
                try:
                    raw = await self._call_with_retry(
                        self.provider.generateStructuredFromText,
                        prompt,
                        text,
                        PaperAnalysisSchema,
                    )
                except Exception as se:
                    if self._is_auth_error(se):
                        # 認証エラーはチャンク/RAGへフォールバックせずに打ち切り
                        raise
                    log.warning(
                        f"Single-shot text inference failed; falling back to chunks. Reason: {str(se)}"
                    )
                    raw = await self._analyze_via_text_pipeline_from_text(text, prompt)
                    used_text_pipeline = True
            else:
                raw = await self._analyze_via_text_pipeline_from_text(text, prompt)
                used_text_pipeline = True

            # フォールバック: 非辞書なら JSON 抽出
            if not isinstance(raw, dict) or not raw:
                if isinstance(raw, str):
                    parsed = parseJsonFromText(raw)
                    if isinstance(parsed, dict) and parsed:
                        raw = parsed
                # 単発の応答が不正だった場合でも、まだチャンク/RAGを使っていないならフォールバック
                if (not isinstance(raw, dict) or not raw) and not used_text_pipeline:
                    log.warning("Structured result invalid (single-shot); retrying via chunk/RAG text pipeline.")
                    raw = await self._analyze_via_text_pipeline_from_text(text, prompt)
                    used_text_pipeline = True
                if not isinstance(raw, dict) or not raw:
                    raise ValueError("LLM returned empty/invalid structured result (text path).")

            normalized = self._normalize_keys(raw)
            clamped = self._clamp_to_schema_limits(normalized)
            return PaperAnalysisSchema.model_validate(clamped)
        except Exception as e:
            log.error(f"Error generating structured content from text: {str(e)}")
            raise

    async def _analyze_via_text_pipeline(self, pdf_path: str, prompt: str) -> dict[str, Any]:
        """テキスト抽出→(単発 or チャンク)推論→統合の簡易RAGパイプライン。
        - 全文が context-window に収まる: 単発で generateStructuredFromText
        - 超過する: チャンクに分割して逐次推論し combine
        """
        text = await extract_pdf_text(pdf_path)
        # 単発で入るならそのまま
        if self._fits_context(text, prompt):
            try:
                d = await self._call_with_retry(
                    self.provider.generateStructuredFromText,
                    prompt,
                    text,
                    PaperAnalysisSchema,
                )
                return d if isinstance(d, dict) else parseJsonFromText(str(d))
            except Exception as se:
                if self._is_auth_error(se):
                    # 認証エラーはチャンクに進まず即時伝播
                    raise
                log.warning(
                    f"Single-shot text inference failed; falling back to chunks. Reason: {str(se)}"
                )
        # それ以外はチャンク
        chunks = splitText(text)
        # FAISS + Qwen Embedding によるベクトル検索で上位チャンクをまとめて単発推論（有効時）
        if settings.RAG_USE_FAISS and chunks:
            try:
                embedder = EmbeddingModel.from_pretrained(settings.RAG_EMBED_MODEL)
                # 可能なら永続化されたインデックスをロード
                store: FaissVectorStore | None = None
                index_dir = Path(settings.RAG_INDEX_DIR)
                if settings.RAG_INDEX_PERSIST:
                    # コンテンツベースのキー（モデル名 + 全文ハッシュ + チャンク数）
                    content_key = hashlib.sha1(
                        f"{settings.RAG_EMBED_MODEL}\x1f{len(chunks)}\x1f{hashlib.sha1(text.encode('utf-8')).hexdigest()}".encode(
                            "utf-8"
                        )
                    ).hexdigest()
                    base = index_dir / content_key
                    idx_path = base.with_suffix(".faiss")
                    payloads_path = base.with_suffix(".payloads.json")
                    try:
                        if idx_path.exists() and payloads_path.exists():
                            store = FaissVectorStore.load(idx_path, payloads_path)
                    except Exception as le:
                        log.warning(f"Failed to load FAISS index; rebuilding. Reason: {str(le)}")

                if store is None:
                    vectors = embedder.embed(chunks)  # shape (N, D), L2 正規化済み
                    dim = int(vectors.shape[1])
                    store = FaissVectorStore(dim)
                    store.add(vectors, chunks)
                    # セーブ
                    if settings.RAG_INDEX_PERSIST:
                        try:
                            base.parent.mkdir(parents=True, exist_ok=True)
                            store.save(idx_path, payloads_path)
                        except Exception as se:
                            log.warning(f"Failed to save FAISS index: {str(se)}")
                # クエリはプロンプト全体（タイトル/要約を含む）をそのまま利用
                qvec = embedder.embed([prompt])[0]
                # トップKで収まらなければ段階的に縮小 + 類似度しきい値でフィルタ + 貪欲パッキング
                initial_top_k = max(1, settings.RAG_TOP_K)
                top_k = initial_top_k
                selected_text = ""
                results_cache = None
                while top_k >= 1:
                    results = store.search(qvec, top_k=top_k)
                    results_cache = results
                    # 類似度しきい値でフィルタ（cosine 類似度を想定）
                    filtered_pairs = [(p, s) for (p, s, _i) in results if s >= float(settings.RAG_MIN_SIMILARITY)]
                    if not filtered_pairs:
                        # しきい値を満たすものがない場合は top_k を縮小
                        next_top_k = max(1, top_k // 2)
                        if next_top_k == top_k:
                            # これ以上縮小できない -> ループ離脱しフォールバックへ
                            break
                        top_k = next_top_k
                        continue
                    # スコア降順は search の結果が既に保証（FAISS は降順返却）と仮定。念のため再ソート可。
                    # 貪欲パッキング: 上位から順に追加し、収まらなくなったら打ち切り
                    buf_pairs: list[tuple[str, float]] = []
                    buf_texts: list[str] = []
                    for chunk, score in filtered_pairs:
                        candidate = "\n\n".join(buf_texts + [chunk])
                        if self._fits_context(candidate, prompt):
                            buf_pairs.append((chunk, score))
                            buf_texts.append(chunk)
                        else:
                            break
                    if buf_pairs:
                        selected_text = "\n\n".join(buf_texts)
                        # メトリクスのログ出力
                        try:
                            selected_count = len(buf_pairs)
                            selected_chars = sum(len(t) for t, _ in buf_pairs)
                            avg_score = sum(s for _, s in buf_pairs) / max(1, selected_count)
                            log.info(
                                "RAG retrieval: init_top_k=%s, used_top_k=%s, threshold=%.3f, filtered=%s, selected_count=%s, selected_chars=%s, avg_score=%.4f",
                                initial_top_k,
                                top_k,
                                float(settings.RAG_MIN_SIMILARITY),
                                len(filtered_pairs),
                                selected_count,
                                selected_chars,
                                avg_score,
                            )
                        except Exception:
                            pass
                        break
                    # 何も詰められない場合は top_k を縮小
                    next_top_k = max(1, top_k // 2)
                    if next_top_k == top_k:
                        # これ以上縮小できない -> ループ離脱しフォールバックへ
                        break
                    top_k = next_top_k
                if not selected_text and results_cache:
                    # それでも収まらない/フィルタ通過なしの場合は最上位1件のみ
                    selected_text = results_cache[0][0]
                    try:
                        log.info(
                            "RAG retrieval: fallback_top1 used (init_top_k=%s, threshold=%.3f, top1_score=%.4f, top1_len=%s)",
                            initial_top_k,
                            float(settings.RAG_MIN_SIMILARITY),
                            float(results_cache[0][1]),
                            len(selected_text),
                        )
                    except Exception:
                        pass
                if selected_text:
                    d = await self._call_with_retry(
                        self.provider.generateStructuredFromText,
                        prompt,
                        selected_text,
                        PaperAnalysisSchema,
                    )
                    if d:
                        return d if isinstance(d, dict) else parseJsonFromText(str(d))
            except Exception as re:
                if self._is_auth_error(re):
                    raise
                log.warning(
                    f"Vector retrieval path failed; falling back to sequential chunking. Reason: {str(re)}"
                )
        chunks = splitText(text)
        partials: list[dict[str, Any]] = []
        for c in chunks:
            try:
                d = await self._call_with_retry(
                    self.provider.generateStructuredFromText,
                    prompt,
                    c,
                    PaperAnalysisSchema,
                )
                if d:
                    partials.append(d)
            except Exception as ce:
                if self._is_auth_error(ce):
                    raise
                log.warning(f"Chunk inference failed: {str(ce)}")
        if not partials:
            return {}
        return combine(partials, {}) if len(partials) > 1 else partials[0]

    async def _analyze_via_text_pipeline_from_text(self, text: str, prompt: str) -> dict[str, Any]:
        """テキストを受け取り、(単発 or チャンク/RAG) 推論を行う。"""
        # 単発で入るならそのまま
        if self._fits_context(text, prompt):
            try:
                d = await self._call_with_retry(
                    self.provider.generateStructuredFromText,
                    prompt,
                    text,
                    PaperAnalysisSchema,
                )
                return d if isinstance(d, dict) else parseJsonFromText(str(d))
            except Exception as se:
                if self._is_auth_error(se):
                    # 認証エラーはチャンクに進まず即時伝播
                    raise
                log.warning(
                    f"Single-shot text inference failed; falling back to chunks. Reason: {str(se)}"
                )
        # チャンク
        chunks = splitText(text)
        if settings.RAG_USE_FAISS and chunks:
            try:
                embedder = EmbeddingModel.from_pretrained(settings.RAG_EMBED_MODEL)
                store: FaissVectorStore | None = None
                index_dir = Path(settings.RAG_INDEX_DIR)
                if settings.RAG_INDEX_PERSIST:
                    content_key = hashlib.sha1(
                        f"{settings.RAG_EMBED_MODEL}\x1f{len(chunks)}\x1f{hashlib.sha1(text.encode('utf-8')).hexdigest()}".encode(
                            "utf-8"
                        )
                    ).hexdigest()
                    base = index_dir / content_key
                    idx_path = base.with_suffix(".faiss")
                    payloads_path = base.with_suffix(".payloads.json")
                    try:
                        if idx_path.exists() and payloads_path.exists():
                            store = FaissVectorStore.load(idx_path, payloads_path)
                    except Exception as le:
                        log.warning(f"Failed to load FAISS index; rebuilding. Reason: {str(le)}")

                if store is None:
                    vectors = embedder.embed(chunks)
                    dim = int(vectors.shape[1])
                    store = FaissVectorStore(dim)
                    store.add(vectors, chunks)
                    if settings.RAG_INDEX_PERSIST:
                        try:
                            base.parent.mkdir(parents=True, exist_ok=True)
                            store.save(idx_path, payloads_path)
                        except Exception as se:
                            log.warning(f"Failed to save FAISS index: {str(se)}")
                qvec = embedder.embed([prompt])[0]
                initial_top_k = max(1, settings.RAG_TOP_K)
                top_k = initial_top_k
                selected_text = ""
                results_cache = None
                while top_k >= 1:
                    results = store.search(qvec, top_k=top_k)
                    results_cache = results
                    filtered_pairs = [(p, s) for (p, s, _i) in results if s >= float(settings.RAG_MIN_SIMILARITY)]
                    if not filtered_pairs:
                        next_top_k = max(1, top_k // 2)
                        if next_top_k == top_k:
                            break
                        top_k = next_top_k
                        continue
                    buf_pairs: list[tuple[str, float]] = []
                    buf_texts: list[str] = []
                    for chunk, score in filtered_pairs:
                        candidate = "\n\n".join(buf_texts + [chunk])
                        if self._fits_context(candidate, prompt):
                            buf_pairs.append((chunk, score))
                            buf_texts.append(chunk)
                        else:
                            break
                    if buf_pairs:
                        selected_text = "\n\n".join(buf_texts)
                        break
                    next_top_k = max(1, top_k // 2)
                    if next_top_k == top_k:
                        break
                    top_k = next_top_k
                if not selected_text and results_cache:
                    selected_text = results_cache[0][0]
                if selected_text:
                    d = await self._call_with_retry(
                        self.provider.generateStructuredFromText,
                        prompt,
                        selected_text,
                        PaperAnalysisSchema,
                    )
                    if d:
                        return d if isinstance(d, dict) else parseJsonFromText(str(d))
            except Exception as re:
                log.warning(
                    f"Vector retrieval path failed; falling back to sequential chunking. Reason: {str(re)}"
                )
        chunks = splitText(text)
        partials: list[dict[str, Any]] = []
        for c in chunks:
            try:
                d = await self._call_with_retry(
                    self.provider.generateStructuredFromText,
                    prompt,
                    c,
                    PaperAnalysisSchema,
                )
                if d:
                    partials.append(d)
            except Exception as ce:
                log.warning(f"Chunk inference failed: {str(ce)}")
        if not partials:
            return {}
        return combine(partials, {}) if len(partials) > 1 else partials[0]

    async def _translate_text(self, text: str) -> str:
        """LLMで英文を日本語訳する"""
        try:
            return await self.provider.translate(text)
        except Exception as e:
            log.error(f"Error translating text: {str(e)}")
            return "翻訳エラー発生"
