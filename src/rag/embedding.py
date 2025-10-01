from __future__ import annotations

from dataclasses import dataclass
from typing import List
import hashlib
from pathlib import Path

from ..utils.config import settings


@dataclass
class EmbeddingModel:
    """軽量な埋め込みラッパー。
    Transformers を遅延インポートし、初回利用時にのみロードします。
    Qwen3-Embedding-0.6B などのテキスト埋め込みモデルを想定。
    """

    modelName: str

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None

    @classmethod
    def from_pretrained(cls, model_name: str) -> "EmbeddingModel":
        return cls(modelName=model_name)

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency at runtime
            raise RuntimeError(
                "Transformers/Torch が見つかりません。requirements.txt をインストールしてください。"
            ) from e
        # tokenizer / model のロード
        self._tokenizer = AutoTokenizer.from_pretrained(self.modelName, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(self.modelName, trust_remote_code=True)
        self._model.eval()
        # 推論デバイス（CUDA があれば CUDA）
        self._device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._torch = torch
        # キャッシュディレクトリの用意
        if settings.RAG_EMBED_CACHE:
            Path(settings.RAG_EMBED_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text: str) -> Path:
        base = Path(settings.RAG_EMBED_CACHE_DIR)
        h = hashlib.sha1(f"{self.modelName}\x1f{text}".encode("utf-8")).hexdigest()
        return base / f"{h}.npy"

    def embed(self, texts: List[str]) -> "numpy.ndarray":
        """テキスト列を埋め込みベクトルに変換（mean pooling）。
        返り値は shape=(N, D) の numpy.ndarray
        ディスクキャッシュが有効な場合、テキスト単位で .npy を読み書きします。
        """
        self._ensure_loaded()
        import numpy as np  # type: ignore

        cache_on = bool(settings.RAG_EMBED_CACHE)
        cached: dict[int, "numpy.ndarray"] = {}
        missing_idx: list[int] = []
        missing_texts: list[str] = []

        if cache_on:
            for i, t in enumerate(texts):
                p = self._cache_path(t)
                if p.exists():
                    try:
                        v = np.load(p)
                        cached[i] = v
                    except Exception:
                        # 壊れている場合は再計算
                        missing_idx.append(i)
                        missing_texts.append(t)
                else:
                    missing_idx.append(i)
                    missing_texts.append(t)
        else:
            missing_idx = list(range(len(texts)))
            missing_texts = list(texts)

        computed: "numpy.ndarray | None" = None
        if missing_texts:
            with self._torch.no_grad():
                toks = self._tokenizer(
                    missing_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=8192,
                )
                toks = {k: v.to(self._device) for k, v in toks.items()}
                out = self._model(**toks)
                last_hidden = out.last_hidden_state  # (B, L, H)
                mask = toks.get("attention_mask")  # (B, L)
                # mean pooling
                mask_exp = mask.unsqueeze(-1).type_as(last_hidden)
                summed = (last_hidden * mask_exp).sum(dim=1)
                counts = mask_exp.sum(dim=1).clamp(min=1e-6)
                embeddings = summed / counts
                emb = embeddings.detach().cpu().numpy()
                # cosine 用に正規化
                norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
                emb = emb / norms
                computed = emb
            if cache_on:
                for j, idx in enumerate(missing_idx):
                    try:
                        np.save(self._cache_path(texts[idx]), computed[j])
                    except Exception:
                        pass

        # 出力をオリジナル順に再構成
        result_list: list["numpy.ndarray"] = []
        for i in range(len(texts)):
            if i in cached:
                result_list.append(cached[i])
            else:
                # computed は None でない前提（missing があるなら）
                result_list.append(computed[missing_idx.index(i)])  # type: ignore[index]
        return np.vstack(result_list) if result_list else np.zeros((0, 0), dtype=np.float32)
