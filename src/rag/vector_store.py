from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
from pathlib import Path
import json


@dataclass
class FaissVectorStore:
    """シンプルな FAISS ベクトルストア（インメモリ）。
    Cosine 類似度（内積、事前に L2 正規化済みを前提）で検索します。
    """

    dim: int

    def __post_init__(self) -> None:
        try:
            import faiss  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency at runtime
            raise RuntimeError(
                "faiss-cpu が見つかりません。requirements.txt をインストールしてください。"
            ) from e
        import numpy as np  # type: ignore

        self._faiss = faiss
        self._np = np
        self._index = faiss.IndexFlatIP(self.dim)
        self._payloads: List[str] = []

    def add(self, vectors: "numpy.ndarray", payloads: Sequence[str]) -> None:
        if len(payloads) != vectors.shape[0]:
            raise ValueError("vectors と payloads の件数が一致していません")
        # float32 要求
        if vectors.dtype != self._np.float32:
            vectors = vectors.astype(self._np.float32)
        self._index.add(vectors)
        self._payloads.extend(payloads)

    def search(self, query: "numpy.ndarray", top_k: int = 8) -> List[Tuple[str, float, int]]:
        if query.ndim == 1:
            query = query[None, :]
        if query.dtype != self._np.float32:
            query = query.astype(self._np.float32)
        scores, idxs = self._index.search(query, top_k)  # (1, k)
        results: List[Tuple[str, float, int]] = []
        for i, s in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            results.append((self._payloads[i], float(s), int(i)))
        return results

    def save(self, index_path: str | Path, payloads_path: str | Path) -> None:
        """FAISS インデックスとペイロードを保存する"""
        p_index = Path(index_path)
        p_payloads = Path(payloads_path)
        p_index.parent.mkdir(parents=True, exist_ok=True)
        p_payloads.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(p_index))
        with open(p_payloads, "w", encoding="utf-8") as f:
            json.dump(self._payloads, f, ensure_ascii=False)

    @classmethod
    def load(cls, index_path: str | Path, payloads_path: str | Path) -> "FaissVectorStore":
        """保存済みインデックスとペイロードを読み込む"""
        from pathlib import Path as _P
        p_index = _P(index_path)
        p_payloads = _P(payloads_path)
        if not p_index.exists() or not p_payloads.exists():
            raise FileNotFoundError("FAISS index or payloads file not found")
        # 一旦インスタンス化して faiss/numpy を初期化
        tmp = cls(dim=1)
        idx = tmp._faiss.read_index(str(p_index))
        # index の次元を採用
        dim = idx.d
        store = cls(dim=dim)
        store._index = idx
        with open(p_payloads, "r", encoding="utf-8") as f:
            payloads = json.load(f)
        if not isinstance(payloads, list):
            raise ValueError("Invalid payloads file: expected list")
        store._payloads = [str(x) for x in payloads]
        return store
