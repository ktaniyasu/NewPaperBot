from pathlib import Path

import numpy as np
import pytest

from src.rag.vector_store import FaissVectorStore

# Skip entire module if faiss is not available in the environment
pytest.importorskip("faiss", reason="faiss-cpu not installed")


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / norms).astype(np.float32)


def test_faiss_save_load_roundtrip(tmp_path: Path):
    # create synthetic normalized vectors
    rng = np.random.default_rng(123)
    dim = 16
    n = 10
    vecs = _normalize_rows(rng.normal(size=(n, dim)))
    payloads = [f"chunk-{i}" for i in range(n)]

    store = FaissVectorStore(dim=dim)
    store.add(vecs, payloads)

    # search a query before saving
    q = _normalize_rows(rng.normal(size=(1, dim)))
    before = store.search(q, top_k=5)

    # persist
    idx_path = tmp_path / "test_index.faiss"
    pl_path = tmp_path / "test_payloads.json"
    store.save(idx_path, pl_path)

    # load back
    loaded = FaissVectorStore.load(idx_path, pl_path)

    # search again and compare payload order (scores may differ by small fp error, but rank should match)
    after = loaded.search(q, top_k=5)

    assert [p for p, _s, _i in before] == [p for p, _s, _i in after]
    assert len(after) == 5


def test_faiss_rejects_mismatched_lengths(tmp_path: Path):
    dim = 8
    store = FaissVectorStore(dim=dim)
    vecs = _normalize_rows(np.random.normal(size=(3, dim)))
    try:
        store.add(vecs, ["a", "b"])  # payload count != vector rows
        assert False, "Expected ValueError for length mismatch"
    except ValueError:
        pass
