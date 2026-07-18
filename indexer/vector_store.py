import json
import pickle
from pathlib import Path
from typing import List
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: List[np.ndarray] = []
        self.metadata: List[dict] = []
        self._index = None

    def add(self, vector: np.ndarray, meta: dict):
        self.vectors.append(vector.astype(np.float32))
        self.metadata.append(meta)

    def build(self):
        mat = np.stack(self.vectors).astype(np.float32) if self.vectors else np.zeros((0, self.dim), dtype=np.float32)
        faiss.normalize_L2(mat)

        index = faiss.IndexFlatIP(self.dim)
        index.add(mat)
        self._index = index
        return self

    def search(self, query_vec: np.ndarray, top_k: int = 10):
    # Cosine-similarity search. Returns [(metadata, score), ...] sorted by score desc.
        q = query_vec.astype(np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(q)
        scores, idxs = self._index.search(q, min(top_k, len(self.metadata)))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def filter_search(self, query_vec: np.ndarray, top_k: int, predicate) -> list:
        # Search then filter by a metadata predicate (e.g. category == 'shirt')
        raw = self.search(query_vec, top_k=min(len(self.metadata), top_k * 20 + 20))
        return [(m, s) for m, s in raw if predicate(m)][:top_k]

    def save(self, path: Path):
        import faiss
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        with open(path / "meta.json", "w") as f:
            json.dump({"dim": self.dim, "count": len(self.metadata)}, f)

    @classmethod
    def load(cls, path: Path):
        import faiss
        path = Path(path)
        with open(path / "meta.json") as f:
            info = json.load(f)
        store = cls(dim=info["dim"])
        store._index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "metadata.pkl", "rb") as f:
            store.metadata = pickle.load(f)
        return store