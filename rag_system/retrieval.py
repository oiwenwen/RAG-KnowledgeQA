from __future__ import annotations

import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """A dependency-free tokenizer that handles Chinese terms and English words."""
    terms: list[str] = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            terms.extend(part)
            terms.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            terms.append(part)
    return terms


class RetrievalEngine:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.by_id = {int(item["id"]): item for item in chunks}
        self.matrix: np.ndarray | None = None
        self.index = None
        if chunks:
            vectors = [np.frombuffer(item["vector"], dtype=np.float32) for item in chunks]
            dimension = len(vectors[0])
            if any(len(vector) != dimension for vector in vectors):
                raise ValueError("知识库向量维度不一致，请重新构建知识库")
            matrix = np.vstack(vectors).astype(np.float32)
            faiss.normalize_L2(matrix)
            self.matrix = matrix
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(matrix)
            corpus = [tokenize(item["content"]) for item in chunks]
            self.bm25 = BM25Okapi(corpus)
        else:
            self.bm25 = None

    def retrieve(self, query: str, query_vector: list[float], *, mode: str = "hybrid",
                 rerank: bool = True, top_k: int = 5, threshold: float = 0.35) -> list[dict]:
        if not self.chunks:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        candidates: dict[int, dict] = {}

        if mode in {"vector", "hybrid"}:
            scores, indexes = self.index.search(q, min(max(top_k * 3, top_k), len(self.chunks)))
            for rank, (score, idx) in enumerate(zip(scores[0], indexes[0]), start=1):
                if idx < 0:
                    continue
                item = self.chunks[int(idx)]
                candidates[int(item["id"])] = {**item, "vector_score": float(score), "vector_rank": rank}

        if mode in {"bm25", "hybrid"} and self.bm25:
            scores = self.bm25.get_scores(tokenize(query))
            order = np.argsort(scores)[::-1][:min(max(top_k * 3, top_k), len(scores))]
            for rank, idx in enumerate(order, start=1):
                item = self.chunks[int(idx)]
                chunk_id = int(item["id"])
                record = candidates.setdefault(chunk_id, {**item, "vector_score": None, "vector_rank": None})
                record["bm25_score"] = float(scores[idx])
                record["bm25_rank"] = rank

        if mode == "hybrid":
            for item in candidates.values():
                item["fusion_score"] = sum(
                    1.0 / (60 + item[key]) for key in ("vector_rank", "bm25_rank") if item.get(key)
                )

        ranked = list(candidates.values())
        if mode == "bm25" and not rerank:
            ranked.sort(key=lambda x: x.get("bm25_score", 0.0), reverse=True)
        elif mode == "vector" and not rerank:
            ranked.sort(key=lambda x: x.get("vector_score", -1.0), reverse=True)
        else:
            ranked.sort(key=lambda x: x.get("fusion_score", x.get("vector_score", -1.0)), reverse=True)

        if rerank:
            q_vec = q[0]
            for item in ranked:
                vector = np.frombuffer(item["vector"], dtype=np.float32)
                denom = float(np.linalg.norm(q_vec) * np.linalg.norm(vector))
                item["rerank_score"] = float(np.dot(q_vec, vector) / denom) if denom else 0.0
            ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            ranked = [item for item in ranked if item["rerank_score"] >= threshold]
        return ranked[:top_k]

