import os
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import Corpus, doc_to_payload


class E5Retriever:
    def __init__(
        self,
        corpus_path: str,
        index_path: str,
        model_name: str,
        max_length: int = 256,
        ef_search: int | None = None,
    ):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"index not found: {index_path}")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.model.max_seq_length = max_length

        self.corpus = Corpus(corpus_path)
        self.index = faiss.read_index(index_path)
        self._configure_index(ef_search=ef_search)
        if self.index.ntotal != len(self.corpus):
            raise ValueError(
                f"index/corpus size mismatch: index={self.index.ntotal}, corpus={len(self.corpus)}"
            )

    def _configure_index(self, ef_search: int | None) -> None:
        if ef_search is None:
            return
        ef = int(ef_search)
        if ef <= 0:
            raise ValueError("ef_search must be > 0")
        if not hasattr(self.index, "hnsw"):
            raise ValueError("ef_search is only valid for HNSW indexes")
        self.index.hnsw.efSearch = ef

    def _encode(self, queries: list[str]) -> np.ndarray:
        if "e5" in self.model_name.lower():
            queries = [f"query: {q}" for q in queries]
        emb = self.model.encode(queries, normalize_embeddings=True)
        return np.asarray(emb, dtype="float32")

    def batch_search(
        self, queries: list[str], topk: int, return_scores: bool
    ) -> list[list[dict[str, Any]]]:
        if topk <= 0:
            raise ValueError("topk must be > 0")
        q_emb = self._encode(queries)
        scores, idxs = self.index.search(q_emb, topk)

        output: list[list[dict[str, Any]]] = []
        for i, row in enumerate(idxs):
            results: list[dict[str, Any]] = []
            for j, doc_idx in enumerate(row):
                if doc_idx < 0:
                    continue
                doc = self.corpus.docs[int(doc_idx)]
                payload = doc_to_payload(doc)
                if return_scores:
                    results.append({"document": payload, "score": float(scores[i][j])})
                else:
                    results.append(payload)
            output.append(results)
        return output
