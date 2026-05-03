import os
from typing import Any

from pyserini.search.lucene import LuceneSearcher

from .base import Corpus, doc_to_payload


class BM25Retriever:
    def __init__(
        self,
        corpus_path: str,
        index_path: str,
        k1: float = 0.9,
        b: float = 0.4,
        threads: int = 8,
    ):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"index not found: {index_path}")
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(float(k1), float(b))
        self.threads = max(1, int(threads))
        self.corpus = Corpus(corpus_path)

    def batch_search(
        self, queries: list[str], topk: int, return_scores: bool
    ) -> list[list[dict[str, Any]]]:
        if topk <= 0:
            raise ValueError("topk must be > 0")
        qids = [str(i) for i in range(len(queries))]
        results_map = self.searcher.batch_search(
            queries=list(queries), qids=qids, k=topk, threads=self.threads
        )

        output: list[list[dict[str, Any]]] = []
        for qid in qids:
            hits = results_map.get(qid, [])
            row: list[dict[str, Any]] = []
            for hit in hits:
                docid = str(hit.docid)
                doc = self.corpus.by_id.get(docid)
                if doc is None:
                    raise KeyError(f"docid not found in corpus: {docid}")
                payload = doc_to_payload(doc)
                if return_scores:
                    row.append({"document": payload, "score": float(hit.score)})
                else:
                    row.append(payload)
            output.append(row)
        return output
