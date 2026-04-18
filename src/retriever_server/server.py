import asyncio
import json
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import faiss
import hydra
import numpy as np
from fastapi import FastAPI, HTTPException
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

_RETRIEVER_CFG_ENV = "RETRIEVER_SERVER_CFG_JSON"


class SearchRequest(BaseModel):
    queries: list[str]
    topk: int = 3
    return_scores: bool = True
    dataset: str | None = None


@dataclass
class DocItem:
    doc_id: str
    title: str
    text: str
    contents: str


class E5Retriever:
    def __init__(
        self,
        corpus_path: str,
        index_path: str,
        model_name: str,
        max_length: int = 256,
        ef_search: int | None = None,
        device: str = "cpu",
        encode_batch_size: int = 64,
    ):
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"corpus not found: {corpus_path}")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"index not found: {index_path}")

        self.model_name = model_name
        self.device = device
        self.encode_batch_size = int(encode_batch_size)
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_length

        self.docs = self._load_corpus(corpus_path)
        self.index = faiss.read_index(index_path)
        self._configure_index(ef_search=ef_search)
        if self.index.ntotal != len(self.docs):
            raise ValueError(
                f"index/corpus size mismatch: index={self.index.ntotal}, corpus={len(self.docs)}"
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

    @staticmethod
    def _load_corpus(corpus_path: str) -> list[DocItem]:
        docs: list[DocItem] = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                doc_id = str(row.get("id", row.get("doc_id", i)))
                contents = str(row.get("contents", "")).strip()
                title = str(row.get("title", "")).strip()
                text = str(row.get("text", "")).strip()
                if not contents:
                    contents = f"{title}\n{text}".strip() if title else text
                if not text:
                    if "\n" in contents:
                        _, text = contents.split("\n", 1)
                    else:
                        text = contents
                if not title and "\n" in contents:
                    title = contents.split("\n", 1)[0]
                docs.append(DocItem(doc_id=doc_id, title=title, text=text, contents=contents))
        return docs

    def _encode(self, queries: list[str]) -> np.ndarray:
        if "e5" in self.model_name.lower():
            queries = [f"query: {q}" for q in queries]
        emb = self.model.encode(
            queries,
            normalize_embeddings=True,
            batch_size=self.encode_batch_size,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype="float32")

    def batch_search(self, queries: list[str], topk: int, return_scores: bool) -> list[list[dict[str, Any]]]:
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
                doc = self.docs[int(doc_idx)]
                doc_payload = {
                    "id": doc.doc_id,
                    "title": doc.title,
                    "text": doc.text,
                    "contents": doc.contents,
                }
                if return_scores:
                    results.append({"document": doc_payload, "score": float(scores[i][j])})
                else:
                    results.append(doc_payload)
            output.append(results)
        return output


class QueryLRUCache:
    def __init__(self, max_size: int):
        self.max_size = max(1, int(max_size))
        self._lock = threading.Lock()
        self._data: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def get(self, key: str) -> list[dict[str, Any]] | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: str, value: list[dict[str, Any]]) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)


def _parse_dataset_names(raw: Any) -> set[str]:
    if isinstance(raw, str):
        return {x.strip() for x in raw.split(",") if x.strip()}
    if isinstance(raw, (list, tuple)):
        return {str(x).strip() for x in raw if str(x).strip()}
    return set()


def build_app_from_cfg(cfg: dict[str, Any]) -> FastAPI:
    retriever = E5Retriever(
        corpus_path=str(cfg["corpus_path"]),
        index_path=str(cfg["index_path"]),
        model_name=str(cfg["model"]),
        max_length=int(cfg["max_length"]),
        ef_search=None if cfg.get("ef_search") is None else int(cfg["ef_search"]),
        device=str(cfg.get("device", "cpu")),
        encode_batch_size=int(cfg.get("encode_batch_size", 64)),
    )
    allowed_datasets = _parse_dataset_names(cfg["datasets"])
    cache_enabled = bool(cfg.get("cache_enabled", True))
    cache_size = int(cfg.get("cache_size", 5000))
    cache = QueryLRUCache(max_size=cache_size) if cache_enabled else None
    executor = ThreadPoolExecutor(max_workers=4)
    app = FastAPI()

    def _sync_retrieve(req: SearchRequest) -> list[list[dict[str, Any]]]:
        if cache is None:
            return retriever.batch_search(req.queries, req.topk, req.return_scores)

        dataset_key = req.dataset or "default"
        result_buffer: list[list[dict[str, Any]] | None] = [None] * len(req.queries)
        miss_queries: list[str] = []
        miss_indices: list[int] = []
        miss_keys: list[str] = []

        for i, q in enumerate(req.queries):
            key = f"{dataset_key}|{req.topk}|{int(req.return_scores)}|{q}"
            hit = cache.get(key)
            if hit is None:
                miss_queries.append(q)
                miss_indices.append(i)
                miss_keys.append(key)
            else:
                result_buffer[i] = hit

        if miss_queries:
            miss_results = retriever.batch_search(miss_queries, req.topk, req.return_scores)
            for idx, key, value in zip(miss_indices, miss_keys, miss_results):
                result_buffer[idx] = value
                cache.put(key, value)

        if any(x is None for x in result_buffer):
            raise RuntimeError("internal cache error: missing query results")
        return [x for x in result_buffer if x is not None]

    @app.post("/retrieve")
    async def retrieve(req: SearchRequest):
        if req.dataset is not None and req.dataset not in allowed_datasets:
            raise HTTPException(status_code=404, detail=f"unknown dataset: {req.dataset}")
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, _sync_retrieve, req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"result": result}

    return app


def app_factory() -> FastAPI:
    raw_cfg = os.environ.get(_RETRIEVER_CFG_ENV)
    if not raw_cfg:
        raise RuntimeError(f"missing env: {_RETRIEVER_CFG_ENV}")
    return build_app_from_cfg(json.loads(raw_cfg))


@hydra.main(config_path="conf", config_name="server_e5", version_base=None)
def main(cfg: DictConfig) -> None:
    import uvicorn

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError("invalid retriever config")

    os.environ[_RETRIEVER_CFG_ENV] = json.dumps(cfg_dict)
    uvicorn.run(
        "retriever_server.server:app_factory",
        host=str(cfg.host),
        port=int(cfg.port),
        workers=int(cfg.workers),
        factory=True,
    )


if __name__ == "__main__":
    main()
