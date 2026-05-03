import asyncio
import json
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import hydra
from fastapi import FastAPI, HTTPException
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from retriever_server.retrievers.base import BaseRetriever

_RETRIEVER_CFG_ENV = "RETRIEVER_SERVER_CFG_JSON"


class SearchRequest(BaseModel):
    queries: list[str]
    topk: int = 3
    return_scores: bool = True
    dataset: str | None = None


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


def _build_retriever(cfg: dict[str, Any]) -> BaseRetriever:
    name = str(cfg.get("retriever", "e5")).lower()
    if name == "e5":
        from retriever_server.retrievers.e5 import E5Retriever

        return E5Retriever(
            corpus_path=str(cfg["corpus_path"]),
            index_path=str(cfg["index_path"]),
            model_name=str(cfg["model"]),
            max_length=int(cfg["max_length"]),
            ef_search=None if cfg.get("ef_search") is None else int(cfg["ef_search"]),
            device=str(cfg.get("device", "cpu")),
            encode_batch_size=int(cfg.get("encode_batch_size", 64)),
        )
    if name == "bm25":
        from retriever_server.retrievers.bm25 import BM25Retriever

        return BM25Retriever(
            corpus_path=str(cfg["corpus_path"]),
            index_path=str(cfg["index_path"]),
            k1=float(cfg.get("k1", 0.9)),
            b=float(cfg.get("b", 0.4)),
            threads=int(cfg.get("threads", 8)),
        )
    raise ValueError(f"unknown retriever: {name}")


def build_app_from_cfg(cfg: dict[str, Any]) -> FastAPI:
    retriever = _build_retriever(cfg)
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


@hydra.main(config_path="conf", config_name="server_e5_cpu", version_base=None)
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
