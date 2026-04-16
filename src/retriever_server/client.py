from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class RetrievalClient:
    url: str
    topk: int
    dataset_name: str
    timeout_sec: int = 30

    def batch_search(self, queries: list[str]) -> list[list[dict[str, Any]]]:
        payload = {
            "queries": queries,
            "topk": self.topk,
            "return_scores": True,
            "dataset": self.dataset_name,
        }
        response = requests.post(self.url, json=payload, timeout=self.timeout_sec)
        response.raise_for_status()
        body = response.json()
        if "result" not in body:
            raise KeyError("retriever response missing `result`")
        return body["result"]

    def search(self, query: str) -> list[dict[str, Any]]:
        return self.batch_search([query])[0]
