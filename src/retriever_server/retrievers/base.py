import json
import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class DocItem:
    doc_id: str
    title: str
    text: str
    contents: str


class Corpus:
    def __init__(self, corpus_path: str):
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"corpus not found: {corpus_path}")
        self.docs: list[DocItem] = []
        self.by_id: dict[str, DocItem] = {}
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
                doc = DocItem(doc_id=doc_id, title=title, text=text, contents=contents)
                self.docs.append(doc)
                self.by_id[doc_id] = doc

    def __len__(self) -> int:
        return len(self.docs)


class BaseRetriever(Protocol):
    def batch_search(
        self, queries: list[str], topk: int, return_scores: bool
    ) -> list[list[dict[str, Any]]]: ...


def doc_to_payload(doc: DocItem) -> dict[str, Any]:
    return {
        "id": doc.doc_id,
        "title": doc.title,
        "text": doc.text,
        "contents": doc.contents,
    }
