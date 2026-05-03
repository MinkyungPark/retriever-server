# Retriever Server

Three retrievers over the wiki-18 corpus, each on its own port:

| Retriever         | Index                         | Port | Compose service     | Dockerfile     |
| ----------------- | ----------------------------- | ---- | ------------------- | -------------- |
| E5 (CPU, HNSW64)  | `e5_HNSW64.index` (FAISS)     | 3000 | `retriever-e5-cpu`  | `Dockerfile.cpu` |
| BM25 (Pyserini)   | `bm25/` (Lucene)              | 3001 | `retriever-bm25`    | `Dockerfile.cpu` |
| E5 (GPU, HNSW64)  | `e5_HNSW64.index` (FAISS)     | 3002 | `retriever-e5-gpu`  | `Dockerfile.gpu` |

`Dockerfile.cpu` ships OpenJDK 21 (required by Pyserini) and is shared by E5-CPU and BM25.
`Dockerfile.gpu` is CUDA 12.2 + cu121 torch and is used only by E5-GPU.

---

## 0. Setup

### 0.1 Environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```
HF_TOKEN=hf_...
CACHE_BASE=/data2/yourname
```

### 0.2 Download data

```bash
# Corpus + E5 HNSW64 index (shared by E5-CPU and E5-GPU)
bash retrieval_data.sh

# BM25 Lucene index
bash bm25_data.sh
```

Outputs under `data/wiki18/`:

- `wiki-18.jsonl` — corpus (shared)
- `e5_HNSW64.index` — used by E5-CPU and E5-GPU
- `bm25/` — used by BM25

---

## 1. Run

### 1.1 E5 (CPU)

```bash
docker compose up -d --build retriever-e5-cpu
```

Or directly:

```bash
python -m retriever_server.server --config-name=server_e5_cpu
```

### 1.2 BM25

```bash
docker compose up -d --build retriever-bm25
```

Or directly:

```bash
python -m retriever_server.server --config-name=server_bm25
```

### 1.3 E5 (GPU)

Requires NVIDIA Container Toolkit on the host.

```bash
docker compose up -d --build retriever-e5-gpu
```

Or directly (inside a CUDA-capable env):

```bash
python -m retriever_server.server --config-name=server_e5_gpu
```

GPU device id is set in `docker-compose.yml` under `device_ids`. Edit there to switch GPUs.

---

## 2. Test

All three expose the same `POST /retrieve` schema. Replace the host as needed (`localhost`, container DNS, or a remote like `degas.snu.vision`).

### 2.1 E5 (CPU) — port 3000

```bash
curl -X POST http://localhost:3000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["Who wrote The Old Man and the Sea?"],
    "topk": 3,
    "return_scores": true,
    "dataset": "hotpotqa"
  }'
```

### 2.2 BM25 — port 3001

```bash
curl -X POST http://localhost:3001/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["Who wrote The Old Man and the Sea?"],
    "topk": 3,
    "return_scores": true,
    "dataset": "hotpotqa"
  }'
```

### 2.3 E5 (GPU) — port 3002

```bash
curl -X POST http://localhost:3002/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["Who wrote The Old Man and the Sea?"],
    "topk": 3,
    "return_scores": true,
    "dataset": "hotpotqa"
  }'
```

### 2.4 Response format (all three)

```json
{
  "result": [
    [
      {
        "document": {
          "id": "13410492",
          "title": "The Old Man and the Sea",
          "text": "The Old Man and the Sea is a short novel ...",
          "contents": "..."
        },
        "score": 0.87
      }
    ]
  ]
}
```

`dataset` must be one of the values listed under `datasets:` in the corresponding config. Allowed sets:

| Retriever  | Allowed datasets |
| ---------- | --- |
| E5 (CPU)   | `hotpotqa`, `2wikimultihopqa`, `musique` |
| BM25       | `hotpotqa`, `2wikimultihopqa`, `musique` |
| E5 (GPU)   | `hotpotqa`, `2wikimultihopqa`, `musique`, `nq` |

Omit `dataset` to bypass the whitelist (treated as `default`).
