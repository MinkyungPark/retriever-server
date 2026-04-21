### 0. Setup Environment

#### 1) Configure environment variables
Docker Compose automatically reads `.env` in the project root. Copy the template and fill in your values:
```bash
cp .env.example .env
```

Edit `.env`:
```
HF_TOKEN=hf_...
WANDB_ENTITY=your-entity
WANDB_API_KEY=your-key

# Base directory for HuggingFace and uv cache.
# Use a path with sufficient disk space (volumes mount as $CACHE_BASE/.cache/{huggingface,uv}).
# e.g. CACHE_BASE=/data2/yourname   or   CACHE_BASE=/home/yourname
CACHE_BASE=/data2/yourname
```

#### 2) Build images and start containers

Retriever server:
```bash
docker compose up -d retriever
```


### 1. Setup Retriever

#### 1) Prepare shared retriever resources (wiki-18 + prebuilt e5 HNSW64 index)
```bash
bash scripts/retrieval_data.sh
```

Outputs:
- `data/wiki18/wiki-18.jsonl`
- `data/wiki18/e5_HNSW64.index`

#### 2) Run shared retriever server
```bash
python -m retriever_server.server \
  index_path=/workspace/data/wiki18/e5_HNSW64.index \
  corpus_path=/workspace/data/wiki18/wiki-18.jsonl \
  datasets=[hotpotqa,2wikimultihopqa,musique] \
  workers=1 \
  port=3000
```

or

```bash
docker compose up -d retriever
```

#### 3) Check server
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

- Response:
  ```json
  {"result":[[{"document":{"id":"13410492","title":"\"The Old Man and the Sea\"","text":"The Old Man and the Sea The Old Man and the Sea is a short novel written by the American author Ernest Hemingway in 1951 in Cuba, and published in 1952...
  ```

- If you run the server directly, modify the `src/agentic_iq/conf/retriever/client_default.yaml`.
  ```yaml
  url: http://retriever:3000/retrieve # http://localhost:3000/retrieve
  topk: 3
  ```



