#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/workspace/data/wiki18}"
mkdir -p "$OUT_DIR"
export OUT_DIR

python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

out_dir = Path(os.environ["OUT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="PeterJinGo/wiki-18-bm25-index",
    repo_type="dataset",
    local_dir=str(out_dir),
    allow_patterns=["bm25/*"],
)
PY

echo "Prepared:"
echo "  bm25 index: $OUT_DIR/bm25"
