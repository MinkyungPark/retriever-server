#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/workspace/data/wiki18}"
mkdir -p "$OUT_DIR"
export OUT_DIR

python - <<'PY'
import glob
import gzip
import os
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

out_dir = Path(os.environ["OUT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
        repo_id="PeterJinGo/wiki-18-e5-index-HNSW64",
        repo_type="dataset",
        local_dir=str(out_dir),
        allow_patterns=["part_*"],
)

hf_hub_download(
    repo_id="PeterJinGo/wiki-18-corpus",
    filename="wiki-18.jsonl.gz",
    repo_type="dataset",
    local_dir=str(out_dir),
)

parts = sorted(glob.glob(str(out_dir / "part_*")))
if not parts:
    raise RuntimeError("No HNSW index shards downloaded (part_* not found).")

index_path = out_dir / "e5_HNSW64.index"
with index_path.open("wb") as w:
    for part in parts:
        with open(part, "rb") as r:
            shutil.copyfileobj(r, w)

gz_path = out_dir / "wiki-18.jsonl.gz"
raw_path = out_dir / "wiki-18.raw"
with gzip.open(gz_path, "rb") as gz, raw_path.open("wb") as raw:
    shutil.copyfileobj(gz, raw)

jsonl_path = out_dir / "wiki-18.jsonl"
if tarfile.is_tarfile(raw_path):
    with tarfile.open(raw_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".jsonl")]
        if not members:
            raise RuntimeError("tar archive has no .jsonl member")
        src = tar.extractfile(members[0])
        if src is None:
            raise RuntimeError("failed to extract jsonl member from tar archive")
        with jsonl_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
else:
    raw_path.replace(jsonl_path)

if raw_path.exists():
    raw_path.unlink()
PY

echo "Prepared:"
echo "  corpus: $OUT_DIR/wiki-18.jsonl"
echo "  index : $OUT_DIR/e5_HNSW64.index"
