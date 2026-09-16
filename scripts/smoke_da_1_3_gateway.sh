#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GATEWAY_VFS_ROOT="${BENCH_ENGINE_GATEWAY_VFS_ROOT:-/mnt/base/agent_workspace/smoke_vascular}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DATA_MOUNT_PATH="${BENCH_ENGINE_DATA_MOUNT_PATH:-$GATEWAY_VFS_ROOT/runs/bench-engine/da-1-3-$RUN_STAMP}"
OUTPUT="${BENCH_ENGINE_OUTPUT:-$GATEWAY_VFS_ROOT/runs/da-1-3-gateway-$RUN_STAMP.jsonl}"
MODEL_ARGS=()

if [[ -n "${BENCH_ENGINE_MODEL:-}" ]]; then
    MODEL_ARGS=(--model "$BENCH_ENGINE_MODEL")
fi

mkdir -p \
    "$GATEWAY_VFS_ROOT/runs" \
    "$(dirname "$DATA_MOUNT_PATH")" \
    "$(dirname "$OUTPUT")"

echo "Gateway VFS root: $GATEWAY_VFS_ROOT"
echo "Data mount path:  $DATA_MOUNT_PATH"
echo "Output:           $OUTPUT"
echo

if [[ "${BENCH_ENGINE_SKIP_GRADER_CHECK:-0}" != "1" ]]; then
    uv run python - <<'PY'
from pathlib import Path
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv(Path(".env"))
api_key = os.environ.get("OPENAI_API_KEY")
base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
if not api_key:
    raise SystemExit("OPENAI_API_KEY is not set; update .env before running the smoke test")

request = urllib.request.Request(
    f"{base_url}/models",
    headers={"Authorization": f"Bearer {api_key}"},
)
try:
    with urllib.request.urlopen(request, timeout=15) as response:
        if response.status != 200:
            raise OSError(f"HTTP {response.status}")
except (OSError, urllib.error.URLError) as exc:
    detail = str(exc)
    if isinstance(exc, urllib.error.HTTPError):
        body = exc.read(500).decode("utf-8", errors="replace")
        detail = f"HTTP {exc.code}: {body or exc.reason}"
    raise SystemExit(
        "OmicOS grader credential check failed. Update OPENAI_API_KEY in .env, "
        f"or set BENCH_ENGINE_SKIP_GRADER_CHECK=1 to run despite this. Detail: {detail}"
    )
PY
fi

exec uv run bench-engine evaluate \
    --benchmark omicos-biomnibench \
    --question-id da-1-3 \
    --grader omicos \
    --autonomics-gateway \
    --data-mount-path "$DATA_MOUNT_PATH" \
    --jobs 1 \
    --timeout 3600 \
    --output "$OUTPUT" \
    "${MODEL_ARGS[@]}"
