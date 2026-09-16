#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.env"
    set +a
fi
GATEWAY_ROOT="${BENCH_ENGINE_GATEWAY_ROOT:-/mnt/base/agent_workspace/smoke_vascular}"
RUN_ID="${BENCH_ENGINE_RUN_ID:-omicos-mass-20260915T141126Z}"
RUN_ROOT="$GATEWAY_ROOT/runs/$RUN_ID"
OUTPUT="${BENCH_ENGINE_OUTPUT:-$RUN_ROOT/results.jsonl}"
OOM_SKIP="${BENCH_ENGINE_OOM_SKIP_FILE:-$RUN_ROOT/results.oom-skipped.jsonl}"
DATA_MOUNT="${BENCH_ENGINE_DATA_MOUNT_PATH:-$GATEWAY_ROOT/runs/bench-engine/$RUN_ID}"
MODEL="${BENCH_ENGINE_MODEL:-zai:glm-5.3-flash}"
JOBS="${BENCH_ENGINE_JOBS:-5}"
TIMEOUT="${BENCH_ENGINE_TIMEOUT:-3600}"
AUTONOMICS="${BENCH_ENGINE_AUTONOMICS:-/mnt/projects/autonomics_projects/autonomics/target/release/autonomics}"
MANAGE_GATEWAY="${BENCH_ENGINE_MANAGE_GATEWAY:-1}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$GATEWAY_ROOT/runs/omicos-mass-$STAMP"
CODEX_BIN="${BENCH_ENGINE_CODEX:-$(command -v codex || true)}"

if [[ -z "$CODEX_BIN" && -x "/home/wjx/.local/bin/codex" ]]; then
    CODEX_BIN="/home/wjx/.local/bin/codex"
fi
if [[ -z "$CODEX_BIN" ]]; then
    echo "codex executable not found; set BENCH_ENGINE_CODEX." >&2
    exit 1
fi

mkdir -p "$RUN_ROOT" "$LOG_DIR" "$(dirname "$DATA_MOUNT")"

# Salvage SIGKILL-style failures into a durable skip sidecar. Other solver
# failures are removed so crashes and cancellations do not count as scores.
if [[ -f "$OUTPUT" ]]; then
    uv run python - "$OUTPUT" "$OOM_SKIP" <<'PY'
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

output, oom_skip = map(Path, sys.argv[1:])
rows = [
    json.loads(line)
    for line in output.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
kept = [row for row in rows if row.get("solver_ok") is True]
removed = [row for row in rows if row.get("solver_ok") is not True]
output.write_text(
    "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in kept),
    encoding="utf-8",
)

existing = set()
if oom_skip.exists():
    existing = {
        json.loads(line)["id"]
        for line in oom_skip.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def oom_killed(row: dict) -> bool:
    if row.get("solver_returncode") == -9 or row.get("solver_oom_killed") is True:
        return True
    diagnostic = " ".join(
        str(row.get(key) or "")
        for key in ("solver_error", "solver_stderr_tail")
    ).lower()
    return any(
        marker in diagnostic
        for marker in (
            "out of memory",
            "oom",
            "killed process",
            "memory cgroup",
        )
    )


new_skips = []
for row in removed:
    task_id = str(row.get("id"))
    if task_id in existing or not oom_killed(row):
        continue
    new_skips.append(
        {
            "schema": 1,
            "id": task_id,
            "recorded_at": datetime.now(UTC).isoformat(),
            "source": "startup-salvage",
            "returncode": row.get("solver_returncode"),
            "agent_name": row.get("solver_agent_name"),
            "error": row.get("solver_error"),
            "category": row.get("category"),
            "answer_type": row.get("answer_type"),
        }
    )

if new_skips:
    oom_skip.parent.mkdir(parents=True, exist_ok=True)
    with oom_skip.open("a", encoding="utf-8") as handle:
        for record in new_skips:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

print(f"kept={len(kept)} removed_solver_failures={len(removed)}")
if removed:
    print("removed:", ", ".join(str(row.get("id")) for row in removed))
if new_skips:
    print("salvaged OOM skips:", ", ".join(row["id"] for row in new_skips))
PY
fi

GATEWAY_PID=""
cleanup_gateway() {
    if [[ -n "$GATEWAY_PID" ]] && kill -0 "$GATEWAY_PID" 2>/dev/null; then
        kill -TERM "$GATEWAY_PID" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "$GATEWAY_PID" 2>/dev/null || break
            sleep 0.2
        done
        kill -KILL "$GATEWAY_PID" 2>/dev/null || true
        wait "$GATEWAY_PID" 2>/dev/null || true
    fi
}
trap 'exit 130' INT
trap 'exit 143' TERM
trap cleanup_gateway EXIT

if [[ "$MANAGE_GATEWAY" == "1" ]]; then
    # A foreground gateway stays in this runner's cgroup. The enclosing service
    # can therefore clean up gateway, frontends, and tools together after OOM.
    if "$AUTONOMICS" serve status >/dev/null 2>&1; then
        "$AUTONOMICS" serve stop >/dev/null 2>&1 || true
        for _ in {1..50}; do
            "$AUTONOMICS" serve status >/dev/null 2>&1 || break
            sleep 0.2
        done
    fi
    "$AUTONOMICS" serve >"$LOG_DIR/gateway.log" 2>&1 &
    GATEWAY_PID=$!
    ready=0
    for _ in {1..100}; do
        if "$AUTONOMICS" run \
            --list-sessions --json \
            --name be_gateway_ready_00000000 >/dev/null 2>&1; then
            ready=1
            break
        fi
        kill -0 "$GATEWAY_PID" 2>/dev/null || break
        sleep 0.2
    done
    if [[ "$ready" != "1" ]]; then
        echo "managed Autonomics gateway failed to become ready" >&2
        exit 1
    fi
else
    "$AUTONOMICS" run \
        --list-sessions --json \
        --name be_gateway_ready_00000000 >/dev/null
    "$AUTONOMICS" serve status >/dev/null
fi

echo "output: $OUTPUT"
echo "data mount: $DATA_MOUNT"
echo "log: $LOG_DIR/run.log"
echo "gateway log: $LOG_DIR/gateway.log"
echo "oom skips: $OOM_SKIP"
echo "model: $MODEL"
echo "jobs: $JOBS"

exec > >(tee -a "$LOG_DIR/run.log") 2>&1
cd "$REPO_ROOT"
set +e
env \
    BENCH_ENGINE_CODEX="$CODEX_BIN" \
    PATH="${CODEX_BIN%/*}:/usr/bin:/bin" \
    uv run bench-engine evaluate \
        --benchmark omicos-biomnibench \
        --model "$MODEL" \
        --grader codex-local \
        --autonomics-gateway \
        --data-mount-path "$DATA_MOUNT" \
        --jobs "$JOBS" \
        --timeout "$TIMEOUT" \
        --output "$OUTPUT" \
        --oom-skip-file "$OOM_SKIP" \
        --resume
STATUS=$?
set -e

if [[ "$MANAGE_GATEWAY" == "1" ]] && ! kill -0 "$GATEWAY_PID" 2>/dev/null; then
    echo "managed Autonomics gateway exited before the benchmark runner" >&2
    STATUS=75
fi
exit "$STATUS"
