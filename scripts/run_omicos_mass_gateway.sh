#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATEWAY_ROOT="${BENCH_ENGINE_GATEWAY_ROOT:-/mnt/base/agent_workspace/smoke_vascular}"
RUN_ID="${BENCH_ENGINE_RUN_ID:-omicos-mass-20260915T141126Z}"
RUN_ROOT="$GATEWAY_ROOT/runs/$RUN_ID"
OUTPUT="${BENCH_ENGINE_OUTPUT:-$RUN_ROOT/results.jsonl}"
DATA_MOUNT="${BENCH_ENGINE_DATA_MOUNT_PATH:-$GATEWAY_ROOT/runs/bench-engine/$RUN_ID}"
MODEL="${BENCH_ENGINE_MODEL:-zai:glm-5.3-flash}"
JOBS="${BENCH_ENGINE_JOBS:-5}"
TIMEOUT="${BENCH_ENGINE_TIMEOUT:-3600}"
SESSION="${BENCH_ENGINE_TMUX_SESSION:-bench_engine_omicos}"
AUTONOMICS="${BENCH_ENGINE_AUTONOMICS:-/mnt/projects/autonomics_projects/autonomics/target/release/autonomics}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$GATEWAY_ROOT/runs/omicos-mass-$STAMP"

mkdir -p \
    "$RUN_ROOT" \
    "$LOG_DIR" \
    "$(dirname "$DATA_MOUNT")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION is already running." >&2
    tmux display-message -p -t "$SESSION" '#{session_name} #{pane_pid} #{pane_current_command}' >&2
    exit 0
fi

# Solver-side failures and cancellations do not count toward benchmark scoring.
if [[ -f "$OUTPUT" ]]; then
    uv run python - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
kept = [row for row in rows if row.get("solver_ok") is True]
removed = [row for row in rows if row.get("solver_ok") is not True]
path.write_text(
    "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in kept),
    encoding="utf-8",
)
print(f"kept={len(kept)} removed_solver_failures={len(removed)}")
if removed:
    print("removed:", ", ".join(str(row.get("id")) for row in removed))
PY
fi

# Cold-start the gateway with one process before launching concurrent runs.
# Concurrent auto-spawns can race while creating the gateway's rolling log.
"$AUTONOMICS" run \
    --list-sessions \
    --json \
    --name be_gateway_ready_00000000 >/dev/null
"$AUTONOMICS" serve status >/dev/null

tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" \
    "env BENCH_ENGINE_CODEX=${BENCH_ENGINE_CODEX:-codex} uv run bench-engine evaluate \
        --benchmark omicos-biomnibench \
        --model '$MODEL' \
        --grader codex-local \
        --autonomics-gateway \
        --data-mount-path '$DATA_MOUNT' \
        --jobs '$JOBS' \
        --timeout '$TIMEOUT' \
        --output '$OUTPUT' \
        --resume >'$LOG_DIR/run.log' 2>&1"

PANE_PID="$(tmux display-message -p -t "$SESSION" '#{pane_pid}')"
printf '%s\n' "$PANE_PID" >"$LOG_DIR/pid"

echo "session: $SESSION"
echo "pane pid: $PANE_PID"
echo "output: $OUTPUT"
echo "log: $LOG_DIR/run.log"
echo "attach: tmux attach-session -t $SESSION"
echo "tail: tail -f $LOG_DIR/run.log"
