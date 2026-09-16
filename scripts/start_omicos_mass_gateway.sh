#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.env"
    set +a
fi
UNIT="${BENCH_ENGINE_SYSTEMD_UNIT:-bench-engine-omicos.service}"
MEMORY_HIGH="${BENCH_ENGINE_MEMORY_HIGH:-24G}"
MEMORY_MAX="${BENCH_ENGINE_MEMORY_MAX:-26G}"
RESTART_SEC="${BENCH_ENGINE_RESTART_SEC:-10}"

if systemctl --user is-active --quiet "$UNIT"; then
    echo "$UNIT is already active." >&2
    systemctl --user --no-pager --full status "$UNIT" >&2
    exit 0
fi

systemctl --user reset-failed "$UNIT" 2>/dev/null || true
systemd-run --user \
    --unit="$UNIT" \
    --property=WorkingDirectory="$REPO_ROOT" \
    --property=KillMode=control-group \
    --property=MemoryHigh="$MEMORY_HIGH" \
    --property=MemoryMax="$MEMORY_MAX" \
    --property=OOMPolicy=stop \
    --property=Restart=on-failure \
    --property=RestartSec="$RESTART_SEC" \
    --property=StartLimitIntervalSec=0 \
    "$REPO_ROOT/scripts/run_omicos_mass_gateway.sh"

echo "started $UNIT"
echo "journal: journalctl --user -u $UNIT -f"
