#!/usr/bin/env bash
# A fresh, detached supplemental recovery; this never dispatches search/select.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${AIME_UV_BIN:-/home/ZZD/.local/bin/uv}"
PYTHON="${AIME_PYTHON:-/data/jxc/envs/rpas-serve/bin/python}"
exec "$UV" run --no-project --python "$PYTHON" python \
  "$ROOT/scripts/run_recovery_worker.py" --launch "$@"
