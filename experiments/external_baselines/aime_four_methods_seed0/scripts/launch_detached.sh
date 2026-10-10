#!/usr/bin/env bash
# Detach the complete launcher and route every inherited stream to disk.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${AIME_UV_BIN:-uv}"

if [[ "${1:-}" == --worker ]]; then
  shift
  LAUNCH_DIR="$1"
  shift
  finish() {
    local code=$?
    trap - EXIT
    printf '%s\n' "$code" > "$LAUNCH_DIR/exit_code.txt"
    "$UV" run --no-project --python "$AIME_PYTHON" python - "$LAUNCH_DIR" "$code" <<'PY'
import datetime, json, pathlib, sys
root, code = pathlib.Path(sys.argv[1]), int(sys.argv[2])
record = {"schema": "aime_detached_completion_v1", "exit_code": code,
          "status": "passed" if code == 0 else "failed",
          "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (root / "completion.json").open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
PY
    exit "$code"
  }
  trap finish EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  printf '%s\n' "$$" > "$LAUNCH_DIR/worker.pid"
  bash "$ROOT/scripts/run_single48g_all.sh" "$@"
  exit 0
fi

STAGE=all
ORDER=aflow,maas,adas,gdesigner
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    all|diagnostics) STAGE="$1"; shift ;;
    --method-order) [[ "$#" -ge 2 ]] || { echo "--method-order needs a value" >&2; exit 2; }; ORDER="$2"; shift 2 ;;
    *) echo "usage: bash scripts/launch_detached.sh [all|diagnostics] [--method-order maas,adas,gdesigner,aflow]" >&2; exit 2 ;;
  esac
done
IFS=, read -r -a METHODS <<< "$ORDER"
[[ "${#METHODS[@]}" == 4 ]] || { echo "All four methods must be present" >&2; exit 2; }
SEEN=,
for method in "${METHODS[@]}"; do
  case "$method" in aflow|maas|adas|gdesigner) ;; *) echo "Unknown method: $method" >&2; exit 2 ;; esac
  [[ "$SEEN" != *",$method,"* ]] || { echo "Duplicate method: $method" >&2; exit 2; }
  SEEN+="$method,"
done
: "${AIME_OUTPUT_ROOT:?Set a fresh AIME_OUTPUT_ROOT}"
for name in AIME_MODEL_PATH AIME_MINILM_PATH AIME_MODEL_VERIFICATION_REPORT AIME_PYTHON \
  AIME_VLLM_BIN AIME_AFLOW_PYTHON AIME_MAAS_PYTHON AIME_ADAS_PYTHON AIME_GDESIGNER_PYTHON; do
  [[ -n "${!name:-}" ]] || { echo "Set $name before launching" >&2; exit 2; }
done
[[ ! -e "$AIME_OUTPUT_ROOT" ]] || { echo "Choose a fresh AIME_OUTPUT_ROOT" >&2; exit 3; }
for executable in "$UV" nohup setsid; do
  command -v "$executable" >/dev/null || { echo "Required executable missing: $executable" >&2; exit 2; }
done
LAUNCH_DIR="${AIME_LAUNCH_DIR:-$ROOT/logs/detached_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$(dirname "$LAUNCH_DIR")"
mkdir "$LAUNCH_DIR"  # Refuse an existing launch/audit directory.
LAUNCH_DIR="$(cd "$LAUNCH_DIR" && pwd)"
nohup setsid bash "$ROOT/scripts/launch_detached.sh" --worker "$LAUNCH_DIR" \
  "$STAGE" --method-order "$ORDER" </dev/null >"$LAUNCH_DIR/run.log" 2>&1 &
WORKER_PID=$!
"$UV" run --no-project --python "$AIME_PYTHON" python - "$LAUNCH_DIR" "$WORKER_PID" "$STAGE" "$ORDER" "$AIME_OUTPUT_ROOT" <<'PY'
import datetime, json, os, pathlib, sys
root, pid, stage, order, output = sys.argv[1:]
record = {"schema": "aime_detached_launch_v1", "pid": int(pid), "stage": stage,
          "method_order": order.split(","), "output_root": str(pathlib.Path(output).resolve()),
          "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "port": os.environ.get("AIME_PORT", "18080"),
          "stdin": "/dev/null", "stdout_stderr": str(pathlib.Path(root) / "run.log"),
          "note": "Process is launched; completion and benchmark quality must be checked separately."}
with (pathlib.Path(root) / "launch.json").open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
PY
printf 'AIME_DETACHED_STARTED pid=%s log=%s output=%s\n' "$WORKER_PID" "$LAUNCH_DIR/run.log" "$AIME_OUTPUT_ROOT"
