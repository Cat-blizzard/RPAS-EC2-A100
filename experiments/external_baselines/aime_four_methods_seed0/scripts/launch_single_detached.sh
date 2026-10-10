#!/usr/bin/env bash
# Run one method's synthetic diagnostic and formal attempt in a detached session.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${AIME_UV_BIN:-uv}"
run_python() { "$UV" run --no-project --python "$AIME_PYTHON" python "$@"; }

if [[ "${1:-}" == --worker ]]; then
  LAUNCH_DIR="$2"
  METHOD="$3"
  STAGE=diagnostic
  finish() {
    local code=$?
    trap - EXIT
    printf '%s\n' "$code" > "$LAUNCH_DIR/exit_code.txt"
    run_python - "$LAUNCH_DIR" "$METHOD" "$STAGE" "$code" <<'PY'
import datetime, json, pathlib, sys
root, method, stage, code = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
record = {"schema": "aime_single_detached_completion_v1", "method": method, "stage": stage,
          "exit_code": code, "status": "passed" if code == 0 and stage == "complete" else "failed",
          "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (root / "completion.json").open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2); stream.write("\n")
PY
    exit "$code"
  }
  trap finish EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  printf '%s\n' "$$" > "$LAUNCH_DIR/worker.pid"
  printf '%s\n' "$STAGE" > "$LAUNCH_DIR/stage.txt"
  export AIME_OUTPUT_DIR="$AIME_OUTPUT_ROOT/diagnostics" AIME_DIAGNOSTICS_ONLY=1
  echo "AIME_SINGLE_DIAGNOSTIC_START method=$METHOD"
  bash "$ROOT/scripts/run_pair.sh" --method "$METHOD"
  STAGE=diagnostic_verification
  printf '%s\n' "$STAGE" > "$LAUNCH_DIR/stage.txt"
  run_python - "$ROOT" "$AIME_OUTPUT_ROOT" "$AIME_DIAGNOSTIC_DIR" "$METHOD" <<'PY'
import hashlib, json, pathlib, sys
root, out, diagnostic, method = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4]
report = json.loads((diagnostic / method / "report.json").read_text())
launcher = json.loads((out / "diagnostics" / f"launcher_result_{method}.json").read_text())
valid = (report.get("method") == method and report.get("status") == "passed"
         and report.get("synthetic_only") is True and report.get("check_only") is False
         and report.get("max_tokens") == 6144 and report.get("concurrency") == 2
         and report.get("script_sha256") == hashlib.sha256((root / "scripts/diagnose_native_methods.py").read_bytes()).hexdigest()
         and launcher.get("method") == method and launcher.get("kind") == "diagnostic"
         and launcher.get("status") == "passed" and launcher.get("compute_exit_code") == 0)
if not valid:
    raise SystemExit(f"DIAGNOSTIC_UNVERIFIED method={method}; formal attempt is blocked")
print(f"AIME_SINGLE_DIAGNOSTIC_VERIFIED method={method}", flush=True)
PY
  unset AIME_DIAGNOSTICS_ONLY
  case "$METHOD" in aflow) PAIR=aflow_maas ;; adas|gdesigner) PAIR=adas_gdesigner ;; esac
  export AIME_OUTPUT_DIR="$AIME_OUTPUT_ROOT/$PAIR"
  STAGE=formal
  printf '%s\n' "$STAGE" > "$LAUNCH_DIR/stage.txt"
  echo "AIME_SINGLE_FORMAL_START method=$METHOD output=$AIME_OUTPUT_DIR"
  bash "$ROOT/scripts/run_pair.sh" --method "$METHOD"
  STAGE=formal_verification
  printf '%s\n' "$STAGE" > "$LAUNCH_DIR/stage.txt"
  run_python - "$AIME_OUTPUT_DIR" "$METHOD" <<'PY'
import json, pathlib, sys
out, method = pathlib.Path(sys.argv[1]), sys.argv[2]
launcher = json.loads((out / f"launcher_result_{method}.json").read_text())
manifest = json.loads((out / method / "seed_0" / "run_manifest.json").read_text())
valid = (launcher.get("method") == method and launcher.get("kind") == "formal"
         and launcher.get("status") == "passed" and launcher.get("preflight_exit_code") == 0
         and launcher.get("compute_exit_code") == 0 and launcher.get("quality_gate_exit_code") == 0
         and manifest.get("method") == method and manifest.get("status") == "complete")
if not valid:
    raise SystemExit(f"FORMAL_UNVERIFIED method={method}; no accepted result")
print(f"AIME_SINGLE_FORMAL_VERIFIED method={method}", flush=True)
PY
  STAGE=complete
  printf '%s\n' "$STAGE" > "$LAUNCH_DIR/stage.txt"
  exit 0
fi

METHOD=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --method) [[ "$#" -ge 2 ]] || { echo "--method needs a value" >&2; exit 2; }; METHOD="$2"; shift 2 ;;
    *) echo "usage: bash scripts/launch_single_detached.sh --method {adas|aflow|gdesigner}" >&2; exit 2 ;;
  esac
done
case "$METHOD" in
  aflow) METHOD_PYTHON=AIME_AFLOW_PYTHON ;;
  adas) METHOD_PYTHON=AIME_ADAS_PYTHON ;;
  gdesigner) METHOD_PYTHON=AIME_GDESIGNER_PYTHON ;;
  *) echo "Choose --method adas, aflow, or gdesigner" >&2; exit 2 ;;
esac
for name in AIME_OUTPUT_ROOT AIME_LAUNCH_DIR AIME_MODEL_PATH AIME_MINILM_PATH \
  AIME_MODEL_VERIFICATION_REPORT AIME_PYTHON AIME_VLLM_BIN AIME_DATA_DIR "$METHOD_PYTHON"; do
  [[ -n "${!name:-}" ]] || { echo "Set $name before launching" >&2; exit 2; }
done
for path in "$AIME_OUTPUT_ROOT" "$AIME_LAUNCH_DIR"; do
  [[ ! -e "$path" ]] || { echo "Choose a fresh directory: $path" >&2; exit 3; }
done
export AIME_DIAGNOSTIC_DIR="${AIME_DIAGNOSTIC_DIR:-$AIME_OUTPUT_ROOT/synthetic_diagnostics}"
[[ ! -e "$AIME_DIAGNOSTIC_DIR" ]] || { echo "Choose a fresh AIME_DIAGNOSTIC_DIR" >&2; exit 3; }
for executable in "$UV" nohup setsid git; do
  command -v "$executable" >/dev/null || { echo "Required executable missing: $executable" >&2; exit 2; }
done
mkdir -p "$(dirname "$AIME_OUTPUT_ROOT")" "$(dirname "$AIME_LAUNCH_DIR")"
mkdir "$AIME_OUTPUT_ROOT" "$AIME_LAUNCH_DIR"
export AIME_OUTPUT_ROOT="$(cd "$AIME_OUTPUT_ROOT" && pwd)"
LAUNCH_DIR="$(cd "$AIME_LAUNCH_DIR" && pwd)"
run_python - "$ROOT" "$AIME_OUTPUT_ROOT" "$LAUNCH_DIR" "$METHOD" <<'PY'
import datetime, hashlib, json, os, pathlib, subprocess, sys
root, out, launch, method = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4]
record = {"schema": "aime_single_detached_launch_v1", "method": method, "status": "started",
          "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "source_root": str(root), "source_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
          "launcher_sha256": hashlib.sha256((root / "scripts/launch_single_detached.sh").read_bytes()).hexdigest(),
          "run_pair_sha256": hashlib.sha256((root / "scripts/run_pair.sh").read_bytes()).hexdigest(),
          "output_root": str(out), "diagnostic_root": os.environ["AIME_DIAGNOSTIC_DIR"],
          "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "0"), "port": os.environ.get("AIME_PORT", "18080"),
          "stdin": "/dev/null", "stdout_stderr": str(launch / "run.log"),
          "protocol": {"search_seed": 0, "data_seed": 2026, "D_search": 60, "D_select": 30,
                       "D_test_each_year": 30, "context": 16384, "max_tokens": 6144,
                       "concurrency": 8, "max_num_seqs": 8},
          "note": "Started records establish no result; completion requires diagnostic, formal runner, and native gate verification."}
with (out / "launch_record.json").open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2); stream.write("\n")
PY
nohup setsid bash "$ROOT/scripts/launch_single_detached.sh" --worker "$LAUNCH_DIR" "$METHOD" \
  </dev/null >"$LAUNCH_DIR/run.log" 2>&1 &
WORKER_PID=$!
run_python - "$AIME_OUTPUT_ROOT" "$LAUNCH_DIR" "$WORKER_PID" <<'PY'
import json, os, pathlib, sys
out, launch, pid = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
record = json.loads((out / "launch_record.json").read_text())
record["pid"] = pid
record["cmd_contains"] = f"{record['source_root']}/scripts/launch_single_detached.sh --worker {launch} {record['method']}"
try:
    # comm may contain spaces or closing parentheses; fields after its final
    # ')' start at stat field 3, so starttime (field 22) is offset 19.
    stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
    record["starttime_ticks"] = int(stat.rsplit(")", 1)[1].split()[19])
except (OSError, ValueError, IndexError):
    # A quick failure can finish before metadata collection; completion.json
    # remains authoritative. An unfinished launch without identity is failed.
    record["starttime_ticks"] = None
temporary = launch / "launch.json.tmp"
with temporary.open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2); stream.write("\n")
os.replace(temporary, launch / "launch.json")
PY
printf 'AIME_SINGLE_DETACHED_STARTED method=%s pid=%s log=%s output=%s\n' \
  "$METHOD" "$WORKER_PID" "$LAUNCH_DIR/run.log" "$AIME_OUTPUT_ROOT"
