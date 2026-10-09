#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE=all
ORDER="${AIME_METHOD_ORDER:-aflow,maas,adas,gdesigner}"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    all|diagnostics) STAGE="$1"; shift ;;
    --method-order) [[ "$#" -ge 2 ]] || { echo "--method-order needs a comma-separated list" >&2; exit 2; }; ORDER="$2"; shift 2 ;;
    *) echo "usage: $0 [all|diagnostics] [--method-order maas,adas,gdesigner,aflow]" >&2; exit 2 ;;
  esac
done
IFS=, read -r -a METHODS <<< "$ORDER"
[[ "${#METHODS[@]}" == 4 ]] || { echo "Method order must contain all four methods exactly once" >&2; exit 2; }
SEEN=,
for method in "${METHODS[@]}"; do
  case "$method" in aflow|maas|adas|gdesigner) ;; *) echo "Unknown method: $method" >&2; exit 2 ;; esac
  [[ "$SEEN" != *",$method,"* ]] || { echo "Duplicate method: $method" >&2; exit 2; }
  SEEN+="$method,"
done
: "${AIME_MODEL_PATH:?Set the verified local Qwen3.5-9B path}"
: "${AIME_MINILM_PATH:?Set the verified local MiniLM path}"
: "${AIME_MODEL_VERIFICATION_REPORT:?Set the model verification report path}"
: "${AIME_PYTHON:?Set the serving environment Python}"
: "${AIME_VLLM_BIN:?Set the serving environment vLLM}"
: "${AIME_AFLOW_PYTHON:?Set AFlow Python}"
: "${AIME_MAAS_PYTHON:?Set MaAS Python}"
: "${AIME_ADAS_PYTHON:?Set ADAS Python}"
: "${AIME_GDESIGNER_PYTHON:?Set G-Designer Python}"
UV="${AIME_UV_BIN:-uv}"
command -v "$UV" >/dev/null || { echo "uv is required for Python execution" >&2; exit 2; }
run_python() { "$UV" run --no-project --python "$1" python "${@:2}"; }
pair_for() { case "$1" in aflow|maas) printf aflow_maas ;; adas|gdesigner) printf adas_gdesigner ;; esac; }
export AIME_DATA_DIR="${AIME_DATA_DIR:-$ROOT/data}"
AIME_OUTPUT_ROOT="${AIME_OUTPUT_ROOT:-$ROOT/outputs/aime_four_methods_seed0_$(date +%Y%m%d_%H%M%S)}"
[[ ! -e "$AIME_OUTPUT_ROOT" ]] || { echo "Choose a fresh AIME_OUTPUT_ROOT" >&2; exit 3; }
mkdir -p "$AIME_OUTPUT_ROOT"
EVENTS="$AIME_OUTPUT_ROOT/method_attempts.tsv"
: > "$EVENTS"
if [[ -n "${AIME_DIAGNOSTIC_DIR:-}" ]]; then
  DIAG="$AIME_DIAGNOSTIC_DIR"
else
  DIAG_TEMP="${AIME_TEMP_DIR:-/data/jxc/tmp/rpas/aime}"
  mkdir -p "$DIAG_TEMP"
  DIAG="$(mktemp -d "$DIAG_TEMP/aime-diagnostics-XXXXXXXX")"
fi
export AIME_DIAGNOSTIC_DIR="$DIAG"
AGGREGATE=not_run
DIAGNOSTIC_EXIT=not_run
SUMMARY_WRITTEN=0
write_summary() {
  run_python "$AIME_PYTHON" - "$AIME_OUTPUT_ROOT" "$STAGE" "$ORDER" "$AGGREGATE" "$DIAGNOSTIC_EXIT" "$1" <<'PY'
import datetime, json, pathlib, sys
root, stage, order, aggregate, diagnostic_exit, exit_code = sys.argv[1:]
root = pathlib.Path(root)
events = {}
for line in (root / "method_attempts.tsv").read_text().splitlines():
    method, status, code = line.split("\t")
    events[method] = {"status": status, "exit_code": int(code)}
methods = {}
for method in order.split(","):
    pair = "aflow_maas" if method in ("aflow", "maas") else "adas_gdesigner"
    item = events.get(method, {"status": "not_attempted", "exit_code": None})
    item["run_dir"] = f"{pair}/{method}/seed_0"
    record = root / pair / f"launcher_result_{method}.json"
    if record.is_file():
        item["launcher_record"] = str(record.relative_to(root))
        item["launcher_result"] = json.loads(record.read_text())
    methods[method] = item
code = int(exit_code)
report = {"schema": "aime_four_methods_attempt_summary_v1", "stage": stage,
          "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "method_order": order.split(","), "overall_exit_code": code,
          "overall_status": ("passed" if stage == "all" and code == 0 and aggregate == "passed"
                             else "diagnostics_passed" if stage == "diagnostics" and code == 0 else "failed"),
          "diagnostic_exit_code": None if diagnostic_exit == "not_run" else int(diagnostic_exit),
          "aggregate_status": aggregate, "methods": methods,
          "note": "Failed or unverified methods are not accepted or aggregated; attempts use fresh output directories."}
with (root / "attempt_summary.json").open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2); stream.write("\n")
PY
  SUMMARY_WRITTEN=1
}
on_exit() {
  code=$?
  if [[ "$SUMMARY_WRITTEN" == 0 ]]; then write_summary "$code" || true; fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_python "$AIME_PYTHON" - "$AIME_OUTPUT_ROOT/launch_record.json" "$ROOT" "$STAGE" "$ORDER" <<'PY'
import datetime, hashlib, json, os, pathlib, subprocess, sys
path, root, stage, order = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
names = ("AIME_MODEL_PATH", "AIME_MINILM_PATH", "AIME_MODEL_VERIFICATION_REPORT", "AIME_UV_BIN",
         "AIME_PYTHON", "AIME_VLLM_BIN", "AIME_AFLOW_PYTHON", "AIME_MAAS_PYTHON",
         "AIME_ADAS_PYTHON", "AIME_GDESIGNER_PYTHON", "AIME_DIAGNOSTIC_DIR", "CUDA_VISIBLE_DEVICES")
record = {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "source_root": str(root), "source_commit": subprocess.check_output(
              ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
          "stage": stage, "method_order": order.split(","),
          "settings": {name: os.environ.get(name, "") for name in names},
          "launcher_sha256": hashlib.sha256((root / "scripts/run_single48g_all.sh").read_bytes()).hexdigest(),
          "protocol": {"search_seed": 0, "data_seed": 2026, "D_search": 60, "D_select": 30,
                       "D_test_each_year": 30, "context": 16384, "max_tokens": 6144, "concurrency": 8},
          "note": "Each independent method runs in its own service session; failures do not stop other methods."}
with path.open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2); stream.write("\n")
PY
echo "AIME_ALL_START $(date -Iseconds) output=$AIME_OUTPUT_ROOT order=$ORDER"
export AIME_OUTPUT_DIR="$AIME_OUTPUT_ROOT/diagnostics"
if bash "$ROOT/scripts/run_pair.sh" diagnostics; then DIAGNOSTIC_EXIT=0; else DIAGNOSTIC_EXIT=$?; fi

diagnostic_verified() {
  run_python "$AIME_PYTHON" - "$DIAG/$1/report.json" "$1" "$ROOT" "$AIME_OUTPUT_ROOT" <<'PY'
import hashlib, json, pathlib, sys
path, method, source, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
try:
    data = json.loads(path.read_text())
    valid = (data.get("status") == "passed" and data.get("method") == method
             and data.get("synthetic_only") is True and data.get("check_only") is False
             and data.get("max_tokens") == 6144 and data.get("concurrency") == 2
             and data.get("script_sha256") == hashlib.sha256((source / "scripts/diagnose_native_methods.py").read_bytes()).hexdigest())
    launch = json.loads((out / "diagnostics" / f"launcher_result_{method}.json").read_text())
    valid = valid and launch.get("kind") == "diagnostic" and launch.get("status") == "passed" and launch.get("compute_exit_code") == 0
except (OSError, ValueError):
    valid = False
if not valid:
    print(f"DIAGNOSTIC_UNVERIFIED method={method} report={path}", file=sys.stderr)
raise SystemExit(0 if valid else 1)
PY
}

FAILED=0
for method in "${METHODS[@]}"; do
  if ! diagnostic_verified "$method"; then
    printf '%s\tdiagnostic_failed\t1\n' "$method" >> "$EVENTS"
    FAILED=$((FAILED + 1))
    continue
  fi
  if [[ "$STAGE" == diagnostics ]]; then
    printf '%s\tdiagnostics_passed\t0\n' "$method" >> "$EVENTS"
    continue
  fi
  pair="$(pair_for "$method")"
  export AIME_OUTPUT_DIR="$AIME_OUTPUT_ROOT/$pair"
  echo "AIME_METHOD_START $(date -Iseconds) method=$method"
  if bash "$ROOT/scripts/run_pair.sh" --method "$method"; then code=0; else code=$?; fi
  if [[ "$code" == 0 ]]; then
    if run_python "$AIME_PYTHON" - "$AIME_OUTPUT_DIR/launcher_result_${method}.json" "$method" <<'PY'
import json, pathlib, sys
try:
    record = json.loads(pathlib.Path(sys.argv[1]).read_text())
    valid = (record.get("method") == sys.argv[2] and record.get("kind") == "formal"
             and record.get("status") == "passed" and record.get("preflight_exit_code") == 0
             and record.get("compute_exit_code") == 0 and record.get("quality_gate_exit_code") == 0)
except (OSError, ValueError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
    then status=passed; else status=unverified; code=1; fi
  else
    status=failed
  fi
  printf '%s\t%s\t%s\n' "$method" "$status" "$code" >> "$EVENTS"
  if [[ "$status" == passed ]]; then
    echo "AIME_METHOD_PASS $(date -Iseconds) method=$method"
  else
    FAILED=$((FAILED + 1))
    echo "AIME_METHOD_FAIL $(date -Iseconds) method=$method exit=$code; continuing independent methods" >&2
  fi
done
if [[ "$FAILED" -gt 0 ]]; then
  AGGREGATE=blocked_incomplete
  write_summary 1
  echo "AIME_ATTEMPTS_FAILED count=$FAILED; no aggregate accepted: $AIME_OUTPUT_ROOT/attempt_summary.json" >&2
  exit 1
fi
if [[ "$STAGE" == diagnostics ]]; then
  write_summary 0
  echo "AIME_DIAGNOSTICS_FINISHED $(date -Iseconds)"
  exit 0
fi
if run_python "$AIME_PYTHON" "$ROOT/scripts/aggregate_aime_seed0.py" --root "$AIME_OUTPUT_ROOT"; then
  AGGREGATE=passed
  write_summary 0
  echo "AIME_FOUR_METHODS_SEED0_COMPLETE $(date -Iseconds) output=$AIME_OUTPUT_ROOT"
else
  AGGREGATE=failed
  write_summary 1
  echo "AIME_AGGREGATE_FAILED output=$AIME_OUTPUT_ROOT" >&2
  exit 1
fi
