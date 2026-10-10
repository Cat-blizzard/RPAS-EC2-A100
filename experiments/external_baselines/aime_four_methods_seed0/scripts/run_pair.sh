#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$ROOT/experiments"
PAIR="${1:-}"
if [[ "$PAIR" == --method ]]; then PAIR="${2:-}"; fi
DATA="${AIME_DATA_DIR:-}"
MODEL="${AIME_MODEL_PATH:-}"
OUT="${AIME_OUTPUT_DIR:-$ROOT/outputs/seed0}"
PORT="${AIME_PORT:-18080}"
VLLM="${AIME_VLLM_BIN:-vllm}"
COMMON_PYTHON="${AIME_PYTHON:-python3}"
AFLOW_PYTHON="${AIME_AFLOW_PYTHON:-$COMMON_PYTHON}"
MAAS_PYTHON="${AIME_MAAS_PYTHON:-$COMMON_PYTHON}"
ADAS_PYTHON="${AIME_ADAS_PYTHON:-$COMMON_PYTHON}"
GDESIGNER_PYTHON="${AIME_GDESIGNER_PYTHON:-$COMMON_PYTHON}"
EMBEDDING="${AIME_MINILM_PATH:-}"
UPSTREAM="$ROOT/upstream"
UV="${AIME_UV_BIN:-uv}"
run_python() { "$UV" run --no-project --python "$1" python "${@:2}"; }
python_for() {
  case "$1" in
    aflow) printf '%s' "$AFLOW_PYTHON" ;;
    maas) printf '%s' "$MAAS_PYTHON" ;;
    adas) printf '%s' "$ADAS_PYTHON" ;;
    gdesigner) printf '%s' "$GDESIGNER_PYTHON" ;;
  esac
}

case "$PAIR" in
  aflow_maas) METHODS=(aflow maas) ;;
  adas_gdesigner) METHODS=(adas gdesigner) ;;
  diagnostics) METHODS=(aflow maas adas gdesigner); export AIME_DIAGNOSTICS_ONLY=1 ;;
  aflow|maas|adas|gdesigner) METHODS=("$PAIR") ;;
  *) echo "usage: bash scripts/run_pair.sh {aflow_maas|adas_gdesigner|diagnostics|aflow|maas|adas|gdesigner}" >&2; exit 2 ;;
esac
SESSION_NAME=run_session.json
if [[ "${#METHODS[@]}" == 1 ]]; then SESSION_NAME="run_session_${PAIR}.json"; fi

[[ -n "$DATA" && -d "$DATA" ]] || { echo "Set AIME_DATA_DIR to the locally authorized frozen AIME data directory" >&2; exit 2; }
[[ -n "$MODEL" && -f "$MODEL/config.json" && -f "$MODEL/tokenizer_config.json" ]] || { echo "Set AIME_MODEL_PATH to local Qwen3.5-9B weights/tokenizer" >&2; exit 2; }
for file in aimo-validation-aime.jsonl aime_2025.jsonl aime_2026.jsonl; do
  [[ -f "$DATA/$file" ]] || { echo "Missing required local AIME file: $DATA/$file" >&2; exit 2; }
done
# Only the validation pool is opened before candidate selection. The test files
# are existence-checked above, but their bytes/rows are not read by this script.
[[ "$(wc -l < "$DATA/aimo-validation-aime.jsonl")" -eq 90 ]] || { echo "Validation split must contain exactly 90 lines" >&2; exit 2; }
for executable in "$COMMON_PYTHON" "$VLLM" "$UV" curl; do
  command -v "$executable" >/dev/null || { echo "Required executable not found: $executable" >&2; exit 2; }
done
for method in "${METHODS[@]}"; do
  command -v "$(python_for "$method")" >/dev/null || { echo "Required Python not found for $method" >&2; exit 2; }
done
command -v setsid >/dev/null || { echo "setsid is required to cleanly stop the vLLM process group" >&2; exit 2; }
for method in "${METHODS[@]}"; do
  case "$method" in aflow) directory=AFlow ;; maas) directory=MaAS ;; adas) directory=ADAS ;; gdesigner) directory=GDesigner ;; esac
  [[ -d "$UPSTREAM/$directory/.git" ]] || { echo "Missing pinned source $UPSTREAM/$directory; run setup_upstreams.sh" >&2; exit 2; }
done
[[ -n "$EMBEDDING" && -d "$EMBEDDING" ]] || { echo "Set AIME_MINILM_PATH to local all-MiniLM-L6-v2 weights for MaAS/G-Designer" >&2; exit 2; }

CONTEXT=16384
OUTPUT=6144
CONCURRENCY=8
MAX_SEQS=8
GPU_UTIL=0.92
DATA_SEED=2026
SEED=0
mkdir -p "$OUT" "$ROOT/logs"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export AIME_TOKENIZER_PATH="$MODEL" AIME_MINILM_PATH="$EMBEDDING" AIME_MAAS_EMBEDDING_PATH="$EMBEDDING"
export AIME_AFLOW_SOURCE="$UPSTREAM/AFlow" AIME_MAAS_SOURCE="$UPSTREAM/MaAS"
export AIME_ADAS_SOURCE="$UPSTREAM/ADAS" AIME_GDESIGNER_SOURCE="$UPSTREAM/GDesigner"
export AIME_MAX_MODEL_LEN="$CONTEXT" AIME_MAX_NUM_SEQS="$MAX_SEQS" AIME_TP_SIZE=1
export AIME_PAIR="$PAIR"
export AIME_AFLOW_REQUEST_TIMEOUT_S="${AIME_AFLOW_REQUEST_TIMEOUT_S:-600}"
export AIME_AFLOW_SAMPLE_TIMEOUT_S="${AIME_AFLOW_SAMPLE_TIMEOUT_S:-1200}"
export AIME_AFLOW_CODE_TIMEOUT_S="${AIME_AFLOW_CODE_TIMEOUT_S:-30}"
export AIME_MAAS_REQUEST_TIMEOUT_S="${AIME_MAAS_REQUEST_TIMEOUT_S:-600}"
export AIME_MAAS_SAMPLE_TIMEOUT_S="${AIME_MAAS_SAMPLE_TIMEOUT_S:-1500}"
export AIME_MAAS_SAMPLE_RETRIES="${AIME_MAAS_SAMPLE_RETRIES:-3}"
export AIME_MAAS_CODE_TIMEOUT_S="${AIME_MAAS_CODE_TIMEOUT_S:-60}"
export AIME_EXTERNAL_REQUEST_TIMEOUT_S="${AIME_EXTERNAL_REQUEST_TIMEOUT_S:-600}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
AIME_CACHE_DIR="${AIME_CACHE_DIR:-/data/jxc/cache/rpas}"
AIME_TEMP_DIR="${AIME_TEMP_DIR:-/data/jxc/tmp/rpas/aime}"
export HF_HOME="$AIME_CACHE_DIR/huggingface" XDG_CACHE_HOME="$AIME_CACHE_DIR/xdg"
export TORCH_HOME="$AIME_CACHE_DIR/torch" TRITON_CACHE_DIR="$AIME_CACHE_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$AIME_CACHE_DIR/torchinductor" VLLM_CACHE_ROOT="$AIME_CACHE_DIR/vllm"
export CUDA_CACHE_PATH="$AIME_CACHE_DIR/cuda" TMPDIR="$AIME_TEMP_DIR"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$CUDA_CACHE_PATH" "$TMPDIR"
AIME_EXISTING_NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
export NO_PROXY="${AIME_EXISTING_NO_PROXY:+$AIME_EXISTING_NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
[[ -n "${AIME_MODEL_VERIFICATION_REPORT:-}" && -f "$AIME_MODEL_VERIFICATION_REPORT" ]] || {
  echo "Set AIME_MODEL_VERIFICATION_REPORT to a fresh scripts/verify_local_models.py report" >&2; exit 2;
}
export RPAS_SERVE_PYTHON="$COMMON_PYTHON" RPAS_AFLOW_PYTHON="$AFLOW_PYTHON" RPAS_MAAS_PYTHON="$MAAS_PYTHON"
export RPAS_VLLM="$VLLM" RPAS_MODEL_PATH="$MODEL" RPAS_MAAS_EMBEDDING_PATH="$EMBEDDING"
export RPAS_MODEL_VERIFICATION_REPORT="$AIME_MODEL_VERIFICATION_REPORT"
export RPAS_MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export RPAS_MAAS_EMBEDDING_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41

# Existing results are refused before any checker can rewrite their audit
# records. A single-method launch only checks its own directory, so other
# independent methods may share the aggregate's pair container safely.
[[ ! -e "$OUT/$SESSION_NAME" ]] || { echo "This selection has a previous service attempt; choose a fresh output root." >&2; exit 3; }
for method in "${METHODS[@]}"; do
  run_dir="$OUT/$method/seed_$SEED"
  if [[ -e "$run_dir" ]]; then
    echo "Existing run detected at $run_dir; choose a fresh AIME_OUTPUT_DIR for any retry." >&2
    exit 3
  fi
  [[ ! -e "$OUT/launcher_attempt_${method}.json" && ! -e "$OUT/launcher_result_${method}.json" ]] || {
    echo "Previous launcher record exists for $method; choose a fresh output root." >&2; exit 3;
  }
done

run_python "$COMMON_PYTHON" - "$PORT" <<'PY'
import socket, sys
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    # Match the server's reuse policy: recently closed connections can remain
    # in TIME_WAIT. An active listener still makes this bind fail.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY
run_python "$COMMON_PYTHON" "$ROOT/scripts/preflight_gpu.py" --role serve
LOG="$(mktemp "$ROOT/logs/vllm-${PAIR}-${PORT}-XXXXXXXX.log")"
GPU_LOG="$(mktemp "$ROOT/logs/gpu-${PAIR}-${PORT}-XXXXXXXX.csv")"
setsid "$VLLM" serve "$MODEL" --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len "$CONTEXT" --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens 16384 --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching --reasoning-parser qwen3 --language-model-only \
  >"$LOG" 2>&1 &
VLLM_PID=$!
nvidia-smi --id "$GPU_ID" --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv -lms 1000 >"$GPU_LOG" 2>&1 &
GPU_PID=$!
METHOD_PID=""
stop_group() {
  local group="${1:-}"
  [[ -n "$group" ]] || return 0
  kill -TERM -- "-$group" 2>/dev/null || true
  for _ in {1..30}; do
    kill -0 -- "-$group" 2>/dev/null || break
    sleep 0.1
  done
  kill -KILL -- "-$group" 2>/dev/null || true
  wait "$group" 2>/dev/null || true
}
run_group() {
  local code
  setsid "$@" &
  METHOD_PID=$!
  if wait "$METHOD_PID"; then code=0; else code=$?; fi
  stop_group "$METHOD_PID"
  METHOD_PID=""
  return "$code"
}
cleanup() {
  set +e
  stop_group "$METHOD_PID"
  stop_group "$VLLM_PID"
  kill "$GPU_PID" 2>/dev/null || true
  wait "$GPU_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
SESSION="${LOG%.log}.json"
export AIME_RUN_SESSION_RECORD="$SESSION"
run_python "$COMMON_PYTHON" "$ROOT/scripts/record_run_session.py" --root "$ROOT" --output "$SESSION" \
  --server-pid "$VLLM_PID" --gpu-monitor-pid "$GPU_PID" --server-log "$LOG" --gpu-log "$GPU_LOG" --port "$PORT"
cp "$SESSION" "$OUT/$SESSION_NAME"

READY=0
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" | run_python "$COMMON_PYTHON" -c 'import json,sys; sys.exit(0 if any(x.get("id")=="Qwen/Qwen3.5-9B" for x in json.load(sys.stdin).get("data",[])) else 1)' 2>/dev/null; then READY=1; break; fi
  kill -0 "$VLLM_PID" 2>/dev/null || { tail -n 200 "$LOG" >&2; exit 1; }
  sleep 5
done
[[ "$READY" == 1 ]] || { echo "Timed out waiting for vLLM; inspect $LOG" >&2; tail -n 200 "$LOG" >&2; exit 1; }

# A disposable endpoint smoke test; it is not part of any search/test metric.
run_python "$COMMON_PYTHON" - "$PORT" <<'PY'
import math, os, sys, time
from openai import OpenAI
try:
    timeout = float(os.environ.get("AIME_ENDPOINT_SMOKE_TIMEOUT_S", "300"))
except ValueError:
    raise SystemExit("AIME_ENDPOINT_SMOKE_TIMEOUT_S must be a finite positive number")
if not math.isfinite(timeout) or timeout <= 0:
    raise SystemExit("AIME_ENDPOINT_SMOKE_TIMEOUT_S must be a finite positive number")
print(f"MODEL_ENDPOINT_SMOKE_START timeout_s={timeout:g}", flush=True)
started = time.monotonic()
client = OpenAI(base_url=f"http://127.0.0.1:{sys.argv[1]}/v1", api_key="EMPTY", timeout=timeout, max_retries=0)
r = client.chat.completions.create(model="Qwen/Qwen3.5-9B", messages=[{"role":"user","content":"Return only OK."}], max_tokens=16, temperature=0, extra_body={"chat_template_kwargs":{"enable_thinking":False}})
assert r.choices and (r.choices[0].message.content or "").strip()
print(f"MODEL_ENDPOINT_SMOKE_PASS elapsed_s={time.monotonic() - started:.3f}", flush=True)
PY

write_method_record() {
  run_python "$COMMON_PYTHON" - "$OUT" "$1" "$2" "$3" "$4" "$5" "$6" "$SESSION" "$ROOT" <<'PY'
import datetime, json, pathlib, sys
out, method, kind, state, preflight, compute, gate, session, root = sys.argv[1:]
name = "launcher_attempt" if state == "started" else "launcher_result"
record = {"schema": "aime_method_launcher_attempt_v1", "method": method, "kind": kind,
          "status": state, "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "source_root": root, "run_session_record": session,
          "preflight_exit_code": None if preflight == "not_run" else int(preflight),
          "compute_exit_code": None if compute == "not_run" else int(compute),
          "quality_gate_exit_code": None if gate == "not_run" else int(gate),
          "note": "Only a successful computation and unchanged native quality gate establish a formal pass."}
with (pathlib.Path(out) / f"{name}_{method}.json").open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2); stream.write("\n")
PY
}

FAILED=0
if [[ "${AIME_DIAGNOSTICS_ONLY:-0}" == 1 ]]; then
  DIAG="${AIME_DIAGNOSTIC_DIR:-$(mktemp -d "$TMPDIR/aime-native-XXXXXXXX")}"
  echo "SYNTHETIC_DIAGNOSTIC_ROOT $DIAG"
  for method in "${METHODS[@]}"; do
    write_method_record "$method" diagnostic started not_run not_run not_run
    echo "DIAGNOSTIC_START method=$method"
    if run_group "$UV" run --no-project --python "$(python_for "$method")" python \
        -u "$ROOT/scripts/diagnose_native_methods.py" --method "$method" --baseline-root "$UPSTREAM" \
        --endpoint "http://127.0.0.1:$PORT/v1" --run-dir "$DIAG/$method" --max-tokens "$OUTPUT"; then
      write_method_record "$method" diagnostic passed not_run 0 not_run
      echo "DIAGNOSTIC_PASS method=$method"
    else
      code=$?
      write_method_record "$method" diagnostic failed not_run "$code" not_run
      FAILED=$((FAILED + 1))
      echo "DIAGNOSTIC_FAIL method=$method exit=$code; continuing independent methods" >&2
    fi
  done
  if [[ "$FAILED" -gt 0 ]]; then echo "AIME_NATIVE_DIAGNOSTICS_FAILED count=$FAILED" >&2; exit 1; fi
  echo "AIME_NATIVE_DIAGNOSTICS_COMPLETE"
  exit 0
fi

for method in "${METHODS[@]}"; do
  PY="$(python_for "$method")"
  run_dir="$OUT/$method/seed_$SEED"
  write_method_record "$method" formal started not_run not_run not_run
  preflight=0
  compute=not_run
  gate=not_run
  status=failed
  export AIME_RUN_ID="aime:$PAIR:$method:seed_$SEED"
  export MASBENCH_RUN_ID="$AIME_RUN_ID"
  echo "START method=$method seed=$SEED data_seed=$DATA_SEED context=$CONTEXT output=$OUTPUT concurrency=$CONCURRENCY run_dir=$run_dir"
  if [[ "$method" == aflow || "$method" == maas ]]; then
    if run_group "$UV" run --no-project --python "$PY" python "$ROOT/scripts/preflight_gpu.py" --role "$method"; then
      preflight=0
    else
      preflight=$?
    fi
  fi
  if [[ "$preflight" == 0 ]]; then
    if [[ "$method" == aflow || "$method" == maas ]]; then
      if run_group "$UV" run --no-project --python "$PY" python -u "$CODE/native_aime_formal.py" \
          --method "$method" --baseline-root "$UPSTREAM" --data-dir "$DATA" --run-dir "$run_dir" \
          --endpoint "http://127.0.0.1:$PORT/v1" --data-seed "$DATA_SEED" --seed "$SEED" \
          --search-size 60 --selection-size 30 --test-size 30 --max-tokens "$OUTPUT" \
          --concurrency "$CONCURRENCY" --aflow-max-rounds 8 --maas-samples 1 --maas-batch-size 8; then
        compute=0
      else
        compute=$?
      fi
      if [[ "$compute" == 0 ]]; then
        if run_group "$UV" run --no-project --python "$PY" python -u "$CODE/aime_quality_gate.py" \
            --run-dir "$run_dir" --method "$method" --seed "$SEED"; then gate=0; else gate=$?; fi
      fi
    else
      if run_group "$UV" run --no-project --python "$PY" python -u "$CODE/native_external_methods.py" \
          --dataset aime --method "$method" --seed "$SEED" --data-dir "$DATA" --run-dir "$run_dir" \
          --endpoint "http://127.0.0.1:$PORT/v1" --adas-generations 30; then compute=0; else compute=$?; fi
      if [[ "$compute" == 0 ]]; then
        if run_group "$UV" run --no-project --python "$PY" python -u "$CODE/aggregate_external_methods.py" \
            --method "$method" --check-run "$run_dir" --dataset aime --seed "$SEED"; then gate=0; else gate=$?; fi
      fi
    fi
  fi
  if [[ "$preflight" == 0 && "$compute" == 0 && "$gate" == 0 ]]; then
    if [[ -e "$run_dir/run_session.json" ]]; then
      if cmp -s "$SESSION" "$run_dir/run_session.json"; then status=passed; else gate=3; fi
    else
      cp "$SESSION" "$run_dir/run_session.json"
      status=passed
    fi
  fi
  write_method_record "$method" formal "$status" "$preflight" "$compute" "$gate"
  if [[ "$status" == passed ]]; then
    echo "PASS method=$method seed=$SEED"
  else
    FAILED=$((FAILED + 1))
    echo "FAIL method=$method preflight_exit=$preflight compute_exit=$compute gate_exit=$gate; continuing independent methods" >&2
  fi
done

if [[ "$FAILED" -gt 0 ]]; then echo "AIME_METHOD_ATTEMPTS_FAILED count=$FAILED pair=$PAIR" >&2; exit 1; fi
echo "AIME_PAIR_COMPLETE pair=$PAIR seed=$SEED"
