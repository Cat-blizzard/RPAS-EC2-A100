# Four external-method AIME runner (seed 0)

The [G-Designer quality-gate waiver continuation](deployment/GDESIGNER_GATE_WAIVER_2026-10-12.md)
reuses its retained controller and selection audit, and evaluates only the
previously unopened 2025/2026 test splits. The user explicitly waived this
method's selection/test quality thresholds. It is a stage-reseed, quality-gate
waiver variant, remains `NOT_FORMAL_PASS`, and preserves the original rejection.

The [2026-10-11 test-stage recovery](deployment/STAGE_RECOVERY_2026-10-11.md)
retains completed 2025 evaluations and can supplement only the unopened 2026
test from a frozen failed AFlow/MaAS attempt. Its outputs are supplementary,
with unknown historical usage and original failure gates retained. The user
initially authorized AFlow, then explicitly accepted a separate MaAS stage
continuation with seed 0 reset. Both remain supplementary results.

The [single RTX 5880 Ada 48 GB host record](deployment/SINGLE_48G_2026-10-01.md)
documents the prepared environments, model verification, runtime limits,
audit fixes, synthetic diagnostics, and four-method orchestrator used by the
2026-10-01 host run. This source variant retains the package's search budgets
and quality gates and discloses G-Designer's role-graph/numerical adapters.
Read that record for the host launch and the preserved rejected AFlow attempt;
the pair and single-method commands below remain available.

This folder contains **only external baseline adapters** for AFlow, MaAS, ADAS,
and G-Designer. It does not contain or invoke the RPAS method. The wrappers
preserve the pinned upstream search/controller logic where possible and adapt
the frozen AIME rows, OpenAI-compatible model transport, shared answer parser,
telemetry, and post-run checks. Therefore report these as *external methods
under a shared AIME adapter protocol*, not as byte-for-byte upstream benchmark
reproductions.

## What is and is not included

- Included: runner/adapters, scoring and quality gates, tests, exact source
  commit pins for all four method repositories, and the authorized frozen AIME
  data files used by this seed-0 pilot.
- Data provenance: AIMO validation plus its canonical `search_60` and
  `select_30`, and the canonical AIME2025/AIME2026 `test_30` files are copied
  from `JiangyueAnn/RPAS` revision
  `e12f58823be5f91a32f05f9af4d36e54838ffe59`. The exact file hashes and row
  counts are in `data/frozen_aime_manifest.json`; runtime verifies the
  manifest and every file before use. `.gitattributes` fixes JSONL line endings
  so these hashes remain portable across Windows and Linux.
- Not included: model weights, tokenizer weights, MiniLM weights, API
  credentials, or complete formal result bundles. Deployment records include
  synthetic diagnostic predictions and execution/rejection audit summaries.
  Publishing the frozen test questions/answers is intentional and was
  authorized by the data owner; users should still avoid using `D_test` for
  tuning or selection.
- Required data directory: `data/`, containing the three raw source JSONL
  files plus the canonical split subdirectories. Search/select load the exact
  frozen 60/30 files (the runner no longer re-shuffles the 90-row source).
  Test files are existence-checked by the launcher; the runner reads and
  verifies their source and canonical split contents only after writing the
  durable `selection_frozen.json` candidate lock.

## Frozen one-seed protocol

| Setting | Value |
|---|---|
| Search seed | `0` (single-seed pilot only; not a 3-seed result) |
| Split seed | `2026` |
| Validation and test split | Canonical frozen `D_search=60` and `D_select=30` files from the pinned manifest; AIME2025 and AIME2026 each have a separate canonical 30-row `D_test` |
| Task model | `Qwen/Qwen3.5-9B` |
| Server context / method output cap | `16384 / 6144` tokens (Final V3 16K profile) |
| Request concurrency | `8` |
| Shared reported score | `0` unparseable, `1` parseable but wrong, `2` exact match |
| Pair execution | Two methods in the selected pair, serially on one visible GPU |

Every selected method requires a fresh output directory and service-attempt
record. Existing complete or partial attempts are refused before any checker
can rewrite them. Independent methods continue after another method fails;
only four verified formal passes produce an accepted complete aggregate.
Freeze/access manifests bind the exact validation
IDs, method-specific selection, baseline provenance, test-file hashes/IDs,
question fingerprints, and timestamps. These checks reduce accidental leakage;
they are not an OS-level audit against a malicious process reading files.

## Install upstream source pins

Run once from this directory:

```bash
bash scripts/setup_upstreams.sh
```

The script clones and verifies these exact commits:

| Method | Upstream | Commit |
|---|---|---|
| AFlow | `FoundationAgents/AFlow` | `3f457218fc716093fe53f6df8a5d5e6379d66346` |
| MaAS | `bingreeky/MaAS` | `987f3c1bc9a96e844fe090db3791446e3ef0f5c7` |
| ADAS | `ShengranHu/ADAS` | `2702bee8fefda42255efc5be9f60e3bd3db96ae4` |
| G-Designer | `yanweiyue/GDesigner` | `a6efcfa3b40bb4d9cbf46f883a95d62020bd8251` |

Before any model request, each wrapper fails closed unless the checkout has the
expected upstream URL, exact commit, and a clean worktree. Upstream source is
not copied into this repository. The setup script does not install Python
dependencies: the four upstream dependency sets and a current vLLM/PyTorch
stack are not a verified single environment. In particular, MaAS’s upstream
requirements pin an old CUDA/PyTorch stack. Use separate method environments
(`AIME_AFLOW_PYTHON`, `AIME_MAAS_PYTHON`, `AIME_ADAS_PYTHON`,
`AIME_GDESIGNER_PYTHON`) and a separate model-serving environment where needed;
do not blindly install upstream requirements over vLLM’s environment.

## Run one pair on one GPU

Prepare a tested Linux/CUDA environment and local model/embedding weights. Set
the method-specific Python paths if the methods cannot share one environment:

```bash
export AIME_MODEL_PATH=/local/models/Qwen3.5-9B
export AIME_DATA_DIR="$PWD/data"
export AIME_MINILM_PATH=/local/models/all-MiniLM-L6-v2
export AIME_VLLM_BIN=/path/to/vllm-environment/bin/vllm
export AIME_PYTHON=/path/to/vllm-environment/bin/python
export AIME_AFLOW_PYTHON=/path/to/aflow-environment/bin/python
export AIME_MAAS_PYTHON=/path/to/maas-environment/bin/python
export AIME_ADAS_PYTHON=/path/to/adas-environment/bin/python
export AIME_GDESIGNER_PYTHON=/path/to/gdesigner-environment/bin/python
export CUDA_VISIBLE_DEVICES=0
export AIME_MODEL_VERIFICATION_REPORT="$PWD/logs/local_model_verification_new.json"
uv run --no-project --python "$AIME_PYTHON" python scripts/verify_local_models.py \
  --qwen "$AIME_MODEL_PATH" --embedding "$AIME_MINILM_PATH" \
  --report "$AIME_MODEL_VERIFICATION_REPORT"
export AIME_OUTPUT_DIR="$PWD/outputs/aime_seed0_pair1"
bash scripts/run_pair.sh aflow_maas
```

For the other pair, select a new output directory and run:

```bash
export AIME_OUTPUT_DIR="$PWD/outputs/aime_seed0_pair2"
bash scripts/run_pair.sh adas_gdesigner
```

For one method, use `bash scripts/run_pair.sh --method maas` with a fresh
`AIME_OUTPUT_DIR`. For all four methods with fresh synthetic diagnostics:

```bash
export AIME_OUTPUT_ROOT="$PWD/outputs/aime_four_methods_seed0_new"
bash scripts/run_single48g_all.sh all --method-order maas,adas,gdesigner,aflow
```

The all-method launcher assigns each formal method its own service session
and records failures without accepting an incomplete comparison.

For long runs, use the detached launcher after exporting the paths above:

```bash
export AIME_OUTPUT_ROOT="$PWD/outputs/aime_four_methods_seed0_new"
export AIME_LAUNCH_DIR="$PWD/logs/aime_four_methods_seed0_new"
bash scripts/launch_detached.sh all --method-order maas,adas,gdesigner,aflow
```

It creates an independent session with stdin at `/dev/null`, stdout/stderr
directly in `AIME_LAUNCH_DIR/run.log`, and launch/PID/exit records in that
directory. `completion.json` records the launcher exit; formal acceptance
still requires the method quality gates and `attempt_summary.json`. Use
`diagnostics` in place of `all` for the synthetic checks only. Both the launch
directory and the output root must be fresh. Do not pipe long runs through
the terminal or a `tee` process tied to the terminal's lifetime.

## Run independent methods on separate GPUs

After exporting the model and Python paths above, use a separate GPU, port,
output root, launch directory, cache, and temporary directory for each method:

```bash
export CUDA_VISIBLE_DEVICES=2 AIME_PORT=18181
export AIME_OUTPUT_ROOT=/absolute/path/adas_attempt
export AIME_LAUNCH_DIR=/absolute/path/adas_launch
export AIME_CACHE_DIR=/absolute/path/adas_cache
export AIME_TEMP_DIR=/absolute/path/adas_temp
export AIME_DIAGNOSTIC_DIR=/absolute/path/adas_diagnostics
bash scripts/launch_single_detached.sh --method adas
```

The same command accepts `aflow` and `gdesigner`. Each worker first runs the
method's synthetic diagnostic on its assigned card, verifies the diagnostic
record, then starts a fresh formal service session. The 16K context, 6144-token
output cap, seed, data, search budgets, and quality gates remain unchanged.
The disposable endpoint smoke waits up to 300 seconds for first-request kernel
compilation; `AIME_ENDPOINT_SMOKE_TIMEOUT_S` overrides this startup-only limit.
Read `stage.txt`, `run.log`, and `completion.json` in the launch directory;
launch records alone establish no accepted result.

`scripts/supervise_parallel.py --plan /absolute/path/parallel_plan.json`
supports the documented [2026-10-10 queue handoff](deployment/PARALLEL_2026-10-10.md).
It observes the independent workers, waits for the retained MaAS task and its
service to exit, retires the paused old dispatcher, and starts G-Designer.
Only four completed formal passes are copied into a fresh aggregate directory;
source results are retained with a file checksum mapping. A cancelled old
dispatcher is recorded as superseded scheduling, separate from MaAS's outcome.

The Final V3 runner starts one local vLLM server (`TP=1`, context 16384, max-num-seqs
8, GPU utilization 0.92), sends a disposable model smoke request, then runs
the two selected methods one at a time. Its smoke request is outside the
experiment telemetry. Logs are under `logs/`; result artifacts remain under
the chosen output directory. The script does not submit to Slurm and offers no
wall-time guarantee. **RTX PRO 6000 hardware/runtime compatibility has not
been tested.** Establish a real model-load and request smoke test on that
machine before starting the long run.

## Local checks

```bash
uv run --no-project --python /path/to/test-environment/bin/python python -m compileall -q experiments
PYTHONPATH=experiments uv run --no-project --python /path/to/test-environment/bin/python python -m pytest -q experiments
```

The authoring-machine checks are CPU/offline tests only. Real-model synthetic
diagnostics on the recorded 48 GB host are documented separately in the host
record; they establish execution viability, not held-out benchmark quality.
Formal methods still need their full quality gates. The 1-seed output is a
pilot; it cannot fill a three-seed mean±standard-deviation table.

See [SCIR reference status](deployment/SCIR_REFERENCE_STATUS.md) for the
distinction between historical SCIR gate evidence, runtime diagnostics, and
a completed formal result.
