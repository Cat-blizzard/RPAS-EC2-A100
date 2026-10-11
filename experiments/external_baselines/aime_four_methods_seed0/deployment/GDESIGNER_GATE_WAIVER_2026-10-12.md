# G-Designer frozen-controller continuation, 2026-10-12

## Authorization and classification

The user explicitly requested: “G-Designer不要门槛，续跑吧”. This supersedes
the earlier instruction to leave G-Designer stopped at its original gate.
It authorizes a separate G-Designer-only continuation, with selection and test
sample/request/truncation quality gates waived. The reported rates, original
5% threshold, original failure, and raw predictions remain in the audit.

This continuation is a **quality-gate waiver stage-recovery variant**, always
`NOT_FORMAL_PASS`. It does not qualify as a formal pass under the original
protocol, even if all newly observed quality rates happen to meet 5%.
The AFlow, MaAS and ADAS gate rules are unchanged.

## Preserved inputs

Original run:

`/data/jxc/RPAS-AIME-Parallel-Seed0-20261010/outputs/aime_parallel_seed0_20261010/gdesigner_attempt/adas_gdesigner/gdesigner/seed_0`

The run has only `run_manifest.json`, `selection_audit.json`, and
`gdesigner_controller.pt`. It failed before creating a test-selection lock or
opening either test split. Selection accuracy was 22/30; 3/30 outputs were
unparseable, and 42/150 calls ended at the output cap. Its recorded search/select
runtime aggregates contain 490 calls and 3,512,680 tokens; they have no original
per-request ledger and are retained as aggregates rather than reconstructed.

Frozen controller SHA256:

`7c1e384e0cca4e92ca845fb350f228bf495e2fdcdbab9f98e2511cd2b04b0fa9`

The checkpoint stores GCN/MLP state, role edges, adapters, and module training
modes. It does **not** store Python/NumPy/Torch RNG or historical graph/node
shortuuid IDs. This continuation resets seed 0 and records newly constructed
native graph/node IDs. Native stochastic topology sampling remains enabled;
the original process trajectory and prompt node identities cannot be exactly
restored. GCN eval mode and MLP train mode are restored from the checkpoint.

## Recovery entry and safeguards

`experiments/recover_gdesigner_tests.py --inspect-only` verifies the old run,
parser, validation IDs/hashes, original runner identity, pinned clean upstream,
all 30 selection rows, aggregate telemetry, and controller hash. It does not
open either blind test split or make a model request.

Execution requires `--allow-selection-gate-waiver`, a fresh run directory and
an exclusive permanent claim beside the source run. No automatic second
recovery is permitted after a claim, including a failed recovery. The three
original artifacts are retained byte for byte and copied into the new run.

The test entry uses the same native `Graph.arun` inference, shared prompt,
role/numerical adapters, frozen data and answer parser. It strictly loads both
controller modules and verifies state, mode, role edges and checkpoint identity
before/after inference. It skips training, optimizer updates, and selection
inference. A new durable candidate lock is written before each 30-row 2025/2026
test is loaded, and E reuses Q without a second evaluation.

Unchanged controls: Qwen3.5-9B; 16,384 context; 6,144 output cap; temperature 0;
top-p 1; thinking disabled; request concurrency 8; native sample batch 4;
one graph round; three native tries; graph node max_time 180 seconds;
external request timeout 600 seconds; local MiniLM embeddings on CPU.

New requests and each completed sample are fsynced to `request_ledger.jsonl`.
Each year's point, returned-response telemetry, and RNG state are persisted
before final scoring/audit checks. Numerical corruption, controller changes,
wrong split identity, duplicate execution and accounting inconsistencies still
stop execution. Failure-rate and truncation measurements are retained without
blocking the second authorized test split.

## Host launch

The launch plan selects only G-Designer, `allow_selection_gate_waiver: true`,
GPU 3 and port 18180. GPU/port locks and two occupancy checks protect the launch.
Model assets use the already verified fixed report from 2026-10-11, with file
existence/sizes rechecked. A disposable 16-token warmup precedes test execution.
The detached worker records source commit, actual recovery-entry SHA, plan,
authorization, model report, service session and exact process identities.

Runtime plan:
`/data/jxc/tmp/rpas/gdesigner_stage_recovery_20261012/plan.json`

Runtime log:
`logs/recovery_gdesigner_20261012/recover_gdesigner.log`

Output:
`/data/jxc/RPAS-AIME-Recovery-Seed0-20261011/outputs/gdesigner_stage_recovery_seed0_20261012/gdesigner/seed_0`

The running ADAS worker on GPU 2 is independent. Completed AFlow/MaAS
continuations are retained and are not queued again. No final accuracy or
completion is asserted by this deployment preparation record; consult the
runtime manifests and supplementary result after execution.

## Pre-launch verification

- Recovery/provenance/scorer/blind-access/launcher regressions: **77 passed**.
- Existing external-adapter regressions: **21 passed, 18 subtests passed**.
- Frozen-controller native-environment offline tests: **19 passed**, including
  normal/exceptional asynchronous client cleanup in the inference event loop.
- Real native Graph/checkpoint plus the complete recovery orchestration:
  60 synthetic sample completions, no real model requests, both phase points
  durable, source unchanged, and result classification `NOT_FORMAL_PASS`.
- Real ModelRuntime/tokenizer plus synthetic SDK responses and the durable
  ledger: 60 synthetic calls and 180 synthetic tokens matched runtime counters;
  no network/model requests. Async client cleanup completed in its original loop.
- The recovery module's standalone CPU durable-write self-test passed.

Synthetic checks establish engineering behavior only; none is an AIME model
accuracy result or evidence that the original quality gates have passed.
