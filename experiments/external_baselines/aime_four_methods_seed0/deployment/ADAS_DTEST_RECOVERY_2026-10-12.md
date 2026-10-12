# ADAS best-candidate D_test recovery, 2026-10-12

## Authorization and status

The user requested: “按照当前最佳的方案跑一下Dtest吧，然后暂时放下门槛校验”.
This authorizes a separate ADAS-only continuation that reuses the best retained
D_select candidate, bypasses the original 5% selection/test quality gates for
execution, and records every observed failure and truncation rate. The run is
**prepared but not launched** in this delivery.

This is a gate-waiver stage-recovery variant and remains `NOT_FORMAL_PASS`.
The original ADAS failure and original 5% rule remain immutable in the source
run. Search and D_select are not repeated.

## Frozen candidate

The original source run was on the execution host at:

`/data/jxc/RPAS-AIME-Parallel-Seed0-20261010/outputs/aime_parallel_seed0_20261010/adas_attempt_v2/adas_gdesigner/adas/seed_0`

The required immutable source artifacts are now also delivered inside this
repository at:

`results/aime_four_methods_seed0_20261012/adas/recovery_source/`

Use that repository-local directory as `--source-run` after cloning. It has
exactly the three files required by the recovery entry; the original absolute
path above is retained only as provenance.

The retained search has 30 generations and 26 archive candidates. The highest
scoring candidate is `archive_21`, named **Constraint-Verified Iteration**:

- selection score: 21/30 (70%);
- sample failure fraction: 16.67%;
- request failure fraction: 0%;
- completion truncation fraction: 12.89%;
- candidate code SHA256:
  `90718551c72c3081a9e7bd9d44ef1bc525d78380a6a5662521740deceacdfb77`.

Those rates are why the original run stopped before D_test. The recovery uses
the exact archive code and verifies its hash before any model request.

## Recovery contract

Entry point: `experiments/recover_adas_dtest.py`.

The read-only inspection checks the failed source manifest, frozen AIME IDs and
hashes, clean pinned ADAS checkout, complete archive, candidate index/code hash,
and absence of any original D_test access. Execution requires
`--allow-selection-gate-waiver`, a fresh output directory, a permanent source
claim, and the fixed seed-0/16K/6144-token/8-request protocol.

The entry point copies only the retained manifest, selection audit and search
archive; creates a new candidate lock; evaluates the same `archive_21` code on
AIME2025 and AIME2026; writes each request and sample durably; saves
`phase_checkpoint.json`, per-year point files, RNG snapshots and
`gate_waiver_audit.json`; and verifies source/upstream hashes after execution.

The checkpoint and ledger preserve completed stage evidence for audit-safe
continuation planning. An interrupted request is recorded as unknown and is
never converted into a successful sample. The permanent source claim prevents
an accidental duplicate run; any restart must inspect the saved phase/ledger
evidence and use an explicitly reviewed fresh recovery session.

## Planned host launch (not executed in this delivery)

The prepared plan uses GPU 8 and port 18183, which were observed free while
preparing this document. The launcher performs its own occupancy and port
checks immediately before starting and stops if another process claims the card.
The plan contains only ADAS with `allow_selection_gate_waiver: true`. Before
launching from a clone, replace its `source_run`, `package_root`, `env_file`,
`output_root` and `launch_dir` with paths on that host:

```bash
bash scripts/launch_recovery_detached.sh \
  --plan /data/jxc/tmp/rpas/adas_dtest_recovery_20261012/plan.json
```

Planned output:
`/data/jxc/RPAS-AIME-Recovery-Seed0-20261011/outputs/adas_dtest_recovery_seed0_20261012/adas/seed_0`

No model service or ADAS process was started for this delivery.
