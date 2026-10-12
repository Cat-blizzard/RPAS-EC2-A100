# AIME four-method seed-0 results bundle

This directory contains the public, auditable result artifacts from the AIME seed-0 runs on 2026-10-12. It includes the three supplementary test continuations (AFlow, MaAS, G-Designer) and the ADAS attempt that stopped before D_test.

## Results

| Method | AIME 2025 | AIME 2026 | Classification |
|---|---:|---:|---|
| AFlow | 19/30 | 25/30 | supplementary, `NOT_FORMAL_PASS` |
| MaAS | 15/30 | 20/30 | stage-reseed variant, `NOT_FORMAL_PASS` |
| G-Designer | 19/30 | 21/30 | quality-gate-waiver variant, `NOT_FORMAL_PASS` |
| ADAS | no D_test | no D_test | D_select gate failed |

The exact measurements, row-level outputs, tokens and failure markers are in each method directory. `RESULTS_INDEX.json` is the machine-readable index.

## Interpretation limits

AFlow and MaAS reuse frozen search/select artifacts and have incomplete historical usage accounting; MaAS also resets seed 0 because the original RNG state was unavailable. G-Designer explicitly waives the selection/test quality gates at the user's direction, and reconstructs new graph/node identities because those identities and RNG were absent from its checkpoint. These are adapter/recovery variants, not formal upstream reproductions or formal passes.

ADAS completed its 30-generation search and evaluated 26 candidates. Its best D_select candidate scored 21/30, but its failure and truncation rates exceeded the original 5% gate, so D_test was never opened.

## Reproducibility

The bundle keeps the frozen controller/candidate artifacts, manifests, selection locks/access audits, per-request ledgers, per-split CSV/log files, and launch completion records. Model weights, embeddings, upstream checkouts, Python caches, temporary GPU/server logs, and credentials are intentionally excluded.

See the adjacent deployment records and the package README for the protocol and recovery contracts.
