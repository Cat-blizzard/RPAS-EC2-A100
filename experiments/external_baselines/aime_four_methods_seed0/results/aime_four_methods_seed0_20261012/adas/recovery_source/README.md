# ADAS recovery source artifact set

This directory is the immutable pre-D_test source artifact set for the ADAS
best-candidate continuation. It is copied byte-for-byte from the original
failed run and contains exactly the three files required by
`experiments/recover_adas_dtest.py`:

- `run_manifest.json`
- `selection_audit.json`
- `native_search/aime_0_run_archive.json`

The original run stopped after D_select because no candidate met the 5% quality
gate. The recovery selects `archive_21` from this archive and records the
user-authorized gate waiver. Upstream ADAS code, frozen AIME data, model
weights and runtime environments remain separate inputs and are not duplicated
here.
