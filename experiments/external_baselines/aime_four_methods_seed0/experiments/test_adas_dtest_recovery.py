"""Offline ADAS best-candidate D_test recovery checks."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import shutil

import pytest

import native_aime_formal as formal
import native_external_methods as external
import recover_adas_dtest as recovery

PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = Path("/data/jxc/RPAS-AIME-Parallel-Seed0-20261010/outputs/aime_parallel_seed0_20261010/") / (
    "adas_attempt_v2/adas_gdesigner/adas/seed_0"
)


def test_real_inspection_selects_archive_21_without_opening_test(tmp_path):
    if not SOURCE.is_dir() or not (PACKAGE / "upstream/ADAS").is_dir():
        pytest.skip("retained ADAS source is unavailable")
    args = SimpleNamespace(method="adas", seed=0, source_run=SOURCE,
        run_dir=tmp_path / "fresh", data_dir=PACKAGE / "data", baseline_root=PACKAGE / "upstream",
        endpoint="http://offline.invalid/v1", inspect_only=True,
        allow_selection_gate_waiver=True, adas_generations=30)
    result = recovery.inspect_source(args)
    assert result["candidate"]["candidate_id"] == "archive_21"
    assert result["candidate"]["correct"] == 21
    assert result["remaining_test_splits"] == ["aime_2025.jsonl", "aime_2026.jsonl"]
    assert result["formal_acceptance"] == "NOT_FORMAL_PASS"
    assert not args.run_dir.exists()


class _OfflineRuntime:
    def __init__(self):
        self.run_id = "offline-adas"
        self.request_timeout_s = 600.0
        self.usage = {"totals": external.empty_usage(), "phases": {}}
        self.sync = SimpleNamespace(close=lambda: None)
        self.async_client = SimpleNamespace(close=lambda: None)

    def summary(self):
        return self.usage


def test_native_recovery_skips_search_and_runs_both_test_splits(tmp_path, monkeypatch):
    if not SOURCE.is_dir() or not (PACKAGE / "upstream/ADAS").is_dir():
        pytest.skip("retained ADAS source is unavailable")
    rows = external.read_task_rows("aime", PACKAGE / "data", 0, None)
    audit = json.loads((SOURCE / "selection_audit.json").read_text())
    selected = next(row for row in audit["candidate_rows"] if row["candidate_id"] == "archive_21")
    retained = tmp_path / "retained_source"
    shutil.copytree(SOURCE / "native_search", retained / "native_search")
    args = SimpleNamespace(method="adas", seed=0, run_dir=tmp_path,
        data_dir=PACKAGE / "data", endpoint="http://offline.invalid/v1", adas_generations=30,
        upstream_provenance=external.verify_upstream_source("adas", PACKAGE / "upstream/ADAS"))
    runtime = _OfflineRuntime(); samples=[]; completed=[]

    class Info:
        def __init__(self, content): self.content = content

    def fake_compile(_code, _agent):
        def forward(*_args): return Info("1")
        return forward

    def fake_archive_path():
        return retained / "native_search/aime_0_run_archive.json"

    with patch.object(external, "compile_candidate", fake_compile):
        result = external.run_adas("aime", rows, args, runtime.run_id,
            runtime, PACKAGE / "upstream/ADAS", test_recovery={
                "archive_path": fake_archive_path(), "candidate_index": 21,
                "candidate_code_sha256": selected["code_sha256"], "selection_row": selected,
                "before_tests": lambda _: None,
                "sample_completed": lambda phase, row: samples.append((phase, row)),
                "checkpoint_completed": lambda name, point, observed: completed.append(name),
            })
    assert set(result["test"]) == {"aime_2025", "aime_2026"}
    assert completed == ["aime_2025", "aime_2026"]
    assert len(samples) == 60
    assert result["recovery_metadata"]["gate_waiver"]
    assert result["recovery_metadata"]["formal_acceptance"] == "NOT_FORMAL_PASS"
    assert not (tmp_path / "native_search/aime_0_run_archive.json").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
