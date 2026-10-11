"""Offline G-Designer recovery: waived quality gates, immutable source and no repeats."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import native_aime_formal as native
import native_external_methods as external
import recover_aime_tests as common
import recover_gdesigner_tests as recovery


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/data/jxc/RPAS-AIME-Parallel-Seed0-20261010/outputs/"
    "aime_parallel_seed0_20261010/gdesigner_attempt/adas_gdesigner/gdesigner/seed_0"
)
SOURCE_FILES = ("run_manifest.json", "selection_audit.json", "gdesigner_controller.pt")
TEST_SPLITS = ["aime_2025.jsonl", "aime_2026.jsonl"]


def _args(source: Path, output: Path, *, inspect_only: bool = False,
          authorized: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        method="gdesigner", seed=0, source_run=source, run_dir=output,
        data_dir=PACKAGE / "data", baseline_root=PACKAGE / "upstream",
        endpoint="http://offline.invalid/v1", inspect_only=inspect_only,
        allow_selection_gate_waiver=authorized,
    )


def _snapshot(source: Path) -> dict[str, bytes]:
    return {str(path.relative_to(source)): path.read_bytes()
            for path in sorted(source.rglob("*")) if path.is_file()}


def _edit_json(path: Path, change) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


@pytest.fixture
def source_args(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    """Use a tiny byte-identical host evidence copy, never the retained run itself."""
    if not SOURCE.is_dir() or not (PACKAGE / "upstream/GDesigner").is_dir():
        pytest.skip("retained G-Designer evidence and pinned source required")
    source = tmp_path / "source" / "seed_0"
    source.mkdir(parents=True)
    for name in SOURCE_FILES:
        shutil.copyfile(SOURCE / name, source / name)
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "0")
    monkeypatch.delenv("AIME_EXTERNAL_REQUEST_TIMEOUT_S", raising=False)
    return _args(source, tmp_path / "new_run")


def _reject_model_and_execute(monkeypatch) -> tuple[Mock, Mock]:
    model = Mock(side_effect=AssertionError("offline inspection must not call a model"))
    execute = Mock(side_effect=AssertionError("offline inspection must not execute D_test"))
    monkeypatch.setattr(native, "verify_served_model", model)
    monkeypatch.setattr(recovery, "execute_tests", execute)
    return model, execute


def test_inspect_is_read_only_and_does_not_read_blind_test_content(source_args, monkeypatch):
    args = source_args
    args.inspect_only = True
    before = _snapshot(args.source_run)
    model, execute = _reject_model_and_execute(monkeypatch)
    original_open = Path.open

    def guarded_open(path: Path, mode="r", *rest, **kwargs):
        if any(flag in mode for flag in "wax+"):
            raise AssertionError(f"inspect attempted a write: {path}")
        if path.suffix == ".jsonl" and any("aime_2025" in part or "aime_2026" in part
                                            for part in path.parts):
            raise AssertionError(f"inspect opened blind D_test content: {path}")
        return original_open(path, mode, *rest, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", guarded_open)
        result = recovery.recover(args)

    assert result["remaining_test_splits"] == TEST_SPLITS
    assert result["selection_gate_waiver"]
    assert result["formal_acceptance"] == "NOT_FORMAL_PASS"
    assert before == _snapshot(args.source_run)
    assert not args.run_dir.exists()
    assert not common.source_claim_path(args.source_run).exists()
    model.assert_not_called()
    execute.assert_not_called()


def test_missing_explicit_gate_waiver_cannot_write_or_compute(source_args, monkeypatch):
    args = source_args
    args.allow_selection_gate_waiver = False
    before = _snapshot(args.source_run)
    model, execute = _reject_model_and_execute(monkeypatch)
    with pytest.raises((RuntimeError, ValueError), match="authoriz|waiver|allow-selection"):
        recovery.recover(args)
    assert before == _snapshot(args.source_run)
    assert not args.run_dir.exists()
    assert not common.source_claim_path(args.source_run).exists()
    model.assert_not_called()
    execute.assert_not_called()


def test_inspect_refuses_existing_output(source_args):
    source_args.run_dir.mkdir()
    with pytest.raises(FileExistsError, match="fresh|exist"):
        recovery.inspect_source(source_args)


@pytest.mark.parametrize("name", [
    "dtest_access_manifest.json", "selection_frozen.json", "native_result.json",
    "unexpected.json", "test_2025.csv",
])
def test_inspect_refuses_prior_test_access_or_unknown_source_artifacts(source_args, name):
    (source_args.source_run / name).write_text("{}\n", encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError)):
        recovery.inspect_source(source_args)


def test_inspect_refuses_modified_controller(source_args):
    path = source_args.source_run / "gdesigner_controller.pt"
    path.write_bytes(path.read_bytes() + b"altered")
    with pytest.raises(RuntimeError, match="controller|checkpoint|SHA|sha|hash"):
        recovery.inspect_source(source_args)


@pytest.mark.parametrize("field,value", [
    ("answer_parser_sha256", "0" * 64),
    ("protocol_version", "other_protocol"),
    ("seed", 1),
    ("context_limit", 8192),
    ("status", "complete"),
])
def test_inspect_refuses_changed_protocol_or_completed_source(source_args, field, value):
    _edit_json(source_args.source_run / "run_manifest.json",
               lambda result: result.update({field: value}))
    with pytest.raises((RuntimeError, ValueError)):
        recovery.inspect_source(source_args)


def test_inspect_refuses_validation_identity_change(source_args):
    def change(value):
        value["split_row_ids"]["select"][0] = "changed_validation_id"
    _edit_json(source_args.source_run / "run_manifest.json", change)
    with pytest.raises((RuntimeError, ValueError)):
        recovery.inspect_source(source_args)


def test_inspect_rechecks_preserved_raw_selection_scoring(source_args):
    def change(value):
        row = next(item for item in value["selection_metrics"]["rows"] if item["correct"])
        row["raw_output"] = "No final numerical answer was produced."
    _edit_json(source_args.source_run / "selection_audit.json", change)
    with pytest.raises((RuntimeError, ValueError)):
        recovery.inspect_source(source_args)


def test_inspect_rechecks_selection_row_id_alignment(source_args):
    def change(value):
        value["selection_metrics"]["rows"][0]["id"] = "other_id"
    _edit_json(source_args.source_run / "selection_audit.json", change)
    with pytest.raises((RuntimeError, ValueError)):
        recovery.inspect_source(source_args)


def test_preserved_source_check_detects_postinspection_mutation(source_args):
    inspection = recovery.inspect_source(source_args)
    _edit_json(source_args.source_run / "selection_audit.json",
               lambda result: result.update(extra="changed after inspection"))
    with pytest.raises(RuntimeError, match="changed|preserv|artifact|hash"):
        recovery.verify_preserved_source(inspection)


def _point(year: str, *, failed: bool = False) -> dict:
    rows = [{
        "id": f"synthetic:{year}:{index}", "prediction": "" if failed else "1",
        "gold": "1", "parser_valid": not failed, "correct": not failed,
        "score_0_1_2": 0 if failed else 2, "sample_failed": failed,
        "raw_output": "unparseable response" if failed else "FINAL ANSWER: 1",
    } for index in range(30)]
    return {
        "num_examples": 30, "correct": 0 if failed else 30,
        "accuracy": 0.0 if failed else 1.0, "parser_valid_rate": 0.0 if failed else 1.0,
        "mean_score_0_1_2": 0.0 if failed else 2.0,
        "failure_fraction": 1.0 if failed else 0.0,
        "rows": rows, "calls": 150, "failed_calls": 15 if failed else 0,
        "completion_truncated": 42 if failed else 0,
        "request_failure_fraction": 0.1 if failed else 0.0,
        "truncation_fraction": 0.28 if failed else 0.0,
        "total_tokens": 1200, "tokens_per_example": 40.0,
        "operating_point": "Q", "selected_candidate": "trained_gdesigner_graph",
    }


def test_waived_quality_failure_does_not_discard_thirty_test_rows():
    recovery.validate_test_point(_point("aime_2025", failed=True))


def test_waiver_does_not_accept_incomplete_test_split():
    point = _point("aime_2025")
    point["num_examples"] = 29
    point["rows"] = point["rows"][:29]
    with pytest.raises((RuntimeError, ValueError)):
        recovery.validate_test_point(point)


def test_waiver_does_not_accept_duplicate_test_identity():
    point = _point("aime_2025")
    point["rows"][1]["id"] = point["rows"][0]["id"]
    with pytest.raises((RuntimeError, ValueError)):
        recovery.validate_test_point(point)


def test_waiver_keeps_the_shared_raw_output_scorer():
    point = _point("aime_2025")
    point["rows"][0]["raw_output"] = "FINAL ANSWER: 2"
    with pytest.raises((RuntimeError, ValueError)):
        recovery.validate_test_point(point)


def test_recovery_only_runs_both_unopened_years_and_retains_original_failure(source_args, monkeypatch, tmp_path):
    args = source_args
    before = _snapshot(args.source_run)
    test = {}
    for year in ("aime_2025", "aime_2026"):
        point = _point(year, failed=True)
        test[year] = {"Q": point, "E": {**point, "operating_point": "E", "same_candidate_as": "Q"}}
    execute = Mock(return_value={
        "test": test, "telemetry": {"fake": True}, "native_method_config": {},
        "recovery_metadata": {},
    })
    model = Mock()
    monkeypatch.setattr(native, "verify_served_model", model)
    monkeypatch.setattr(recovery, "execute_tests", execute)
    result = recovery.recover(args)

    execute.assert_called_once()
    called_args, inspection, ledger = execute.call_args.args
    assert called_args.run_dir == args.run_dir
    assert inspection["remaining_test_splits"] == TEST_SPLITS
    assert ledger.path == args.run_dir / "request_ledger.jsonl"
    assert set(result["test"]) == {"aime_2025", "aime_2026"}
    assert all(result["test"][year]["Q"]["num_examples"] == 30 for year in result["test"])
    assert result["formal_acceptance"] == "NOT_FORMAL_PASS"
    assert result["gate_waiver"]["original_gate_pass"] is False
    assert before == _snapshot(args.source_run)
    assert json.loads((args.source_run / "run_manifest.json").read_text())["status"] == "failed"
    assert not (args.run_dir / "native_result.json").exists()
    assert not (args.run_dir / "quality_gate.json").exists()
    assert (args.run_dir / "supplementary_result.json").exists()
    assert common.source_claim_path(args.source_run).exists()
    model.assert_called_once_with(args.endpoint)

    second = _args(args.source_run, tmp_path / "second_recovery")
    with pytest.raises(RuntimeError, match="claimed|claim|repeat"):
        recovery.recover(second)
    execute.assert_called_once()
    model.assert_called_once()


def test_phase_callback_persists_raw_work_before_late_validation_failure(source_args, monkeypatch):
    """Expensive phase outputs survive a scorer failure and the second year never starts."""
    args = source_args
    inspection = recovery.inspect_source(args)
    args.run_dir.mkdir()
    shutil.copyfile(args.source_run / "gdesigner_controller.pt", args.run_dir / "gdesigner_controller.pt")
    ledger = common.DurableLedger(args.run_dir / "request_ledger.jsonl")
    closes = []
    completed_phases = []

    class OfflineRuntime:
        def __init__(self, *_):
            self.run_id = "synthetic-offline"
            self.usage = {"totals": external.empty_usage(), "phases": {}}
            self.sync = SimpleNamespace(close=lambda: closes.append("sync"))
            async def close_async():
                closes.append("async")
            self.async_client = SimpleNamespace(close=close_async)

        def summary(self):
            return self.usage

    def save_synthetic_rng(path: Path):
        path.write_bytes(b"synthetic offline RNG snapshot")

    def fail_after_persist(point):
        completed_phases.append("validated:first")
        raise RuntimeError("synthetic late scorer failure")

    def offline_adapter(dataset, rows, called_args, run_id, runtime, source, *, test_recovery):
        assert dataset == "aime"
        assert set(rows) == {"search", "select"}
        assert test_recovery["checkpoint_sha256"] == inspection["frozen_selection"]["controller_sha256"]
        test_recovery["before_tests"]({"graph_id": "synthetic", "nodes": []})
        point = _point("aime_2025", failed=True)
        runtime.usage["phases"]["test:aime_2025"] = external.empty_usage()
        for row in point["rows"]:
            test_recovery["sample_completed"]("test:aime_2025", row)
        test_recovery["checkpoint_completed"]("aime_2025", point, runtime)
        raise AssertionError("late validation failure must stop before the second year")

    monkeypatch.setattr(external, "ModelRuntime", OfflineRuntime)
    monkeypatch.setattr(external, "run_gdesigner", offline_adapter)
    monkeypatch.setattr(recovery, "install_request_ledger", lambda _: None)
    monkeypatch.setattr(recovery, "save_rng", save_synthetic_rng)
    monkeypatch.setattr(recovery, "validate_test_point", fail_after_persist)
    with pytest.raises(RuntimeError, match="synthetic late scorer failure"):
        recovery.execute_tests(args, inspection, ledger)

    persisted = json.loads((args.run_dir / "aime_2025_point.json").read_text())
    assert persisted["num_examples"] == 30
    assert persisted["failure_fraction"] == 1.0
    assert (args.run_dir / "recovery_telemetry.json").exists()
    assert (args.run_dir / "rng_before_tests.pkl").exists()
    assert (args.run_dir / "rng_after_aime_2025.pkl").exists()
    assert not (args.run_dir / "aime_2026_point.json").exists()
    events = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert len(events) == 30
    assert {event["phase"] for event in events} == {"test:aime_2025"}
    assert completed_phases == ["validated:first"]
    assert closes == ["sync", "async"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
