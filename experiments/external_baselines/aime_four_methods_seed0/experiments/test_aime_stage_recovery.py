"""Offline provenance, orchestration, and durable-event recovery regressions."""

from __future__ import annotations

import asyncio
import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import native_aime_formal as native
import recover_aime_tests as recovery


PACKAGE = Path(__file__).resolve().parents[1]
BASELINE_ROOT = Path(
    "/data/jxc/RPAS-AIME-FourMethods-Seed0-v3-16k/experiments/external_baselines/aime_four_methods_seed0/upstream"
)
REAL_RUNS = {
    "aflow": Path(
        "/data/jxc/RPAS-AIME-Parallel-Seed0-20261010/outputs/aime_parallel_seed0_20261010/"
        "aflow_attempt_v2/aflow_maas/aflow/seed_0"
    ),
    "maas": Path(
        "/data/jxc/RPAS-AIME-FourMethods-Seed0-v3-16k/outputs/"
        "aime_finalv3_detached_seed0_20261010_recovery/aflow_maas/maas/seed_0"
    ),
}


def _args(source: Path, output: Path, method: str, inspect_only: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        source_run=source, run_dir=output, method=method, seed=0,
        data_dir=PACKAGE / "data", baseline_root=BASELINE_ROOT,
        endpoint="http://offline.invalid/v1", inspect_only=inspect_only,
        allow_stage_reseed=True,
    )


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _change_json(path: Path, mutate) -> None:
    value = recovery.read_json(path)
    mutate(value)
    _json(path, value)


def _mini_source(tmp_path: Path, monkeypatch, method: str = "aflow") -> SimpleNamespace:
    """A minimal artificial native repo and audit, never a copy of a real run."""
    source = tmp_path / "source" / "seed_0"
    source.mkdir(parents=True)
    args = _args(source, tmp_path / "recovered", method)
    provenance = {"synthetic_fixture": True}
    monkeypatch.setattr(native, "verify_pinned_upstream", lambda *_: provenance)
    for name in recovery.TIMEOUTS:
        monkeypatch.delenv(name, raising=False)
    protocol = {
        "method": method, "protocol_version": native.PROTOCOL_VERSION,
        "search_seed": 0, "data_seed": 2026, "model": native.MODEL,
        "context_limit": 16384, "output_limit": 6144,
        "search_size": 60, "selection_size": 30,
        "answer_protocol": native.ANSWER_PARSER,
        "answer_protocol_sha256": native.sha256_file(Path(native.__file__).with_name("answer_protocol.py")),
        "task_contract": native.task_contract_manifest(),
        "upstream_provenance": provenance,
    }
    _json(source / "run_manifest.json", {
        **protocol, "status": "failed", "test_size_each": 30,
        "failure": "RuntimeError: AIME evaluator row order/content mismatch at synthetic row 0",
    })
    _json(source / "split_manifest.json", protocol)
    search, select = native.freeze_split(args.data_dir, 2026, 60, 30)
    search = native.namespace_aime_rows(search, "validation")
    select = native.namespace_aime_rows(select, "validation")
    repo = source / f"{method}_repo"
    if method == "aflow":
        candidate = repo / recovery.NATIVE_ROOT / "workflows/round_9"
        candidate.mkdir(parents=True)
        (candidate / "graph.py").write_text("def synthetic_forward():\n    return 1\n", encoding="utf-8")
        (candidate / "prompt.py").write_text("PROMPT = 'offline'\n", encoding="utf-8")
        selected = {"Q": {"round": 9, "score": 0.7}, "E": {"round": 9, "score": 0.7}}
        selection = {"candidate_rows": [{
            "round": 9, "status": "evaluated", "shared_metrics": {"num_examples": 30, "score": 0.7},
            "output_audit": {"failure_fraction": 0.0},
        }]}
        input_path = repo / "data/datasets/math_test.jsonl"
        log_dir = source / "aflow_test/aime_2025_Q"
    else:
        checkpoint = repo / recovery.NATIVE_ROOT / "train/round_1/MATH_controller_sample1.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic controller; never loaded by a model")
        controller_hash = native.sha256_file(checkpoint)
        selected = {"controller_sha256": controller_hash, "selection_score": 0.7, "num_examples": 30}
        selection = {
            "controller_sha256": controller_hash,
            "selection_metrics": {"num_examples": 30, "score": 0.7},
            "output_audit": {"failure_fraction": 0.0},
        }
        input_path = repo / "maas/ext/maas/data/math_test.jsonl"
        log_dir = repo / recovery.NATIVE_ROOT / "test/round_1"
    native.freeze_aime_selection(source, method, search, select, selected, source_provenance=provenance)
    _json(source / "selection_audit.json", selection)
    rows = native.load_aime_test_rows_after_freeze(args.data_dir, source, method, recovery.YEAR_OLD, 30)
    rendered = native.native_math_rows(rows, Path(recovery.YEAR_OLD).stem)
    native.write_jsonl(input_path, rendered)
    log_dir.mkdir(parents=True)
    with (log_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["question", "expected_output", "prediction"])
        writer.writeheader()
        writer.writerows({
            "question": row["problem"], "expected_output": row["solution"],
            "prediction": f"FINAL ANSWER: {row['answer']}",
        } for row in rendered)
    return args


@pytest.mark.parametrize(("method", "correct", "sample_gate"), [("aflow", 19, False), ("maas", 15, True)])
def test_real_inspect_only_reuses_thirty_rows_without_writes_or_model_calls(tmp_path: Path, monkeypatch, method, correct, sample_gate):
    retained = REAL_RUNS[method]
    if not retained.is_dir() or not BASELINE_ROOT.is_dir():
        pytest.skip("retained host run and pinned upstream sources are needed for read-only inspection")
    retained_bytes = (retained / "run_manifest.json").read_bytes()
    # Host execution later creates a permanent claim beside the real run.
    # Inspect an identical temporary copy so this regression remains runnable
    # after that claim; the real failure records are still checked unchanged.
    source = tmp_path / "source_copy"
    shutil.copytree(retained, source, ignore=shutil.ignore_patterns("__pycache__"))
    args = _args(source, tmp_path / "must_stay_absent", method, inspect_only=True)
    model = Mock(side_effect=AssertionError("inspect-only must not call a model"))
    execute = Mock(side_effect=AssertionError("inspect-only must not execute a test"))
    monkeypatch.setattr(native, "verify_served_model", model)
    monkeypatch.setattr(recovery, "execute_remaining", execute)
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "0")
    before = {str(path.relative_to(source)): (path.stat().st_size, path.stat().st_mtime_ns)
              for path in source.rglob("*") if path.is_file()}
    original_open = Path.open

    def read_only_open(path, mode="r", *rest, **kwargs):
        if any(flag in mode for flag in "wax+"):
            raise AssertionError(f"inspection attempted a write: {path}")
        return original_open(path, mode, *rest, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", read_only_open)
        result = recovery.recover(args)
    after = {str(path.relative_to(source)): (path.stat().st_size, path.stat().st_mtime_ns)
             for path in source.rglob("*") if path.is_file()}
    assert before == after
    assert (retained / "run_manifest.json").read_bytes() == retained_bytes
    assert not args.run_dir.exists()
    assert not recovery.source_claim_path(source).exists()
    assert result["reused_2025"]["num_examples"] == 30
    assert result["reused_2025"]["correct"] == correct
    assert result["reused_2025"]["sample_gate_pass"] is sample_gate
    assert result["reused_2025"]["total_tokens"] is None
    assert result["remaining_test_split"] == recovery.YEAR_NEW
    assert result["formal_acceptance"] == "NOT_FORMAL_PASS"
    model.assert_not_called()
    execute.assert_not_called()


def test_duplicate_source_claim_is_permanent_and_exclusive(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    first = recovery.claim_source(source, tmp_path / "first")
    original = first.read_bytes()
    with pytest.raises(FileExistsError):
        recovery.claim_source(source, tmp_path / "second")
    assert first.read_bytes() == original


def test_inspection_refuses_existing_output(tmp_path: Path, monkeypatch):
    args = _mini_source(tmp_path, monkeypatch)
    args.run_dir.mkdir()
    with pytest.raises(FileExistsError, match="must be fresh"):
        recovery.inspect_source(args)


def test_inspection_refuses_already_opened_2026(tmp_path: Path, monkeypatch):
    args = _mini_source(tmp_path, monkeypatch)
    _change_json(args.source_run / "dtest_access_manifest.json", lambda value: value["test_splits"].update({recovery.YEAR_NEW: {}}))
    with pytest.raises(RuntimeError, match="unopened 2026"):
        recovery.inspect_source(args)


def test_inspection_refuses_changed_maas_checkpoint(tmp_path: Path, monkeypatch):
    args = _mini_source(tmp_path, monkeypatch, "maas")
    checkpoint = args.source_run / "maas_repo" / recovery.NATIVE_ROOT / "train/round_1/MATH_controller_sample1.pth"
    checkpoint.write_bytes(b"modified artificial state")
    with pytest.raises(RuntimeError, match="controller/selection gate"):
        recovery.inspect_source(args)


@pytest.mark.parametrize("method", ["aflow", "maas"])
def test_inspection_refuses_failed_original_selection(tmp_path: Path, monkeypatch, method):
    args = _mini_source(tmp_path, monkeypatch, method)
    def fail(value):
        audit = value["candidate_rows"][0]["output_audit"] if method == "aflow" else value["output_audit"]
        audit["failure_fraction"] = 0.1
    _change_json(args.source_run / "selection_audit.json", fail)
    with pytest.raises(RuntimeError, match="selection.*gate|controller/selection gate"):
        recovery.inspect_source(args)


def test_inspection_refuses_changed_protocol_hash(tmp_path: Path, monkeypatch):
    args = _mini_source(tmp_path, monkeypatch)
    _change_json(args.source_run / "run_manifest.json", lambda value: value.update(answer_protocol_sha256="0" * 64))
    with pytest.raises(RuntimeError, match="protocol differs"):
        recovery.inspect_source(args)


def test_preserved_source_verification_detects_candidate_change(tmp_path: Path, monkeypatch):
    args = _mini_source(tmp_path, monkeypatch)
    inspection = recovery.inspect_source(args)
    graph = args.source_run / "aflow_repo" / recovery.NATIVE_ROOT / "workflows/round_9/graph.py"
    graph.write_text("def synthetic_forward():\n    return 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifacts changed"):
        recovery.verify_preserved_source(inspection)


@pytest.mark.parametrize("method", ["aflow", "maas"])
def test_orchestrator_only_executes_remaining_2026_and_never_claims_formal_pass(tmp_path: Path, monkeypatch, method):
    args = _mini_source(tmp_path, monkeypatch, method)
    original_manifest = (args.source_run / "run_manifest.json").read_bytes()
    model = Mock()
    execute = Mock(return_value={"num_examples": 30, "correct": 17, "total_tokens": 100, "sample_gate_pass": True})
    monkeypatch.setattr(native, "verify_served_model", model)
    monkeypatch.setattr(recovery, "execute_remaining", execute)
    result = recovery.recover(args)
    execute.assert_called_once()
    called_args, inspection, ledger = execute.call_args.args
    assert called_args.run_dir == args.run_dir
    assert inspection["remaining_test_split"] == recovery.YEAR_NEW
    assert ledger.path == args.run_dir / "request_ledger.jsonl"
    assert result["test"]["aime_2025"]["Q"]["reused_without_model_requests"]
    assert result["test"]["aime_2025"]["Q"]["total_tokens"] is None
    assert all(value is None for value in result["test"]["aime_2025"]["Q"]["resource_usage"].values())
    assert result["formal_acceptance"] == "NOT_FORMAL_PASS"
    assert result["historical_usage_complete"] is False
    assert (args.source_run / "run_manifest.json").read_bytes() == original_manifest
    assert recovery.read_json(args.source_run / "run_manifest.json")["status"] == "failed"
    assert not (args.run_dir / "native_result.json").exists()
    assert not (args.run_dir / "quality_gate.json").exists()
    manifest = recovery.read_json(args.run_dir / "recovery_manifest.json")
    assert manifest["status"] == "completed_supplementary"
    assert manifest["formal_quality_gate"] == "NOT_RUN_INCOMPLETE_HISTORICAL_USAGE"
    assert recovery.read_json(recovery.source_claim_path(args.source_run))["status"] == "completed_supplementary"
    model.assert_called_once_with(args.endpoint)
    with pytest.raises(RuntimeError, match="already claimed"):
        recovery.recover(_args(args.source_run, tmp_path / "second_recovery", method))
    execute.assert_called_once()


def test_orchestrator_failure_retains_claim_and_original_failed_run(tmp_path: Path, monkeypatch):
    args = _mini_source(tmp_path, monkeypatch)
    original_manifest = (args.source_run / "run_manifest.json").read_bytes()
    monkeypatch.setattr(native, "verify_served_model", Mock())
    execute = Mock(side_effect=RuntimeError("synthetic interrupted 2026"))
    monkeypatch.setattr(recovery, "execute_remaining", execute)
    with pytest.raises(RuntimeError, match="interrupted 2026"):
        recovery.recover(args)
    assert recovery.read_json(args.run_dir / "recovery_manifest.json")["status"] == "failed"
    assert recovery.read_json(recovery.source_claim_path(args.source_run))["status"] == "failed"
    assert (args.source_run / "run_manifest.json").read_bytes() == original_manifest
    assert not (args.run_dir / "native_result.json").exists()
    assert not (args.run_dir / "supplementary_result.json").exists()


def test_maas_requires_disclosed_stage_reseed_before_claim_or_model(tmp_path: Path, monkeypatch):
    args = _mini_source(tmp_path, monkeypatch, "maas")
    args.allow_stage_reseed = False
    model = Mock()
    monkeypatch.setattr(native, "verify_served_model", model)
    with pytest.raises(RuntimeError, match="allow-stage-reseed"):
        recovery.recover(args)
    assert not args.run_dir.exists()
    assert not recovery.source_claim_path(args.source_run).exists()
    model.assert_not_called()


@pytest.mark.parametrize("outcome", ["success", "missing_usage", "cancelled"])
def test_request_ledger_persists_success_unknown_usage_and_cancellation(tmp_path: Path, monkeypatch, outcome):
    response = SimpleNamespace(
        id="server-fixture", choices=[SimpleNamespace(finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10) if outcome == "success" else None,
    )
    class Completion:
        async def create(self, **kwargs):
            if outcome == "cancelled":
                raise asyncio.CancelledError()
            return response
    ledger = recovery.DurableLedger(tmp_path / "requests.jsonl")
    fsync = Mock(wraps=recovery.os.fsync)
    monkeypatch.setattr(recovery.os, "fsync", fsync)
    recovery.install_request_ledger(ledger, Completion)
    async def invoke():
        with native.phase_scope("test:aime_2026"):
            return await Completion().create(model=native.MODEL, max_tokens=6144)
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(invoke())
    else:
        assert asyncio.run(invoke()) is response
    events = native.read_jsonl(ledger.path)
    assert [event["sequence"] for event in events] == [1, 2]
    assert events[0]["event"] == "request_started"
    assert events[0]["phase"] == "test:aime_2026"
    assert events[0]["request_id"] == events[1]["request_id"]
    assert fsync.call_count == 2
    if outcome == "success":
        assert events[1]["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        assert events[1]["usage_status"] == "REPORTED"
    elif outcome == "missing_usage":
        assert all(value is None for value in events[1]["usage"].values())
        assert events[1]["usage_status"] == "UNKNOWN_INCOMPLETE_RESPONSE"
    else:
        assert events[1]["event"] == "request_failed"
        assert events[1]["exception_type"] == "CancelledError"
        assert events[1]["usage"] is None
    summary = ledger.summary()
    assert summary["calls"] == 1
    assert summary["unknown_usage_calls"] == int(outcome != "success")
    assert summary["usage_complete"] is (outcome == "success")
    assert summary["failed_or_cancelled_calls"] == int(outcome == "cancelled")
    assert summary["reported_response_usage"]["total_tokens"] == (10 if outcome == "success" else 0)


@pytest.mark.parametrize("cancelled", [False, True])
def test_sample_ledger_persists_complete_output_or_cancellation(tmp_path: Path, cancelled):
    class Benchmark:
        async def evaluate_problem(self, problem, graph):
            if cancelled:
                raise asyncio.CancelledError()
            return ("fixture question", "FINAL ANSWER: 7", r"\boxed{7}", 1)
    ledger = recovery.DurableLedger(tmp_path / "samples.jsonl")
    recovery.install_sample_ledger(Benchmark, ledger)
    invoke = Benchmark().evaluate_problem({"aime_id": "aime_2026:fixture", "problem": "fixture"}, None)
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(invoke)
    else:
        asyncio.run(invoke)
    event, = native.read_jsonl(ledger.path)
    assert event["id"] == "aime_2026:fixture"
    assert event["event"] == ("sample_interrupted" if cancelled else "sample_completed")
    assert event.get("exception_type") == ("CancelledError" if cancelled else None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
