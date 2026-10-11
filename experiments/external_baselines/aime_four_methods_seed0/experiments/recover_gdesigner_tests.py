#!/usr/bin/env python3
"""Test a retained G-Designer controller under an explicit quality-gate waiver.

The immutable inputs are a failed selection audit, controller, and run manifest.
Search/select never run again. Exactly two frozen 30-row test splits are opened
after a new candidate lock. Original failures and all new quality measurements
remain visible; this stage-reseed continuation never becomes a formal pass.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import native_aime_formal as native
import native_external_methods as external
from recover_aime_tests import (
    DurableLedger, atomic_json, install_request_ledger, read_json, save_rng,
    source_claim_path,
)

YEARS = ("aime_2025.jsonl", "aime_2026.jsonl")
SOURCE_FILES = ("run_manifest.json", "selection_audit.json", "gdesigner_controller.pt")
RANDOM_POLICY = "stage_reseed_seed0; original_rng_and_node_ids_unavailable; not_exact_process_resume"


def inspect_source(args: argparse.Namespace) -> dict[str, Any]:
    """Only inspect validation/provenance/checkpoint; never open D_test here."""
    source = args.source_run.resolve()
    if args.method != "gdesigner" or args.seed != 0:
        raise ValueError("this recovery supports only G-Designer seed 0")
    if source_claim_path(source).exists():
        raise RuntimeError("G-Designer test recovery already claimed")
    if args.run_dir.exists():
        raise FileExistsError(f"recovery output must be fresh: {args.run_dir}")
    if {path.name for path in source.iterdir()} != set(SOURCE_FILES):
        raise RuntimeError("source must have only its three retained pre-test artifacts")
    run = read_json(source / "run_manifest.json")
    selection = read_json(source / "selection_audit.json")
    required = {
        "protocol_version": external.AIME_PROTOCOL, "dataset": "aime", "method": "gdesigner",
        "seed": 0, "data_seed": 2026, "split_counts": {"search": 60, "select": 30},
        "model": external.MODEL, "context_limit": 16384, "output_limit": 6144,
        "concurrency": 8, "selection_split": "D_select", "test_split": "D_test",
        "frozen_data_manifest_sha256": native.FROZEN_AIME_MANIFEST_SHA256,
        "answer_parser": "answer_protocol.py",
        "answer_parser_sha256": external.sha256_file(Path(external.aime_answer_protocol.__file__)),
        "status": "failed",
        "failure": "RuntimeError: G-Designer D_select failure rate exceeds 5%; refusing to open D_test.",
    }
    if any(run.get(key) != value for key, value in required.items()):
        raise RuntimeError("source run protocol/parser/failure differs from the retained pre-test attempt")
    if (selection.get("schema") != "aime_fourway_selection_audit_v1"
            or selection.get("method") != "gdesigner" or selection.get("split") != "D_select"
            or selection.get("dtest_opened") is not False
            or selection.get("failure_limit") != external.MAX_SAMPLE_FAILURE_RATE):
        raise RuntimeError("source selection audit differs or D_test was already opened")
    rows = external.read_task_rows("aime", args.data_dir, 0, None)
    if run.get("split_row_ids") != {key: [str(row["id"]) for row in batch] for key, batch in rows.items()}:
        raise RuntimeError("retained validation IDs differ from the frozen splits")
    frozen = native.load_frozen_aime_manifest(args.data_dir)
    expected_data = {
        "aimo-validation-aime.jsonl": frozen["validation"]["source"]["sha256_git_content"],
        frozen["validation"]["search"]["path"]: frozen["validation"]["search"]["sha256_git_content"],
        frozen["validation"]["select"]["path"]: frozen["validation"]["select"]["sha256_git_content"],
    }
    if run.get("data_sha256") != expected_data:
        raise RuntimeError("retained validation data hashes differ")
    baseline = (args.baseline_root / "GDesigner").resolve()
    provenance = external.verify_upstream_source("gdesigner", baseline)
    hashes = external.sha256_tree(baseline)
    tree_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if (run.get("upstream_provenance") != provenance or run.get("baseline_source_sha256") != hashes
            or run.get("baseline_source_tree_sha256") != tree_hash):
        raise RuntimeError("retained upstream source identity differs from the clean pinned baseline")
    original_runner = Path(run["data_dir"]).parent / "experiments/native_external_methods.py"
    if external.sha256_file(original_runner) != run.get("runner_sha256"):
        raise RuntimeError("retained runner SHA cannot be verified against its original source")
    checkpoint_path = source / "gdesigner_controller.pt"
    controller_hash = external.sha256_file(checkpoint_path)
    if controller_hash != selection.get("controller_sha256"):
        raise RuntimeError("controller checkpoint differs from the selection audit")
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema") != "aime_gdesigner_controller_v1" or checkpoint.get("seed") != 0
            or checkpoint.get("frozen_phase") != "after D_search and before D_select/D_test"
            or checkpoint.get("module_training_modes") != {"gcn": False, "mlp": True}
            or set(checkpoint.get("state_dicts", {})) != {"gcn", "mlp"}):
        raise RuntimeError("controller metadata differs from the frozen original configuration")
    for module in checkpoint["state_dicts"].values():
        if not module or any(not torch.isfinite(value).all().item() for value in module.values()):
            raise RuntimeError("controller checkpoint contains non-finite or empty parameters")
    metrics = selection["selection_metrics"]
    audits = metrics.get("rows", [])
    if len(audits) != 30 or metrics.get("num_examples") != 30:
        raise RuntimeError("retained selection must contain all thirty rows")
    for row, audit in zip(rows["select"], audits):
        score = external.score_row("aime", audit.get("raw_output", ""), str(row["answer"]))
        if (audit.get("id") != str(row["id"]) or audit.get("gold") != score["gold"]
                or any(audit.get(key) != score[key] for key in ("prediction", "parser_valid", "correct", "score_0_1_2"))
                or audit.get("sample_failed") != (not bool(score["parser_valid"]))):
            raise RuntimeError("retained selection row order/golds/shared scores differ")
    recomputed = {
        "correct": sum(int(row["correct"]) for row in audits),
        "accuracy": sum(int(row["correct"]) for row in audits) / 30,
        "parser_valid_rate": sum(int(row["parser_valid"]) for row in audits) / 30,
        "failure_fraction": sum(int(row["sample_failed"]) for row in audits) / 30,
        "mean_score_0_1_2": sum(int(row["score_0_1_2"]) for row in audits) / 30,
    }
    if any(metrics.get(key) != value for key, value in recomputed.items()):
        raise RuntimeError("retained selection aggregate differs from its original rows")
    telemetry = selection.get("telemetry")
    if telemetry != run.get("telemetry") or telemetry.get("request_timeout_s") != 600.0:
        raise RuntimeError("retained runtime telemetry/timeout differs")
    phase_usage = telemetry["phases"]["select"]
    if any(metrics.get(key) != phase_usage[key] for key in ("calls", "failed_calls", "completion_truncated", "total_tokens")):
        raise RuntimeError("retained selection usage differs from its runtime audit")
    for key in external.empty_usage():
        if telemetry["totals"][key] != sum(phase[key] for phase in telemetry["phases"].values()):
            raise RuntimeError("historical aggregate usage does not match its phases")
    original_failed = any(float(metrics[key]) > 0.05 for key in (
        "failure_fraction", "request_failure_fraction", "truncation_fraction"))
    if not original_failed:
        raise RuntimeError("source selection did not fail the quality gate being waived")
    return {
        "schema": "aime_gdesigner_gate_waiver_recovery_inspection_v1", "method": "gdesigner",
        "source_run": str(source), "baseline_source": str(baseline), "baseline_provenance": provenance,
        "baseline_source_tree_sha256": tree_hash,
        "source_artifacts_sha256": {name: external.sha256_file(source / name) for name in SOURCE_FILES},
        "original_runner_path": str(original_runner), "original_runner_sha256": run["runner_sha256"],
        "original_status": "failed", "original_failure": run["failure"],
        "frozen_selection": {"candidate_id": "trained_gdesigner_graph", "controller_sha256": controller_hash,
                             "selection_accuracy": metrics["accuracy"], "selection_rows": 30},
        "selection_metrics": metrics, "historical_runtime_telemetry": telemetry,
        "historical_usage_audit": "RECORDED_RUNTIME_AGGREGATES; no historical per-request ledger",
        "remaining_test_splits": list(YEARS), "formal_acceptance": "NOT_FORMAL_PASS",
        "selection_gate_waiver": {"original_threshold": 0.05, "original_gate_pass": False,
                                  "scope": "G-Designer selection and test sample/request/truncation quality gates"},
        "random_state_policy": RANDOM_POLICY,
    }


def verify_preserved_source(inspection: dict[str, Any]) -> None:
    source = Path(inspection["source_run"])
    if any(external.sha256_file(source / name) != sha for name, sha in inspection["source_artifacts_sha256"].items()):
        raise RuntimeError("retained source artifacts changed during recovery")
    if external.sha256_file(Path(inspection["original_runner_path"])) != inspection["original_runner_sha256"]:
        raise RuntimeError("original runner changed during recovery")
    baseline = Path(inspection["baseline_source"])
    provenance = external.verify_upstream_source("gdesigner", baseline)
    observed = hashlib.sha256(json.dumps(external.sha256_tree(baseline), sort_keys=True).encode()).hexdigest()
    if provenance != inspection["baseline_provenance"] or observed != inspection["baseline_source_tree_sha256"]:
        raise RuntimeError("pinned upstream changed during recovery")


def validate_test_point(point: dict[str, Any]) -> None:
    audits = point["rows"]
    if len(audits) != 30 or point.get("num_examples") != 30 or len({row["id"] for row in audits}) != 30:
        raise RuntimeError("test point does not contain exactly thirty unique rows")
    correct = 0
    for row in audits:
        score = external.score_row("aime", row.get("raw_output", ""), row["gold"])
        if any(row.get(key) != score[key] for key in ("prediction", "parser_valid", "correct", "score_0_1_2")):
            raise RuntimeError("test raw predictions differ from the shared scorer")
        correct += int(score["correct"])
    recomputed = {
        "correct": correct, "accuracy": correct / 30,
        "failure_fraction": sum(int(not row["parser_valid"]) for row in audits) / 30,
        "parser_valid_rate": sum(int(row["parser_valid"]) for row in audits) / 30,
        "mean_score_0_1_2": sum(int(row["score_0_1_2"]) for row in audits) / 30,
    }
    if any(point.get(key) != value for key, value in recomputed.items()):
        raise RuntimeError("test aggregate differs from its sample records")


def execute_tests(args: argparse.Namespace, inspection: dict[str, Any], ledger: DurableLedger) -> dict[str, Any]:
    runtime = external.ModelRuntime(args.endpoint, "gdesigner", f"aime:gate_waiver:gdesigner:{args.run_dir}")
    install_request_ledger(ledger)
    args.upstream_provenance = inspection["baseline_provenance"]
    phases: dict[str, Any] = {}

    def before_tests(identity: dict[str, Any]) -> None:
        atomic_json(args.run_dir / "reconstructed_graph_identity.json", identity)
        save_rng(args.run_dir / "rng_before_tests.pkl")
        atomic_json(args.run_dir / "phase_checkpoint.json", {"status": "running", "phase": "test:aime_2025"})
        print("RECOVERY_TEST_START method=gdesigner splits=2025,2026 rows_each=30 reused_search_select=true gate_waiver=true", flush=True)

    def sample_completed(phase: str, row: dict[str, Any]) -> None:
        ledger.append({"event": "sample_completed", "phase": phase, **row})

    def checkpoint_completed(name: str, point: dict[str, Any], observed_runtime: Any) -> None:
        # Save raw predictions and counters first: no late scorer can erase work.
        phases[name] = point
        atomic_json(args.run_dir / f"{name}_point.json", point)
        atomic_json(args.run_dir / "recovery_telemetry.json", {
            "runtime": observed_runtime.summary(), "request_ledger_audit": ledger.summary(),
        })
        save_rng(args.run_dir / f"rng_after_{name}.pkl")
        validate_test_point(point)
        atomic_json(args.run_dir / "phase_checkpoint.json", {
            "status": "phase_completed", "completed_test_splits": list(phases), "phase": f"test:{name}",
        })
        print(f"RECOVERY_TEST_COMPLETE method=gdesigner split={name} rows=30 correct={point['correct']}", flush=True)

    try:
        result = external.run_gdesigner("aime", external.read_task_rows("aime", args.data_dir, 0, None), args,
            runtime.run_id, runtime, Path(inspection["baseline_source"]), test_recovery={
                "controller_path": args.run_dir / "retained_source/gdesigner_controller.pt",
                "checkpoint_sha256": inspection["frozen_selection"]["controller_sha256"],
                "selection_metrics": inspection["selection_metrics"], "before_tests": before_tests,
                "sample_completed": sample_completed, "checkpoint_completed": checkpoint_completed,
                "close_async_client": runtime.async_client.close,
            })
        audit = ledger.summary()
        totals = runtime.usage["totals"]
        if audit["calls"] != totals["calls"] or any(audit["reported_response_usage"][key] != totals[key]
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
            raise RuntimeError("durable request usage does not match runtime counters")
        if set(runtime.usage["phases"]) != {"test:aime_2025", "test:aime_2026"} or set(phases) != {"aime_2025", "aime_2026"}:
            raise RuntimeError("recovery executed phases other than exactly the two test splits")
        access = read_json(args.run_dir / "dtest_access_manifest.json")
        lock = read_json(args.run_dir / "selection_frozen.json")
        if (set(access.get("test_splits", {})) != set(YEARS)
                or access.get("selection_lock_sha256") != external.sha256_file(args.run_dir / "selection_frozen.json")
                or lock["selected"]["controller_sha256"] != inspection["frozen_selection"]["controller_sha256"]):
            raise RuntimeError("test access audit differs from the two authorized frozen splits")
        for filename, evidence in access["test_splits"].items():
            if (evidence["row_count"] != 30 or evidence["opened_at_epoch"] < lock["created_at_epoch"]
                    or evidence["ids"] != [row["id"] for row in phases[Path(filename).stem]["rows"]]):
                raise RuntimeError("test sample IDs/access timing differ from the post-freeze audit")
        if external.sha256_file(args.run_dir / "gdesigner_controller.pt") != inspection["frozen_selection"]["controller_sha256"]:
            raise RuntimeError("copied frozen controller changed during testing")
        result["request_ledger_audit"] = audit
        return result
    finally:
        atomic_json(args.run_dir / "recovery_telemetry.json", {
            "runtime": runtime.summary(), "request_ledger_audit": ledger.summary(),
        })
        runtime.sync.close()
        if not getattr(runtime.async_client, "is_closed", lambda: False)():
            asyncio.run(runtime.async_client.close())


def recover(args: argparse.Namespace) -> dict[str, Any]:
    args.source_run = args.source_run.resolve()
    args.run_dir = args.run_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    inspection = inspect_source(args)
    if args.inspect_only:
        return inspection
    if not args.allow_selection_gate_waiver:
        raise RuntimeError("G-Designer needs explicit --allow-selection-gate-waiver authorization")
    if float(os.environ.get("AIME_EXTERNAL_REQUEST_TIMEOUT_S", "600")) != 600:
        raise RuntimeError("recovery refuses a changed external request timeout")
    os.environ["AIME_EXTERNAL_REQUEST_TIMEOUT_S"] = "600"
    native.verify_served_model(args.endpoint)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    claim = source_claim_path(args.source_run)
    claim.parent.mkdir(parents=True, exist_ok=True)
    claim_payload = {"source_run": str(args.source_run), "run_dir": str(args.run_dir),
                     "test_splits": list(YEARS), "status": "claimed", "created_at_epoch": time.time()}
    with claim.open("x", encoding="utf-8") as stream:
        json.dump(claim_payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    claim_directory = os.open(claim.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(claim_directory)
    finally:
        os.close(claim_directory)
    manifest = {**inspection, "run_dir": str(args.run_dir), "claim": str(claim), "status": "preparing",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_waiver_authorized": True, "formal_quality_gate": "WAIVED_BY_USER; NOT_FORMAL_PASS",
        "runner_sha256": external.sha256_file(Path(__file__)),
        "native_adapter_sha256": external.sha256_file(Path(external.__file__)),
        "recovery_source_commit": subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip(),
        "model_verification_report": os.environ.get("AIME_MODEL_VERIFICATION_REPORT"),
        "model_verification_report_sha256": external.sha256_file(Path(os.environ["AIME_MODEL_VERIFICATION_REPORT"]))
            if os.environ.get("AIME_MODEL_VERIFICATION_REPORT") else None,
        "run_session_record": os.environ.get("AIME_RUN_SESSION_RECORD"),
        "controls": {"model": external.MODEL, "seed": 0, "data_seed": 2026, "concurrency": 8,
                     "max_model_len": 16384, "max_tokens": 6144, "temperature": 0.0, "top_p": 1.0,
                     "thinking": False, "request_timeout_s": 600, "batch_size": 4,
                     "num_rounds": 1, "max_tries": 3, "max_time": 180},
    }
    atomic_json(args.run_dir / "recovery_manifest.json", manifest)
    try:
        retained = args.run_dir / "retained_source"
        retained.mkdir()
        for name in SOURCE_FILES:
            shutil.copyfile(args.source_run / name, retained / name)
        verify_preserved_source(inspection)
        manifest["status"] = "running"
        atomic_json(args.run_dir / "recovery_manifest.json", manifest)
        raw = execute_tests(args, inspection, DurableLedger(args.run_dir / "request_ledger.jsonl"))
        verify_preserved_source(inspection)
        result = {
            "schema": "aime_gdesigner_quality_gate_waiver_stage_result_v1", "method": "gdesigner",
            "formal_acceptance": "NOT_FORMAL_PASS", "variant": "selection_and_test_gate_waiver_stage_recovery",
            "random_state_policy": RANDOM_POLICY, "frozen_selection": inspection["frozen_selection"],
            "gate_waiver": inspection["selection_gate_waiver"], "test": raw["test"],
            "native_method_config": raw.get("native_method_config"), "recovery_metadata": raw.get("recovery_metadata"),
            "historical_runtime_telemetry": inspection["historical_runtime_telemetry"],
            "historical_usage_audit": inspection["historical_usage_audit"],
            "test_telemetry": raw["telemetry"], "request_ledger_audit": raw.get("request_ledger_audit"),
            "limitations": ["original selection/test quality gates explicitly waived",
                            "stage seed reset; original RNG and node IDs unavailable",
                            "historical resource data are runtime aggregates without a per-request ledger"],
        }
        atomic_json(args.run_dir / "supplementary_result.json", result)
        manifest.update(status="completed_supplementary", source_verified_unchanged_after_run=True,
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(args.run_dir / "recovery_manifest.json", manifest)
        atomic_json(claim, {**claim_payload, "status": manifest["status"]})
        return result
    except BaseException as exc:
        manifest.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        atomic_json(args.run_dir / "recovery_manifest.json", manifest)
        native.persist_failure_traceback(args.run_dir, "gdesigner_stage_recovery", exc)
        atomic_json(claim, {**claim_payload, "status": "failed", "failure": manifest["failure"]})
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("gdesigner",), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--allow-selection-gate-waiver", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    """CPU-only durable-write sanity check, without data or model access."""
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        atomic_json(root / "checkpoint.json", {"status": "synthetic"})
        assert read_json(root / "checkpoint.json") == {"status": "synthetic"}
        ledger = DurableLedger(root / "events.jsonl")
        ledger.append({"event": "request_started", "request_id": "synthetic"})
        assert ledger.summary()["unknown_usage_calls"] == 1
        ledger.append({"event": "request_completed", "request_id": "synthetic",
                       "usage_status": "REPORTED", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
        assert ledger.summary()["calls"] == 1 and ledger.summary()["usage_complete"]
    print("gdesigner recovery durable checkpoint self-test passed", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        print(json.dumps(recover(parse_args()), ensure_ascii=False, indent=2), flush=True)
