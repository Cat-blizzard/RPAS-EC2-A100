#!/usr/bin/env python3
"""Run D_test for one retained ADAS candidate under an explicit gate waiver."""
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
from functools import wraps
from typing import Any

import native_aime_formal as native
import native_external_methods as external
from recover_aime_tests import DurableLedger, atomic_json, install_request_ledger, read_json, save_rng, source_claim_path

YEARS = ("aime_2025.jsonl", "aime_2026.jsonl")
SOURCE_FILES = ("run_manifest.json", "selection_audit.json", "native_search/aime_0_run_archive.json")


def install_sync_request_ledger(ledger: DurableLedger) -> None:
    """ADAS uses the synchronous OpenAI client; record the same durable events."""
    from openai.resources.chat.completions import Completions
    original = Completions.create

    @wraps(original)
    def tracked(self: Any, *args: Any, **kwargs: Any) -> Any:
        import uuid
        request_id = uuid.uuid4().hex
        phase = native.current_phase()
        ledger.append({"event": "request_started", "request_id": request_id, "phase": phase,
                       "model": kwargs.get("model"), "max_tokens": kwargs.get("max_tokens")})
        try:
            response = original(self, *args, **kwargs)
        except BaseException as exc:
            ledger.append({"event": "request_failed", "request_id": request_id, "phase": phase,
                           "exception_type": type(exc).__name__, "usage": None,
                           "usage_status": "UNKNOWN_NO_RESPONSE"})
            raise
        usage = getattr(response, "usage", None)
        values = {key: getattr(usage, key, None) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        choices = getattr(response, "choices", []) or []
        ledger.append({"event": "request_completed", "request_id": request_id, "phase": phase,
                       "usage": values,
                       "usage_status": "REPORTED" if all(value is not None for value in values.values())
                       else "UNKNOWN_INCOMPLETE_RESPONSE",
                       "finish_reason": getattr(choices[0], "finish_reason", None) if choices else None})
        return response

    Completions.create = tracked


def inspect_source(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_run.resolve()
    if args.method != "adas" or args.seed != 0:
        raise ValueError("this recovery supports only ADAS AIME seed 0")
    if source_claim_path(source).exists():
        raise RuntimeError("ADAS D_test recovery already claimed")
    if args.run_dir.exists():
        raise FileExistsError(f"recovery output must be fresh: {args.run_dir}")
    if {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()} != set(SOURCE_FILES):
        raise RuntimeError("source must contain only the retained ADAS pre-test artifacts")
    run = read_json(source / "run_manifest.json")
    audit = read_json(source / "selection_audit.json")
    rows = external.read_task_rows("aime", args.data_dir, 0, None)
    required = {
        "protocol_version": external.AIME_PROTOCOL, "dataset": "aime", "method": "adas",
        "seed": 0, "data_seed": 2026, "split_counts": {"search": 60, "select": 30},
        "model": external.MODEL, "context_limit": 16384, "output_limit": 6144,
        "concurrency": 8, "selection_split": "D_select", "test_split": "D_test",
        "frozen_data_manifest_sha256": native.FROZEN_AIME_MANIFEST_SHA256,
        "answer_parser": "answer_protocol.py",
        "answer_parser_sha256": external.sha256_file(Path(external.aime_answer_protocol.__file__)),
        "status": "failed",
        "failure": "RuntimeError: ADAS has no valid candidate on D_select (failure rate must be <=5%).",
    }
    if any(run.get(k) != v for k, v in required.items()):
        raise RuntimeError("retained ADAS run protocol or failure differs")
    if run.get("split_row_ids") != {k: [str(row["id"]) for row in batch] for k, batch in rows.items()}:
        raise RuntimeError("retained validation IDs differ from frozen data")
    if (audit.get("schema") != "aime_fourway_selection_audit_v1" or audit.get("method") != "adas"
            or audit.get("split") != "D_select" or audit.get("dtest_opened") is not False
            or audit.get("failure_limit") != external.MAX_SAMPLE_FAILURE_RATE):
        raise RuntimeError("ADAS selection audit is not a pre-test audit")
    frozen = native.load_frozen_aime_manifest(args.data_dir)
    expected_data = {
        "aimo-validation-aime.jsonl": frozen["validation"]["source"]["sha256_git_content"],
        frozen["validation"]["search"]["path"]: frozen["validation"]["search"]["sha256_git_content"],
        frozen["validation"]["select"]["path"]: frozen["validation"]["select"]["sha256_git_content"],
    }
    if run.get("data_sha256") != expected_data:
        raise RuntimeError("retained validation data hashes differ")
    baseline = (args.baseline_root / "ADAS").resolve()
    provenance = external.verify_upstream_source("adas", baseline)
    source_hashes = external.sha256_tree(baseline)
    tree_hash = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    if (run.get("upstream_provenance") != provenance or run.get("baseline_source_sha256") != source_hashes
            or run.get("baseline_source_tree_sha256") != tree_hash):
        raise RuntimeError("retained ADAS upstream provenance differs")
    candidate_rows = audit.get("candidate_rows", [])
    selected = next((row for row in candidate_rows if row.get("candidate_id") == "archive_21"), None)
    if selected is None or selected.get("status") != "evaluated" or selected.get("correct") != 21:
        raise RuntimeError("authorized ADAS best candidate archive_21 is absent or changed")
    archive_path = source / "native_search/aime_0_run_archive.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    index = int(selected["archive_index"])
    code = str(archive[index].get("code", "")) if 0 <= index < len(archive) else ""
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    if not code or code_hash != selected.get("code_sha256"):
        raise RuntimeError("archive_21 code hash does not match the selection audit")
    metrics = {k: selected.get(k) for k in (
        "candidate_id", "archive_index", "code_sha256", "accuracy", "correct", "num_examples",
        "failure_fraction", "request_failure_fraction", "truncation_fraction", "total_calls",
        "total_tokens", "completion_truncated", "native_name")}
    return {
        "schema": "aime_adas_dtest_gate_waiver_inspection_v1", "method": "adas",
        "source_run": str(source), "baseline_source": str(baseline), "baseline_provenance": provenance,
        "baseline_source_tree_sha256": tree_hash,
        "source_artifacts_sha256": {name: external.sha256_file(source / name) for name in SOURCE_FILES},
        "original_status": "failed", "original_failure": run["failure"],
        "candidate": metrics, "candidate_code_sha256": code_hash, "candidate_index": index,
        "selection_metrics": selected, "remaining_test_splits": list(YEARS),
        "formal_acceptance": "NOT_FORMAL_PASS", "original_gate_threshold": external.MAX_SAMPLE_FAILURE_RATE,
        "gate_waiver_scope": "ADAS D_select candidate gate and D_test sample/request/truncation gates",
        "random_state_policy": "stage_reseed_seed0; original search RNG unavailable; not_exact_process_resume",
    }


def verify_preserved_source(inspection: dict[str, Any]) -> None:
    source = Path(inspection["source_run"])
    for name, expected in inspection["source_artifacts_sha256"].items():
        if external.sha256_file(source / name) != expected:
            raise RuntimeError("retained ADAS source artifact changed")
    baseline = Path(inspection["baseline_source"])
    if (external.verify_upstream_source("adas", baseline) != inspection["baseline_provenance"]
            or hashlib.sha256(json.dumps(external.sha256_tree(baseline), sort_keys=True).encode()).hexdigest()
            != inspection["baseline_source_tree_sha256"]):
        raise RuntimeError("pinned ADAS upstream changed")


def execute_tests(args: argparse.Namespace, inspection: dict[str, Any], ledger: DurableLedger) -> dict[str, Any]:
    runtime = external.ModelRuntime(args.endpoint, "adas", f"aime:gate_waiver:adas:{args.run_dir}")
    install_request_ledger(ledger)
    install_sync_request_ledger(ledger)
    args.upstream_provenance = inspection["baseline_provenance"]
    phases: dict[str, Any] = {}

    def before_tests(identity: dict[str, Any]) -> None:
        atomic_json(args.run_dir / "reconstructed_candidate.json", identity)
        save_rng(args.run_dir / "rng_before_tests.pkl")
        atomic_json(args.run_dir / "phase_checkpoint.json", {"status": "running", "phase": "test:aime_2025"})
        print("RECOVERY_TEST_START method=adas candidate=archive_21 splits=2025,2026 rows_each=30 gate_waiver=true", flush=True)

    def sample_completed(phase: str, row: dict[str, Any]) -> None:
        ledger.append({"event": "sample_completed", "phase": phase, **row})

    def checkpoint_completed(name: str, point: dict[str, Any], observed_runtime: Any) -> None:
        phases[name] = point
        atomic_json(args.run_dir / f"{name}_point.json", point)
        atomic_json(args.run_dir / "recovery_telemetry.json", {
            "runtime": observed_runtime.summary(), "request_ledger_audit": ledger.summary(),
        })
        save_rng(args.run_dir / f"rng_after_{name}.pkl")
        atomic_json(args.run_dir / "phase_checkpoint.json", {
            "status": "phase_completed", "completed_test_splits": list(phases), "phase": f"test:{name}",
        })
        print(f"RECOVERY_TEST_COMPLETE method=adas split={name} rows=30 correct={point['correct']}", flush=True)

    try:
        result = external.run_adas("aime", external.read_task_rows("aime", args.data_dir, 0, None), args,
            runtime.run_id, runtime, Path(inspection["baseline_source"]), test_recovery={
                "archive_path": args.run_dir / "retained_source/native_search/aime_0_run_archive.json",
                "candidate_index": inspection["candidate_index"],
                "candidate_code_sha256": inspection["candidate_code_sha256"],
                "selection_row": inspection["selection_metrics"], "before_tests": before_tests,
                "sample_completed": sample_completed, "checkpoint_completed": checkpoint_completed,
            })
        audit = ledger.summary()
        totals = runtime.usage["totals"]
        if audit["calls"] != totals["calls"] or any(audit["reported_response_usage"][key] != totals[key]
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
            raise RuntimeError("durable request usage does not match runtime counters")
        if set(runtime.usage["phases"]) != {"test:aime_2025", "test:aime_2026"} or set(phases) != {"aime_2025", "aime_2026"}:
            raise RuntimeError("ADAS recovery executed an unexpected phase")
        result["request_ledger_audit"] = audit
        return result
    finally:
        atomic_json(args.run_dir / "recovery_telemetry.json", {
            "runtime": runtime.summary(), "request_ledger_audit": ledger.summary(),
        })
        runtime.sync.close()
        asyncio.run(runtime.async_client.close())


def recover(args: argparse.Namespace) -> dict[str, Any]:
    args.source_run = args.source_run.resolve(); args.run_dir = args.run_dir.resolve(); args.data_dir = args.data_dir.resolve()
    inspection = inspect_source(args)
    if args.inspect_only:
        return inspection
    if not args.allow_selection_gate_waiver:
        raise RuntimeError("ADAS needs explicit --allow-selection-gate-waiver authorization")
    if float(os.environ.get("AIME_EXTERNAL_REQUEST_TIMEOUT_S", "600")) != 600:
        raise RuntimeError("recovery refuses a changed external request timeout")
    os.environ["AIME_EXTERNAL_REQUEST_TIMEOUT_S"] = "600"
    native.verify_served_model(args.endpoint)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    claim = source_claim_path(args.source_run); claim.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source_run": str(args.source_run), "run_dir": str(args.run_dir), "test_splits": list(YEARS),
               "candidate": "archive_21", "status": "claimed", "created_at_epoch": time.time()}
    with claim.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    manifest = {**inspection, "run_dir": str(args.run_dir), "claim": str(claim), "status": "preparing",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "gate_waiver_authorized": True,
        "formal_quality_gate": "WAIVED_BY_USER; NOT_FORMAL_PASS", "runner_sha256": external.sha256_file(Path(__file__)),
        "native_adapter_sha256": external.sha256_file(Path(external.__file__)),
        "recovery_source_commit": subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip(),
        "model_verification_report": os.environ.get("AIME_MODEL_VERIFICATION_REPORT"),
        "model_verification_report_sha256": external.sha256_file(Path(os.environ["AIME_MODEL_VERIFICATION_REPORT"]))
            if os.environ.get("AIME_MODEL_VERIFICATION_REPORT") else None,
        "run_session_record": os.environ.get("AIME_RUN_SESSION_RECORD"),
        "controls": {"model": external.MODEL, "seed": 0, "data_seed": 2026, "concurrency": 8,
            "max_model_len": 16384, "max_tokens": 6144, "temperature": 0.0, "top_p": 1.0,
            "thinking": False, "request_timeout_s": 600, "candidate": "archive_21"}}
    atomic_json(args.run_dir / "recovery_manifest.json", manifest)
    try:
        retained = args.run_dir / "retained_source"; retained.mkdir()
        for name in ("run_manifest.json", "selection_audit.json"):
            shutil.copyfile(args.source_run / name, retained / name)
        shutil.copytree(args.source_run / "native_search", retained / "native_search")
        atomic_json(args.run_dir / "selection_audit.json", read_json(args.source_run / "selection_audit.json"))
        verify_preserved_source(inspection)
        manifest["status"] = "running"; atomic_json(args.run_dir / "recovery_manifest.json", manifest)
        raw = execute_tests(args, inspection, DurableLedger(args.run_dir / "request_ledger.jsonl"))
        verify_preserved_source(inspection)
        result = {"schema": "aime_adas_quality_gate_waiver_stage_result_v1", "method": "adas",
            "formal_acceptance": "NOT_FORMAL_PASS", "variant": "best-candidate-dtest-gate-waiver",
            "candidate": inspection["candidate"], "selection_gate_waiver": True,
            "random_state_policy": inspection["random_state_policy"], "test": raw["test"],
            "native_method_config": raw.get("native_method_config"), "test_telemetry": raw["telemetry"],
            "request_ledger_audit": raw.get("request_ledger_audit"),
            "limitations": ["D_select 5% gate explicitly waived by user", "D_test sample/request/truncation gates explicitly waived",
                            "stage seed reset; original search RNG unavailable", "original search/select usage retained as historical audit"]}
        atomic_json(args.run_dir / "supplementary_result.json", result)
        manifest.update(status="completed_supplementary", source_verified_unchanged_after_run=True,
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(args.run_dir / "recovery_manifest.json", manifest)
        atomic_json(claim, {**payload, "status": manifest["status"]})
        return result
    except BaseException as exc:
        manifest.update(status="failed", failure=f"{type(exc).__name__}: {exc}"); atomic_json(args.run_dir / "recovery_manifest.json", manifest)
        native.persist_failure_traceback(args.run_dir, "adas_dtest_recovery", exc)
        atomic_json(claim, {**payload, "status": "failed", "failure": manifest["failure"]})
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("adas",), required=True)
    parser.add_argument("--source-run", type=Path, required=True); parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True); parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--endpoint", required=True); parser.add_argument("--seed", type=int, choices=(0,), default=0)
    parser.add_argument("--inspect-only", action="store_true"); parser.add_argument("--allow-selection-gate-waiver", action="store_true")
    parser.add_argument("--adas-generations", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(recover(parse_args()), ensure_ascii=False, indent=2), flush=True)
