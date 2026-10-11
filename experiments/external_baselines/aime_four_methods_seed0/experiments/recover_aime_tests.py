#!/usr/bin/env python3
"""Supplement only the unopened 2026 test of a frozen, failed AIME run.

Contracts: search/select and the complete 2025 CSV are immutable inputs. Only
2026 receives model requests. Missing historical usage remains null. This is
a stage continuation, never a formal pass or an exact process/RNG restore.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
from typing import Any, Callable
import uuid

import native_aime_formal as native

YEAR_OLD = "aime_2025.jsonl"
YEAR_NEW = "aime_2026.jsonl"
NATIVE_ROOT = Path("workspace/rpas_aime_native/MATH")
NULL_USAGE = {key: None for key in native.empty_phase_usage()}
TIMEOUTS = {
    "AIME_AFLOW_REQUEST_TIMEOUT_S": "600", "AIME_AFLOW_SAMPLE_TIMEOUT_S": "1200",
    "AIME_AFLOW_CODE_TIMEOUT_S": "30", "AIME_MAAS_REQUEST_TIMEOUT_S": "600",
    "AIME_MAAS_SAMPLE_TIMEOUT_S": "1500", "AIME_MAAS_CODE_TIMEOUT_S": "60",
    "AIME_MAAS_SAMPLE_RETRIES": "3", "AIME_MAX_NUM_SEQS": "8", "AIME_TP_SIZE": "1",
}


def atomic_json(path: Path, payload: Any) -> None:
    """Durable atomic checkpoints; readers never see half a JSON document."""
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def source_claim_path(source_run: Path) -> Path:
    identity = hashlib.sha256(str(source_run.resolve()).encode()).hexdigest()
    return source_run.resolve().parent / "test_recovery_claims" / f"{identity}.json"


def claim_source(source_run: Path, run_dir: Path) -> Path:
    """A permanent exclusive claim forbids accidentally computing 2026 twice."""
    path = source_claim_path(source_run)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_run": str(source_run.resolve()), "run_dir": str(run_dir.resolve()),
        "test_split": YEAR_NEW, "status": "claimed", "created_at_epoch": time.time(),
    }
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def verified_old_rows(data_dir: Path, lock: dict[str, Any], access: dict[str, Any]) -> list[dict[str, Any]]:
    """Verify the already opened split, without changing/reopening its audit."""
    manifest = native.load_frozen_aime_manifest(data_dir)
    spec = manifest["test"][YEAR_OLD]
    rows, source_hash = native.read_verified_jsonl(
        data_dir, YEAR_OLD, spec["source_sha256_git_content"], 30,
    )
    frozen, frozen_hash = native.read_verified_jsonl(
        data_dir, spec["split_path"], spec["split_sha256_git_content"], 30,
    )
    if native._question_answer_map(rows, YEAR_OLD) != native._question_answer_map(frozen, spec["split_path"]):
        raise RuntimeError("preserved 2025 source differs from its canonical split")
    rows = native.namespace_aime_rows(rows, Path(YEAR_OLD).stem)
    evidence = access["test_splits"][YEAR_OLD]
    expected = {
        "sha256": source_hash, "frozen_split_path": spec["split_path"],
        "frozen_split_sha256": frozen_hash, "row_count": 30,
        "frozen_data_manifest_sha256": native.FROZEN_AIME_MANIFEST_SHA256,
        "ids": [str(row["id"]) for row in rows],
        "question_sha256": [native._question_sha256(row) for row in rows],
    }
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise RuntimeError("preserved 2025 access manifest does not match frozen data")
    if evidence.get("opened_at_epoch", 0) <= lock["created_at_epoch"]:
        raise RuntimeError("test access predates candidate freeze")
    if set(expected["question_sha256"]).intersection(lock["validation_question_sha256"]):
        raise RuntimeError("2025 test overlaps validation")
    return rows


def inspect_source(args: argparse.Namespace) -> dict[str, Any]:
    """Read-only, CPU provenance/data/scoring checks; no model or file writes."""
    source = args.source_run.resolve()
    if args.method not in {"aflow", "maas"} or args.seed != 0:
        raise ValueError("this recovery supports only AFlow/MaAS seed 0")
    if source_claim_path(source).exists():
        raise RuntimeError(f"2026 recovery already claimed: {source_claim_path(source)}")
    if args.run_dir.exists():
        raise FileExistsError(f"recovery output must be fresh: {args.run_dir}")
    run = read_json(source / "run_manifest.json")
    split = read_json(source / "split_manifest.json")
    lock = read_json(source / "selection_frozen.json")
    access = read_json(source / "dtest_access_manifest.json")
    selection = read_json(source / "selection_audit.json")
    required = {
        "method": args.method, "protocol_version": native.PROTOCOL_VERSION,
        "search_seed": 0, "data_seed": 2026, "model": native.MODEL,
        "context_limit": 16384, "output_limit": 6144,
        "search_size": 60, "selection_size": 30,
        "answer_protocol": native.ANSWER_PARSER,
        "answer_protocol_sha256": native.sha256_file(Path(native.__file__).with_name("answer_protocol.py")),
        "task_contract": native.task_contract_manifest(),
    }
    if any(run.get(k) != v or split.get(k) != v for k, v in required.items()):
        raise RuntimeError("source run/split protocol differs from the frozen seed-0 protocol")
    if run.get("status") != "failed" or run.get("test_size_each") != 30:
        raise RuntimeError("source must be the failed formal 30-row attempt")
    if "AIME evaluator row order/content mismatch" not in run.get("failure", ""):
        raise RuntimeError("recovery accepts only the diagnosed D_test scoring crash")
    if (source / "native_result.json").exists():
        raise RuntimeError("source has a result already; refusing repeated testing")
    search, select = native.freeze_split(args.data_dir, 2026, 60, 30)
    search = native.namespace_aime_rows(search, "validation")
    select = native.namespace_aime_rows(select, "validation")
    lock_required = {
        "schema": "aime_dtest_blind_selection_lock_v2", "method": args.method,
        "protocol_version": native.PROTOCOL_VERSION, "dtest_loaded_before_lock": False,
        "frozen_data_manifest_sha256": native.FROZEN_AIME_MANIFEST_SHA256,
        "search_size": 60, "selection_size": 30,
        "search_ids_sha256": native._sha256_ids(search),
        "selection_ids_sha256": native._sha256_ids(select),
        "validation_question_sha256": sorted({native._question_sha256(row) for row in search + select}),
    }
    if any(lock.get(k) != v for k, v in lock_required.items()):
        raise RuntimeError("source candidate lock differs from frozen validation")
    provenance = native.verify_pinned_upstream(args.method, native.baseline_source_path(args, args.method))
    if any(provenance != value for value in (run.get("upstream_provenance"), split.get("upstream_provenance"), lock.get("source_provenance"))):
        raise RuntimeError("source provenance differs from pinned clean upstream")
    if (access.get("schema") != "aime_dtest_access_manifest_v1"
            or access.get("method") != args.method
            or access.get("selection_lock_sha256") != native.sha256_file(source / "selection_frozen.json")
            or access.get("selection_frozen_at_epoch") != lock["created_at_epoch"]
            or set(access.get("test_splits", {})) != {YEAR_OLD}):
        raise RuntimeError("only completed 2025 and unopened 2026 may be recovered")
    old_rows = verified_old_rows(args.data_dir, lock, access)
    repo = source / f"{args.method}_repo"
    artifacts = {name: native.sha256_file(source / name) for name in (
        "run_manifest.json", "split_manifest.json", "selection_frozen.json",
        "selection_audit.json", "dtest_access_manifest.json",
    )}
    if args.method == "aflow":
        q, e = lock["selected"]["Q"], lock["selected"]["E"]
        if q["round"] != e["round"]:
            raise RuntimeError("recovery currently requires AFlow Q/E to share one frozen candidate")
        candidate = next((row for row in selection["candidate_rows"] if row.get("round") == q["round"]), None)
        if (not candidate or candidate.get("status") != "evaluated"
                or candidate["shared_metrics"]["num_examples"] != 30
                or candidate["shared_metrics"]["score"] != q["score"]
                or e["score"] != q["score"]
                or candidate["output_audit"]["failure_fraction"] > native.MAX_SAMPLE_FAILURE_RATE):
            raise RuntimeError("frozen AFlow selection did not pass its original sample gate")
        candidate_paths = [repo / NATIVE_ROOT / "workflows" / f"round_{q['round']}" / name for name in ("graph.py", "prompt.py")]
        old_log = source / "aflow_test/aime_2025_Q"
        original_input = repo / "data/datasets/math_test.jsonl"
    else:
        checkpoint = repo / NATIVE_ROOT / "train/round_1/MATH_controller_sample1.pth"
        if (native.sha256_file(checkpoint) != lock["selected"]["controller_sha256"]
                or selection.get("controller_sha256") != lock["selected"]["controller_sha256"]
                or selection["selection_metrics"]["num_examples"] != 30
                or selection["selection_metrics"]["score"] != lock["selected"]["selection_score"]
                or selection["output_audit"]["failure_fraction"] > native.MAX_SAMPLE_FAILURE_RATE):
            raise RuntimeError("MaAS frozen controller/selection gate differs from original")
        candidate_paths = [checkpoint]
        old_log = repo / NATIVE_ROOT / "test/round_1"
        original_input = repo / "maas/ext/maas/data/math_test.jsonl"
    rendered = native.native_math_rows(old_rows, Path(YEAR_OLD).stem)
    if native.read_jsonl(original_input) != rendered:
        raise RuntimeError("preserved native test input differs from the frozen 2025 order/golds")
    metrics = native.shared_aime_metrics(old_log, rendered)
    audit = native.audit_aflow_outputs(old_log)
    for path in candidate_paths + [original_input, Path(audit["csv"])]:
        artifacts[str(path.relative_to(source))] = native.sha256_file(path)
    python_hash = native.sha256_python_tree(repo)
    old_point = {
        **metrics, "output_audit": audit, "resource_usage": dict(NULL_USAGE),
        "calls": None, "prompt_tokens": None, "completion_tokens": None,
        "total_tokens": None, "avg_cost": None, "total_cost": None,
        "cost_unit": "model_tokens", "resource_audit_status": "UNKNOWN_HISTORICAL",
        "reused_without_model_requests": True,
        "sample_gate_pass": audit["failure_fraction"] <= native.MAX_SAMPLE_FAILURE_RATE,
    }
    return {
        "schema": "aime_stage_recovery_inspection_v1", "method": args.method,
        "source_run": str(source), "source_artifacts_sha256": artifacts,
        "source_repo_python_sha256": python_hash, "baseline_provenance": provenance,
        "frozen_selection": lock["selected"], "reused_2025": old_point,
        "remaining_test_split": YEAR_NEW, "original_status": run["status"],
        "original_failure": run["failure"], "missing_historical_usage": ["search", "test:aime_2025"],
        "original_runner_sha256": run.get("runner_sha256"),
        "historical_candidate_hash_lock": "controller_checkpoint" if args.method == "maas" else "round_only; Python hashes first recorded at recovery",
        "formal_acceptance": "NOT_FORMAL_PASS",
        "random_state_policy": "stage_reseed_seed0; original_rng_state_unavailable; not_exact_process_resume",
    }


def verify_preserved_source(inspection: dict[str, Any]) -> None:
    source = Path(inspection["source_run"])
    if any(native.sha256_file(source / relative) != expected for relative, expected in inspection["source_artifacts_sha256"].items()):
        raise RuntimeError("preserved source artifacts changed during recovery")
    if native.sha256_python_tree(source / f"{inspection['method']}_repo") != inspection["source_repo_python_sha256"]:
        raise RuntimeError("preserved source Python changed during recovery")


class DurableLedger:
    """Fsync each request/sample event, including cancellation and unknown usage."""
    def __init__(self, path: Path):
        self.path = path
        self.sequence = 0

    def append(self, event: dict[str, Any]) -> None:
        self.sequence += 1
        payload = {"sequence": self.sequence, "at_utc": datetime.now(timezone.utc).isoformat(), **event}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def summary(self) -> dict[str, Any]:
        requests: dict[str, dict[str, Any]] = {}
        lines = self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []
        for line in lines:
            event = json.loads(line)
            if event["event"].startswith("request_"):
                requests[event["request_id"]] = event
        totals = {key: 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        unknown = 0
        failures = 0
        for event in requests.values():
            if event["event"] == "request_failed":
                failures += 1
            values = event.get("usage")
            if event.get("usage_status") != "REPORTED":
                unknown += 1
            if values:
                for key in totals:
                    if values.get(key) is not None:
                        totals[key] += int(values[key])
        return {"calls": len(requests), "failed_or_cancelled_calls": failures,
                "unknown_usage_calls": unknown, "reported_response_usage": totals,
                "usage_complete": unknown == 0}


def install_request_ledger(ledger: DurableLedger, completion_class: Any = None) -> None:
    if completion_class is None:
        from openai.resources.chat.completions import AsyncCompletions
        completion_class = AsyncCompletions
    original = completion_class.create

    @wraps(original)
    async def tracked(self: Any, *args: Any, **kwargs: Any) -> Any:
        identity = uuid.uuid4().hex
        phase = native.current_phase()
        ledger.append({"event": "request_started", "request_id": identity, "phase": phase,
                       "model": kwargs.get("model"), "max_tokens": kwargs.get("max_tokens")})
        try:
            response = await original(self, *args, **kwargs)
        except BaseException as exc:
            ledger.append({"event": "request_failed", "request_id": identity, "phase": phase,
                           "exception_type": type(exc).__name__, "usage": None,
                           "usage_status": "UNKNOWN_NO_RESPONSE"})
            raise
        usage = getattr(response, "usage", None)
        values = {key: getattr(usage, key, None) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        choices = getattr(response, "choices", []) or []
        ledger.append({"event": "request_completed", "request_id": identity, "phase": phase,
                       "server_request_id": getattr(response, "id", None), "usage": values,
                       "usage_status": "REPORTED" if all(value is not None for value in values.values()) else "UNKNOWN_INCOMPLETE_RESPONSE",
                       "finish_reason": getattr(choices[0], "finish_reason", None) if choices else None})
        return response

    completion_class.create = tracked


def install_sample_ledger(benchmark_class: Any, ledger: DurableLedger) -> None:
    original = benchmark_class.evaluate_problem

    @wraps(original)
    async def tracked(self: Any, problem: dict[str, Any], graph: Any) -> Any:
        identity = problem.get("aime_id", native._question_sha256(problem))
        try:
            result = await original(self, problem, graph)
        except BaseException as exc:
            ledger.append({"event": "sample_interrupted", "id": identity, "exception_type": type(exc).__name__})
            raise
        ledger.append({"event": "sample_completed", "id": identity, "question": result[0],
                       "prediction": result[1], "expected_output": result[2], "native_score": result[3]})
        return result

    benchmark_class.evaluate_problem = tracked


def save_rng(path: Path) -> None:
    import random
    payload: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np
        payload["numpy"] = np.random.get_state()
    except ImportError:
        pass
    try:
        import torch
        payload["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    with path.open("xb") as stream:
        pickle.dump(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())


def execute_remaining(args: argparse.Namespace, inspection: dict[str, Any], ledger: DurableLedger) -> dict[str, Any]:
    """Reuse the isolated native evaluation logic, call only 2026 exactly once."""
    repo = args.run_dir / f"{args.method}_repo"
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    if args.method == "maas":
        os.environ["METAGPT_PROJECT_ROOT"] = str(repo)
        native.make_maas_config(repo, args.endpoint, 6144)
    usage = native.patch_aflow_runtime(6144, 8) if args.method == "aflow" else native.patch_maas_runtime(8, 6144)
    # Patches are the existing protocol's executor adapters. They must be
    # idempotent on the retained Python bytes, including selected templates.
    if native.sha256_python_tree(repo) != inspection["source_repo_python_sha256"]:
        raise RuntimeError("runtime patch changed retained native Python; refusing model requests")
    install_request_ledger(ledger)
    native.seed_everything(0)
    test_name = Path(YEAR_NEW).stem
    log_dir = args.run_dir / "test_2026"
    log_dir.mkdir()
    if args.method == "aflow":
        from benchmarks.math import MATHBenchmark
        from scripts.evaluator import Evaluator
        from scripts.optimizer_utils.graph_utils import GraphUtils
        install_sample_ledger(MATHBenchmark, ledger)
        config = native.make_aflow_config(args.endpoint)
        round_number = inspection["frozen_selection"]["Q"]["round"]
        utils = GraphUtils(str(NATIVE_ROOT))
        graph = utils.load_graph(round_number, str(NATIVE_ROOT / "workflows"))
        evaluator = Evaluator(eval_path=str(log_dir))
        dataset_path = repo / "data/datasets/math_test.jsonl"
        phase = f"test:{test_name}:Q"
        evaluate: Callable[[], Any] = lambda: native.evaluate_aflow_graph(
            evaluator=evaluator, graph=graph, config=config, dataset_path=dataset_path,
            log_dir=log_dir, is_test=True,
        )
    else:
        from maas.configs.models_config import ModelsConfig
        from maas.ext.maas.benchmark.experiment_configs import EXPERIMENT_CONFIGS
        from maas.ext.maas.benchmark.math import MATHBenchmark
        from maas.ext.maas.scripts.optimizer import Optimizer
        install_sample_ledger(MATHBenchmark, ledger)
        config = EXPERIMENT_CONFIGS["MATH"]
        llm_config = ModelsConfig.default().get("qwen35_9b")
        if llm_config is None:
            raise RuntimeError("MaAS model config was not loaded")
        optimizer = Optimizer(
            dataset=config.dataset, question_type=config.question_type,
            opt_llm_config=llm_config, exec_llm_config=llm_config, operators=config.operators,
            optimized_path=str(NATIVE_ROOT.parent), sample=1, round=1, batch_size=8,
            lr=0.01, is_textgrad=False,
        )
        # Only redirect output bytes. Graph imports and controller load paths
        # remain those of the retained frozen workflow/checkpoint.
        original_directory = optimizer.graph_utils.create_round_directory
        def recovery_directory(graph_path: str, round_number: int) -> str:
            if Path(graph_path) == NATIVE_ROOT / "test":
                return str(log_dir)
            return original_directory(graph_path, round_number)
        optimizer.graph_utils.create_round_directory = recovery_directory
        dataset_path = repo / "maas/ext/maas/data/math_test.jsonl"
        phase = f"test:{test_name}"
        evaluate = lambda: asyncio.run(optimizer.test())
    save_rng(args.run_dir / "rng_before_2026.pkl")
    rows = native.load_aime_test_rows_after_freeze(args.data_dir, args.run_dir, args.method, YEAR_NEW, 30)
    native.write_jsonl(dataset_path, native.native_math_rows(rows, test_name))
    atomic_json(args.run_dir / "phase_checkpoint.json", {"phase": phase, "status": "running", "num_examples": 30})
    print(f"RECOVERY_TEST_START method={args.method} phase={phase} rows=30 reused_2025=true", flush=True)
    with native.phase_scope(phase, usage["phase_wall_seconds"]):
        native_value = evaluate()
    native_score = float(native_value[0]) if args.method == "aflow" else float(native_value)
    save_rng(args.run_dir / "rng_after_2026.pkl")
    # Persist usage before CSV scoring, so a scorer failure cannot lose it.
    guard = usage.pop("context_guard", None) or usage.pop("_context_guard", None)
    request_audit = ledger.summary()
    snapshot = {**usage, "context_guard": guard.summary() if guard is not None else None,
                "request_ledger_audit": request_audit}
    atomic_json(args.run_dir / "recovery_telemetry.json", snapshot)
    if request_audit["calls"] != usage["calls"] or any(
        request_audit["reported_response_usage"][key] != usage[key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        raise RuntimeError("durable request usage does not match native runtime counters")
    metrics = native.shared_aime_metrics(log_dir, native.native_math_rows(rows, test_name))
    audit = native.audit_aflow_outputs(log_dir)
    resource = native.phase_snapshot(usage, phase)
    point = {
        **metrics, "output_audit": audit, "native_score": native_score,
        "resource_usage": resource, **resource, "cost_unit": "model_tokens",
        "avg_cost": resource["total_tokens"] / 30, "total_cost": resource["total_tokens"],
        "phase": phase, "sample_gate_pass": audit["failure_fraction"] <= native.MAX_SAMPLE_FAILURE_RATE,
        "request_failure_fraction": usage["failed_calls"] / max(1, usage["calls"]),
        "truncation_fraction": usage["completion_truncated"] / max(1, usage["calls"]),
        "resource_audit_status": "COMPLETE_RETURNED_RESPONSES" if request_audit["usage_complete"] else "REPORTED_RESPONSES_ONLY",
        "request_ledger_audit": request_audit,
    }
    if args.method == "aflow":
        point["native_avg_cost"] = float(native_value[1])
        point["native_total_cost"] = float(native_value[2])
    atomic_json(args.run_dir / "phase_checkpoint.json", {"phase": phase, "status": "complete", "point": point})
    return point


def recover(args: argparse.Namespace) -> dict[str, Any]:
    args.source_run = args.source_run.resolve()
    args.run_dir = args.run_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    inspection = inspect_source(args)
    if args.inspect_only:
        return inspection
    if args.method == "maas" and not args.allow_stage_reseed:
        raise RuntimeError("MaAS needs explicit --allow-stage-reseed authorization; historical RNG was not saved")
    for key, value in TIMEOUTS.items():
        if key in os.environ and float(os.environ[key]) != float(value):
            raise RuntimeError(f"recovery refuses protocol timeout/sequence changes: {key}")
        os.environ[key] = value
    native.verify_served_model(args.endpoint)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    claim = claim_source(args.source_run, args.run_dir)
    manifest = {
        **inspection, "run_dir": str(args.run_dir), "status": "preparing",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "claim": str(claim),
        "runner_sha256": native.sha256_file(Path(__file__)),
        "native_adapter_sha256": native.sha256_file(Path(native.__file__)),
        "recovery_source_commit": subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "model_verification_report": os.environ.get("AIME_MODEL_VERIFICATION_REPORT"),
        "model_verification_report_sha256": native.sha256_file(Path(os.environ["AIME_MODEL_VERIFICATION_REPORT"]))
            if os.environ.get("AIME_MODEL_VERIFICATION_REPORT") else None,
        "run_session_record": os.environ.get("AIME_RUN_SESSION_RECORD"),
        "stage_reseed_authorized": args.allow_stage_reseed,
        "controls": {"model": native.MODEL, "seed": 0, "data_seed": 2026,
                     "max_model_len": 16384, "max_tokens": 6144, "concurrency": 8,
                     "temperature": 0.0, "top_p": 1.0, "thinking": False, **TIMEOUTS},
        "formal_quality_gate": "NOT_RUN_INCOMPLETE_HISTORICAL_USAGE",
    }
    atomic_json(args.run_dir / "recovery_manifest.json", manifest)
    atomic_json(args.run_dir / "reused_2025_metrics.json", inspection["reused_2025"])
    try:
        for name in ("selection_frozen.json", "dtest_access_manifest.json", "selection_audit.json"):
            shutil.copyfile(args.source_run / name, args.run_dir / name)
        repo_name = f"{args.method}_repo"
        shutil.copytree(args.source_run / repo_name, args.run_dir / repo_name,
                        ignore=shutil.ignore_patterns("__pycache__", ".git"))
        verify_preserved_source(inspection)
        manifest["status"] = "running"
        atomic_json(args.run_dir / "recovery_manifest.json", manifest)
        os.environ["AIME_RUN_ID"] = f"aime:recovery:{args.method}:{args.run_dir.name}"
        os.environ["MASBENCH_RUN_ID"] = os.environ["AIME_RUN_ID"]
        new_point = execute_remaining(args, inspection, DurableLedger(args.run_dir / "request_ledger.jsonl"))
        verify_preserved_source(inspection)
        old_point = inspection["reused_2025"]
        result = {
            "schema": "aime_supplementary_stage_recovery_result_v1", "method": args.method,
            "formal_acceptance": "NOT_FORMAL_PASS", "historical_usage_complete": False,
            "random_state_policy": inspection["random_state_policy"],
            "frozen_selection": inspection["frozen_selection"],
            "test": {Path(YEAR_OLD).stem: {"Q": old_point, "E": {**old_point, "same_candidate_as": "Q"}},
                     Path(YEAR_NEW).stem: {"Q": new_point, "E": {**new_point, "same_candidate_as": "Q"}}},
            "limitations": ["historical search/2025 precise usage unavailable", "stage seed reset; original RNG unavailable"]
                + (["2025 sample failure rate exceeds original 5% gate"] if not old_point["sample_gate_pass"] else []),
        }
        atomic_json(args.run_dir / "supplementary_result.json", result)
        manifest["status"] = "completed_supplementary"
        manifest["source_verified_unchanged_after_run"] = True
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.run_dir / "recovery_manifest.json", manifest)
        atomic_json(claim, {"source_run": str(args.source_run), "run_dir": str(args.run_dir), "status": manifest["status"], "test_split": YEAR_NEW})
        return result
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        atomic_json(args.run_dir / "recovery_manifest.json", manifest)
        native.persist_failure_traceback(args.run_dir, "stage_recovery", exc)
        atomic_json(claim, {"source_run": str(args.source_run), "run_dir": str(args.run_dir), "status": "failed", "test_split": YEAR_NEW, "failure": manifest["failure"]})
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("aflow", "maas"), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--allow-stage-reseed", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(recover(parse_args()), ensure_ascii=False, indent=2), flush=True)
