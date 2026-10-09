#!/usr/bin/env python3
"""Fail-closed validation and mean/sample-std aggregation for one method's 18 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

METHODS = ("adas", "gdesigner")
AXES = ("breadth", "depth", "horizon", "parallel", "robustness")
MODEL = "Qwen/Qwen3.5-9B"
FROZEN_AIME_MANIFEST_SHA256 = "7e6501210c7689e1702e9786a9652222c5ca33193d11d4cde531ea84bdee2cfe"
UPSTREAMS = {
    "adas": ("https://github.com/ShengranHu/ADAS.git", "2702bee8fefda42255efc5be9f60e3bd3db96ae4"),
    "gdesigner": ("https://github.com/yanweiyue/GDesigner.git", "a6efcfa3b40bb4d9cbf46f883a95d62020bd8251"),
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode("utf-8")).hexdigest()


def _verify_aime_freeze(run_dir: Path, manifest: dict[str, Any], result: dict[str, Any]) -> None:
    """Verify blind-selection lock, access ledger, and pinned source provenance."""
    lock_path = run_dir / "selection_frozen.json"
    access_path = run_dir / "dtest_access_manifest.json"
    if not lock_path.is_file() or not access_path.is_file():
        raise ValueError(f"{run_dir}: missing D_test blind-freeze/access evidence")
    lock, access = _read(lock_path), _read(access_path)
    method = manifest["method"]
    expected_repo, expected_commit = UPSTREAMS[method]
    provenance = manifest.get("upstream_provenance", {})
    if (provenance.get("repository") != expected_repo
            or provenance.get("commit") != expected_commit
            or provenance.get("clean_worktree") is not True
            or lock.get("source_provenance") != provenance):
        raise ValueError(f"{run_dir}: unpinned or inconsistent upstream provenance")
    lock_hash = _sha256_file(lock_path)
    if (lock.get("schema") != "aime_dtest_blind_selection_lock_v2"
            or lock.get("protocol_version") != "aime_external_methods_v5_16k_shared_task_input"
            or lock.get("frozen_data_manifest_sha256") != FROZEN_AIME_MANIFEST_SHA256
            or manifest.get("frozen_data_manifest_sha256") != FROZEN_AIME_MANIFEST_SHA256
            or lock.get("method") != method
            or lock.get("dtest_loaded_before_lock") is not False
            or lock.get("search_size") != 60 or lock.get("selection_size") != 30
            or access.get("method") != method
            or access.get("selection_lock_sha256") != lock_hash
            or access.get("selection_frozen_at_epoch") != lock.get("created_at_epoch")):
        raise ValueError(f"{run_dir}: invalid method-specific selection lock/access ledger")
    if (manifest.get("selection_lock_sha256") != lock_hash
            or manifest.get("dtest_access_manifest_sha256") != _sha256_file(access_path)):
        raise ValueError(f"{run_dir}: D_test evidence hash differs from run manifest")
    data_dir_value = manifest.get("data_dir")
    if not data_dir_value:
        raise ValueError(f"{run_dir}: run manifest has no frozen AIME data directory")
    data_dir = Path(data_dir_value)
    manifest_file = data_dir / "frozen_aime_manifest.json"
    if not manifest_file.is_file() or _sha256_file(manifest_file) != FROZEN_AIME_MANIFEST_SHA256:
        raise ValueError(f"{run_dir}: frozen AIME data manifest is missing or has changed")
    frozen_data = _read(manifest_file)
    expected_data_hashes = {
        "aimo-validation-aime.jsonl": frozen_data["validation"]["source"]["sha256_git_content"],
        frozen_data["validation"]["search"]["path"]: frozen_data["validation"]["search"]["sha256_git_content"],
        frozen_data["validation"]["select"]["path"]: frozen_data["validation"]["select"]["sha256_git_content"],
    }
    for filename, spec in frozen_data["test"].items():
        expected_data_hashes[filename] = spec["source_sha256_git_content"]
        expected_data_hashes[spec["split_path"]] = spec["split_sha256_git_content"]
    if manifest.get("data_sha256") != expected_data_hashes:
        raise ValueError(f"{run_dir}: run manifest data hashes differ from pinned AIME manifest")
    for relative_path, expected_hash in manifest.get("data_sha256", {}).items():
        data_file = data_dir / relative_path
        if not data_file.is_file() or _sha256_file(data_file) != expected_hash:
            raise ValueError(f"{run_dir}: frozen AIME data hash mismatch: {relative_path}")
    ids_by_split = manifest.get("split_row_ids", {})
    canonical_validation = frozen_data["validation"]
    for name, split_key in (("search", "search"), ("select", "select")):
        spec = canonical_validation[split_key]
        rows = [json.loads(line) for line in (data_dir / spec["path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        expected_ids = ["validation:" + str(row.get("id", row.get("problem_idx", ""))) for row in rows]
        if ids_by_split.get(name) != expected_ids:
            raise ValueError(f"{run_dir}: {name} IDs do not match the canonical frozen AIME split")
    search_ids = ids_by_split.get("search", [])
    select_ids = ids_by_split.get("select", [])
    if (len(search_ids) != 60 or len(set(search_ids)) != 60
            or len(select_ids) != 30 or len(set(select_ids)) != 30
            or set(search_ids) & set(select_ids)
            or lock.get("search_ids_sha256") != _sha256_ids(search_ids)
            or lock.get("selection_ids_sha256") != _sha256_ids(select_ids)):
        raise ValueError(f"{run_dir}: lock does not match disjoint frozen D_search/D_select IDs")
    tests = access.get("test_splits", {})
    if set(tests) != {"aime_2025.jsonl", "aime_2026.jsonl"}:
        raise ValueError(f"{run_dir}: expected exactly the 2025 and 2026 AIME audit entries")
    validation_ids = set(search_ids) | set(select_ids)
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    validation_questions = set(lock.get("validation_question_sha256", []))
    for filename, record in tests.items():
        ids = list(map(str, record.get("ids", [])))
        questions = set(record.get("question_sha256", []))
        if (record.get("row_count") != 30 or len(ids) != 30 or len(set(ids)) != 30
                or record.get("opened_at_epoch", 0) < lock.get("created_at_epoch", float("inf"))
                or validation_ids.intersection(ids) or seen_ids.intersection(ids)
                or len(questions) != 30 or validation_questions.intersection(questions)
                or seen_questions.intersection(questions)):
            raise ValueError(f"{run_dir}: invalid, overlapping, or pre-freeze D_test audit for {filename}")
        if ids_by_split.get(Path(filename).stem) != ids:
            raise ValueError(f"{run_dir}: run manifest D_test IDs differ from access ledger for {filename}")
        if manifest.get("data_sha256", {}).get(filename) != record.get("sha256"):
            raise ValueError(f"{run_dir}: test data hash differs from access ledger for {filename}")
        test_spec = frozen_data["test"].get(filename)
        if (test_spec is None
                or record.get("frozen_data_manifest_sha256") != FROZEN_AIME_MANIFEST_SHA256
                or record.get("frozen_split_path") != test_spec["split_path"]
                or record.get("frozen_split_sha256") != test_spec["split_sha256_git_content"]
                or manifest.get("data_sha256", {}).get(test_spec["split_path"])
                != test_spec["split_sha256_git_content"]):
            raise ValueError(f"{run_dir}: canonical frozen test split differs from audit for {filename}")
        seen_ids.update(ids)
        seen_questions.update(questions)
    selected = lock.get("selected", {})
    if method == "adas":
        expected_selected = {key: value.get("candidate_id") for key, value in selected.items()}
        if expected_selected != {"Q": result.get("selected_Q"), "E": result.get("selected_E")}:
            raise ValueError(f"{run_dir}: ADAS evaluated candidates differ from the durable D_select freeze")
    else:
        if (selected.get("candidate_id") != result.get("selected_Q")
                or result.get("selected_E") not in ("same_candidate_as_Q", result.get("selected_Q"))):
            raise ValueError(f"{run_dir}: G-Designer test graph differs from the durable D_select freeze")
    for split_name, points in result.get("test", {}).items():
        for point_name in ("Q", "E"):
            observed = points.get(point_name, {}).get("selected_candidate")
            expected = (result.get("selected_Q") if point_name == "Q"
                        or result.get("selected_E") == "same_candidate_as_Q"
                        else result.get("selected_E"))
            if observed not in (expected, "same_candidate_as_Q"):
                raise ValueError(f"{run_dir}: {split_name}/{point_name} candidate does not match D_select lock")


def _check_rows(point: dict[str, Any], expected_ids: list[str], expected_count: int,
                label: str) -> None:
    rows = point.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"{label}: expected {expected_count} per-example audits")
    ids = [str(row.get("id", "")) for row in rows]
    if ids != expected_ids or len(set(ids)) != expected_count:
        raise ValueError(f"{label}: test audit IDs differ from frozen D_test order")
    for row in rows:
        correct = bool(row.get("correct"))
        parsed = bool(row.get("parser_valid"))
        expected_score = 2 if correct else (1 if parsed else 0)
        if int(row.get("score_0_1_2", -1)) != expected_score:
            raise ValueError(f"{label}: row {row['id']} violates the shared 0/1/2 score")
    mean_score = sum(int(row["score_0_1_2"]) for row in rows) / expected_count
    if not math.isclose(float(point.get("mean_score_0_1_2", math.nan)), mean_score, abs_tol=1e-12):
        raise ValueError(f"{label}: mean score differs from the shared 0/1/2 row audit")
    correct_count = sum(bool(row["correct"]) for row in rows)
    if int(point.get("correct", -1)) != correct_count:
        raise ValueError(f"{label}: aggregate correct count differs from row audit")
    accuracy = float(point.get("accuracy", math.nan))
    if not math.isfinite(accuracy) or not math.isclose(accuracy, correct_count / expected_count, abs_tol=1e-12):
        raise ValueError(f"{label}: accuracy is inconsistent with exact-match row counts")
    failure = float(point.get("failure_fraction", math.nan))
    if not math.isfinite(failure) or failure > 0.05:
        raise ValueError(f"{label}: failure fraction is invalid or exceeds 5%")
    if int(point.get("total_tokens", -1)) < 0 or int(point.get("calls", -1)) < 0:
        raise ValueError(f"{label}: missing/non-numeric token or call telemetry")
    calls = int(point.get("calls", 0))
    failed_calls = int(point.get("failed_calls", -1))
    truncated = int(point.get("completion_truncated", -1))
    if calls <= 0 or failed_calls < 0 or truncated < 0:
        raise ValueError(f"{label}: missing request-failure/truncation telemetry")
    if failed_calls / calls > 0.05 or truncated / calls > 0.05:
        raise ValueError(f"{label}: request failure or truncation rate exceeds 5%")
    expected_tpe = float(point["total_tokens"]) / expected_count
    if not math.isclose(float(point.get("tokens_per_example", expected_tpe)), expected_tpe, abs_tol=1e-9):
        raise ValueError(f"{label}: tokens-per-example does not match total token count")


def load_run(run_dir: Path, method: str, dataset: str, seed: int,
             axis: str | None = None) -> list[dict[str, Any]]:
    manifest_path = run_dir / "run_manifest.json"
    result_path = run_dir / "native_result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        raise ValueError(f"{run_dir}: missing run manifest or native result")
    manifest, result = _read(manifest_path), _read(result_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"{run_dir}: status is not complete")
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    if digest != manifest.get("native_result_sha256"):
        raise ValueError(f"{run_dir}: result checksum mismatch")
    runner_path = Path(__file__).with_name("native_external_methods.py")
    if manifest.get("runner_sha256") != _sha256_file(runner_path):
        raise ValueError(f"{run_dir}: external-method runner differs from the hashed execution source")
    expected_protocol = "aime_external_methods_v5_16k_shared_task_input" if dataset == "aime" else "masbench_external_five_axis_v1"
    if (manifest.get("method") != method or manifest.get("dataset") != dataset
            or int(manifest.get("seed", -1)) != seed or manifest.get("protocol_version") != expected_protocol
            or manifest.get("model") != MODEL or int(manifest.get("context_limit", -1)) != 16384
            or int(manifest.get("output_limit", -1)) != 6144 or int(manifest.get("data_seed", -1)) != 2026
            or manifest.get("selection_split") != "D_select" or manifest.get("test_split") != "D_test"):
        raise ValueError(f"{run_dir}: method/model/seed/protocol controls do not match the frozen experiment")
    if manifest.get("score_mapping") != "2=exact match; 1=parseable wrong; 0=unparseable":
        raise ValueError(f"{run_dir}: answer score mapping is not the shared 0/1/2 protocol")
    if result.get("method") != method:
        raise ValueError(f"{run_dir}: result method differs from its manifest")
    if result.get("controls", {}).get("score_mapping") != manifest.get("score_mapping"):
        raise ValueError(f"{run_dir}: result controls and manifest disagree on answer scoring")
    if dataset == "aime":
        _verify_aime_freeze(run_dir, manifest, result)
        expected_tests = {"aime_2025": 30, "aime_2026": 30}
    else:
        if axis not in AXES or manifest.get("axis") != axis:
            raise ValueError(f"{run_dir}: invalid/mismatched MASBench axis")
        expected_tests = {"test": 60}
    outputs = result.get("test", {})
    if set(outputs) != set(expected_tests):
        raise ValueError(f"{run_dir}: final test split set is incomplete or unexpected")
    expected_ids_by_split = manifest.get("split_row_ids", {})
    records: list[dict[str, Any]] = []
    for split_name, count in expected_tests.items():
        expected_ids = [str(item) for item in expected_ids_by_split.get(split_name, [])]
        if len(expected_ids) != count:
            raise ValueError(f"{run_dir}: manifest is missing frozen IDs for {split_name}")
        output = outputs[split_name]
        for operating_point in ("Q", "E"):
            point = output.get(operating_point)
            if not isinstance(point, dict):
                raise ValueError(f"{run_dir}: missing {split_name}/{operating_point} result")
            _check_rows(point, expected_ids, count, f"{run_dir.name}/{split_name}/{operating_point}")
            records.append({"dataset": dataset, "axis": axis or "-", "test_split": split_name,
                "operating_point": operating_point, "seed": seed,
                "accuracy": float(point["accuracy"]), "total_tokens": int(point["total_tokens"]),
                "tokens_per_example": float(point["tokens_per_example"]), "calls": int(point["calls"]),
                "failure_fraction": float(point["failure_fraction"])})
    return records


def iter_runs(root: Path, method: str):
    for seed in (0, 1, 2):
        yield root / "aime" / method / f"seed_{seed}", "aime", seed, None
    for axis in AXES:
        for seed in (0, 1, 2):
            yield root / "masbench" / method / axis / f"seed_{seed}", "masbench", seed, axis


def summarize(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["dataset"], record["axis"], record["test_split"], record["operating_point"])
        groups.setdefault(key, []).append(record)
    expected_groups = {
        ("aime", "-", split, point)
        for split in ("aime_2025", "aime_2026") for point in ("Q", "E")
    } | {
        ("masbench", axis, "test", point)
        for axis in AXES for point in ("Q", "E")
    }
    if set(groups) != expected_groups:
        raise ValueError(f"expected all {len(expected_groups)} AIME/MASBench reporting groups")
    summary_rows = []
    for key, values in sorted(groups.items()):
        if len(values) != 3 or {int(value["seed"]) for value in values} != {0, 1, 2}:
            raise ValueError(f"{key}: expected exactly seeds 0/1/2")
        accuracies = [value["accuracy"] for value in values]
        tokens = [value["tokens_per_example"] for value in values]
        calls = [value["calls"] for value in values]
        summary_rows.append({"method": method, "dataset": key[0], "axis": key[1],
            "test_split": key[2], "operating_point": key[3], "n_seeds": 3,
            "accuracy_mean": statistics.mean(accuracies), "accuracy_sample_std": statistics.stdev(accuracies),
            "tokens_per_example_mean": statistics.mean(tokens), "tokens_per_example_sample_std": statistics.stdev(tokens),
            "calls_mean": statistics.mean(calls), "calls_sample_std": statistics.stdev(calls)})
    verified_runs = len({(record["dataset"], record["axis"], record["seed"]) for record in records})
    if verified_runs != 18:
        raise ValueError(f"expected all 18 method runs, found {verified_runs}")
    return {"method": method, "model": MODEL, "context_limit": 16384, "output_limit": 6144,
        "data_seed": 2026, "search_seeds": [0, 1, 2], "expected_runs": 18,
        "verified_runs": verified_runs, "tables": summary_rows, "raw_runs": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/external_native_methods_a10040_v1"))
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--check-run", type=Path)
    parser.add_argument("--dataset", choices=("aime", "masbench"))
    parser.add_argument("--axis", choices=AXES)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    if args.check_run:
        if args.dataset is None or args.seed is None or (args.dataset == "masbench" and args.axis is None):
            parser.error("--check-run requires --dataset, --seed, and --axis for MASBench")
        load_run(args.check_run, args.method, args.dataset, args.seed, args.axis)
        print(f"PASS verified complete run: {args.check_run}")
        return 0
    records = []
    for run_dir, dataset, seed, axis in iter_runs(args.root, args.method):
        records.extend(load_run(run_dir, args.method, dataset, seed, axis))
    report = summarize(records, args.method)
    output_dir = args.root / args.method
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = list(report["tables"][0])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["tables"])
    print(f"PASS: verified 18 runs; wrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
