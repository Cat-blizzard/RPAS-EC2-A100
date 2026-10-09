#!/usr/bin/env python3
"""Post-run structural and quality gate for one native AIME baseline run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ANSWER_PARSER = "answer_protocol_v4.extract_aime_answer"
PROTOCOL_VERSION = "aime_main_protocol_v8_16k_shared_task_input"
FROZEN_AIME_MANIFEST_SHA256 = "7e6501210c7689e1702e9786a9652222c5ca33193d11d4cde531ea84bdee2cfe"
MAX_SAMPLE_FAILURE_RATE = 0.05
UPSTREAMS = {
    "aflow": ("https://github.com/FoundationAgents/AFlow.git", "3f457218fc716093fe53f6df8a5d5e6379d66346"),
    "maas": ("https://github.com/bingreeky/MaAS.git", "987f3c1bc9a96e844fe090db3791446e3ef0f5c7"),
}


def load(path: Path):
    if not path.exists():
        raise SystemExit(f"MISSING: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_ids(values: list[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_ordinal_scores(metrics: dict, label: str, expected_count: int = 30) -> list[str]:
    errors: list[str] = []
    rows = metrics.get("rows")
    ordinal = metrics.get("score_0_1_2")
    if not isinstance(rows, list) or len(rows) != expected_count or not isinstance(ordinal, list) or len(ordinal) != expected_count:
        return [f"{label} is missing {expected_count} per-example 0/1/2 score audits"]
    expected_scores = [2 if bool(row.get("correct")) else (1 if bool(row.get("parser_valid")) else 0)
                       for row in rows]
    try:
        actual_scores = [int(value) for value in ordinal]
    except (TypeError, ValueError):
        actual_scores = []
    if actual_scores != expected_scores or any(value not in (0, 1, 2) for value in actual_scores):
        errors.append(f"{label} per-example scores violate the shared 0/1/2 mapping")
    mean_score = sum(expected_scores) / len(expected_scores)
    try:
        actual_mean = float(metrics.get("mean_score_0_1_2", -1))
    except (TypeError, ValueError):
        actual_mean = -1.0
    if abs(actual_mean - mean_score) > 1e-12:
        errors.append(f"{label} mean 0/1/2 score differs from its per-example audit")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("aflow", "maas"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    root = args.run_dir.resolve()
    manifest = load(root / "run_manifest.json")
    split = load(root / "split_manifest.json")
    result = load(root / "native_result.json")
    errors: list[str] = []
    warnings: list[str] = []

    if manifest.get("protocol_version") != PROTOCOL_VERSION or split.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("run/split manifest protocol version mismatch")
    if manifest.get("method") != args.method or manifest.get("search_seed") != args.seed:
        errors.append("run manifest method/seed mismatch")
    if manifest.get("data_seed") != 2026:
        errors.append(f"data_seed={manifest.get('data_seed')} != 2026")
    if manifest.get("search_size") != 60 or manifest.get("selection_size") != 30:
        errors.append("search/selection split is not 60/30")
    if len(split.get("search_ids", [])) != 60 or len(split.get("selection_ids", [])) != 30:
        errors.append("split manifest does not contain 60/30 disjoint rows")
    if set(split.get("search_ids", [])) & set(split.get("selection_ids", [])):
        errors.append("search and selection IDs overlap")

    lock_path = root / "selection_frozen.json"
    access_path = root / "dtest_access_manifest.json"
    lock = {}
    access = {}
    if not lock_path.is_file() or not access_path.is_file():
        errors.append("missing durable D_test blind-freeze/access artifacts")
    else:
        lock_bytes = lock_path.read_bytes()
        lock = json.loads(lock_bytes)
        access = load(access_path)
        lock_hash = hashlib.sha256(lock_bytes).hexdigest()
        if (lock.get("schema") != "aime_dtest_blind_selection_lock_v2"
                or lock.get("method") != args.method
                or lock.get("protocol_version") != PROTOCOL_VERSION
                or lock.get("frozen_data_manifest_sha256") != FROZEN_AIME_MANIFEST_SHA256
                or lock.get("dtest_loaded_before_lock") is not False
                or lock.get("search_size") != 60 or lock.get("selection_size") != 30):
            errors.append("selection lock does not match the frozen method/protocol/split sizes")
        if (lock.get("search_ids_sha256") != sha256_ids(split.get("search_ids", []))
                or lock.get("selection_ids_sha256") != sha256_ids(split.get("selection_ids", []))):
            errors.append("selection lock is not tied to the exact frozen D_search/D_select IDs")
        expected_repo, expected_commit = UPSTREAMS[args.method]
        provenance = manifest.get("upstream_provenance", {})
        if (provenance.get("repository") != expected_repo
                or provenance.get("commit") != expected_commit
                or provenance.get("clean_worktree") is not True
                or lock.get("source_provenance") != provenance):
            errors.append("upstream baseline provenance is missing, unpinned, or differs from the selection lock")
        if access.get("method") != args.method or access.get("selection_lock_sha256") != lock_hash:
            errors.append("D_test access manifest is not tied to this method's selection lock")
        if (split.get("selection_lock_sha256") != lock_hash
                or manifest.get("selection_lock_sha256") != lock_hash):
            errors.append("run/split manifest selection-lock hash mismatch")
        if (split.get("dtest_access_manifest_sha256") != sha256_file(access_path)
                or manifest.get("dtest_access_manifest_sha256") != sha256_file(access_path)):
            errors.append("run/split manifest D_test access-manifest hash mismatch")
        if access.get("selection_frozen_at_epoch") != lock.get("created_at_epoch"):
            errors.append("D_test access manifest freeze timestamp mismatch")
        test_splits = access.get("test_splits", {})
        if set(test_splits) != {"aime_2025.jsonl", "aime_2026.jsonl"}:
            errors.append("D_test access manifest does not record exactly both AIME years")
        validation_question_hashes = set(lock.get("validation_question_sha256", []))
        seen_question_hashes = set()
        for filename, record in test_splits.items():
            if record.get("opened_at_epoch", 0) < lock.get("created_at_epoch", float("inf")):
                errors.append(f"D_test opened before selection freeze: {filename}")
            if record.get("row_count") != 30 or len(record.get("ids", [])) != 30:
                errors.append(f"D_test audit has wrong row count: {filename}")
            question_hashes = set(record.get("question_sha256", []))
            if (len(question_hashes) != 30 or validation_question_hashes & question_hashes
                    or seen_question_hashes & question_hashes):
                errors.append(f"D_test question overlap/duplicate audit failed: {filename}")
            seen_question_hashes.update(question_hashes)

    if manifest.get("context_limit") != 16384 or split.get("context_limit") != 16384:
        errors.append("manifest context limit is not 8192")
    if manifest.get("output_limit") != 6144 or split.get("output_limit") != 6144:
        errors.append("manifest output limit is not 6144")
    if manifest.get("answer_protocol") != ANSWER_PARSER or split.get("answer_protocol") != ANSWER_PARSER:
        errors.append("manifest answer parser does not match the frozen shared parser")

    parser_path = Path(__file__).with_name("answer_protocol.py")
    parser_hash = sha256_file(parser_path)
    if manifest.get("answer_protocol_sha256") != parser_hash or split.get("answer_protocol_sha256") != parser_hash:
        errors.append("answer parser source hash differs from the run manifest")
    runner_path = Path(__file__).with_name("native_aime_formal.py")
    if manifest.get("runner_sha256") != sha256_file(runner_path):
        errors.append("native runner source hash differs from the run manifest")

    data_dir_value = split.get("data_dir")
    data_hashes = split.get("data_sha256", {})
    if (manifest.get("frozen_data_manifest_sha256") != FROZEN_AIME_MANIFEST_SHA256
            or split.get("frozen_data_manifest_sha256") != FROZEN_AIME_MANIFEST_SHA256):
        errors.append("run/split manifest does not bind the published frozen AIME data manifest")
    if not data_dir_value:
        errors.append("split manifest is missing data_dir")
    else:
        data_dir = Path(data_dir_value)
        frozen_manifest_path = data_dir / "frozen_aime_manifest.json"
        if (not frozen_manifest_path.is_file()
                or sha256_file(frozen_manifest_path) != FROZEN_AIME_MANIFEST_SHA256):
            errors.append("frozen AIME data manifest is missing or has the wrong hash")
        data_manifest = None
        if frozen_manifest_path.is_file():
            try:
                data_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
                expected_hashes = {
                    "aimo-validation-aime.jsonl": data_manifest["validation"]["source"]["sha256_git_content"],
                    data_manifest["validation"]["search"]["path"]: data_manifest["validation"]["search"]["sha256_git_content"],
                    data_manifest["validation"]["select"]["path"]: data_manifest["validation"]["select"]["sha256_git_content"],
                }
                for filename, spec in data_manifest["test"].items():
                    expected_hashes[filename] = spec["source_sha256_git_content"]
                    expected_hashes[spec["split_path"]] = spec["split_sha256_git_content"]
                for filename, expected_hash in expected_hashes.items():
                    if data_hashes.get(filename) != expected_hash:
                        errors.append(f"run/split manifest does not match pinned AIME hash: {filename}")
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                errors.append(f"frozen AIME data manifest has invalid structure: {exc}")
        for filename in (
            "aimo-validation-aime.jsonl", "aimo_validation/search_60.jsonl",
            "aimo_validation/select_30.jsonl", "aime_2025.jsonl",
            "aime2025/test_30.jsonl", "aime_2026.jsonl", "aime2026/test_30.jsonl",
        ):
            path = data_dir / filename
            if not path.exists():
                if filename in data_hashes:
                    errors.append(f"frozen input data missing: {path}")
                continue
            actual_hash = sha256_file(path)
            if data_hashes.get(filename) != actual_hash:
                if filename in data_hashes:
                    errors.append(f"input data hash mismatch: {filename}")
        if manifest.get("data_sha256") != data_hashes:
            errors.append("run and split manifest data hashes differ")

        try:
            if data_manifest is None:
                raise ValueError("frozen AIME data manifest could not be loaded")
            validation = read_jsonl(data_dir / data_manifest["validation"]["source"]["path"])
            canonical_search = read_jsonl(data_dir / data_manifest["validation"]["search"]["path"])
            canonical_select = read_jsonl(data_dir / data_manifest["validation"]["select"]["path"])
            if len(validation) != 90 or len(canonical_search) != 60 or len(canonical_select) != 30:
                errors.append("frozen validation source or canonical 60/30 split has an unexpected row count")
            expected_search = ["validation:" + str(row.get("id", row.get("problem_idx", "")))
                               for row in canonical_search]
            expected_select = ["validation:" + str(row.get("id", row.get("problem_idx", "")))
                               for row in canonical_select]
            if split.get("search_ids") != expected_search or split.get("selection_ids") != expected_select:
                errors.append("D_search/D_select IDs do not match the canonical frozen 60/30 files")
            normalize = lambda row: (" ".join(str(row.get("problem", row.get("input", ""))).casefold().split()),
                                     str(row.get("answer", "")).strip())
            raw_map = {normalize(row)[0]: normalize(row)[1] for row in validation}
            canonical_rows = canonical_search + canonical_select
            canonical_map = {normalize(row)[0]: normalize(row)[1] for row in canonical_rows}
            if len(raw_map) != len(validation) or canonical_map != raw_map:
                errors.append("canonical validation split does not exactly partition the pinned source questions/answers")
            for filename in ("aime_2025.jsonl", "aime_2026.jsonl"):
                rows = read_jsonl(data_dir / filename)
                namespace = Path(filename).stem
                expected_ids = []
                for row in rows[:30]:
                    source_id = str(row.get("source_row_id", row.get("id", row.get("problem_idx", ""))))
                    expected_ids.append(source_id if source_id.startswith(namespace + ":")
                                        else namespace + ":" + source_id)
                actual_ids = split.get("test_ids", {}).get(namespace)
                if len(rows) < 30 or actual_ids != expected_ids:
                    errors.append(f"D_test IDs mismatch for {filename}")
                access_record = (access.get("test_splits", {}) if access_path.is_file() else {}).get(filename, {})
                if access_record.get("sha256") != sha256_file(data_dir / filename):
                    errors.append(f"D_test opened-data hash mismatch for {filename}")
                spec = data_manifest["test"][filename]
                frozen_path = data_dir / spec["split_path"]
                if (not frozen_path.is_file()
                        or access_record.get("frozen_split_path") != spec["split_path"]
                        or access_record.get("frozen_split_sha256") != sha256_file(frozen_path)
                        or data_hashes.get(spec["split_path"]) != sha256_file(frozen_path)):
                    errors.append(f"D_test canonical frozen split hash mismatch for {filename}")
                if access_record.get("frozen_data_manifest_sha256") != FROZEN_AIME_MANIFEST_SHA256:
                    errors.append(f"D_test audit does not bind the frozen data manifest for {filename}")
                frozen_test_rows = read_jsonl(frozen_path)
                if {normalize(row) for row in rows} != {normalize(row) for row in frozen_test_rows}:
                    errors.append(f"D_test source and canonical frozen split differ for {filename}")
        except Exception as exc:
            errors.append(f"cannot verify frozen split IDs: {type(exc).__name__}: {exc}")

    controls = result.get("controls", {})
    for key, expected in (("model", "Qwen/Qwen3.5-9B"), ("data_seed", 2026),
                          ("search_seed", args.seed), ("search_size", 60),
                          ("selection_size", 30), ("test_size_each", 30)):
        if controls.get(key) != expected:
            errors.append(f"controls.{key}={controls.get(key)!r} != {expected!r}")
    if controls.get("thinking") is not False or controls.get("temperature") != 0.0 or controls.get("top_p") != 1.0:
        errors.append("decoding controls are not deterministic/no-thinking")
    for key, expected in (("concurrency", 8), ("max_num_seqs", 8),
                          ("max_model_len", 16384), ("max_tokens", 6144),
                          ("tensor_parallel_size", 1)):
        if controls.get(key) != expected:
            errors.append(f"controls.{key}={controls.get(key)!r} != protocol value {expected!r}")
    expected_gold_protocol = f"{ANSWER_PARSER}; normalized integer exact match"
    if controls.get("gold_protocol") != expected_gold_protocol:
        errors.append(f"controls.gold_protocol={controls.get('gold_protocol')!r} != {expected_gold_protocol!r}")
    if controls.get("score_mapping") != "2=exact match; 1=parseable wrong; 0=unparseable":
        errors.append("controls.score_mapping does not match the shared 0/1/2 protocol")

    def check_point(point: dict, label: str) -> None:
        if point.get("answer_parser") != ANSWER_PARSER:
            errors.append(f"{label} answer parser is missing or mismatched")
        if point.get("num_examples") != 30:
            errors.append(f"{label} has {point.get('num_examples')} examples, expected 30")
        score = point.get("score")
        if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
            errors.append(f"{label} has invalid score={score!r}")
        errors.extend(check_ordinal_scores(point, label))
        audit = point.get("output_audit")
        if not isinstance(audit, dict) or audit.get("csv_rows") != 30:
            errors.append(f"{label} output audit is missing or has the wrong row count")
            return
        failure_fraction = float(audit.get("failure_fraction", 1.0) or 0.0)
        if failure_fraction > MAX_SAMPLE_FAILURE_RATE:
            errors.append(f"{label} workflow failure rate {failure_fraction:.1%} exceeds {MAX_SAMPLE_FAILURE_RATE:.1%}")
        total_tokens = int(point.get("total_tokens", 0) or 0)
        if total_tokens <= 0:
            errors.append(f"{label} has no model-token usage recorded")
        if point.get("cost_unit") != "model_tokens":
            errors.append(f"{label} cost is not expressed in model tokens")

    tests = result.get("test", {})
    for name in ("aime_2025", "aime_2026"):
        row = tests.get(name)
        if not isinstance(row, dict):
            errors.append(f"missing {name} test result")
            continue
        for point_name in ("Q", "E"):
            point = row.get(point_name)
            if not isinstance(point, dict):
                errors.append(f"missing {name}/{point_name} operating point")
                continue
            check_point(point, f"{name}/{point_name}")
            if point_name == "E" and point.get("same_candidate_as") not in (None, "Q"):
                errors.append(f"{name}/E incorrectly declares its shared candidate")

    if args.method == "aflow":
        rounds = result.get("search_rounds", [])
        selection = result.get("selection_rows", [])
        search_audits = result.get("search_shared_metrics", [])
        if len(search_audits) != len(rounds):
            errors.append("AFlow D_search does not contain one shared-score audit per search round")
        for audit in search_audits:
            round_number = audit.get("round")
            label = f"AFlow D_search/round_{round_number}"
            if round_number not in rounds:
                errors.append(f"{label} is not in the recorded AFlow search rounds")
            if audit.get("status") == "invalid":
                if not audit.get("errors"):
                    errors.append(f"{label} is marked invalid without a reason")
                continue
            if audit.get("status") != "evaluated":
                errors.append(f"{label} has unknown evaluation status {audit.get('status')!r}")
                continue
            if audit.get("answer_parser") != ANSWER_PARSER:
                errors.append(f"{label} answer parser is missing or mismatched")
            if audit.get("num_examples") != 60:
                errors.append(f"{label} has {audit.get('num_examples')} examples, expected 60")
            errors.extend(check_ordinal_scores(audit, label, expected_count=60))
        if len(rounds) < 3 or not selection:
            errors.append(f"AFlow native search/selection incomplete: rounds={rounds}")
        if result.get("selected_round", {}).get("round") not in rounds:
            errors.append("AFlow selected round is not in generated rounds")
        invalid = result.get("invalid_rounds", [])
        if invalid:
            warnings.append(f"AFlow skipped {len(invalid)}/{len(selection)} invalid generated candidates")
            if len(invalid) * 2 > len(selection):
                errors.append("more than half of AFlow candidates were invalid")
        if result.get("selected_round", {}).get("status") != "evaluated":
            errors.append("AFlow selected round was not a validated/evaluated candidate")
        selected_e = result.get("selected_efficiency_round", {})
        locked_selected = lock.get("selected", {})
        if (locked_selected.get("Q", {}).get("round") != result.get("selected_round", {}).get("round")
                or locked_selected.get("E", {}).get("round") != selected_e.get("round")):
            errors.append("AFlow post-selection test candidates differ from the durable D_select freeze")
        if selected_e.get("status") != "evaluated":
            errors.append("AFlow efficiency operating point was not a validated/evaluated candidate")
        for label, selection in (("Q", result.get("selected_round", {})), ("E", selected_e)):
            if selection.get("shared_metrics", {}).get("answer_parser") != ANSWER_PARSER:
                errors.append(f"AFlow D_select/{label} did not use the frozen shared parser")
            errors.extend(check_ordinal_scores(selection.get("shared_metrics", {}), f"AFlow D_select/{label}"))
            if float(selection.get("output_audit", {}).get("failure_fraction", 1.0) or 0.0) > MAX_SAMPLE_FAILURE_RATE:
                errors.append(f"AFlow D_select/{label} exceeds the sample-failure limit")
    else:
        selection = result.get("selection", {})
        if selection.get("num_examples") != 30:
            errors.append("MaAS selection is incomplete")
        if selection.get("answer_parser") != ANSWER_PARSER:
            errors.append("MaAS D_select did not use the frozen shared parser")
        errors.extend(check_ordinal_scores(selection, "MaAS D_select"))
        selection_audit = selection.get("output_audit", {})
        selection_failure_fraction = float(selection_audit.get("failure_fraction", 1.0) or 0.0)
        if selection_audit.get("csv_rows") != 30:
            errors.append("MaAS D_select output audit is missing or has the wrong row count")
        if selection_failure_fraction > MAX_SAMPLE_FAILURE_RATE:
            errors.append(f"MaAS D_select workflow failure rate {selection_failure_fraction:.1%} exceeds {MAX_SAMPLE_FAILURE_RATE:.1%}")
        checkpoint = result.get("controller_checkpoint")
        if not checkpoint or not Path(checkpoint).exists():
            errors.append("MaAS controller checkpoint is missing")
        elif lock.get("selected", {}).get("controller_sha256") != sha256_file(Path(checkpoint)):
            errors.append("MaAS test controller differs from the durable D_select freeze")

    telemetry = result.get("telemetry", {})
    truncated = int(telemetry.get("completion_truncated", 0) or 0)
    calls = int(telemetry.get("calls", 0) or 0)
    if calls > 0 and truncated / calls > MAX_SAMPLE_FAILURE_RATE:
        errors.append(f"truncation rate {truncated}/{calls} exceeds {MAX_SAMPLE_FAILURE_RATE:.1%}")
    elif truncated:
        warnings.append(f"LLM completions hit the output limit {truncated} times")
    failed = int(telemetry.get("failed_calls", 0) or 0)
    if calls <= 0:
        errors.append("no model calls were recorded")
    elif failed / calls > 0.05:
        errors.append(f"failed call rate is {failed}/{calls} > 5%")
    elif failed:
        warnings.append(f"small number of failed calls: {failed}/{calls}")

    audit = {
        "status": "PASS" if not errors else "FAIL",
        "method": args.method,
        "seed": args.seed,
        "errors": errors,
        "warnings": warnings,
        "test_scores": {
            name: {
                point: (tests.get(name, {}).get(point) or {}).get("score")
                for point in ("Q", "E")
            }
            for name in ("aime_2025", "aime_2026")
        },
        "calls": calls,
        "failed_calls": failed,
    }
    (root / "quality_gate.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["status"] = "complete" if not errors else "rejected"
    manifest["quality_gate"] = {
        "status": audit["status"],
        "path": "quality_gate.json",
        "errors": len(errors),
        "warnings": len(warnings),
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    return 0 if not errors else 42


if __name__ == "__main__":
    raise SystemExit(main())
