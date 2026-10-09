import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

import aime_quality_gate as gate
from aime_quality_gate import check_ordinal_scores


def test_aime_quality_gate_accepts_uniform_ordinal_audit():
    metrics = {
        "rows": [
            {"correct": True, "parser_valid": True},
            {"correct": False, "parser_valid": True},
            {"correct": False, "parser_valid": False},
        ],
        "score_0_1_2": [2, 1, 0],
        "mean_score_0_1_2": 1.0,
    }
    assert check_ordinal_scores(metrics, "fixture", expected_count=3) == []


def test_aime_quality_gate_rejects_inconsistent_ordinal_audit():
    metrics = {
        "rows": [
            {"correct": True, "parser_valid": True},
            {"correct": False, "parser_valid": False},
        ],
        "score_0_1_2": [1, 0],
        "mean_score_0_1_2": 1.0,
    }
    errors = check_ordinal_scores(metrics, "fixture", expected_count=2)
    assert errors == ["fixture per-example scores violate the shared 0/1/2 mapping"]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _synthetic_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a complete synthetic gate input without any benchmark file reads."""
    data = tmp_path / "data"
    run = tmp_path / "run"
    hashes = {}

    def jsonl(filename: str, rows: list[dict], *, pretty: bool = False) -> None:
        path = data / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        separators = None if pretty else (",", ":")
        path.write_text("".join(json.dumps(row, separators=separators) + "\n" for row in rows), encoding="utf-8")
        hashes[filename] = gate.sha256_file(path)

    validation = [{"id": index, "problem": f"Synthetic validation {index}", "answer": str(index)}
                  for index in range(90)]
    jsonl("aimo-validation-aime.jsonl", validation)
    jsonl("aimo_validation/search_60.jsonl", validation[:60])
    jsonl("aimo_validation/select_30.jsonl", validation[60:])
    test_rows = {}
    for year in (2025, 2026):
        rows = [{"id": index, "problem": f"Synthetic {year} example {index}", "answer": str(index)}
                for index in range(30)]
        test_rows[year] = rows
        # Source and canonical files have equal rows but different bytes.
        jsonl(f"aime_{year}.jsonl", rows, pretty=True)
        jsonl(f"aime{year}/test_30.jsonl", rows)
    frozen_manifest = {
        "validation": {
            "source": {"path": "aimo-validation-aime.jsonl", "sha256_git_content": hashes["aimo-validation-aime.jsonl"]},
            "search": {"path": "aimo_validation/search_60.jsonl", "sha256_git_content": hashes["aimo_validation/search_60.jsonl"]},
            "select": {"path": "aimo_validation/select_30.jsonl", "sha256_git_content": hashes["aimo_validation/select_30.jsonl"]},
        },
        "test": {f"aime_{year}.jsonl": {
            "source_sha256_git_content": hashes[f"aime_{year}.jsonl"],
            "split_path": f"aime{year}/test_30.jsonl",
            "split_sha256_git_content": hashes[f"aime{year}/test_30.jsonl"],
        } for year in (2025, 2026)},
    }
    _write_json(data / "frozen_aime_manifest.json", frozen_manifest)
    manifest_hash = gate.sha256_file(data / "frozen_aime_manifest.json")
    monkeypatch.setattr(gate, "FROZEN_AIME_MANIFEST_SHA256", manifest_hash)
    checkpoint = tmp_path / "synthetic_controller.pth"
    checkpoint.write_bytes(b"Synthetic controller bytes; never loaded as a model")
    provenance = {"repository": gate.UPSTREAMS["maas"][0], "commit": gate.UPSTREAMS["maas"][1], "clean_worktree": True}
    search_ids = [f"validation:{index}" for index in range(60)]
    select_ids = [f"validation:{index}" for index in range(60, 90)]
    question_hash = lambda row: hashlib.sha256(row["problem"].encode()).hexdigest()
    lock = {
        "schema": "aime_dtest_blind_selection_lock_v2", "method": "maas", "protocol_version": gate.PROTOCOL_VERSION,
        "frozen_data_manifest_sha256": manifest_hash, "dtest_loaded_before_lock": False,
        "search_size": 60, "selection_size": 30, "search_ids_sha256": gate.sha256_ids(search_ids),
        "selection_ids_sha256": gate.sha256_ids(select_ids), "source_provenance": provenance,
        "created_at_epoch": 100, "validation_question_sha256": [question_hash(row) for row in validation],
        "selected": {"controller_sha256": gate.sha256_file(checkpoint)},
    }
    _write_json(run / "selection_frozen.json", lock)
    lock_hash = gate.sha256_file(run / "selection_frozen.json")
    access = {
        "method": "maas", "selection_lock_sha256": lock_hash, "selection_frozen_at_epoch": 100,
        "test_splits": {f"aime_{year}.jsonl": {
            "opened_at_epoch": 101, "row_count": 30, "ids": [f"aime_{year}:{i}" for i in range(30)],
            "question_sha256": [question_hash(row) for row in test_rows[year]],
            "sha256": hashes[f"aime_{year}.jsonl"], "frozen_split_path": f"aime{year}/test_30.jsonl",
            "frozen_split_sha256": hashes[f"aime{year}/test_30.jsonl"], "frozen_data_manifest_sha256": manifest_hash,
        } for year in (2025, 2026)},
    }
    _write_json(run / "dtest_access_manifest.json", access)
    common = {
        "protocol_version": gate.PROTOCOL_VERSION, "context_limit": 16384, "output_limit": 6144,
        "answer_protocol": gate.ANSWER_PARSER, "answer_protocol_sha256": gate.sha256_file(Path(gate.__file__).with_name("answer_protocol.py")),
        "frozen_data_manifest_sha256": manifest_hash, "data_sha256": hashes,
        "selection_lock_sha256": lock_hash, "dtest_access_manifest_sha256": gate.sha256_file(run / "dtest_access_manifest.json"),
    }
    _write_json(run / "run_manifest.json", {
        **common, "method": "maas", "search_seed": 0, "data_seed": 2026, "search_size": 60, "selection_size": 30,
        "upstream_provenance": provenance, "runner_sha256": gate.sha256_file(Path(gate.__file__).with_name("native_aime_formal.py")),
        "status": "results_ready",
    })
    _write_json(run / "split_manifest.json", {
        **common, "data_dir": str(data), "search_ids": search_ids, "selection_ids": select_ids,
        "test_ids": {f"aime_{year}": [f"aime_{year}:{i}" for i in range(30)] for year in (2025, 2026)},
    })
    metrics = {
        "answer_parser": gate.ANSWER_PARSER, "num_examples": 30, "score": 1.0,
        "rows": [{"correct": True, "parser_valid": True} for _ in range(30)], "score_0_1_2": [2] * 30,
        "mean_score_0_1_2": 2.0, "output_audit": {"csv_rows": 30, "failure_fraction": 0.0},
        "total_tokens": 100, "cost_unit": "model_tokens",
    }
    _write_json(run / "native_result.json", {
        "controls": {
            "model": "Qwen/Qwen3.5-9B", "data_seed": 2026, "search_seed": 0, "search_size": 60,
            "selection_size": 30, "test_size_each": 30, "thinking": False, "temperature": 0.0, "top_p": 1.0,
            "concurrency": 8, "max_num_seqs": 8, "max_model_len": 16384, "max_tokens": 6144, "tensor_parallel_size": 1,
            "gold_protocol": f"{gate.ANSWER_PARSER}; normalized integer exact match",
            "score_mapping": "2=exact match; 1=parseable wrong; 0=unparseable",
        },
        "selection": copy.deepcopy(metrics), "controller_checkpoint": str(checkpoint),
        "test": {f"aime_{year}": {point: copy.deepcopy(metrics) for point in ("Q", "E")} for year in (2025, 2026)},
        "telemetry": {"calls": 100, "failed_calls": 0, "completion_truncated": 0},
    })
    monkeypatch.setattr(sys, "argv", ["aime_quality_gate.py", "--run-dir", str(run), "--method", "maas", "--seed", "0"])
    return run


def test_gate_accepts_distinct_year_source_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _synthetic_run(tmp_path, monkeypatch)
    access = json.loads((run / "dtest_access_manifest.json").read_text())
    assert access["test_splits"]["aime_2025.jsonl"]["sha256"] != access["test_splits"]["aime_2026.jsonl"]["sha256"]
    assert gate.main() == 0
    assert json.loads((run / "quality_gate.json").read_text())["errors"] == []


@pytest.mark.parametrize("year", (2025, 2026))
def test_gate_rejects_wrong_year_source_access_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, year: int) -> None:
    run = _synthetic_run(tmp_path, monkeypatch)
    access_path = run / "dtest_access_manifest.json"
    access = json.loads(access_path.read_text())
    access["test_splits"][f"aime_{year}.jsonl"]["sha256"] = "0" * 64
    _write_json(access_path, access)
    for filename in ("run_manifest.json", "split_manifest.json"):
        path = run / filename
        value = json.loads(path.read_text())
        value["dtest_access_manifest_sha256"] = gate.sha256_file(access_path)
        _write_json(path, value)
    assert gate.main() == 42
    assert json.loads((run / "quality_gate.json").read_text())["errors"] == [f"D_test opened-data hash mismatch for aime_{year}.jsonl"]


def test_gate_keeps_five_percent_truncation_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _synthetic_run(tmp_path, monkeypatch)
    result_path = run / "native_result.json"
    result = json.loads(result_path.read_text())
    result["telemetry"]["completion_truncated"] = 6
    _write_json(result_path, result)
    assert gate.main() == 42
    assert json.loads((run / "quality_gate.json").read_text())["errors"] == ["truncation rate 6/100 exceeds 5.0%"]
