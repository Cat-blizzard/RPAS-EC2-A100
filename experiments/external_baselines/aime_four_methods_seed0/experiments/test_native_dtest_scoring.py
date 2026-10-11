"""CPU regressions for native D_test CSVs and the production scoring callsites.

Execute the dataset write and scoring statements from the actual test loops;
the expensive search, generated workflows, and model calls are excluded. This
keeps the regression sensitive to the original raw-versus-rendered call bug,
rather than repeating the corrected scorer arguments only inside a fixture.
"""

from __future__ import annotations

import ast
import csv
import inspect
from pathlib import Path
from typing import Any

import pytest

import native_aime_formal as runner


def _production_dtest_statements(method: str) -> tuple[ast.stmt, ast.stmt]:
    function = getattr(runner, f"run_{method}")
    tree = ast.parse(inspect.getsource(function))
    test_loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
        and node.target.id == "filename" and isinstance(node.iter, ast.Tuple)
        and [item.value for item in node.iter.elts if isinstance(item, ast.Constant)]
        == ["aime_2025.jsonl", "aime_2026.jsonl"]
    ]
    assert len(test_loops) == 1, "Expected the production loop over both frozen D_test years"
    writes = [
        node for node in ast.walk(test_loops[0])
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == "write_jsonl"
    ]
    scores = [
        node for node in ast.walk(test_loops[0])
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == "shared_aime_metrics"
    ]
    assert len(writes) == len(scores) == 1
    return writes[0], scores[0]


def _execute(statement: ast.stmt, context: dict[str, Any]) -> None:
    module = ast.fix_missing_locations(ast.Module(body=[statement], type_ignores=[]))
    exec(compile(module, str(Path(runner.__file__)), "exec"), vars(runner), context)


def _prepare_dtest(tmp_path: Path, method: str, year: int) -> tuple[dict[str, Any], ast.stmt, list[dict[str, str]]]:
    data_dir = Path(runner.__file__).resolve().parents[1] / "data"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    search, select = runner.freeze_split(data_dir, 2026, 60, 30)
    search = runner.namespace_aime_rows(search, "validation")
    select = runner.namespace_aime_rows(select, "validation")
    runner.freeze_aime_selection(run_dir, method, search, select, {"offline_contract_fixture": True})
    filename = f"aime_{year}.jsonl"
    # These are the same raw problem rows returned to the formal D_test loop,
    # verified against the pinned source, canonical split, and selection lock.
    expected_rows = runner.load_aime_test_rows_after_freeze(data_dir, run_dir, method, filename, 30)
    native_path = tmp_path / "native" / "math_test.jsonl"
    log_dir = tmp_path / "csv"
    log_dir.mkdir()
    context = {
        "filename": filename,
        "test_rows": expected_rows,
        "expected_test_rows": expected_rows,
        "test_file": native_path,
        "data_path": native_path.parent,
        "log_dir": log_dir,
        "maas_eval_dir": log_dir,
    }
    write, score = _production_dtest_statements(method)
    _execute(write, context)
    native_rows = runner.read_jsonl(native_path)
    assert len(native_rows) == 30
    csv_rows = [
        {"question": row["problem"], "expected_output": row["solution"],
         "prediction": f"FINAL ANSWER: {row['answer']}"}
        for row in native_rows
    ]
    return context, score, csv_rows


def _write_native_csv(context: dict[str, Any], rows: list[dict[str, str]]) -> None:
    with (context["log_dir"] / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["question", "expected_output", "prediction"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("method", ["aflow", "maas"])
@pytest.mark.parametrize("year", [2025, 2026])
def test_production_dtest_call_scores_native_rendered_csv_from_frozen_raw_rows(tmp_path: Path, method: str, year: int):
    context, score, csv_rows = _prepare_dtest(tmp_path, method, year)
    expected_rows = context["test_rows"]
    wrong = (int(expected_rows[1]["answer"]) + 1) % 1000
    csv_rows[1]["prediction"] = f"FINAL ANSWER: {wrong}"
    csv_rows[2]["prediction"] = "reasoning without a final answer"
    _write_native_csv(context, csv_rows)

    # Reproduce the original exception with raw frozen rows before exercising
    # the actual production callsite, which must supply the native rendering.
    with pytest.raises(RuntimeError, match="row order/content mismatch.*row 0"):
        runner.shared_aime_metrics(context["log_dir"], expected_rows)
    _execute(score, context)
    metrics = context["shared_metrics"]
    assert metrics["num_examples"] == 30
    assert metrics["correct"] == 28
    assert metrics["score_0_1_2"] == [2, 1, 0] + [2] * 27
    assert [row["id"] for row in metrics["rows"]] == [row["id"] for row in expected_rows]


@pytest.mark.parametrize("method", ["aflow", "maas"])
@pytest.mark.parametrize("corruption", ["swapped", "wrong_gold", "missing", "wrong_prefix"])
def test_production_dtest_call_rejects_corrupt_native_csv(tmp_path: Path, method: str, corruption: str):
    context, score, csv_rows = _prepare_dtest(tmp_path, method, 2025)
    if corruption == "swapped":
        csv_rows[0], csv_rows[1] = csv_rows[1], csv_rows[0]
        message = "row order/content mismatch"
    elif corruption == "wrong_gold":
        wrong = (int(context["test_rows"][0]["answer"]) + 1) % 1000
        csv_rows[0]["expected_output"] = rf"\boxed{{{wrong}}}"
        message = "gold mismatch"
    elif corruption == "missing":
        csv_rows.pop()
        message = "output count mismatch"
    else:
        csv_rows[0]["question"] = "Incorrect task instruction\n\n" + context["test_rows"][0]["problem"]
        message = "row order/content mismatch"
    _write_native_csv(context, csv_rows)
    with pytest.raises(RuntimeError, match=message):
        _execute(score, context)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
