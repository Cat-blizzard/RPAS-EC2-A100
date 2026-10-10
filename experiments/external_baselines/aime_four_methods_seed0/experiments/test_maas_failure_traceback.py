"""CPU regressions for traceback persistence across MaAS's native catch."""

from __future__ import annotations

import argparse
import ast
import asyncio
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import native_aime_formal as runner


def read_failures(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "failure_tracebacks.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_inner_traceback_survives_native_optimizer_catch(tmp_path: Path):
    # Execute the actual pinned optimize method without importing its model,
    # provider, or GPU dependencies. Its try/catch/retry code stays intact.
    upstream = Path(runner.__file__).resolve().parents[1] / "upstream/MaAS/maas/ext/maas/scripts/optimizer.py"
    if not upstream.is_file():
        pytest.skip("requires the fixed MaAS source checkout")
    tree = ast.parse(upstream.read_text(encoding="utf-8"), filename=str(upstream))
    native_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Optimizer")
    native_optimize = next(
        node for node in native_class.body if isinstance(node, ast.FunctionDef) and node.name == "optimize"
    )
    loops = []
    sleeps = []
    native_logs = []

    def new_event_loop():
        loop = asyncio.new_event_loop()
        loops.append(loop)
        return loop

    namespace = {
        "asyncio": SimpleNamespace(new_event_loop=new_event_loop, set_event_loop=lambda loop: None),
        "time": SimpleNamespace(sleep=sleeps.append),
        "logger": SimpleNamespace(info=native_logs.append),
        "OptimizerType": str,
    }
    exec(compile(ast.Module(body=[native_optimize], type_ignores=[]), str(upstream), "exec"), namespace)
    native_error = BrokenPipeError("closed progress log pipe")

    class CpuOptimizer:
        optimize = namespace["optimize"]

        def __init__(self):
            self.calls = 0

        async def _optimize_graph_maas(self):
            self.calls += 1
            try:
                raise ValueError("fixture writer closed")
            except ValueError as cause:
                raise native_error from cause

    optimizer = CpuOptimizer()
    original_class_method = CpuOptimizer._optimize_graph_maas
    runner.install_maas_optimizer_failure_traceback(optimizer, tmp_path)
    try:
        assert optimizer.optimize("Graph") is None  # The native catch swallows the exception.
    finally:
        for loop in loops:
            loop.close()

    assert CpuOptimizer._optimize_graph_maas is original_class_method
    assert optimizer.calls == 1
    assert sleeps == [5, 5]  # Unchanged native retry and trailing sleeps.
    assert any("Attempt 1/1" in message for message in native_logs)
    assert native_logs[-1] == "Score for round 1: None"
    failures = read_failures(tmp_path)
    assert len(failures) == 1
    record = failures[0]
    assert record["stage"] == "maas._optimize_graph_maas"
    assert record["exception_type"] == "builtins.BrokenPipeError"
    assert record["exception_message"] == "closed progress log pipe"
    assert "Traceback (most recent call last):" in record["traceback"]
    assert "ValueError: fixture writer closed" in record["traceback"]
    assert "BrokenPipeError: closed progress log pipe" in record["traceback"]
    assert "_optimize_graph_maas" in record["traceback"]
    assert datetime.fromisoformat(record["created_at_utc"]).utcoffset().total_seconds() == 0
    assert record["created_at_epoch"] > 0
    assert record["pid"] > 0


def test_successful_optimizer_keeps_arguments_and_return_value(tmp_path: Path):
    sentinel = object()

    class CpuOptimizer:
        async def _optimize_graph_maas(self, value, *, mode):
            assert value == 7
            assert mode == "native"
            return sentinel

    optimizer = CpuOptimizer()
    runner.install_maas_optimizer_failure_traceback(optimizer, tmp_path)
    assert asyncio.run(optimizer._optimize_graph_maas(7, mode="native")) is sentinel
    assert not (tmp_path / "failure_tracebacks.jsonl").exists()


def test_traceback_file_appends_without_replacing_existing_failure(tmp_path: Path):
    for message in ("first", "second"):
        try:
            raise RuntimeError(message)
        except RuntimeError as exc:
            assert runner.persist_failure_traceback(tmp_path, "fixture", exc) == tmp_path / "failure_tracebacks.jsonl"
    assert [item["exception_message"] for item in read_failures(tmp_path)] == ["first", "second"]


def test_traceback_storage_failure_preserves_original_native_exception(tmp_path: Path):
    invalid_directory = tmp_path / "ordinary-file"
    invalid_directory.write_text("not a directory", encoding="utf-8")
    original_error = BrokenPipeError("closed native log pipe")

    class CpuOptimizer:
        async def _optimize_graph_maas(self):
            raise original_error

    optimizer = CpuOptimizer()
    runner.install_maas_optimizer_failure_traceback(optimizer, invalid_directory)
    with pytest.raises(BrokenPipeError) as raised:
        asyncio.run(optimizer._optimize_graph_maas())
    assert raised.value is original_error


def test_main_failure_persists_traceback_and_failed_manifest(tmp_path: Path, monkeypatch):
    args = argparse.Namespace(
        method="maas", data_seed=2026, seed=0, search_size=60, selection_size=30,
        test_size=30, max_tokens=runner.OUTPUT_LIMIT, concurrency=8, maas_resume=False,
        run_dir=tmp_path / "run", data_dir=tmp_path / "data", endpoint="http://fixture.invalid/v1",
    )
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "baseline_source_path", lambda args, method: tmp_path / "source")
    monkeypatch.setattr(runner, "verify_pinned_upstream", lambda method, path: {"commit": "fixture"})
    monkeypatch.setattr(runner, "seed_everything", lambda seed: None)
    rows = [{"id": str(index)} for index in range(90)]
    monkeypatch.setattr(runner, "freeze_split", lambda *values: (rows[:60], rows[60:]))
    monkeypatch.setattr(runner, "namespace_aime_rows", lambda rows, namespace: rows)
    monkeypatch.setattr(runner, "verify_served_model", lambda endpoint: None)
    monkeypatch.setattr(runner, "load_frozen_aime_manifest", lambda directory: {
        "validation": {
            key: {"path": f"{key}.jsonl", "sha256_git_content": f"fixture-{key}"}
            for key in ("source", "search", "select")
        },
    })
    original_error = RuntimeError("native controller checkpoint missing")

    def fail_maas(*unused):
        raise original_error

    monkeypatch.setattr(runner, "run_maas", fail_maas)
    with pytest.raises(RuntimeError) as raised:
        runner.main()
    assert raised.value is original_error
    manifest = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"] == "RuntimeError: native controller checkpoint missing"
    assert manifest["failure_tracebacks_path"] == str(args.run_dir / "failure_tracebacks.jsonl")
    records = read_failures(args.run_dir)
    assert len(records) == 1
    assert records[0]["stage"] == "runner.main"
    assert records[0]["exception_type"] == "builtins.RuntimeError"
    assert "fail_maas" in records[0]["traceback"]
    assert "RuntimeError: native controller checkpoint missing" in records[0]["traceback"]
