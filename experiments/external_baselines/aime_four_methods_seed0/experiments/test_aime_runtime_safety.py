"""Offline regression tests for bounded generated-code execution and AFlow guards."""

from __future__ import annotations

import asyncio
import ast
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
from types import ModuleType
import time
from unittest.mock import patch

import pytest

import native_aime_formal as runner
from scripts import record_run_session


def fake_run_code(code: str) -> tuple[str, str]:
    if code.startswith("hang:"):
        Path(code.split(":", 1)[1]).write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(30)
    if code == "hang":
        time.sleep(30)
    if code == "raise":
        raise ValueError("fixture failure")
    return "Success", code


def test_aime_contract_is_self_contained_and_not_masbench_metadata():
    contract = runner.task_contract_manifest()
    assert contract["version"] == "AIME_TASK_CONTRACT_v2_shared_input"
    assert "MASBENCH" not in contract["version"]
    assert contract["sha256"] == runner.AIME_TASK_CONTRACT_SHA256


def test_run_session_records_selected_server_profile(monkeypatch):
    monkeypatch.setenv("AIME_MAX_MODEL_LEN", "16384")
    monkeypatch.setenv("AIME_MAX_NUM_SEQS", "8")
    monkeypatch.setenv("AIME_MAX_NUM_BATCHED_TOKENS", "16384")
    settings = record_run_session.server_settings()
    assert settings["max_model_len"] == 16384
    assert settings["max_num_seqs"] == 8
    assert settings["max_num_batched_tokens"] == 16384


def test_run_session_profile_defaults_preserve_8k_fallback(monkeypatch):
    for name in ("AIME_MAX_MODEL_LEN", "AIME_MAX_NUM_SEQS", "AIME_MAX_NUM_BATCHED_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    settings = record_run_session.server_settings()
    assert settings["max_model_len"] == 8192
    assert settings["max_num_seqs"] == 24
    assert settings["max_num_batched_tokens"] == 16384


def test_output_audit_counts_native_attribute_and_worker_failures_only(tmp_path: Path):
    failures = [
        "'Workflow' object has no attribute 'custom'",
        "AttributeError: 'Workflow' object has no attribute 'custom'",
        "Code execution failed: ValueError: fixture failure",
    ]
    solutions = [
        "An error in the first calculation is corrected below.\nFINAL ANSWER: 7",
        "Error bounds give the same integer.\nFINAL ANSWER: 7",
        "A discarded approach reported 'Workflow' object has no attribute 'custom'.\nFINAL ANSWER: 7",
        "The previous code execution failed: its assumption was wrong.\nFINAL ANSWER: 7",
    ]
    output = tmp_path / "predictions.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["prediction"])
        writer.writeheader()
        writer.writerows({"prediction": value} for value in failures + solutions)

    audit = runner.audit_aflow_outputs(tmp_path)
    assert audit["csv_rows"] == len(failures) + len(solutions)
    assert audit["failure_predictions"] == len(failures)
    assert audit["failure_markers"] == failures
    assert audit["failure_fraction"] == len(failures) / (len(failures) + len(solutions))


def test_output_audit_recognizes_validation_and_bare_name_errors(tmp_path: Path):
    failures = [
        "1 validation error for ScEnsembleOp_AN\n  Missing fields: {'solution_letter'}",
        "Error: 1 validation error for ScEnsembleOp_AN",
        "NameError: name 'answer' is not defined",
        "name 'answer' is not defined",
    ]
    solutions = [
        "FINAL ANSWER: 7",
        "A solution that mentions validation error in ordinary prose. FINAL ANSWER: 7",
    ]
    output = tmp_path / "predictions.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["prediction"])
        writer.writeheader()
        writer.writerows({"prediction": row} for row in failures + solutions)
    audit = runner.audit_aflow_outputs(tmp_path)
    assert audit["csv_rows"] == 6
    assert audit["failure_predictions"] == 4
    assert audit["failure_fraction"] == 4 / 6


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"AIME_RUN_ID": "aime:seed_0", "MASBENCH_RUN_ID": "legacy"}, "aime:seed_0"),
        ({"AIME_RUN_ID": "aime:seed_0"}, "aime:seed_0"),
        ({"MASBENCH_RUN_ID": "legacy"}, "legacy"),
        ({"AIME_RUN_ID": "", "MASBENCH_RUN_ID": "legacy"}, "legacy"),
        ({}, ""),
    ],
)
def test_aime_run_id_precedence_and_legacy_fallback(environment, expected):
    with patch.dict(os.environ, environment, clear=True):
        assert runner.current_run_id() == expected


def test_process_executor_success_and_failure_are_cleanly_closed():
    limiter = runner.LoopBoundSemaphore(2)

    async def exercise():
        ok = await runner.execute_generated_code(fake_run_code, "ok", 5, limiter)
        bad = await runner.execute_generated_code(fake_run_code, "raise", 5, limiter)
        return ok, bad

    ok, bad = asyncio.run(exercise())
    assert ok == ("Success", "ok")
    assert bad[0] == "Error"
    assert "ValueError" in bad[1]


def track_terminated_workers():
    observed = []
    original = runner._shutdown_process_executor

    def wrapped(executor, *, terminate):
        processes = list((getattr(executor, "_processes", None) or {}).values())
        original(executor, terminate=terminate)
        if terminate:
            observed.extend(process.is_alive() for process in processes)

    return observed, wrapped


def test_process_executor_timeout_kills_runaway_worker_promptly(tmp_path: Path):
    limiter = runner.LoopBoundSemaphore(1)
    pid_file = tmp_path / "timeout-worker.pid"
    started = time.monotonic()
    async def exercise():
        task = asyncio.create_task(
            runner.execute_generated_code(fake_run_code, f"hang:{pid_file}", 1.2, limiter)
        )
        for _ in range(80):
            if pid_file.exists():
                break
            await asyncio.sleep(0.05)
        assert pid_file.exists(), "worker did not start before the timeout window"
        return await task

    observed, wrapped = track_terminated_workers()
    with patch.object(runner, "_shutdown_process_executor", wrapped):
        result = asyncio.run(exercise())
    elapsed = time.monotonic() - started
    assert result == ("Error", "Code execution timed out")
    assert elapsed < 6.0
    assert observed and not any(observed)


def test_process_executor_cancellation_kills_runaway_worker(tmp_path: Path):
    limiter = runner.LoopBoundSemaphore(1)
    pid_file = tmp_path / "cancelled-worker.pid"

    async def exercise():
        task = asyncio.create_task(
            runner.execute_generated_code(fake_run_code, f"hang:{pid_file}", 30, limiter)
        )
        for _ in range(80):
            if pid_file.exists():
                break
            await asyncio.sleep(0.05)
        assert pid_file.exists(), "worker did not start before cancellation"
        started = time.monotonic()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return time.monotonic() - started, int(pid_file.read_text(encoding="utf-8"))
        raise AssertionError("cancellation was swallowed")

    observed, wrapped = track_terminated_workers()
    with patch.object(runner, "_shutdown_process_executor", wrapped):
        elapsed, pid = asyncio.run(exercise())
    assert elapsed < 6.0
    assert observed and not any(observed)


def test_loop_limiter_can_be_reused_across_native_event_loops():
    limiter = runner.LoopBoundSemaphore(2)

    async def get_one():
        async with limiter.current():
            return True

    assert asyncio.run(get_one())
    assert asyncio.run(get_one())


def test_cli_executor_alias_rejects_a_different_runner_source():
    other = ModuleType("native_aime_formal")
    other.__file__ = "/different/frozen/native_aime_formal.py"
    with patch.dict(runner.__dict__, {"__name__": "__main__"}), patch.dict(sys.modules, {"native_aime_formal": other}):
        with pytest.raises(RuntimeError, match="different runner source"):
            runner.configure_code_executor("aflow", {}, 30, 1)


def _write_fixture_module(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent
    while parent != root:
        (parent / "__init__.py").touch()
        parent = parent.parent
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def _stage_real_programmer_templates(root: Path, method: str) -> None:
    """Use full pinned operator source; stub only unused model dependencies."""
    upstream = Path(__file__).resolve().parents[1] / "upstream" / ("AFlow" if method == "aflow" else "MaAS")
    if not upstream.is_dir():
        pytest.skip("pinned native sources are needed for real-template integration")
    _write_fixture_module(root, "tenacity.py", """
        def retry(*args, **kwargs):
            return lambda function: function
        def stop_after_attempt(*args):
            return None
        def wait_fixed(*args):
            return None
    """)
    _write_fixture_module(root, "huggingface_hub.py", """
        def snapshot_download(*args, **kwargs):
            return 'offline-tokenizer'
    """)
    _write_fixture_module(root, "transformers.py", """
        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return object()
    """)
    _write_fixture_module(root, "maas_context_guard.py", """
        class PromptGuard:
            def __init__(self, tokenizer, context_limit):
                pass
    """)
    benchmark = """
        class BaseBenchmark:
            async def evaluate_all_problems(self, *args, **kwargs):
                raise AssertionError('synthetic executor test must not evaluate problems')
            async def evaluate_all_problems_test(self, *args, **kwargs):
                raise AssertionError('synthetic executor test must not evaluate problems')
    """
    logger = "import logging\nlogger = logging.getLogger('offline-programmer-fixture')\n"
    if method == "aflow":
        native_paths = ["workspace/MATH/workflows/template/operator.py", "scripts/operators.py"]
        _write_fixture_module(root, "scripts/async_llm.py", "class AsyncLLM: pass\n")
        _write_fixture_module(root, "scripts/logs.py", logger)
        _write_fixture_module(root, "scripts/formatter.py", """
            class BaseFormatter: pass
            class FormatError(Exception): pass
            XmlFormatter = CodeFormatter = TextFormatter = BaseFormatter
        """)
        _write_fixture_module(root, "scripts/operator_an.py", "\n".join(
            f"class {name}: pass" for name in (
                "AnswerGenerateOp", "CodeGenerateOp", "FormatOp", "GenerateOp", "MdEnsembleOp",
                "ReflectionTestOp", "ReviewOp", "ReviseOp", "ScEnsembleOp",
            )
        ))
        _write_fixture_module(root, "scripts/prompts/prompt.py", "\n".join(
            f"{name} = ''" for name in (
                "ANSWER_GENERATION_PROMPT", "FORMAT_PROMPT", "MD_ENSEMBLE_PROMPT",
                "PYTHON_CODE_VERIFIER_PROMPT", "REFLECTION_ON_PUBLIC_TEST_PROMPT", "REVIEW_PROMPT",
                "REVISE_PROMPT", "SC_ENSEMBLE_PROMPT",
            )
        ))
        _write_fixture_module(root, "scripts/utils/code.py", """
            def extract_test_cases_from_jsonl(*args): pass
            def test_case_2_test_function(*args): pass
        """)
        _write_fixture_module(root, "benchmarks/benchmark.py", benchmark)
        _write_fixture_module(root, "benchmarks/math.py", "from benchmarks.benchmark import BaseBenchmark\nclass MATHBenchmark(BaseBenchmark): pass\n")
        _write_fixture_module(root, "tqdm/asyncio.py", "class tqdm_asyncio: pass\n")
    else:
        native_paths = [
            f"maas/ext/maas/scripts/optimized/MATH/{phase}/template/operator.py"
            for phase in ("train", "test")
        ]
        _write_fixture_module(root, "maas/logs.py", logger)
        _write_fixture_module(root, "maas/llm.py", "class LLM: pass\n")
        _write_fixture_module(root, "maas/actions/action_node.py", """
            class ActionNode:
                async def xml_fill(self, *args, **kwargs):
                    return {}
        """)
        _write_fixture_module(root, "maas/ext/maas/benchmark/benchmark.py", benchmark)
        _write_fixture_module(root, "maas/ext/maas/benchmark/math.py", "from maas.ext.maas.benchmark.benchmark import BaseBenchmark\nclass MATHBenchmark(BaseBenchmark): pass\n")
        _write_fixture_module(root, "maas/ext/maas/scripts/optimizer_utils/data_utils.py", "class DataUtils: pass\n")
        _write_fixture_module(root, "maas/provider/openai_api.py", """
            class OpenAILLM:
                def _cons_kwargs(self, *args, **kwargs):
                    raise AssertionError('no model requests allowed')
                async def _achat_completion(self, *args, **kwargs):
                    raise AssertionError('no model requests allowed')
                async def _achat_completion_function(self, *args, **kwargs):
                    raise AssertionError('no model requests allowed')
        """)
    for relative in native_paths:
        _write_fixture_module(root, relative, (upstream / relative).read_text(encoding="utf-8"))
        if "/template/" in relative:
            for sibling in ("operator_an.py", "op_prompt.py"):
                _write_fixture_module(root, str(Path(relative).with_name(sibling)), "# unused prompt dependency\n")
            if method == "aflow":
                target = "workspace/rpas_aime_native/MATH/workflows/template/operator.py"
            else:
                phase = Path(relative).parts[-3]
                target = f"workspace/rpas_aime_native/MATH/{phase}/template/operator.py"
            _write_fixture_module(root, target, (upstream / relative).read_text(encoding="utf-8"))


_REAL_TEMPLATE_DRIVER = r"""
import argparse
import asyncio
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import shutil
import sys

def exercise_cli_entry(parser, *args, **kwargs):
    # main() is genuinely entered under __main__; replace parsing with CPU-only
    # fixtures before frozen data or the model endpoint can be accessed.
    native = sys.modules['__main__']
    assert Path(native.__file__).resolve() == Path(sys.argv[1]).resolve()
    repo = Path(sys.argv[2])
    method = sys.argv[3]
    if method == 'aflow':
        copied_names = ['workspace.rpas_aime_native.MATH.workflows.template.operator']
        source_names = ['scripts.operators', 'workspace.MATH.workflows.template.operator']
    else:
        copied_names = [f'workspace.rpas_aime_native.MATH.{phase}.template.operator' for phase in ('train', 'test')]
        source_names = [f'maas.ext.maas.scripts.optimized.MATH.{phase}.template.operator' for phase in ('train', 'test')]
    # An already imported copy must be repaired as well as later imports.
    copied = [importlib.import_module(name) for name in copied_names]
    usage = native.patch_aflow_runtime(6144, 8) if method == 'aflow' else native.patch_maas_runtime(8, 6144)
    assert sys.modules['native_aime_formal'] is native
    assert sys.modules['native_aime_formal']._CODE_EXECUTOR_RUNTIMES is native._CODE_EXECUTOR_RUNTIMES
    assert usage['code_executor_adapter']['timeout_s'] == 1.2
    sources = [importlib.import_module(name) for name in source_names]
    dynamic = []
    for module in copied:
        relative = Path(module.__file__).relative_to(repo)
        target = repo / str(relative).replace('rpas_aime_native/', 'rpas_aime_native/dynamic/')
        target.parent.mkdir(parents=True, exist_ok=True)
        parent = target.parent
        while parent != repo:
            (parent / '__init__.py').touch()
            parent = parent.parent
        shutil.copy2(module.__file__, target)
        name = module.__name__.replace('rpas_aime_native.', 'rpas_aime_native.dynamic.')
        dynamic.append(importlib.import_module(name))
    modules = sources + copied + dynamic
    programmers = [module.Programmer(None) for module in modules]
    assert all(programmer._rpas_bounded_code_executor == method for programmer in programmers)
    ok = 'def solve():\n    return 42\n'
    async def first_loop():
        results = await asyncio.gather(*(programmer.exec_code(ok) for programmer in programmers))
        assert results == [('Success', '42')] * len(programmers)
    asyncio.run(first_loop())
    target = programmers[len(sources)]
    async def code_generate(*args, **kwargs):
        return {'code': ok}
    target.code_generate = code_generate
    async def second_loop():
        called, direct = await asyncio.gather(target('synthetic fixture'), programmers[-1].exec_code(ok))
        assert called == {'code': ok, 'output': '42'}
        assert direct == ('Success', '42')
    asyncio.run(second_loop())
    worker_pids = []
    def hanging_code(path):
        # The native evaluator forbids a literal import os. This CPU fixture
        # writes the actual worker PID while preserving its run_code semantics.
        return "import time\ndef solve():\n    with open(" + repr(str(path)) + ", 'w') as stream:\n        stream.write(str(__import__('os').getpid()))\n    time.sleep(60)\n"
    async def control_cases():
        result = await target.exec_code("def solve():\n    raise ValueError('synthetic failure')\n")
        assert result[0] == 'Error' and 'Execution error: synthetic failure' in result[1]
        assert await target.exec_code('import os\ndef solve():\n    return 42\n') == ('Error', 'Prohibited import: os and graphing functionalities')
        threaded = 'import threading\nimport time\ndef solve():\n    threading.Thread(target=lambda: time.sleep(60)).start()\n    return 42\n'
        assert await target.exec_code(threaded) == ('Success', '42')
        timeout_pid = repo / 'timeout.pid'
        assert await target.exec_code(hanging_code(timeout_pid), timeout=600) == ('Error', 'Code execution timed out')
        assert timeout_pid.is_file(), 'native run_code worker did not execute'
        worker_pids.append(int(timeout_pid.read_text()))
        cancel_pid = repo / 'cancel.pid'
        task = asyncio.create_task(programmers[-1].exec_code(hanging_code(cancel_pid), timeout=600))
        for _ in range(100):
            if cancel_pid.is_file():
                break
            await asyncio.sleep(0.01)
        assert cancel_pid.is_file(), 'native run_code worker did not start before cancellation'
        worker_pids.append(int(cancel_pid.read_text()))
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError('Programmer swallowed cancellation')
        assert await target.exec_code(ok) == ('Success', '42')
    asyncio.run(control_cases())
    for pid in worker_pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError(f'runaway worker {pid} is still alive')
    assert not multiprocessing.active_children()
    assert usage['code_execution_calls'] == len(programmers) + 8
    assert usage['code_execution_timeouts'] == 1
    assert usage['code_execution_errors'] == 2
    assert usage['code_execution_cancellations'] == 1
    assert set(usage['code_executor_adapter']['bound_modules']) == {module.__name__ for module in modules}
    print(json.dumps({key: usage[key] for key in ('code_execution_calls', 'code_execution_timeouts', 'code_execution_errors', 'code_execution_cancellations')}))
    raise SystemExit(0)

if __name__ == '__main__':
    sys.path.insert(0, sys.argv[2])
    sys.path.append(str(Path(sys.argv[1]).parent))
    os.environ['AIME_AFLOW_CODE_TIMEOUT_S'] = '1.2'
    os.environ['AIME_MAAS_CODE_TIMEOUT_S'] = '1.2'
    os.environ['AIME_CODE_EXEC_WORKERS'] = '1'
    argparse.ArgumentParser.parse_args = exercise_cli_entry
    runpy.run_path(sys.argv[1], run_name='__main__')
"""


@pytest.mark.parametrize("method", ["aflow", "maas"])
def test_real_template_programmers_cli_timeout_cancel_and_exit(tmp_path: Path, method: str):
    repo = tmp_path / method
    repo.mkdir()
    _stage_real_programmer_templates(repo, method)
    driver = _write_fixture_module(tmp_path, "offline_cli_fixture.py", _REAL_TEMPLATE_DRIVER)
    command = [
        "uv", "run", "--no-project", "--python", sys.executable,
        "python", str(driver), str(Path(runner.__file__).resolve()), str(repo), method,
    ]
    # Isolate only synthetic fixture processes so a failing regression cannot
    # leak workers, while never touching any formal benchmark process group.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail(f"native {method} template did not exit within 10s: {stdout}\n{stderr}")
    assert process.returncode == 0, f"{stdout}\n{stderr}"
    report = json.loads(stdout.strip().splitlines()[-1])
    assert report["code_execution_timeouts"] == report["code_execution_cancellations"] == 1
    assert report["code_execution_errors"] == 2


def test_aflow_retry_patch_is_bounded_and_fails_closed(tmp_path: Path):
    optimizer_dir = tmp_path / "scripts"
    optimizer_dir.mkdir()
    optimizer_path = optimizer_dir / "optimizer.py"
    optimizer_path.write_text(
        """import asyncio

class Optimizer:
    async def _optimize_graph(self):
        # Create a loop until the generated graph meets the check conditions
        while True:
            if check:
                break
        # Save the graph and evaluate
        return 1
""",
        encoding="utf-8",
    )

    runner.patch_aflow_optimizer_retry_guard(tmp_path)
    patched = optimizer_path.read_text(encoding="utf-8")
    compile(patched, str(optimizer_path), "exec")
    assert "import os\n" in patched
    assert "while True:" not in patched
    assert "range(max_mutation_attempts)" in patched
    assert "failed to produce an accepted graph mutation" in patched
    runner.patch_aflow_optimizer_retry_guard(tmp_path)  # idempotent


def test_aflow_round_limit_includes_initial_round_and_last_mutation(tmp_path: Path):
    for round_number in range(1, 11):
        round_dir = tmp_path / "workflows" / f"round_{round_number}"
        round_dir.mkdir(parents=True)
        (round_dir / "graph.py").write_text("# offline fixture\n", encoding="utf-8")

    assert runner.aflow_search_rounds(tmp_path / "workflows", max_mutations=8) == list(range(1, 10))


def test_shared_aime_scorer_exports_uniform_ordinal_scores(tmp_path: Path):
    output = tmp_path / "predictions.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["question", "expected_output", "prediction"])
        writer.writeheader()
        writer.writerows([
            {"question": "p1", "expected_output": r"\boxed{7}", "prediction": "FINAL ANSWER: 7"},
            {"question": "p2", "expected_output": r"\boxed{8}", "prediction": "FINAL ANSWER: 9"},
            {"question": "p3", "expected_output": r"\boxed{3}", "prediction": "reasoning only: 3"},
        ])
    metrics = runner.shared_aime_metrics(output.parent, [
        {"problem": "p1", "answer": "7", "aime_id": "id1"},
        {"problem": "p2", "answer": "8", "aime_id": "id2"},
        {"problem": "p3", "answer": "3", "aime_id": "id3"},
    ])
    assert metrics["score_0_1_2"] == [2, 1, 0]
    assert metrics["mean_score_0_1_2"] == 1.0
    assert [row["id"] for row in metrics["rows"]] == ["id1", "id2", "id3"]

def test_aflow_native_search_score_uses_shared_aime_parser():
    class DummyBenchmark:
        calculate_score = runner.aflow_aime_calculate_score

    benchmark = DummyBenchmark()
    assert benchmark.calculate_score(r"\boxed{7}", "FINAL ANSWER: 7") == (1, "7")
    assert benchmark.calculate_score(r"\boxed{7}", "FINAL ANSWER: 8") == (0, "8")
    assert benchmark.calculate_score(r"\boxed{7}", "reasoning without an answer") == (0, "")
    try:
        benchmark.calculate_score("unparseable reference", "FINAL ANSWER: 7")
    except ValueError as exc:
        assert "reference answer is not parseable" in str(exc)
    else:
        raise AssertionError("unparseable AIME gold must fail closed")
