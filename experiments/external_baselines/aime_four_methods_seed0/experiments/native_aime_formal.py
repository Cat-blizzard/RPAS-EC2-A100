#!/usr/bin/env python3
"""Run an official native AFlow or MaAS search on the frozen AIME split.

The runner deliberately keeps the baseline source trees isolated per run.  It
adapts only the dataset container (AIME rows -> the native MATH JSONL schema)
and the endpoint/decoding configuration; the native optimizer and workflow
code remain in control of search and execution.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import csv
import concurrent.futures
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import weakref
from urllib.request import urlopen
from answer_protocol import extract_aime_answer, score_aime
from aime_selection import efficiency_operating_point, quality_operating_point
from pathlib import Path
from typing import Any


MODEL = "Qwen/Qwen3.5-9B"
CONTEXT_LIMIT = 16384
OUTPUT_LIMIT = 6144
PROTOCOL_VERSION = "aime_main_protocol_v8_16k_shared_task_input"
EXTERNAL_AIME_PROTOCOL_VERSION = "aime_external_methods_v5_16k_shared_task_input"
ANSWER_PARSER = "answer_protocol_v4.extract_aime_answer"
MAX_SAMPLE_FAILURE_RATE = 0.05
FROZEN_AIME_MANIFEST = "frozen_aime_manifest.json"
FROZEN_AIME_MANIFEST_SHA256 = "7e6501210c7689e1702e9786a9652222c5ca33193d11d4cde531ea84bdee2cfe"
UPSTREAMS = {
    "aflow": {
        "repo": "https://github.com/FoundationAgents/AFlow.git",
        "commit": "3f457218fc716093fe53f6df8a5d5e6379d66346",
    },
    "maas": {
        "repo": "https://github.com/bingreeky/MaAS.git",
        "commit": "987f3c1bc9a96e844fe090db3791446e3ef0f5c7",
    },
}
AIME_SHARED_INPUT_INSTRUCTION = (
    "Solve this AIME problem. Give a concise derivation and put the integer "
    "answer on the last line as FINAL ANSWER: <integer>."
)
AIME_TASK_CONTRACT = {
    "version": "AIME_TASK_CONTRACT_v2_shared_input",
    "input": "shared AIME instruction plus original problem statement; gold answer and solution excluded",
    "gold": "integer answer from the frozen AIME dataset",
    "scoring": f"{ANSWER_PARSER}; normalized integer exact match",
}
AIME_TASK_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(AIME_TASK_CONTRACT, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
PHASES = ("search", "select", "test:aime_2025", "test:aime_2026")
BRIDGE_LOGS = (
    Path(__file__).resolve().parents[1] / "outputs" / "aime_tinker_9b_protocol_v2_main_bridge.jsonl",
    Path(__file__).resolve().parents[1] / "outputs" / "aime_tinker_9b_protocol_v2_bridge.jsonl",
)


def format_aime_task(problem: str) -> str:
    """Identical AIME user-task input across ALL four benchmark adapters."""
    return f"{AIME_SHARED_INPUT_INSTRUCTION}\n\n{problem}"


def task_contract_manifest() -> dict[str, str]:
    """Return AIME-only audit metadata; it is never appended to model prompts."""
    return {**AIME_TASK_CONTRACT, "sha256": AIME_TASK_CONTRACT_SHA256}


class LoopBoundSemaphore:
    """Provide a separate semaphore for each asyncio loop used by native code."""

    def __init__(self, limit: int):
        self.limit = max(1, int(limit))
        self._by_loop: weakref.WeakKeyDictionary[Any, asyncio.Semaphore] = weakref.WeakKeyDictionary()

    def current(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        semaphore = self._by_loop.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.limit)
            self._by_loop[loop] = semaphore
        return semaphore


def _shutdown_process_executor(executor: concurrent.futures.ProcessPoolExecutor, *, terminate: bool) -> None:
    """Stop workers promptly on timeout/cancellation; otherwise join normally."""
    if terminate:
        processes = list((getattr(executor, "_processes", None) or {}).values())
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
    executor.shutdown(wait=not terminate, cancel_futures=True)


async def execute_generated_code(
    run_code: Any,
    code: str,
    timeout: float,
    limiter: LoopBoundSemaphore,
) -> tuple[str, str]:
    """Run untrusted generated code in a bounded, disposable worker process."""
    async with limiter.current():
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        future = None
        try:
            future = asyncio.get_running_loop().run_in_executor(executor, run_code, code)
            result = await asyncio.wait_for(future, timeout=max(0.01, float(timeout)))
        except asyncio.TimeoutError:
            if future is not None:
                future.cancel()
            _shutdown_process_executor(executor, terminate=True)
            return "Error", "Code execution timed out"
        except asyncio.CancelledError:
            if future is not None:
                future.cancel()
            _shutdown_process_executor(executor, terminate=True)
            raise
        except Exception as exc:
            if future is not None:
                future.cancel()
            _shutdown_process_executor(executor, terminate=True)
            return "Error", f"Code execution failed: {type(exc).__name__}: {exc}"
        else:
            # A completed solve() may leave background threads in its worker.
            # Its result is already received; disposable workers must still end.
            _shutdown_process_executor(executor, terminate=True)
            return result


_CODE_EXECUTOR_RUNTIMES: dict[str, dict[str, Any]] = {}
_CODE_EXECUTOR_TEMPLATE_MARKER = "# RPAS bounded Programmer executor v2"


def configure_code_executor(
    method: str, telemetry: dict[str, Any], timeout_s: float, workers: int,
) -> None:
    """Set shared limits/counters before any native or copied template is used."""
    if method not in UPSTREAMS:
        raise ValueError(f"Unknown generated-code method: {method}")
    timeout_s = float(timeout_s)
    workers = int(workers)
    if not math.isfinite(timeout_s) or timeout_s <= 0 or workers <= 0:
        raise ValueError("Generated-code timeout and worker count must be positive and finite")
    # Direct CLI execution otherwise creates a second, unconfigured module when
    # a workflow imports the binder below under its ordinary module name.
    if __name__ == "__main__":
        existing = sys.modules.get("native_aime_formal")
        if existing is not None and Path(getattr(existing, "__file__", "") or "").resolve() != Path(__file__).resolve():
            raise RuntimeError("Refusing to bind generated-code templates to a different runner source")
        sys.modules["native_aime_formal"] = sys.modules[__name__]
    for key in (
        "code_execution_calls", "code_execution_timeouts", "code_execution_errors",
        "code_execution_cancellations",
    ):
        telemetry.setdefault(key, 0)
    telemetry["code_executor_adapter"] = {
        "version": "bounded_programmer_v2_all_template_paths",
        "timeout_s": timeout_s,
        "workers_per_event_loop": workers,
        "bound_modules": [],
        "template_repairs": [],
    }
    _CODE_EXECUTOR_RUNTIMES[method] = {
        "telemetry": telemetry, "timeout_s": timeout_s,
        "limiter": LoopBoundSemaphore(workers),
    }


def install_bounded_programmer(
    programmer: Any, run_code: Any, method: str, module_name: str,
) -> None:
    """Bind the actual class to its own native run_code and shared runtime.

    Binding is also safe during spawned-worker imports: runtime configuration
    is required only when exec_code is called, not when run_code is imported.
    """
    if method not in UPSTREAMS:
        raise ValueError(f"Unknown generated-code method: {method}")

    def record_module(telemetry: dict[str, Any]) -> None:
        modules = telemetry["code_executor_adapter"]["bound_modules"]
        if module_name not in modules:
            modules.append(module_name)

    async def bounded_exec_code(self: Any, code: str, timeout: float = 30 if method == "aflow" else 600):
        runtime = _CODE_EXECUTOR_RUNTIMES.get(method)
        if runtime is None:
            raise RuntimeError(f"{method} generated-code runtime was not configured")
        telemetry = runtime["telemetry"]
        record_module(telemetry)
        telemetry["code_execution_calls"] += 1
        try:
            result = await execute_generated_code(
                run_code, code, min(float(timeout), runtime["timeout_s"]), runtime["limiter"],
            )
        except asyncio.CancelledError:
            telemetry["code_execution_cancellations"] += 1
            raise
        if result[0] != "Success":
            field = "code_execution_timeouts" if result[1] == "Code execution timed out" else "code_execution_errors"
            telemetry[field] += 1
        return result

    programmer.exec_code = bounded_exec_code
    programmer._rpas_bounded_code_executor = method
    runtime = _CODE_EXECUTOR_RUNTIMES.get(method)
    if runtime is not None:
        record_module(runtime["telemetry"])


def patch_programmer_templates(repo: Path, method: str) -> list[dict[str, str]]:
    """Repair isolated templates before import/copy, including future aliases.

    This is a run-local execution adapter. Never call it on the pinned upstream
    checkout or an existing formal run. Template run_code and retry loops remain
    native; only their executor binding is appended.
    """
    repo = repo.resolve()
    if method == "aflow":
        originals = [repo / "workspace/MATH/workflows/template/operator.py"]
    elif method == "maas":
        originals = [
            repo / f"maas/ext/maas/scripts/optimized/MATH/{phase}/template/operator.py"
            for phase in ("train", "test")
        ]
    else:
        raise ValueError(f"Unknown generated-code method: {method}")
    isolated = repo / "workspace/rpas_aime_native"
    paths = sorted(set(originals + list(isolated.rglob("template/operator.py"))))
    repairs = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        if not any(isinstance(node, ast.ClassDef) and node.name == "Programmer" for node in tree.body):
            raise RuntimeError(f"Native Programmer class missing from isolated template: {path}")
        if not any(isinstance(node, ast.FunctionDef) and node.name == "run_code" for node in tree.body):
            raise RuntimeError(f"Native run_code missing from isolated template: {path}")
        before_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if _CODE_EXECUTOR_TEMPLATE_MARKER not in source:
            source += (
                f"\n\n{_CODE_EXECUTOR_TEMPLATE_MARKER}\n"
                "from native_aime_formal import install_bounded_programmer as _rpas_bind_programmer\n"
                f"_rpas_bind_programmer(Programmer, run_code, {method!r}, __name__)\n"
                "del _rpas_bind_programmer\n"
            )
            compile(source, str(path), "exec")
            path.write_text(source, encoding="utf-8")
        repairs.append({
            "path": str(path.relative_to(repo)), "source_sha256": before_sha,
            "patched_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        })
    runtime = _CODE_EXECUTOR_RUNTIMES[method]
    runtime["telemetry"]["code_executor_adapter"]["template_repairs"] = repairs
    # Native classes may already have been imported while constructing the
    # runtime. Apply the same binder to those objects, not just their files.
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and Path(file).resolve() in paths:
            install_bounded_programmer(module.Programmer, module.run_code, method, name)
    return repairs


def patch_aflow_optimizer_retry_guard(repo: Path) -> None:
    """Bound AFlow's native retry loop for rejected graph mutations."""
    path = repo / "scripts" / "optimizer.py"
    source = path.read_text(encoding="utf-8")
    if not re.search(r"(?m)^import os$", source):
        import_marker = "import asyncio\n"
        if source.count(import_marker) != 1:
            raise RuntimeError("Cannot safely add os import to AFlow optimizer; refusing the patch")
        source = source.replace(import_marker, import_marker + "import os\n", 1)
    start_marker = "        # Create a loop until the generated graph meets the check conditions\n"
    end_marker = "        # Save the graph and evaluate\n"
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise RuntimeError("Unexpected AFlow optimizer mutation-loop layout; refusing an unsafe patch")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    region = source[start:end]
    if "for _mutation_attempt in range(max_mutation_attempts):" in region:
        return
    if region.count("        while True:\n") != 1 or region.count("            if check:\n                break\n") != 1:
        raise RuntimeError("Unexpected AFlow mutation retry body; refusing an unsafe patch")
    region = region.replace(
        start_marker + "        while True:\n",
        start_marker
        + "        max_mutation_attempts = max(1, int(os.environ.get(\"AIME_AFLOW_MUTATION_RETRIES\", \"8\")))\n"
        + "        for _mutation_attempt in range(max_mutation_attempts):\n",
        1,
    )
    region = region.replace(
        "            if check:\n                break\n",
        "            if check:\n                break\n"
        "        else:\n"
        "            raise RuntimeError(\n"
        "                f\"AFlow failed to produce an accepted graph mutation after {max_mutation_attempts} attempts\"\n"
        "            )\n",
        1,
    )
    patched = source[:start] + region + source[end:]
    compile(patched, str(path), "exec")
    path.write_text(patched, encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_python_tree(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*.py")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        entries.append((str(path.relative_to(root)), sha256_file(path)))
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode("utf-8")).hexdigest()


def _canonical_repo_url(value: str) -> str:
    value = value.strip()
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    elif value.startswith("ssh://git@github.com/"):
        value = "https://github.com/" + value.removeprefix("ssh://git@github.com/")
    return value.removesuffix(".git").rstrip("/").lower()


def verify_pinned_upstream(method: str, root: Path) -> dict[str, Any]:
    """Require the known upstream origin, exact commit, and a clean source tree."""
    expected = UPSTREAMS[method]

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args], check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()

    if not root.is_dir():
        raise FileNotFoundError(f"{method} upstream source not found: {root}")
    commit = git("rev-parse", "HEAD")
    origin = git("remote", "get-url", "origin")
    dirty = git("status", "--porcelain", "--untracked-files=normal")
    if _canonical_repo_url(origin) != _canonical_repo_url(expected["repo"]):
        raise RuntimeError(f"{method} source origin is not the audited upstream: {origin}")
    if commit != expected["commit"]:
        raise RuntimeError(f"{method} source must be pinned to {expected['commit']}, got {commit}")
    if dirty:
        raise RuntimeError(f"{method} upstream source checkout is dirty; refusing an unaudited run")
    return {
        "repository": expected["repo"], "origin": origin,
        "commit": commit, "clean_worktree": True,
        "python_tree_sha256": sha256_python_tree(root),
    }


def baseline_source_path(args: argparse.Namespace, method: str) -> Path:
    env_name = "AIME_AFLOW_SOURCE" if method == "aflow" else "AIME_MAAS_SOURCE"
    directory = "AFlow" if method == "aflow" else "MaAS"
    return Path(os.environ.get(env_name, str(args.baseline_root / directory))).resolve()


def verify_served_model(endpoint: str) -> None:
    url = endpoint.rstrip("/") + "/models"
    try:
        with urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot verify task-model endpoint {url}: {type(exc).__name__}: {exc}") from exc
    served = {str(item.get("id", "")) for item in payload.get("data", [])}
    if MODEL not in served:
        raise RuntimeError(f"endpoint serves {sorted(served)!r}, expected exactly-compatible model {MODEL!r}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("aflow", "maas"), required=True)
    parser.add_argument(
        "--baseline-root", type=Path,
        default=Path(os.environ.get("AIME_UPSTREAM_ROOT", Path(__file__).resolve().parent / "upstream")),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--search-size", type=int, default=60)
    parser.add_argument("--selection-size", type=int, default=30)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--aflow-max-rounds", type=int, default=8)
    parser.add_argument("--maas-samples", type=int, default=1)
    parser.add_argument("--maas-batch-size", type=int, default=8)
    parser.add_argument(
        "--maas-resume",
        action="store_true",
        help="reuse an existing MaAS repo/controller and only run selection/test",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_frozen_aime_manifest(data_dir: Path) -> dict[str, Any]:
    """Load the published, hash-pinned manifest; reject incomplete data bundles."""
    path = data_dir / FROZEN_AIME_MANIFEST
    actual = sha256_file(path)
    if actual != FROZEN_AIME_MANIFEST_SHA256:
        raise ValueError(f"frozen AIME manifest hash mismatch: {actual}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "rpas_frozen_aime_external_v1"
            or manifest.get("source_repository") != "JiangyueAnn/RPAS"
            or manifest.get("source_revision") != "e12f58823be5f91a32f05f9af4d36e54838ffe59"
            or manifest.get("data_seed") != 2026):
        raise ValueError("frozen AIME manifest has an unsupported source or split protocol")
    return manifest


def read_verified_jsonl(
    data_dir: Path, relative_path: str, expected_sha256: str, expected_rows: int
) -> tuple[list[dict[str, Any]], str]:
    """Read one manifest-pinned JSONL file and enforce its exact row count."""
    path = data_dir / relative_path
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"frozen AIME data hash mismatch for {relative_path}: {actual}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != expected_rows:
        raise ValueError(f"{relative_path} has {len(rows)} rows; expected {expected_rows}")
    return rows, actual


def _question_answer_map(rows: list[dict[str, Any]], label: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in rows:
        question = " ".join(str(row.get("problem", row.get("input", ""))).casefold().split())
        answer = str(row.get("answer", "")).strip()
        if not question or not answer or question in mapping:
            raise ValueError(f"{label} contains an empty answer/question or duplicate question")
        mapping[question] = answer
    return mapping


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def freeze_split(data_dir: Path, seed: int, search_size: int, selection_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (seed, search_size, selection_size) != (2026, 60, 30):
        raise ValueError("the published AIME protocol requires data_seed=2026 and frozen 60/30 splits")
    manifest = load_frozen_aime_manifest(data_dir)
    validation = manifest["validation"]
    raw_rows, _ = read_verified_jsonl(
        data_dir, validation["source"]["path"],
        validation["source"]["sha256_git_content"], 90,
    )
    search_rows, _ = read_verified_jsonl(
        data_dir, validation["search"]["path"],
        validation["search"]["sha256_git_content"], 60,
    )
    select_rows, _ = read_verified_jsonl(
        data_dir, validation["select"]["path"],
        validation["select"]["sha256_git_content"], 30,
    )
    raw_mapping = _question_answer_map(raw_rows, "AIME validation source")
    search_mapping = _question_answer_map(search_rows, "frozen D_search")
    select_mapping = _question_answer_map(select_rows, "frozen D_select")
    if (set(search_mapping) & set(select_mapping)
            or {**search_mapping, **select_mapping} != raw_mapping):
        raise ValueError("canonical D_search/D_select do not partition the exact frozen validation source")
    return search_rows, select_rows


def canonical_aime_id(row: dict[str, Any], namespace: str) -> str:
    """Qualify source-local numeric IDs so different AIME files cannot collide."""
    value = row.get("source_row_id", row.get("id", row.get("problem_idx", "")))
    value = str(value).strip()
    if not value:
        return ""
    prefix = f"{namespace}:"
    return value if value.startswith(prefix) else prefix + value


def namespace_aime_rows(rows: list[dict[str, Any]], namespace: str) -> list[dict[str, Any]]:
    qualified = []
    for row in rows:
        item = dict(row)
        item.setdefault("source_row_id", str(row.get("id", row.get("problem_idx", ""))))
        item["id"] = canonical_aime_id(item, namespace)
        qualified.append(item)
    return qualified


def validate_aime_partitions(partitions: dict[str, list[dict[str, Any]]]) -> None:
    """Check unique qualified IDs and reject cross-split duplicate problem text."""
    ids_by_partition: dict[str, set[str]] = {}
    content_by_partition: dict[str, set[str]] = {}
    for name, rows in partitions.items():
        ids = [str(row.get("id", "")).strip() for row in rows]
        if any(not row_id for row_id in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"AIME partition {name} must have unique nonempty qualified IDs")
        contents = []
        for row in rows:
            question = " ".join(str(row.get("problem", "")).casefold().split())
            if not question:
                raise ValueError(f"AIME partition {name} contains an empty problem")
            contents.append(hashlib.sha256(question.encode("utf-8")).hexdigest())
        if len(set(contents)) != len(contents):
            raise ValueError(f"AIME partition {name} contains duplicate problem text")
        ids_by_partition[name] = set(ids)
        content_by_partition[name] = set(contents)
    names = list(partitions)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            a, b = names[left], names[right]
            if ids_by_partition[a] & ids_by_partition[b]:
                raise ValueError(f"AIME partition IDs overlap: {a} vs {b}")
            if content_by_partition[a] & content_by_partition[b]:
                raise ValueError(f"AIME problem content overlaps: {a} vs {b}")


def native_math_rows(rows: list[dict[str, Any]], namespace: str = "validation") -> list[dict[str, Any]]:
    # Native MATH evaluators accept a solution string and extract the final
    # boxed answer.  The AIME gold answer is kept only in the gold container;
    # it is never added to the problem prompt.
    return [
        {
            "problem": format_aime_task(str(row["problem"])),
            "solution": rf"\boxed{{{row['answer']}}}",
            "aime_id": canonical_aime_id(row, namespace),
            "answer": str(row["answer"]),
        }
        for row in rows
    ]


def test_file_rows(data_dir: Path, filename: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(data_dir / filename)
    return native_math_rows(rows[:limit] if limit is not None else rows, namespace=Path(filename).stem)


def _sha256_ids(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(str(row.get("id", "")) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _question_sha256(row: dict[str, Any]) -> str:
    question = " ".join(str(row.get("problem", "")).casefold().split())
    if not question:
        raise ValueError("AIME row has an empty problem statement")
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def freeze_aime_selection(
    run_dir: Path,
    method: str,
    search_rows: list[dict[str, Any]],
    select_rows: list[dict[str, Any]],
    selected: dict[str, Any],
    source_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Durably lock candidates from validation splits before any D_test open."""
    path = run_dir / "selection_frozen.json"
    if (run_dir / "dtest_access_manifest.json").exists():
        raise RuntimeError("cannot freeze selection after any D_test access has been recorded")
    if method not in {"aflow", "maas", "adas", "gdesigner"}:
        raise ValueError(f"unsupported AIME method for selection freeze: {method}")
    method_protocol = (PROTOCOL_VERSION if method in {"aflow", "maas"}
                       else EXTERNAL_AIME_PROTOCOL_VERSION)
    payload = {
        "schema": "aime_dtest_blind_selection_lock_v2",
        "protocol_version": method_protocol,
        "frozen_data_manifest_sha256": FROZEN_AIME_MANIFEST_SHA256,
        "method": method,
        "search_size": len(search_rows),
        "selection_size": len(select_rows),
        "search_ids_sha256": _sha256_ids(search_rows),
        "selection_ids_sha256": _sha256_ids(select_rows),
        "validation_question_sha256": sorted(
            {_question_sha256(row) for row in search_rows + select_rows}
        ),
        "selected": selected,
        "source_provenance": source_provenance,
        "created_at_epoch": time.time(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dtest_loaded_before_lock": False,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    # Exclusive creation prevents an old/partial selection from being
    # silently replaced before test evaluation or during a resume attempt.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def load_aime_test_rows_after_freeze(
    data_dir: Path, run_dir: Path, method: str, filename: str, expected_count: int = 30
) -> list[dict[str, Any]]:
    """Open one blind test split only after a durable candidate-freeze lock."""
    lock_path = run_dir / "selection_frozen.json"
    if not lock_path.is_file():
        raise RuntimeError("refusing D_test access before selection_frozen.json exists")
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    expected_protocol = (PROTOCOL_VERSION if method in {"aflow", "maas"}
                         else EXTERNAL_AIME_PROTOCOL_VERSION)
    if (lock.get("schema") != "aime_dtest_blind_selection_lock_v2"
            or lock.get("method") != method
            or lock.get("protocol_version") != expected_protocol
            or lock.get("frozen_data_manifest_sha256") != FROZEN_AIME_MANIFEST_SHA256
            or lock.get("dtest_loaded_before_lock") is not False):
        raise RuntimeError("D_test selection lock is invalid or belongs to another method")
    if filename not in {"aime_2025.jsonl", "aime_2026.jsonl"}:
        raise ValueError(f"unsupported frozen AIME test split: {filename}")

    frozen_manifest = load_frozen_aime_manifest(data_dir)
    test_spec = frozen_manifest["test"].get(filename)
    if test_spec is None:
        raise ValueError(f"{filename} is absent from the frozen AIME data manifest")
    if expected_count != test_spec["rows"]:
        raise ValueError(f"{filename} expected_count conflicts with the frozen data manifest")
    path = data_dir / filename
    opened_at = time.time()
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    source_hash = hashlib.sha256(raw).hexdigest()
    if source_hash != test_spec["source_sha256_git_content"] or len(rows) != expected_count:
        raise ValueError(f"{filename} differs from the exact frozen AIME source")
    frozen_rows, frozen_hash = read_verified_jsonl(
        data_dir, test_spec["split_path"], test_spec["split_sha256_git_content"], expected_count
    )
    if _question_answer_map(rows, filename) != _question_answer_map(frozen_rows, test_spec["split_path"]):
        raise ValueError(f"{filename} does not match its canonical frozen D_test split")
    namespace = Path(filename).stem
    qualified = namespace_aime_rows(rows, namespace)
    question_hashes = [_question_sha256(row) for row in qualified]
    validation_hashes = set(lock.get("validation_question_sha256", []))
    if validation_hashes.intersection(question_hashes):
        raise ValueError(f"{filename} contains question text overlapping D_search/D_select")
    audit_path = run_dir / "dtest_access_manifest.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("selection_lock_sha256") != hashlib.sha256(lock_bytes).hexdigest():
            raise RuntimeError("D_test access manifest is tied to a different selection lock")
    else:
        audit = {
            "schema": "aime_dtest_access_manifest_v1",
            "method": method,
            "selection_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "selection_frozen_at_epoch": lock["created_at_epoch"],
            "test_splits": {},
        }
    if filename in audit["test_splits"]:
        raise RuntimeError(f"refusing to reopen already-recorded D_test file: {filename}")
    earlier_question_hashes = {
        value for split in audit["test_splits"].values()
        for value in split.get("question_sha256", [])
    }
    if earlier_question_hashes.intersection(question_hashes):
        raise ValueError(f"{filename} contains duplicate problem text from another D_test year")
    audit["test_splits"][filename] = {
        "opened_at_epoch": opened_at,
        "sha256": source_hash,
        "frozen_split_path": test_spec["split_path"],
        "frozen_split_sha256": frozen_hash,
        "frozen_data_manifest_sha256": FROZEN_AIME_MANIFEST_SHA256,
        "ids": [str(row["id"]) for row in qualified],
        "question_sha256": question_hashes,
        "row_count": len(qualified),
    }
    dump_json(audit_path, audit)
    return qualified


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.lock"),
    )


def patch_aflow_code_formatter(repo: Path) -> None:
    """Normalize AFlow's formatter key in this isolated run copy."""
    path = repo / "scripts" / "formatter.py"
    source = path.read_text(encoding="utf-8")
    old = 'result = {"response": sanitized_code}'
    new = 'result = {"code": sanitized_code}'
    if source.count(old) == 1:
        path.write_text(source.replace(old, new), encoding="utf-8")
    elif source.count(new) == 1:
        pass
    else:
        raise RuntimeError(f"Unexpected AFlow CodeFormatter implementation: {path}")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def ensure_package_tree(path: Path, stop_at: Path) -> None:
    current = path
    while True:
        init_file = current / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
        if current == stop_at:
            break
        current = current.parent


def patch_aflow_graph_namespace(graph_utils: Any) -> None:
    """Keep generated AFlow graphs inside this run's isolated package tree.

    The upstream MATH template hard-codes ``workspace.MATH`` in generated
    imports.  Our formal runner gives each baseline an isolated namespace so
    artifacts from another run cannot leak into the search; rewrite only
    those generated imports to the corresponding isolated namespace.
    """
    original_write = graph_utils.write_graph_files

    def write_graph_files(self: Any, directory: str, response: dict[str, Any], round_number: int, dataset: str) -> None:
        original_write(self, directory, response, round_number, dataset)
        directory_path = Path(directory)
        for filename in ("graph.py", "prompt.py"):
            path = directory_path / filename
            if path.exists():
                text = path.read_text(encoding="utf-8")
                text = text.replace("workspace.MATH", "workspace.rpas_aime_native.MATH")
                # Some Qwen outputs wrap an otherwise valid prompt module in
                # line comments (e.g. ``# STRUCTURED_REASONING_PROMPT =``),
                # which makes the generated graph fail only at evaluation
                # time.  Uncomment that whole prompt module before import.
                if filename == "prompt.py" and re.search(
                    r"(?m)^#\s*[A-Z][A-Z0-9_]*_PROMPT\s*=\s*['\"]{1,3}", text
                ):
                    normalized_lines = []
                    for line in text.splitlines():
                        if line.startswith("#"):
                            line = line[1:]
                            if line.startswith(" "):
                                line = line[1:]
                        normalized_lines.append(line)
                    text = "\n".join(normalized_lines) + "\n"
                path.write_text(text, encoding="utf-8")

    graph_utils.write_graph_files = write_graph_files


def validate_aflow_workflow_round(workflow_root: Path, round_number: int) -> list[str]:
    """Validate generated graph/prompt interfaces before evaluation.

    AFlow's optimizer may generate arbitrary Python workflow code. With a
    smaller local model, a graph can reference a prompt constant that its
    generated prompt module did not define. The upstream evaluator retries
    every sample and silently records skipped examples; reject that candidate
    before sending requests so it cannot produce a misleading score.
    """
    round_root = workflow_root / f"round_{round_number}"
    graph_path = round_root / "graph.py"
    prompt_path = round_root / "prompt.py"
    if not graph_path.exists():
        return ["missing graph.py"]
    if not prompt_path.exists():
        return ["missing prompt.py"]
    try:
        graph_tree = ast.parse(graph_path.read_text(encoding="utf-8"), filename=str(graph_path))
        prompt_tree = ast.parse(prompt_path.read_text(encoding="utf-8"), filename=str(prompt_path))
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} at line {exc.lineno}"]

    referenced: set[str] = set()
    for node in ast.walk(graph_tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "prompt_custom" and node.attr.endswith("_PROMPT"):
                referenced.add(node.attr)
    defined: set[str] = set()
    for node in ast.walk(prompt_tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id.endswith("_PROMPT"):
                    defined.add(target.id)
    missing = sorted(referenced - defined)
    return ["missing prompt constants: " + ", ".join(missing)] if missing else []


def aflow_search_rounds(workflow_root: Path, max_mutations: int) -> list[int]:
    """List the initial workflow plus every native mutation round."""
    return sorted(
        int(match.group(1))
        for path in workflow_root.glob("round_*/graph.py")
        if (match := re.fullmatch(r"round_(\d+)", path.parent.name))
        and int(match.group(1)) <= 1 + int(max_mutations)
    )


def audit_aflow_outputs(log_dir: Path) -> dict[str, Any]:
    """Audit explicit workflow failure sentinels without matching prose mentions."""
    csv_paths = sorted(log_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime)
    if not csv_paths:
        return {
            "csv_rows": 0,
            "empty_predictions": 0,
            "failure_predictions": 0,
            "failure_fraction": 1.0,
            "sample_timeouts": 0,
            "failure_markers": [],
        }
    latest = csv_paths[-1]
    with latest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    marker_prefixes = (
        "aflow sample timeout",
        "maas sample failed",
        "no code generated",
        "maximum retries reached",
        "traceback (most recent call last)",
        "execution error:",
        "code execution timed out",
        "code execution exited with code",
        "code execution returned invalid output",
        "code execution failed:",
        "unknown error:",
        "error:",
    )
    failure_rows = []
    sample_timeouts = 0
    empty = 0
    for row in rows:
        prediction = str(row.get("prediction", "") or "").strip()
        lowered = prediction.lower()
        # Native evaluators sometimes persist str(exc) without the exception
        # class. Count complete error shapes, not ordinary explanatory prose.
        bare_attribute_error = re.fullmatch(
            r"(?:attributeerror:\s*)?'[\w.]+' object has no attribute '\w+'", lowered
        ) is not None
        bare_name_error = re.fullmatch(
            r"(?:nameerror:\s*)?name '[^']+' is not defined", lowered
        ) is not None
        pydantic_validation_error = re.match(
            r"^(?:error:\s*)?(?:\d+\s+)?validation error\b", lowered
        ) is not None
        if not prediction:
            empty += 1
            failure_rows.append("<empty prediction>")
        elif bare_attribute_error or bare_name_error or pydantic_validation_error or any(
            lowered == marker or lowered.startswith(marker) for marker in marker_prefixes
        ):
            failure_rows.append(prediction[:160])
            if lowered.startswith("aflow sample timeout"):
                sample_timeouts += 1
    count = len(rows)
    return {
        "csv": str(latest),
        "csv_rows": count,
        "empty_predictions": empty,
        "failure_predictions": len(failure_rows),
        "failure_fraction": (len(failure_rows) / count) if count else 1.0,
        "sample_timeouts": sample_timeouts,
        "failure_markers": failure_rows[:10],
    }


def shared_aime_metrics(log_dir: Path, expected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score native outputs with the same frozen AIME parser across methods."""
    csv_paths = sorted(log_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime)
    if not csv_paths:
        raise RuntimeError(f"native AIME evaluator produced no CSV output in {log_dir}")
    with csv_paths[-1].open(newline="", encoding="utf-8") as stream:
        outputs = list(csv.DictReader(stream))
    if len(outputs) != len(expected_rows):
        raise RuntimeError(
            f"AIME output count mismatch in {log_dir}: {len(outputs)} != {len(expected_rows)}"
        )
    normalize = lambda value: re.sub(r"\s+", " ", str(value or "")).strip()
    scored = []
    for index, (output, expected) in enumerate(zip(outputs, expected_rows)):
        question = normalize(output.get("question", ""))
        if question and question != normalize(expected.get("problem", "")):
            raise RuntimeError(f"AIME evaluator row order/content mismatch at {log_dir} row {index}")
        csv_gold = extract_aime_answer(output.get("expected_output", ""))
        raw_expected_gold = str(expected.get("answer", "")).strip()
        expected_gold = extract_aime_answer(raw_expected_gold)
        if not expected_gold:
            raise RuntimeError(f"AIME frozen gold is unparseable at {log_dir} row {index}")
        if output.get("expected_output") and not csv_gold:
            raise RuntimeError(f"AIME evaluator gold is unparseable at {log_dir} row {index}")
        if csv_gold and csv_gold != expected_gold:
            raise RuntimeError(f"AIME evaluator gold mismatch at {log_dir} row {index}")
        item = score_aime(output.get("prediction", ""), expected_gold)
        item["id"] = str(expected.get("aime_id", expected.get("id", expected.get("problem_idx", index))))
        scored.append(item)
    correct = sum(bool(row["correct"]) for row in scored)
    valid = sum(bool(row["parser_valid"]) for row in scored)
    count = len(scored)
    return {
        "score": correct / count if count else 0.0,
        "accuracy": correct / count if count else 0.0,
        "correct": correct,
        "num_examples": count,
        "valid_answer_rate": valid / count if count else 0.0,
        "mean_score_0_1_2": sum(int(row["score_0_1_2"]) for row in scored) / count if count else 0.0,
        "score_0_1_2": [int(row["score_0_1_2"]) for row in scored],
        "rows": scored,
        "answer_parser": ANSWER_PARSER,
    }


def usage_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in after}


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def empty_phase_usage() -> dict[str, int]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "failed_calls": 0,
    }


def new_phase_totals() -> dict[str, dict[str, int]]:
    return {phase: empty_phase_usage() for phase in PHASES}


def current_run_id() -> str:
    """Prefer the AIME launcher ID while retaining legacy bridge callers."""
    return os.environ.get("AIME_RUN_ID", "") or os.environ.get("MASBENCH_RUN_ID", "")


def current_phase() -> str:
    return os.environ.get("MASBENCH_PHASE", "unattributed") or "unattributed"


@contextmanager
def phase_scope(phase: str, phase_wall_seconds: dict[str, float] | None = None):
    previous = os.environ.get("MASBENCH_PHASE")
    started = time.perf_counter()
    os.environ["MASBENCH_PHASE"] = phase
    try:
        yield
    finally:
        if phase_wall_seconds is not None:
            phase_wall_seconds[phase] = phase_wall_seconds.get(phase, 0.0) + (
                time.perf_counter() - started
            )
        if previous is None:
            os.environ.pop("MASBENCH_PHASE", None)
        else:
            os.environ["MASBENCH_PHASE"] = previous


def begin_call(telemetry: dict[str, Any]) -> str:
    phase = current_phase()
    telemetry["calls"] += 1
    phase_usage = telemetry.setdefault("phase_totals", {}).setdefault(phase, empty_phase_usage())
    phase_usage["calls"] += 1
    return phase


def failed_call(telemetry: dict[str, Any], phase: str) -> None:
    telemetry["failed_calls"] += 1
    telemetry["phase_totals"][phase]["failed_calls"] += 1


def record_usage(
    telemetry: dict[str, Any],
    phase: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> None:
    prompt_tokens = max(0, int(prompt_tokens))
    completion_tokens = max(0, int(completion_tokens))
    reported_total = int(total_tokens)
    total_tokens = reported_total if reported_total > 0 else prompt_tokens + completion_tokens
    telemetry["prompt_tokens"] += prompt_tokens
    telemetry["completion_tokens"] += completion_tokens
    telemetry["total_tokens"] += total_tokens
    phase_usage = telemetry["phase_totals"][phase]
    phase_usage["prompt_tokens"] += prompt_tokens
    phase_usage["completion_tokens"] += completion_tokens
    phase_usage["total_tokens"] += total_tokens


def phase_snapshot(telemetry: dict[str, Any], phase: str) -> dict[str, int]:
    return dict(telemetry.get("phase_totals", {}).get(phase, empty_phase_usage()))

def aflow_aime_calculate_score(self: Any, expected_output: str, prediction: str) -> tuple[int, str]:
    """Use the frozen AIME answer parser for AFlow's native search fitness."""
    expected_answer = extract_aime_answer(expected_output)
    if not expected_answer:
        raise ValueError("AFlow AIME reference answer is not parseable")
    scored = score_aime(prediction, expected_answer)
    # Preserve AFlow's native exact-match accuracy reward while eliminating
    # its incompatible MATH boxed-answer extraction on AIME outputs.
    return int(bool(scored["correct"])), str(scored["prediction"])


def patch_aflow_aime_search_scorer(benchmark_class: Any) -> None:
    benchmark_class.calculate_score = aflow_aime_calculate_score

def bridge_phase_usage(run_id: str, phase: str) -> dict[str, int]:
    """Read completed request usage for one phase from the audited bridge log."""
    usage = empty_phase_usage()
    seen: set[str] = set()
    for path in BRIDGE_LOGS:
        if not path.exists():
            continue
        try:
            stream = path.open(encoding="utf-8")
        except OSError:
            continue
        with stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("run_id") != run_id or row.get("phase") != phase:
                    continue
                request_id = str(row.get("request_id", ""))
                if request_id and request_id in seen:
                    continue
                if request_id:
                    seen.add(request_id)
                usage["calls"] += 1
                usage["prompt_tokens"] += int(row.get("prompt_tokens", 0) or 0)
                usage["completion_tokens"] += int(row.get("completion_tokens", 0) or 0)
                usage["total_tokens"] += int(row.get("total_tokens", 0) or 0)
                usage["failed_calls"] += int(bool(row.get("error")))
    return usage


def merge_phase_usage(telemetry: dict[str, Any], phase: str, usage: dict[str, int]) -> None:
    """Add an audited phase snapshot to in-process telemetry."""
    phase_usage = telemetry.setdefault("phase_totals", {}).setdefault(phase, empty_phase_usage())
    for field in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "failed_calls"):
        value = int(usage.get(field, 0) or 0)
        phase_usage[field] += value
        telemetry[field] += value


def patch_aflow_runtime(max_tokens: int, concurrency: int) -> dict[str, int]:
    """Use the native AFlow objects with the common endpoint contract."""
    from scripts.async_llm import AsyncLLM
    import scripts.operators as operators
    from benchmarks.benchmark import BaseBenchmark
    from benchmarks.math import MATHBenchmark
    from tqdm.asyncio import tqdm_asyncio
    from maas_context_guard import PromptGuard
    patch_aflow_aime_search_scorer(MATHBenchmark)

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    telemetry = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "failed_calls": 0,
        "sample_timeouts": 0,
        "phase_totals": new_phase_totals(),
        "finish_reason_counts": {},
        "completion_truncated": 0,
        "phase_wall_seconds": {},
        "code_execution_timeouts": 0,
        "code_execution_errors": 0,
    }
    code_timeout = float(os.environ.get("AIME_AFLOW_CODE_TIMEOUT_S", "30"))
    configure_code_executor(
        "aflow", telemetry, code_timeout,
        int(os.environ.get("AIME_CODE_EXEC_WORKERS", str(min(8, os.cpu_count() or 1)))),
    )
    install_bounded_programmer(operators.Programmer, operators.run_code, "aflow", operators.__name__)
    # Workflows import a distinct template.Programmer rather than the scripts
    # class. Patch isolated template bytes before subsequent namespace copies.
    patch_programmer_templates(Path(operators.__file__).resolve().parents[1], "aflow")
    tokenizer_path = os.environ.get("AIME_TOKENIZER_PATH", "") or snapshot_download(
        MODEL, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    context_guard = PromptGuard(tokenizer, context_limit=CONTEXT_LIMIT)

    async def controlled_call(self: Any, prompt: str) -> str:
        messages = []
        if self.sys_msg is not None:
            messages.append({"content": self.sys_msg, "role": "system"})
        messages.append({"role": "user", "content": prompt})
        # AFlow can feed previous operator outputs back into later prompts.
        # Keep the requested 6144 output budget when it fits, but reserve room
        # inside the configured context window for long prompts instead of
        # letting vLLM reject the request and silently skip the sample.
        guarded_messages, guard_event = context_guard.prepare(messages, max_tokens)
        context_limit = CONTEXT_LIMIT
        effective_max_tokens = max(
            1, min(max_tokens, context_limit - int(guard_event["final_input_tokens"]))
        )
        phase = begin_call(telemetry)
        try:
            request = self.aclient.chat.completions.create(
                model=self.config.model,
                messages=guarded_messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=effective_max_tokens,
                extra_body={
                    "audit_metadata": {
                        "run_id": current_run_id(),
                        "method": "aflow",
                        "agent": "task_model",
                        "phase": phase,
                        "context_guard": guard_event,
                    },
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            request_timeout = float(os.environ.get("AIME_AFLOW_REQUEST_TIMEOUT_S", "600"))
            if request_timeout > 0:
                response = await asyncio.wait_for(request, timeout=request_timeout)
            else:
                response = await request
        except Exception:
            failed_call(telemetry, phase)
            raise
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        record_usage(
            telemetry,
            phase,
            input_tokens,
            output_tokens,
            int(getattr(usage, "total_tokens", 0) or 0),
        )
        finish_reason = str(getattr(response.choices[0], "finish_reason", "unknown") or "unknown")
        telemetry["finish_reason_counts"][finish_reason] = telemetry["finish_reason_counts"].get(finish_reason, 0) + 1
        if finish_reason == "length":
            telemetry["completion_truncated"] += 1
        self.usage_tracker.add_usage(self.config.model, input_tokens, output_tokens)
        return response.choices[0].message.content or ""

    # Keep the telemetry object separate from each native AFlow usage tracker;
    # this covers all operator instances in one search/evaluation process.
    AsyncLLM.__call__ = controlled_call

    original_evaluate_all = BaseBenchmark.evaluate_all_problems

    async def bounded_evaluate_all(self: Any, data: list[dict[str, Any]], agent: Any, max_concurrent_tasks: int = 50):
        # The native MATH evaluator retries each problem, while generated
        # AFlow workflows may contain several serial LLM calls.  Bound one
        # whole sample so a pathological generated workflow cannot hold the
        # complete D_select/test pass indefinitely.
        sample_timeout = float(os.environ.get("AIME_AFLOW_SAMPLE_TIMEOUT_S", "1200"))
        semaphore = asyncio.Semaphore(concurrency)

        async def evaluate_one(problem: dict[str, Any]):
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        self.evaluate_problem(problem, agent), timeout=sample_timeout
                    )
                except asyncio.TimeoutError:
                    telemetry["sample_timeouts"] += 1
                    return (
                        problem.get("problem", ""),
                        "AFlow sample timeout",
                        problem.get("solution", ""),
                        0.0,
                        0.0,
                    )

        tasks = [evaluate_one(problem) for problem in data]
        return await tqdm_asyncio.gather(
            *tasks,
            desc=f"Evaluating {self.name} problems",
            total=len(data),
        )

    BaseBenchmark.evaluate_all_problems = bounded_evaluate_all
    telemetry["context_guard"] = context_guard
    return telemetry


def make_aflow_config(endpoint: str) -> Any:
    from scripts.async_llm import LLMConfig

    return LLMConfig(
        {
            "model": "Qwen/Qwen3.5-9B",
            "key": "EMPTY",
            "base_url": endpoint,
            "temperature": 0.0,
            "top_p": 1.0,
        }
    )


def evaluate_aflow_graph(
    *,
    evaluator: Any,
    graph: Any,
    config: Any,
    dataset_path: Path,
    log_dir: Path,
    is_test: bool,
) -> tuple[float, float, float]:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    # The upstream evaluator writes its CSV directly under ``log_dir`` and
    # assumes the caller has created the round-specific directory.
    log_dir.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(
        evaluator.graph_evaluate(
            "MATH",
            graph,
            {"dataset": "MATH", "llm_config": config},
            str(log_dir),
            is_test=is_test,
        )
    )
    return tuple(float(value) for value in result)


def run_aflow(args: argparse.Namespace, search_rows: list[dict[str, Any]], select_rows: list[dict[str, Any]]) -> None:
    source = baseline_source_path(args, "aflow")
    run_started = time.time()
    repo = args.run_dir / "aflow_repo"
    copy_tree(source, repo)
    patch_aflow_code_formatter(repo)
    patch_aflow_optimizer_retry_guard(repo)
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    validate_file = repo / "data/datasets/math_validate.jsonl"
    test_file = repo / "data/datasets/math_test.jsonl"
    write_jsonl(validate_file, native_math_rows(search_rows))
    # Keep the native test file absent until the D_select decision is locked.

    aflow_usage = patch_aflow_runtime(args.max_tokens, args.concurrency)
    from scripts.optimizer import Optimizer
    from scripts.evaluator import Evaluator
    from scripts.optimizer_utils.graph_utils import GraphUtils

    # The official AFlow MATH seed workflow is the initial node of the native
    # MCTS loop.  Each run receives an isolated copy, so no saved graph from a
    # previous experiment can silently become the result.
    relative_root = Path("workspace") / "rpas_aime_native" / "MATH"
    source_seed = repo / "workspace/MATH/workflows/round_1"
    target_seed = repo / relative_root / "workflows/round_1"
    target_seed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_seed, target_seed)
    shutil.copytree(
        repo / "workspace/MATH/workflows/template",
        repo / relative_root / "workflows/template",
        dirs_exist_ok=True,
    )
    ensure_package_tree(repo / relative_root / "workflows", repo)
    # The upstream MATH seed hard-codes the non-isolated ``workspace.MATH``
    # namespace; keep the seed consistent with generated search rounds.
    for filename in ("graph.py", "prompt.py"):
        path = target_seed / filename
        if path.exists():
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace("workspace.MATH", "workspace.rpas_aime_native.MATH"),
                encoding="utf-8",
            )
            prompt_text = path.read_text(encoding="utf-8")
            if re.search(r"(?m)^#\s*[A-Z][A-Z0-9_]*_PROMPT\s*=\s*['\"]{1,3}", prompt_text):
                path.write_text(
                    "\n".join(
                        line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line
                        for line in prompt_text.splitlines()
                    )
                    + "\n",
                    encoding="utf-8",
                )
    patch_aflow_graph_namespace(GraphUtils)

    config = make_aflow_config(args.endpoint)
    seed_everything(args.seed)

    optimizer = Optimizer(
        dataset="MATH",
        question_type="math",
        opt_llm_config=config,
        exec_llm_config=config,
        operators=["Custom", "ScEnsemble", "Programmer"],
        sample=4,
        check_convergence=False,
        optimized_path=str(relative_root.parent),
        initial_round=1,
        max_rounds=args.aflow_max_rounds,
        validation_rounds=1,
    )
    started = time.time()
    with phase_scope("search", aflow_usage["phase_wall_seconds"]):
        optimizer.optimize("Graph")
    search_seconds = time.time() - started

    workflow_root = repo / relative_root / "workflows"
    rounds = aflow_search_rounds(workflow_root, args.aflow_max_rounds)
    if not rounds:
        raise RuntimeError("AFlow native search produced no workflow rounds")
    search_shared_metrics = []
    search_expected_rows = native_math_rows(search_rows)
    for round_number in rounds:
        search_log_dir = workflow_root / f"round_{round_number}"
        validation_errors = validate_aflow_workflow_round(workflow_root, round_number)
        if validation_errors:
            search_shared_metrics.append({
                "round": round_number,
                "status": "invalid",
                "errors": validation_errors,
            })
            continue
        search_shared_metrics.append({
            "round": round_number,
            "status": "evaluated",
            **shared_aime_metrics(search_log_dir, search_expected_rows),
        })

    # Freeze the search output, then use the disjoint selection split to pick
    # the quality operating point.  Selection is not visible during MCTS.
    write_jsonl(validate_file, native_math_rows(select_rows))
    evaluator = Evaluator(eval_path=str(args.run_dir / "aflow_selection"))
    graph_utils = GraphUtils(str(relative_root))
    selection_rows = []
    invalid_rounds = []
    with phase_scope("select", aflow_usage["phase_wall_seconds"]):
        for round_number in rounds:
            log_dir = args.run_dir / "aflow_selection" / f"round_{round_number}"
            validation_errors = validate_aflow_workflow_round(workflow_root, round_number)
            if validation_errors:
                row = {
                    "round": round_number,
                    "status": "invalid",
                    "score": 0.0,
                    "avg_cost": 0.0,
                    "total_cost": 0.0,
                    "errors": validation_errors,
                }
                selection_rows.append(row)
                invalid_rounds.append(row)
                continue
            try:
                graph = graph_utils.load_graph(round_number, str(relative_root / "workflows"))
                usage_before = phase_snapshot(aflow_usage, "select")
                score, avg_cost, total_cost = evaluate_aflow_graph(
                    evaluator=evaluator,
                    graph=graph,
                    config=config,
                    dataset_path=validate_file,
                    log_dir=log_dir,
                    is_test=False,
                )
                usage_after = phase_snapshot(aflow_usage, "select")
                shared_metrics = shared_aime_metrics(log_dir, native_math_rows(select_rows))
                candidate_usage = usage_delta(usage_after, usage_before)
                output_audit = audit_aflow_outputs(log_dir)
                native_avg_cost = avg_cost
                native_total_cost = total_cost
                total_cost = float(candidate_usage["total_tokens"])
                avg_cost = total_cost / max(1, len(select_rows))
                is_valid = output_audit["failure_fraction"] <= MAX_SAMPLE_FAILURE_RATE
                row = {
                    "round": round_number,
                    "status": "evaluated" if is_valid else "invalid",
                    "native_score": score,
                    "score": shared_metrics["score"],
                    "shared_metrics": shared_metrics,
                    "valid": is_valid,
                    "candidate_id": f"round_{round_number}",
                    "total_calls": candidate_usage["calls"],
                    "total_tokens": candidate_usage["total_tokens"],
                    "avg_cost": avg_cost,
                    "total_cost": total_cost,
                    "cost_unit": "model_tokens",
                    "native_avg_cost": native_avg_cost,
                    "native_total_cost": native_total_cost,
                    "output_audit": output_audit,
                }
                if not is_valid:
                    row["errors"] = [
                        f"workflow failure rate {output_audit['failure_fraction']:.1%} "
                        f"exceeds {MAX_SAMPLE_FAILURE_RATE:.1%}"
                    ]
                    invalid_rounds.append(row)
                selection_rows.append(row)
            except Exception as exc:
                row = {
                    "round": round_number,
                    "status": "invalid",
                    "score": 0.0,
                    "avg_cost": 0.0,
                    "total_cost": 0.0,
                    "errors": [f"load/evaluate error: {type(exc).__name__}: {exc}"],
                }
                selection_rows.append(row)
                invalid_rounds.append(row)
    # Persist ALL candidate-level D_select outcomes before any quality-gate
    # exception. This is diagnostic only and never unlocks D_test.
    dump_json(args.run_dir / "selection_audit.json", {
        "schema": "aime_fourway_selection_audit_v1",
        "method": "aflow", "split": "D_select", "dtest_opened": False,
        "candidate_rows": selection_rows, "invalid_rounds": invalid_rounds,
        "failure_limit": MAX_SAMPLE_FAILURE_RATE,
    })
    valid_selection_rows = [row for row in selection_rows if row.get("status") == "evaluated"]
    if not valid_selection_rows:
        raise RuntimeError("AFlow generated no valid workflow candidates for selection")
    selected_q = quality_operating_point(valid_selection_rows)
    selected_e = efficiency_operating_point(valid_selection_rows, delta=0.05)
    if selected_q is None or selected_e is None:
        raise RuntimeError("AFlow could not select both frozen Q/E operating points from D_select")
    selected = next(row for row in valid_selection_rows if row["round"] == selected_q["round"])
    selected_efficiency = next(
        row for row in valid_selection_rows if row["round"] == selected_e["round"]
    )
    for label, candidate in (("Q", selected), ("E", selected_efficiency)):
        if (candidate.get("shared_metrics", {}).get("num_examples") != len(select_rows)
                or float(candidate.get("output_audit", {}).get("failure_fraction", 1.0) or 0.0)
                > MAX_SAMPLE_FAILURE_RATE):
            raise RuntimeError(f"AFlow D_select/{label} quality gate failed; refusing to open D_test")

    freeze_aime_selection(
        args.run_dir, "aflow", search_rows, select_rows,
        {
            "Q": {"round": int(selected["round"]), "score": float(selected["shared_metrics"]["score"])},
            "E": {"round": int(selected_efficiency["round"]),
                  "score": float(selected_efficiency["shared_metrics"]["score"])},
        },
        source_provenance=args.baseline_provenance,
    )

    test_results = {}
    for filename in ("aime_2025.jsonl", "aime_2026.jsonl"):
        test_rows = load_aime_test_rows_after_freeze(
            args.data_dir, args.run_dir, "aflow", filename, args.test_size
        )
        write_jsonl(test_file, native_math_rows(test_rows, Path(filename).stem))
        test_name = Path(filename).stem
        point_rows = {}
        for point, candidate in (("Q", selected), ("E", selected_efficiency)):
            candidate_round = int(candidate["round"])
            if point == "E" and candidate_round == int(selected["round"]):
                point_rows[point] = {**point_rows["Q"], "same_candidate_as": "Q"}
                continue
            test_phase = f"test:{test_name}:{point}"
            log_dir = args.run_dir / "aflow_test" / f"{test_name}_{point}"
            graph = graph_utils.load_graph(candidate_round, str(relative_root / "workflows"))
            with phase_scope(test_phase, aflow_usage["phase_wall_seconds"]):
                native_score, avg_cost, total_cost = evaluate_aflow_graph(
                    evaluator=evaluator,
                    graph=graph,
                    config=config,
                    dataset_path=test_file,
                    log_dir=log_dir,
                    is_test=True,
                )
            shared_metrics = shared_aime_metrics(log_dir, test_rows)
            output_audit = audit_aflow_outputs(log_dir)
            resource_usage = phase_snapshot(aflow_usage, test_phase)
            total_tokens = int(resource_usage["total_tokens"])
            point_rows[point] = {
                **shared_metrics,
                "native_score": native_score,
                "avg_cost": total_tokens / max(1, len(test_rows)),
                "total_cost": total_tokens,
                "cost_unit": "model_tokens",
                "native_avg_cost": avg_cost,
                "native_total_cost": total_cost,
                "phase": test_phase,
                "resource_usage": resource_usage,
                "calls": resource_usage["calls"],
                "prompt_tokens": resource_usage["prompt_tokens"],
                "completion_tokens": resource_usage["completion_tokens"],
                "total_tokens": resource_usage["total_tokens"],
                "failed_calls": resource_usage["failed_calls"],
                "output_audit": output_audit,
                "selected_round": candidate_round,
                "operating_point": point,
            }
        test_results[test_name] = {
            **point_rows["Q"],
            "Q": point_rows["Q"],
            "E": point_rows["E"],
        }

    context_guard = aflow_usage.pop("context_guard", None)

    dump_json(
        args.run_dir / "native_result.json",
        {
            "method": "aflow",
            "search_entrypoint": "AFlow scripts.optimizer.Optimizer.optimize(Graph)",
            "search_seconds": search_seconds,
            "search_rounds": rounds,
            "search_shared_metrics": search_shared_metrics,
            "selection_rows": selection_rows,
            "invalid_rounds": invalid_rounds,
            "selected_round": selected,
            "selected_efficiency_round": selected_efficiency,
            "test": test_results,
            "telemetry": {
                **aflow_usage,
                "context_guard": context_guard.summary() if context_guard is not None else None,
                "phase_telemetry_version": "native_phase_telemetry_v1",
                "wall_time_seconds": time.time() - run_started,
                "gpu_hours": (time.time() - run_started) / 3600.0,
                "gpu_usd_per_hour": float(os.environ["AIME_GPU_USD_PER_HOUR"])
                if os.environ.get("AIME_GPU_USD_PER_HOUR")
                else None,
                "estimated_gpu_cost_usd": (
                    (time.time() - run_started) / 3600.0 * float(os.environ["AIME_GPU_USD_PER_HOUR"])
                    if os.environ.get("AIME_GPU_USD_PER_HOUR")
                    else None
                ),
                "cost_formula": "gpu_hours * gpu_usd_per_hour; null rate means rate not configured",
            },
            "controls": {
                "model": MODEL,
                "endpoint": args.endpoint,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": args.max_tokens,
                "thinking": False,
                "concurrency": args.concurrency,
                "max_model_len": CONTEXT_LIMIT,
                "max_num_seqs": int(os.environ.get("AIME_MAX_NUM_SEQS", "8")),
                "tensor_parallel_size": int(os.environ.get("AIME_TP_SIZE", "1")),
                "sample_timeout_s": float(os.environ.get("AIME_AFLOW_SAMPLE_TIMEOUT_S", "1200")),
                "request_timeout_s": float(os.environ.get("AIME_AFLOW_REQUEST_TIMEOUT_S", "600")),
                "data_seed": args.data_seed,
                "search_seed": args.seed,
                "search_size": len(search_rows),
                "selection_size": len(select_rows),
                "test_size_each": args.test_size,
                "gold_protocol": f"{ANSWER_PARSER}; normalized integer exact match",
                "score_mapping": "2=exact match; 1=parseable wrong; 0=unparseable",
                "context_guard_version": "maas_context_guard_v1",
            },
        },
    )


def patch_maas_runtime(concurrency: int, max_tokens: int) -> dict[str, int]:
    from maas.ext.maas.benchmark.benchmark import BaseBenchmark
    from maas.ext.maas.benchmark.math import MATHBenchmark
    from maas.ext.maas.scripts.optimizer_utils.data_utils import DataUtils
    import maas.ext.maas.scripts.optimized.MATH.train.template.operator as train_operators
    import maas.ext.maas.scripts.optimized.MATH.test.template.operator as test_operators
    from maas.provider.openai_api import OpenAILLM
    from maas_context_guard import PromptGuard

    usage = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "failed_calls": 0,
        "finish_reason_counts": {},
        "completion_truncated": 0,
        "phase_totals": new_phase_totals(),
        "phase_wall_seconds": {},
        "code_execution_timeouts": 0,
        "code_execution_errors": 0,
    }

    # Fail closed on malformed ScEnsemble XML. No guessed candidate letters;
    # a second native request is allowed only after schema parsing fails.
    from maas.actions.action_node import ActionNode
    from maas_xml_guard import install_scensemble_xml_guard
    install_scensemble_xml_guard(ActionNode, usage, max_attempts=2)

    # Load the same local Qwen tokenizer used by the formal bridge.  The guard
    # counts the public output-policy suffix as well, so a request that passes
    # this boundary has deterministic room for model output under 8192.
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizer_path = os.environ.get("AIME_TOKENIZER_PATH", "")
    if not tokenizer_path:
        tokenizer_path = snapshot_download(MODEL, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    context_guard = PromptGuard(tokenizer, context_limit=CONTEXT_LIMIT)

    original_train = BaseBenchmark.evaluate_all_problems
    original_test = BaseBenchmark.evaluate_all_problems_test

    async def bounded_train(self: Any, data: list[dict[str, Any]], graph: Any, max_concurrent_tasks: int = 30, repetitions: int = 4, is_textgrad: bool = False):
        return await original_train(
            self,
            data,
            graph,
            max_concurrent_tasks=concurrency,
            repetitions=repetitions,
            is_textgrad=is_textgrad,
        )

    async def bounded_test(self: Any, data: list[dict[str, Any]], graph: Any, max_concurrent_tasks: int = 10):
        return await original_test(self, data, graph, max_concurrent_tasks=concurrency)

    BaseBenchmark.evaluate_all_problems = bounded_train
    BaseBenchmark.evaluate_all_problems_test = bounded_test

    # The released MaAS MATH benchmark permits 20 retries with a 1500-second
    # per-sample timeout.  That is appropriate for a broad benchmark sweep,
    # but a single malformed generated program would otherwise consume most
    # of the experiment wall time.  Keep native MaAS search/controller logic
    # intact while bounding each sample and retaining a small retry budget.
    sample_timeout = float(os.environ.get("AIME_MAAS_SAMPLE_TIMEOUT_S", "300"))
    sample_retries = max(1, int(os.environ.get("AIME_MAAS_SAMPLE_RETRIES", "3")))

    async def bounded_generate_output(self: Any, graph: Any, input_text: str):
        last_error: Exception | None = None
        for _ in range(sample_retries):
            try:
                return await asyncio.wait_for(graph(input_text), timeout=sample_timeout)
            except Exception as exc:  # framework records exhausted samples as score 0
                last_error = exc
        raise last_error or RuntimeError("MaAS sample failed without an exception")

    MATHBenchmark._generate_output = bounded_generate_output

    # Cover native train/test classes plus all copied workflow templates. Each
    # keeps its own native run_code implementation and operator retry budget.
    code_timeout = float(os.environ.get("AIME_MAAS_CODE_TIMEOUT_S", "60"))
    configure_code_executor(
        "maas", usage, code_timeout,
        int(os.environ.get("AIME_CODE_EXEC_WORKERS", str(min(8, os.cpu_count() or 1)))),
    )
    for module in (train_operators, test_operators):
        install_bounded_programmer(module.Programmer, module.run_code, "maas", module.__name__)
    patch_programmer_templates(Path(train_operators.__file__).resolve().parents[8], "maas")

    original_cons_kwargs = OpenAILLM._cons_kwargs

    def controlled_kwargs(self: Any, messages: list[dict[str, Any]], timeout: int = 0, **extra_kwargs: Any) -> dict[str, Any]:
        requested_max_tokens = max_tokens
        try:
            guarded_messages, guard_event = context_guard.prepare(messages, requested_max_tokens)
        except Exception:
            raise
        kwargs = original_cons_kwargs(self, guarded_messages, timeout=timeout, **extra_kwargs)
        kwargs["temperature"] = 0.0
        kwargs["top_p"] = 1.0
        # Match the bridge's effective budget calculation.  The requested
        # method cap remains 6144; long prompts receive the remaining context
        # room rather than an invalid request or a hidden server truncation.
        effective_max_tokens = max(1, min(max_tokens, CONTEXT_LIMIT - int(guard_event["final_input_tokens"])))
        kwargs["messages"] = guarded_messages
        kwargs["max_tokens"] = effective_max_tokens
        kwargs["extra_body"] = {
            "audit_metadata": {
                "run_id": current_run_id(),
                "method": "maas",
                "agent": "task_model",
                "phase": current_phase(),
                "context_guard": guard_event,
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        return kwargs

    OpenAILLM._cons_kwargs = controlled_kwargs

    # MaAS's native evaluator exposes accuracy and an internal cost manager,
    # but the released local-model path does not persist request telemetry.
    # Keep the native workflow unchanged while recording the usage fields
    # required by the experiment protocol.
    original_completion = OpenAILLM._achat_completion
    original_function_completion = OpenAILLM._achat_completion_function

    def record_finish_reason(response: Any) -> None:
        choices = getattr(response, "choices", None) or []
        finish_reason = str(getattr(choices[0], "finish_reason", "unknown") or "unknown") if choices else "unknown"
        usage["finish_reason_counts"][finish_reason] = usage["finish_reason_counts"].get(finish_reason, 0) + 1
        if finish_reason == "length":
            usage["completion_truncated"] += 1

    async def tracked_completion(self: Any, messages: list[dict[str, Any]], timeout: Any = 0):
        phase = begin_call(usage)
        try:
            response = await original_completion(self, messages, timeout=timeout)
        except Exception:
            failed_call(usage, phase)
            raise
        response_usage = getattr(response, "usage", None)
        if response_usage is not None:
            record_usage(
                usage,
                phase,
                int(getattr(response_usage, "prompt_tokens", 0) or 0),
                int(getattr(response_usage, "completion_tokens", 0) or 0),
                int(getattr(response_usage, "total_tokens", 0) or 0),
            )
        record_finish_reason(response)
        return response

    async def tracked_function_completion(self: Any, messages: list[dict[str, Any]], timeout: Any = 0, **kwargs: Any):
        phase = begin_call(usage)
        try:
            response = await original_function_completion(self, messages, timeout=timeout, **kwargs)
        except Exception:
            failed_call(usage, phase)
            raise
        response_usage = getattr(response, "usage", None)
        if response_usage is not None:
            record_usage(
                usage,
                phase,
                int(getattr(response_usage, "prompt_tokens", 0) or 0),
                int(getattr(response_usage, "completion_tokens", 0) or 0),
                int(getattr(response_usage, "total_tokens", 0) or 0),
            )
        record_finish_reason(response)
        return response

    OpenAILLM._achat_completion = tracked_completion
    OpenAILLM._achat_completion_function = tracked_function_completion

    # The released MaAS optimizer calls this helper with only (round, score),
    # although its helper signature requires additional telemetry arguments.
    # Keep the native algorithm unchanged and make the compatibility boundary
    # explicit so training can finish and save its controller checkpoint.
    def compatible_result_data(self: Any, round_number: int, score: float, avg_cost: float = 0.0, total_cost: float = 0.0, token: float = 0.0) -> dict[str, Any]:
        return {
            "round": round_number,
            "score": score,
            "avg_cost": avg_cost,
            "total_cost": total_cost,
            "token": token,
        }

    DataUtils.create_result_data = compatible_result_data
    usage["_context_guard"] = context_guard
    return usage


def make_maas_config(repo: Path, endpoint: str, max_tokens: int) -> None:
    model_config = {
                "api_type": "openai",
                "model": "Qwen/Qwen3.5-9B",
                "base_url": endpoint,
                "api_key": "EMPTY",
                "max_token": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "timeout": int(os.environ.get("AIME_MAAS_REQUEST_TIMEOUT_S", "180")),
                "stream": False,
                "calc_usage": True,
                "use_system_prompt": True,
    }
    # MaAS imports MetaGPT's global Config before ModelsConfig.  The upstream
    # example only documents the named ``models`` map, but this Config schema
    # also requires a top-level ``llm`` entry.
    config = {
        "llm": dict(model_config),
        "models": {"qwen35_9b": dict(model_config)},
    }
    dump_json(repo / "config/config2.yaml", config)


def patch_maas_optional_provider_import(repo: Path) -> None:
    """Ignore an unused provider whose released dependency API has drifted.

    MaAS imports every provider at package import time.  The installed
    DashScope package no longer exposes the private helper used by the
    upstream DashScope adapter, while this experiment uses only the local
    OpenAI-compatible endpoint.  Keep the provider name available but make
    its optional import non-fatal inside the isolated copy.
    """
    path = repo / "maas/provider/__init__.py"
    text = path.read_text(encoding="utf-8")
    old = "from maas.provider.dashscope_api import DashScopeLLM"
    new = "try:\n    from maas.provider.dashscope_api import DashScopeLLM\nexcept ImportError:\n    DashScopeLLM = None"
    if old not in text:
        if "from maas.provider.openai_api import OpenAILLM" in text:
            return
        raise RuntimeError("MaAS provider import layout changed; refusing an unverified patch")
    lines = text.splitlines()
    dash_index = next(i for i, line in enumerate(lines) if old in line)
    start = dash_index
    while start > 0 and lines[start - 1].strip() == "try:":
        start -= 1
    end = next(
        (i for i in range(dash_index + 1, len(lines)) if lines[i].startswith("from maas.provider.anthropic_api")),
        None,
    )
    if end is None:
        raise RuntimeError("MaAS provider import boundary changed; refusing an unverified patch")
    lines[start:end] = new.splitlines()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def patch_maas_provider_surface(repo: Path) -> None:
    """Expose only the OpenAI provider needed by the native MATH run."""
    path = repo / "maas/provider/__init__.py"
    path.write_text(
        """from maas.provider.openai_api import OpenAILLM

__all__ = [\"OpenAILLM\"]
""",
        encoding="utf-8",
    )


def patch_maas_optional_backoff(repo: Path) -> None:
    """Make the unused TextGrad helper importable without backoff installed."""
    path = repo / "maas/ext/maas/scripts/textgrad/textual_gradient.py"
    text = path.read_text(encoding="utf-8")
    old = "import backoff"
    new = (
        "try:\n"
        "    import backoff\n"
        "except ModuleNotFoundError:\n"
        "    class _BackoffCompat:\n"
        "        expo = object()\n"
        "        @staticmethod\n"
        "        def on_exception(*_args, **_kwargs):\n"
        "            return lambda function: function\n"
        "    backoff = _BackoffCompat()"
    )
    if old not in text:
        if "class _BackoffCompat:" in text:
            return
        raise RuntimeError("MaAS TextGrad import layout changed; refusing an unverified patch")
    lines = text.splitlines()
    backoff_index = next(i for i, line in enumerate(lines) if line.strip() == old)
    start = backoff_index
    while start > 0 and lines[start - 1].strip() == "try:":
        start -= 1
    end = next(
        (i for i in range(backoff_index + 1, len(lines)) if lines[i].strip() == "import openai"),
        None,
    )
    if end is None:
        raise RuntimeError("MaAS TextGrad import boundary changed; refusing an unverified patch")
    lines[start:end] = new.splitlines()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def patch_maas_optional_tools(repo: Path) -> None:
    """Avoid importing unused browser and mail tool providers in the MATH run."""
    path = repo / "maas/tools/__init__.py"
    path.write_text(
        """from enum import Enum


class SearchEngineType(Enum):
    SERPAPI_GOOGLE = \"serpapi\"
    SERPER_GOOGLE = \"serper\"
    DIRECT_GOOGLE = \"google\"
    DUCK_DUCK_GO = \"ddg\"
    CUSTOM_ENGINE = \"custom\"
    BING = \"bing\"


class WebBrowserEngineType(Enum):
    PLAYWRIGHT = \"playwright\"
    SELENIUM = \"selenium\"
    CUSTOM = \"custom\"

    @classmethod
    def __missing__(cls, key):
        return cls.CUSTOM


class SearchInterface:
    async def asearch(self, *args, **kwargs):
        ...
""",
        encoding="utf-8",
    )
    (repo / "maas/actions/__init__.py").write_text(
        """from maas.actions.action import Action
from maas.actions.action_output import ActionOutput

__all__ = [\"Action\", \"ActionOutput\"]
""",
        encoding="utf-8",
    )


def patch_maas_embedding_path(repo: Path) -> None:
    """Use the host-local copy of MaAS's declared MiniLM encoder."""
    path = repo / "maas/ext/maas/models/utils.py"
    text = path.read_text(encoding="utf-8")
    local = os.environ.get(
        "AIME_MAAS_EMBEDDING_PATH",
        str(Path(os.environ.get("AIME_EMBEDDINGS_ROOT", Path(__file__).resolve().parent / "embeddings")) / "all-MiniLM-L6-v2"),
    )
    if "SentenceTransformer(" not in text:
        raise RuntimeError("MaAS embedding model layout changed; refusing an unverified patch")
    lines = text.splitlines()
    if not any("SentenceTransformer(" in line for line in lines):
        raise RuntimeError("MaAS SentenceEncoder layout changed; refusing an unverified patch")
    in_encoder_class = False
    for target, line in enumerate(lines):
        if line.startswith("class SentenceEncoder"):
            in_encoder_class = True
        if "SentenceTransformer(" not in line:
            continue
        indent = line[: len(line) - len(line.lstrip())]
        variable = "self.model" if in_encoder_class else "model"
        lines[target] = f"{indent}{variable} = SentenceTransformer({local!r})"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_maas(args: argparse.Namespace, search_rows: list[dict[str, Any]], select_rows: list[dict[str, Any]]) -> None:
    source = baseline_source_path(args, "maas")
    run_started = time.time()
    repo = args.run_dir / "maas_repo"
    if args.maas_resume:
        if not repo.exists():
            raise RuntimeError(f"cannot resume MaAS; isolated repo does not exist: {repo}")
    else:
        copy_tree(source, repo)
    patch_maas_optional_provider_import(repo)
    patch_maas_provider_surface(repo)
    patch_maas_optional_backoff(repo)
    patch_maas_optional_tools(repo)
    patch_maas_embedding_path(repo)
    make_maas_config(repo, args.endpoint, args.max_tokens)

    data_path = repo / "maas/ext/maas/data"
    write_jsonl(data_path / "math_train.jsonl", native_math_rows(search_rows))
    write_jsonl(data_path / "math_test.jsonl", native_math_rows(select_rows))

    relative_root = Path("workspace") / "rpas_aime_native" / "MATH"
    train_target = repo / relative_root / "train"
    test_target = repo / relative_root / "test"
    train_source = repo / "maas/ext/maas/scripts/optimized/MATH/train"
    test_source = repo / "maas/ext/maas/scripts/optimized/MATH/test"

    # Repair the released prompt literal bug and install the conservative v2
    # wording before either native graph is imported.  This changes prompt
    # text only; controller topology, operator set, and optimizer remain
    # native MaAS.  Keep the manifests so the exact prompt variant is audited.
    from maas_context_guard import patch_template_directory

    prompt_repairs = [
        patch_template_directory(train_source / "template", concise=False),
        patch_template_directory(test_source / "template", concise=False),
    ]
    shutil.copytree(train_source, train_target, dirs_exist_ok=True)
    shutil.copytree(test_source, test_target, dirs_exist_ok=True)
    ensure_package_tree(repo / relative_root, repo)

    os.environ["METAGPT_PROJECT_ROOT"] = str(repo)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    maas_usage = patch_maas_runtime(args.concurrency, args.max_tokens)

    resumed_search_telemetry = None
    if args.maas_resume:
        source_run_id = os.environ.get("MASBENCH_RESUME_FROM_RUN_ID", "")
        if not source_run_id:
            raise RuntimeError("MaAS resume requires MASBENCH_RESUME_FROM_RUN_ID")
        resumed_search_telemetry = bridge_phase_usage(source_run_id, "search")
        if resumed_search_telemetry["calls"] <= 0:
            raise RuntimeError(
                f"cannot resume MaAS; no audited search calls found for {source_run_id}"
            )
        merge_phase_usage(maas_usage, "search", resumed_search_telemetry)
        maas_usage["resumed_phase_telemetry"] = {
            "source_run_id": source_run_id,
            "search": resumed_search_telemetry,
        }

    from maas.configs.models_config import ModelsConfig
    from maas.ext.maas.benchmark.experiment_configs import EXPERIMENT_CONFIGS
    from maas.ext.maas.scripts.optimizer import Optimizer

    config = EXPERIMENT_CONFIGS["MATH"]
    models_config = ModelsConfig.default()
    llm_config = models_config.get("qwen35_9b")
    if llm_config is None:
        raise RuntimeError("MaAS qwen35_9b model config was not loaded")

    seed_everything(args.seed)
    started = time.time()
    optimizer = Optimizer(
        dataset=config.dataset,
        question_type=config.question_type,
        opt_llm_config=llm_config,
        exec_llm_config=llm_config,
        operators=config.operators,
        optimized_path=str(relative_root.parent),
        sample=args.maas_samples,
        round=1,
        batch_size=args.maas_batch_size,
        lr=0.01,
        is_textgrad=False,
    )
    if not args.maas_resume:
        with phase_scope("search", maas_usage["phase_wall_seconds"]):
            optimizer.optimize("Graph")
    search_seconds = time.time() - started if not args.maas_resume else 0.0

    controller_path = repo / relative_root / "train" / "round_1" / f"MATH_controller_sample{args.maas_samples}.pth"
    if not controller_path.exists():
        raise RuntimeError(f"MaAS native search did not save controller: {controller_path}")

    # MaAS has one native learned controller checkpoint per search round.  The
    # disjoint D_select score freezes that checkpoint before either test year.
    # Reuse one event loop for all three native test passes; repeatedly calling
    # asyncio.run closes the OpenAI transport underneath MaAS and emits a
    # non-fatal but confusing "Event loop is closed" cleanup traceback.
    test_loop = asyncio.new_event_loop()
    try:
        with phase_scope("select", maas_usage["phase_wall_seconds"]):
            native_selection_score = float(test_loop.run_until_complete(optimizer.test()))
        maas_eval_dir = repo / relative_root / "test" / "round_1"
        selection_metrics = shared_aime_metrics(maas_eval_dir, native_math_rows(select_rows))
        selection_score = float(selection_metrics["score"])
        selection_audit = audit_aflow_outputs(maas_eval_dir)
        selection_usage = phase_snapshot(maas_usage, "select")
        selection_failure_fraction = float(selection_audit.get("failure_fraction", 1.0) or 0.0)
        # Failures previously discarded this entire costly native select run.
        # Keep its unchanged exact-match score, parser audit and cost on disk,
        # while preserving the hard 5% gate and leaving D_test unopened.
        dump_json(args.run_dir / "selection_audit.json", {
            "schema": "aime_fourway_selection_audit_v1", "method": "maas",
            "split": "D_select", "dtest_opened": False,
            "controller_sha256": sha256_file(controller_path),
            "selection_metrics": selection_metrics, "output_audit": selection_audit,
            "resource_usage": selection_usage,
            "failure_limit": MAX_SAMPLE_FAILURE_RATE,
        })
        if (selection_metrics.get("num_examples") != len(select_rows)
                or selection_failure_fraction > MAX_SAMPLE_FAILURE_RATE):
            raise RuntimeError(
                "MaAS D_select quality gate failed; refusing to open D_test. "
                f"rows={selection_metrics.get('num_examples')}, failures={selection_failure_fraction:.1%}"
            )
        freeze_aime_selection(
            args.run_dir, "maas", search_rows, select_rows,
            {"controller_sha256": sha256_file(controller_path),
             "selection_score": selection_score, "num_examples": len(select_rows)},
            source_provenance=args.baseline_provenance,
        )
        test_results = {}
        for filename in ("aime_2025.jsonl", "aime_2026.jsonl"):
            expected_test_rows = load_aime_test_rows_after_freeze(
                args.data_dir, args.run_dir, "maas", filename, args.test_size
            )
            write_jsonl(data_path / "math_test.jsonl", native_math_rows(expected_test_rows, Path(filename).stem))
            test_name = Path(filename).stem
            test_phase = f"test:{test_name}"
            with phase_scope(test_phase, maas_usage["phase_wall_seconds"]):
                native_score = float(test_loop.run_until_complete(optimizer.test()))
            shared_metrics = shared_aime_metrics(maas_eval_dir, expected_test_rows)
            output_audit = audit_aflow_outputs(maas_eval_dir)
            resource_usage = phase_snapshot(maas_usage, test_phase)
            total_tokens = int(resource_usage["total_tokens"])
            point = {
                **shared_metrics,
                "native_score": native_score,
                "num_examples": args.test_size,
                "avg_cost": total_tokens / max(1, args.test_size),
                "total_cost": total_tokens,
                "cost_unit": "model_tokens",
                "phase": test_phase,
                "resource_usage": resource_usage,
                "calls": resource_usage["calls"],
                "prompt_tokens": resource_usage["prompt_tokens"],
                "completion_tokens": resource_usage["completion_tokens"],
                "total_tokens": resource_usage["total_tokens"],
                "failed_calls": resource_usage["failed_calls"],
                "output_audit": output_audit,
                "selected_controller": str(controller_path),
                "operating_point": "Q",
            }
            test_results[test_name] = {
                **point,
                "Q": point,
                "E": {**point, "same_candidate_as": "Q"},
            }
    finally:
        test_loop.run_until_complete(test_loop.shutdown_asyncgens())
        test_loop.close()

    wall_time_seconds = time.time() - run_started
    gpu_hours = wall_time_seconds / 3600.0
    gpu_rate = os.environ.get("AIME_GPU_USD_PER_HOUR", "")
    estimated_gpu_cost = gpu_hours * float(gpu_rate) if gpu_rate else None
    context_guard = maas_usage.pop("_context_guard", None)

    dump_json(
        args.run_dir / "native_result.json",
        {
            "method": "maas",
            "search_entrypoint": "MaAS examples.maas.optimize.Optimizer.optimize(Graph)",
            "search_seconds": search_seconds,
            "telemetry": {
                **maas_usage,
                "context_guard": context_guard.summary() if context_guard is not None else None,
                "phase_telemetry_version": "native_phase_telemetry_v1",
                "wall_time_seconds": wall_time_seconds,
                "gpu_hours": gpu_hours,
                "gpu_usd_per_hour": float(gpu_rate) if gpu_rate else None,
                "estimated_gpu_cost_usd": estimated_gpu_cost,
                "cost_formula": "gpu_hours * gpu_usd_per_hour; null rate means rate not configured",
            },
            "controller_checkpoint": str(controller_path),
            "selection": {
                "score": selection_score,
                "native_score": native_selection_score,
                **selection_metrics,
                "split": "D_select",
                "num_examples": len(select_rows),
                "resource_usage": selection_usage,
                "output_audit": selection_audit,
            },
            "test": test_results,
            "controls": {
                "model": MODEL,
                "endpoint": args.endpoint,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": args.max_tokens,
                "thinking": False,
                "concurrency": args.concurrency,
                "max_model_len": CONTEXT_LIMIT,
                "max_num_seqs": int(os.environ.get("AIME_MAX_NUM_SEQS", "8")),
                "tensor_parallel_size": int(os.environ.get("AIME_TP_SIZE", "1")),
                "sample_timeout_s": float(os.environ.get("AIME_MAAS_SAMPLE_TIMEOUT_S", "300")),
                "request_timeout_s": int(os.environ.get("AIME_MAAS_REQUEST_TIMEOUT_S", "180")),
                "code_timeout_s": float(os.environ.get("AIME_MAAS_CODE_TIMEOUT_S", "60")),
                "sample_retries": int(os.environ.get("AIME_MAAS_SAMPLE_RETRIES", "3")),
                "data_seed": args.data_seed,
                "search_seed": args.seed,
                "search_size": len(search_rows),
                "selection_size": len(select_rows),
                "test_size_each": 30,
                "sample_repetitions": args.maas_samples,
                "batch_size": args.maas_batch_size,
                "resumed_controller": args.maas_resume,
                "resumed_search_run_id": (
                    resumed_search_telemetry["source_run_id"]
                    if resumed_search_telemetry is not None
                    else None
                ),
                "prompt_repairs": prompt_repairs,
                "context_guard_version": "maas_context_guard_v1",
                "gold_protocol": f"{ANSWER_PARSER}; normalized integer exact match",
                "score_mapping": "2=exact match; 1=parseable wrong; 0=unparseable",
            },
        },
    )


def main() -> None:
    args = parse_args()
    frozen = {
        "data_seed": (args.data_seed, 2026),
        "search_size": (args.search_size, 60),
        "selection_size": (args.selection_size, 30),
        "test_size": (args.test_size, 30),
        "max_tokens": (args.max_tokens, OUTPUT_LIMIT),
        "concurrency": (args.concurrency, 8),
    }
    mismatches = [f"{name}={actual} (required {expected})" for name, (actual, expected) in frozen.items() if actual != expected]
    if mismatches:
        raise ValueError("refusing non-protocol AIME run: " + "; ".join(mismatches))
    if args.maas_resume and args.method != "maas":
        raise ValueError("--maas-resume is valid only for MaAS")
    # The SLURM launcher passes repository-relative output paths.  Resolve them
    # before either native baseline changes cwd; otherwise imports and final
    # artifacts resolve under a duplicated nested output path.
    args.run_dir = args.run_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.baseline_provenance = verify_pinned_upstream(
        args.method, baseline_source_path(args, args.method)
    )
    if args.run_dir.exists():
        if not args.maas_resume:
            raise FileExistsError(f"refusing to overwrite existing run artifacts: {args.run_dir}")
        if (args.run_dir / "selection_frozen.json").exists() or (args.run_dir / "dtest_access_manifest.json").exists():
            raise RuntimeError("cannot resume after the D_test lock/access phase; use a fresh run directory")
        prior_path = args.run_dir / "run_manifest.json"
        if not prior_path.is_file():
            raise RuntimeError("MaAS resume requires an existing run_manifest.json")
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior.get("method") != "maas" or prior.get("search_seed") != args.seed:
            raise RuntimeError("MaAS resume manifest does not match the requested method/seed")
    else:
        args.run_dir.mkdir(parents=True)
    seed_everything(args.seed)
    search_rows, select_rows = freeze_split(
        args.data_dir, args.data_seed, args.search_size, args.selection_size
    )
    search_rows = namespace_aime_rows(search_rows, "validation")
    select_rows = namespace_aime_rows(select_rows, "validation")
    validation_ids = [str(row["id"]) for row in search_rows + select_rows]
    if len(set(validation_ids)) != 90:
        raise ValueError("the frozen AIME search/selection pool must have 90 unique IDs")
    verify_served_model(args.endpoint)
    frozen_data_manifest = load_frozen_aime_manifest(args.data_dir)
    validation_hashes = {
        frozen_data_manifest["validation"][key]["path"]:
            frozen_data_manifest["validation"][key]["sha256_git_content"]
        for key in ("source", "search", "select")
    }
    validation_hashes["aimo-validation-aime.jsonl"] = validation_hashes.pop(
        frozen_data_manifest["validation"]["source"]["path"]
    )
    split_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "task_contract": task_contract_manifest(),
        "method": args.method,
        "frozen_data_manifest_sha256": FROZEN_AIME_MANIFEST_SHA256,
        "upstream_provenance": args.baseline_provenance,
        "search_seed": args.seed,
        "data_seed": args.data_seed,
        "data_dir": str(args.data_dir.resolve()),
        "search_size": len(search_rows),
        "selection_size": len(select_rows),
        "search_ids": [str(row.get("id", "")) for row in search_rows],
        "selection_ids": [str(row.get("id", "")) for row in select_rows],
        "model": MODEL,
        "context_limit": CONTEXT_LIMIT,
        "output_limit": OUTPUT_LIMIT,
        "answer_protocol": ANSWER_PARSER,
        "answer_protocol_sha256": sha256_file(Path(__file__).with_name("answer_protocol.py")),
        # D_test IDs and hashes are intentionally absent until after selection lock.
        "data_sha256": validation_hashes,
    }
    run_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "task_contract": task_contract_manifest(),
        "method": args.method,
        "upstream_provenance": args.baseline_provenance,
        "search_seed": args.seed,
        "data_seed": args.data_seed,
        "model": MODEL,
        "context_limit": CONTEXT_LIMIT,
        "output_limit": OUTPUT_LIMIT,
        "answer_protocol": ANSWER_PARSER,
        "answer_protocol_sha256": split_manifest["answer_protocol_sha256"],
        "data_sha256": dict(split_manifest["data_sha256"]),
        "search_size": len(search_rows),
        "selection_size": len(select_rows),
        "test_size_each": args.test_size,
        "endpoint": args.endpoint,
        "run_id": current_run_id(),
        "attempt": os.environ.get("MASBENCH_ATTEMPT", "main"),
        "status": "running",
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    dump_json(args.run_dir / "split_manifest.json", split_manifest)
    dump_json(args.run_dir / "run_manifest.json", run_manifest)
    try:
        if args.method == "aflow":
            run_aflow(args, search_rows, select_rows)
        else:
            run_maas(args, search_rows, select_rows)
        access_path = args.run_dir / "dtest_access_manifest.json"
        lock_path = args.run_dir / "selection_frozen.json"
        if not access_path.is_file() or not lock_path.is_file():
            raise RuntimeError("AIME method finished without a selection lock and D_test access audit")
        access = json.loads(access_path.read_text(encoding="utf-8"))
        if set(access.get("test_splits", {})) != {"aime_2025.jsonl", "aime_2026.jsonl"}:
            raise RuntimeError("AIME run did not open exactly the two frozen test files")
        search_ids = set(split_manifest["search_ids"])
        select_ids = set(split_manifest["selection_ids"])
        test_ids_by_name = {}
        for filename, record in access["test_splits"].items():
            ids = record.get("ids", [])
            if len(ids) != args.test_size or len(set(ids)) != args.test_size:
                raise RuntimeError(f"invalid D_test IDs/count in access audit for {filename}")
            if set(ids) & (search_ids | select_ids):
                raise RuntimeError(f"D_test overlaps a validation split: {filename}")
            if record.get("opened_at_epoch", 0) < access.get("selection_frozen_at_epoch", float("inf")):
                raise RuntimeError(f"D_test was opened before candidate freeze: {filename}")
            test_ids_by_name[Path(filename).stem] = ids
            split_manifest["data_sha256"][filename] = record["sha256"]
            split_manifest["data_sha256"][record["frozen_split_path"]] = record["frozen_split_sha256"]
        split_manifest["frozen_data_manifest_sha256"] = FROZEN_AIME_MANIFEST_SHA256
        split_manifest["test_ids"] = test_ids_by_name
        split_manifest["selection_lock_sha256"] = sha256_file(lock_path)
        split_manifest["dtest_access_manifest_sha256"] = sha256_file(access_path)
        run_manifest["data_sha256"] = dict(split_manifest["data_sha256"])
        run_manifest["frozen_data_manifest_sha256"] = FROZEN_AIME_MANIFEST_SHA256
        run_manifest["selection_lock_sha256"] = split_manifest["selection_lock_sha256"]
        run_manifest["dtest_access_manifest_sha256"] = split_manifest["dtest_access_manifest_sha256"]
        run_manifest["status"] = "results_ready"
        dump_json(args.run_dir / "split_manifest.json", split_manifest)
        dump_json(args.run_dir / "run_manifest.json", run_manifest)
    except BaseException as exc:
        run_manifest["status"] = "failed"
        run_manifest["failure"] = f"{type(exc).__name__}: {exc}"
        dump_json(args.run_dir / "run_manifest.json", run_manifest)
        raise


if __name__ == "__main__":
    main()
