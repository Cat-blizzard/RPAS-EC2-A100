#!/usr/bin/env python3
"""Protocol adapters for the unmodified ADAS and G-Designer baselines.

The upstream search/graph implementations are imported from clean, pinned
repositories. This file only adapts dataset rows, model transport, answer
scoring, and audited run artifacts. It supports AIME and all five repository-
frozen MASBench axes; it does not submit jobs itself.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

if __package__:
    from . import answer_protocol as aime_answer_protocol
    from .answer_protocol import extract_aime_answer, score_aime
    from .masbench_answer_protocol import score_masbench
else:
    import answer_protocol as aime_answer_protocol
    from answer_protocol import extract_aime_answer, score_aime
    from masbench_answer_protocol import score_masbench

MODEL = "Qwen/Qwen3.5-9B"
CONTEXT_LIMIT = 16384
OUTPUT_LIMIT = 6144
CONCURRENCY = 8
MAX_SAMPLE_FAILURE_RATE = 0.05
METHODS = ("adas", "gdesigner")
MASBENCH_AXES = ("breadth", "depth", "horizon", "parallel", "robustness")
MASBENCH_PROTOCOL = "masbench_external_five_axis_v1"
AIME_PROTOCOL = "aime_external_methods_v5_16k_shared_task_input"
UPSTREAM_COMMITS = {
    "adas": "2702bee8fefda42255efc5be9f60e3bd3db96ae4",
    "gdesigner": "a6efcfa3b40bb4d9cbf46f883a95d62020bd8251",
}
UPSTREAM_REPOSITORIES = {
    "adas": "https://github.com/ShengranHu/ADAS.git",
    "gdesigner": "https://github.com/yanweiyue/GDesigner.git",
}


def _canonical_repo_url(value: str) -> str:
    value = value.strip()
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    elif value.startswith("ssh://git@github.com/"):
        value = "https://github.com/" + value.removeprefix("ssh://git@github.com/")
    return value.removesuffix(".git").rstrip("/").lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*.py"))
        if ".git" not in path.parts and "__pycache__" not in path.parts
    }


def verify_upstream_source(method: str, root: Path) -> dict[str, str]:
    """Require the audited upstream revision and a clean checkout before any run."""
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(root), *args], check=True,
                                capture_output=True, text=True)
        return result.stdout.strip()
    commit = git("rev-parse", "HEAD")
    origin = git("remote", "get-url", "origin")
    dirty = git("status", "--porcelain")
    if _canonical_repo_url(origin) != _canonical_repo_url(UPSTREAM_REPOSITORIES[method]):
        raise RuntimeError(f"{method} source origin is not the audited upstream: {origin}")
    if commit != UPSTREAM_COMMITS[method]:
        raise RuntimeError(f"{method} source must be pinned to {UPSTREAM_COMMITS[method]}, got {commit}")
    if dirty:
        raise RuntimeError(f"{method} upstream source checkout is dirty; refusing an unaudited run")
    return {"commit": commit, "clean_worktree": True, "origin": origin,
            "repository": UPSTREAM_REPOSITORIES[method]}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_masbench_splits(rows: dict[str, list[dict[str, Any]]], manifest: dict[str, Any], axis: str) -> None:
    """Fail closed against RPAS's five-axis, 24/24/60 frozen split manifests."""
    if axis not in MASBENCH_AXES:
        raise ValueError(f"unknown MASBench axis {axis!r}; expected one of {MASBENCH_AXES}")
    counts = {"search": 24, "select": 24, "test": 60}
    files = {"search": f"{axis}/search_24.jsonl", "select": f"{axis}/select_24.jsonl",
             "test": f"{axis}/test_60.jsonl"}
    if manifest.get("schema") != "rpas_frozen_splits_v1" or manifest.get("group") != axis:
        raise ValueError(f"unexpected frozen MASBench manifest for {axis}")
    if int(manifest.get("seed", -1)) != 2026:
        raise ValueError("MASBench data seed must remain fixed at 2026 across search seeds")
    if manifest.get("counts") != counts or manifest.get("files") != files:
        raise ValueError(f"{axis} manifest must freeze D_search/D_select/D_test at 24/24/60")
    source = manifest.get("source", {})
    if source.get("dataset") != "Salesforce/MASBench" or source.get("axis") != axis:
        raise ValueError(f"{axis} manifest does not identify the official MASBench source")
    ids_by_split: dict[str, set[str]] = {}
    for split, expected_n in counts.items():
        current = rows[split]
        ids = [str(row.get("id", "")) for row in current]
        if len(current) != expected_n or any(not item for item in ids) or len(set(ids)) != expected_n:
            raise ValueError(f"{axis}/{split} must have {expected_n} unique nonempty row IDs")
        official_split = "test" if split == "test" else "train"
        for row in current:
            if row.get("dataset") != "masbench" or row.get("axis") != axis:
                raise ValueError(f"row {row.get('id')} has mismatched dataset/axis metadata")
            if row.get("official_split") != official_split:
                raise ValueError(f"row {row.get('id')} is from the wrong official split")
            if not str(row.get("input", "")).strip() or not str(row.get("answer", "")).strip():
                raise ValueError(f"row {row.get('id')} is missing its task or gold answer")
        ids_by_split[split] = set(ids)
        # Keep the repository's identity digests as provenance. They are
        # produced by a separate exporter and are not reproducible from the
        # committed JSONL serialization; raw file SHA-256 is recorded below.
        for digest_field in ("id_sha256", "content_sha256"):
            digest = str(manifest.get(digest_field, {}).get(split, ""))
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
                raise ValueError(f"{axis}/{split} is missing its frozen {digest_field}")
    if (ids_by_split["search"] & ids_by_split["select"] or
            ids_by_split["search"] & ids_by_split["test"] or
            ids_by_split["select"] & ids_by_split["test"]):
        raise ValueError("MASBench D_search, D_select, and D_test must be pairwise disjoint")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


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


def read_task_rows(dataset: str, data_dir: Path, seed: int, axis: str | None) -> dict[str, list[dict[str, Any]]]:
    if dataset == "masbench":
        if axis not in MASBENCH_AXES:
            raise ValueError(f"--axis must be one of {MASBENCH_AXES} for MASBench")
        axis_dir = data_dir / axis
        rows = {"search": read_jsonl(axis_dir / "search_24.jsonl"),
                "select": read_jsonl(axis_dir / "select_24.jsonl"),
                "test": read_jsonl(axis_dir / "test_60.jsonl")}
        manifest = json.loads((axis_dir / "manifest.json").read_text(encoding="utf-8"))
        validate_masbench_splits(rows, manifest, axis)
        return rows

    if dataset == "aime":
        from native_aime_formal import (
            freeze_split, namespace_aime_rows, read_jsonl as read_aime_jsonl,
            validate_aime_partitions,
        )
        # Freeze data partitions across search seeds; only the optimizer seed changes.
        search, select = freeze_split(data_dir, 2026, 60, 30)
        search = namespace_aime_rows(search, "validation")
        select = namespace_aime_rows(select, "validation")
        validate_aime_partitions({"D_search": search, "D_select": select})
        # The AIME test files are deliberately not opened here. Each method
        # loads them only after its D_select candidate freeze is durable.
        return {"search": search, "select": select}
    raise ValueError(f"unsupported dataset: {dataset}")


def task_text(dataset: str, row: dict[str, Any]) -> str:
    if dataset == "masbench":
        from phase2_wan_agent_search import MASBENCH_INSTRUCTION
        instruction = MASBENCH_INSTRUCTION
        if row.get("axis") == "robustness":
            instruction += ("For explicitly injected factual or magic-number values, preserve the value exactly as stated; "
                            "do not reduce those raw values modulo 23. Apply Z_23 only to algebraic quantities.\n")
        return f"{instruction}\nProblem: {row['input']}"
    # Same user-facing task text as AFlow/MaAS, not a stronger
    # ADAS/G-Designer-only instruction. Native optimizers remain unchanged.
    from native_aime_formal import format_aime_task
    return format_aime_task(str(row["problem"]))


def score_row(dataset: str, output: Any, gold: Any) -> dict[str, Any]:
    if dataset == "masbench":
        return score_masbench(output, gold)
    exact = score_aime(output, gold)
    exact["component_accuracy"] = 1.0 if exact["correct"] else 0.0
    exact["component_correct"] = int(bool(exact["correct"]))
    exact["component_total"] = 1
    return exact


def empty_usage() -> dict[str, int]:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "failed_calls": 0,
            "completion_truncated": 0}


class ModelRuntime:
    """One audited OpenAI-compatible client with an enforced 8192-token window."""

    def __init__(self, endpoint: str, method: str, run_id: str, *, temperature: float = 0.0):
        try:
            request_timeout_s = float(os.environ.get("AIME_EXTERNAL_REQUEST_TIMEOUT_S", "180"))
        except ValueError as exc:
            raise ValueError("AIME_EXTERNAL_REQUEST_TIMEOUT_S must be a positive finite number") from exc
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError("AIME_EXTERNAL_REQUEST_TIMEOUT_S must be a positive finite number")
        self.request_timeout_s = request_timeout_s

        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from maas_context_guard import PromptGuard
        from openai import OpenAI, AsyncOpenAI

        tokenizer_path = os.environ.get("AIME_TOKENIZER_PATH", "") or snapshot_download(MODEL, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.guard = PromptGuard(tokenizer, context_limit=CONTEXT_LIMIT)
        self.endpoint, self.method, self.run_id = endpoint.rstrip("/"), method, run_id
        self.sync = OpenAI(base_url=self.endpoint, api_key="EMPTY", timeout=self.request_timeout_s, max_retries=0)
        self.async_client = AsyncOpenAI(base_url=self.endpoint, api_key="EMPTY", timeout=self.request_timeout_s, max_retries=0)
        self.temperature = temperature
        self.usage: dict[str, Any] = {"totals": empty_usage(), "phases": {}, "finish_reason_counts": {},
                                     "guard_events": 0, "context_limit": CONTEXT_LIMIT}
        self.lock = threading.Lock()
        self.async_semaphore: asyncio.Semaphore | None = None

    def _prepare(self, messages: list[dict[str, str]], max_tokens: int) -> tuple[list[dict[str, str]], dict[str, Any], int]:
        guarded, event = self.guard.prepare(messages, max_tokens)
        effective = max(1, min(max_tokens, CONTEXT_LIMIT - int(event["final_input_tokens"])))
        return guarded, event, effective

    def _record(self, phase: str, response: Any, error: bool = False) -> None:
        usage = getattr(response, "usage", None) if response is not None else None
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
        choices = getattr(response, "choices", []) or []
        finish = str(getattr(choices[0], "finish_reason", "unknown") or "unknown") if choices else "unknown"
        with self.lock:
            target = self.usage["totals"]
            part = self.usage["phases"].setdefault(phase, empty_usage())
            for bucket in (target, part):
                bucket["calls"] += 1
                bucket["prompt_tokens"] += prompt
                bucket["completion_tokens"] += completion
                bucket["total_tokens"] += total
                bucket["failed_calls"] += int(error)
                bucket["completion_truncated"] += int(finish == "length")
            self.usage["finish_reason_counts"][finish] = self.usage["finish_reason_counts"].get(finish, 0) + 1

    def _messages_call(self, messages: list[dict[str, str]], *, temperature: float, json_mode: bool, phase: str) -> str:
        guarded, event, effective = self._prepare(messages, OUTPUT_LIMIT)
        with self.lock:
            self.usage["guard_events"] += 1
        try:
            kwargs: dict[str, Any] = {
                "model": MODEL, "messages": guarded, "temperature": temperature, "top_p": 1.0,
                "max_tokens": effective,
                "extra_body": {"audit_metadata": {"run_id": self.run_id, "method": self.method,
                    "agent": "task_model", "phase": phase, "context_guard": event},
                    "chat_template_kwargs": {"enable_thinking": False}},
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = self.sync.chat.completions.create(**kwargs)
        except Exception:
            self._record(phase, None, error=True)
            raise
        self._record(phase, response)
        return response.choices[0].message.content or ""

    async def call_async(self, messages: list[dict[str, str]], *, temperature: float, phase: str) -> str:
        guarded, event, effective = self._prepare(messages, OUTPUT_LIMIT)
        with self.lock:
            self.usage["guard_events"] += 1
        if self.async_semaphore is None:
            self.async_semaphore = asyncio.Semaphore(CONCURRENCY)
        try:
            async with self.async_semaphore:
                response = await self.async_client.chat.completions.create(
                    model=MODEL, messages=guarded, temperature=temperature, top_p=1.0,
                    max_tokens=effective,
                    extra_body={"audit_metadata": {"run_id": self.run_id, "method": self.method,
                        "agent": "task_model", "phase": phase, "context_guard": event},
                        "chat_template_kwargs": {"enable_thinking": False}},
                )
        except Exception:
            self._record(phase, None, error=True)
            raise
        self._record(phase, response)
        return response.choices[0].message.content or ""

    def summary(self) -> dict[str, Any]:
        return {**self.usage, "context_limit": CONTEXT_LIMIT, "output_limit": OUTPUT_LIMIT,
                "max_concurrency": CONCURRENCY, "request_timeout_s": self.request_timeout_s}


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Some compatible servers still wrap JSON in a code fence.
        left, right = content.find("{"), content.rfind("}")
        if left < 0 or right <= left:
            raise
        result = json.loads(content[left:right + 1])
    if not isinstance(result, dict):
        raise ValueError("model response must be a JSON object")
    return result


class UnsafeCandidate(ValueError):
    pass


_SAFE_AST_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign,
    ast.If, ast.For, ast.Expr, ast.Call, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.Dict, ast.List, ast.Tuple, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.comprehension, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Subscript, ast.Slice, ast.Attribute, ast.keyword, ast.JoinedStr, ast.FormattedValue,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.ImportFrom, ast.alias, ast.Pass,
    ast.Break, ast.Continue, ast.Lambda,
)
_SAFE_BUILTINS = {"range", "len", "str", "int", "float", "min", "max", "sum", "enumerate",
                  "zip", "list", "tuple", "set", "sorted", "any", "all", "abs", "bool", "dict",
                  "Counter", "LLMAgentBase"}
_SAFE_ATTRIBUTES = {"content", "most_common", "append", "extend", "strip", "split", "join", "lower", "upper"}


class CandidateGuard(ast.NodeVisitor):
    def __init__(self) -> None:
        self.functions: list[ast.FunctionDef] = []
        self.defined_names: set[str] = set()
        self.callable_names: set[str] = set(_SAFE_BUILTINS)
        self.agent_list_names: set[str] = set()

    @staticmethod
    def _target_names(target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return set().union(*(CandidateGuard._target_names(item) for item in target.elts))
        return set()

    @staticmethod
    def _is_agent_constructor(node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "LLMAgentBase")

    @classmethod
    def _is_agent_collection(cls, node: ast.AST) -> bool:
        if isinstance(node, ast.ListComp):
            return cls._is_agent_constructor(node.elt)
        return isinstance(node, ast.List) and bool(node.elts) and all(
            cls._is_agent_constructor(item) for item in node.elts
        )

    def configure_call_targets(self, tree: ast.AST) -> None:
        """Permit only safe builtins, local helpers, constructed agents, and agent-list dispatch."""
        function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        assignments: dict[str, list[str]] = {}
        loop_targets: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                safe_kind = None
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    if self._is_agent_constructor(node.value):
                        safe_kind = "agent"
                    elif self._is_agent_collection(node.value):
                        safe_kind = "agent_list"
                for target in node.targets:
                    for name in self._target_names(target):
                        assignments.setdefault(name, []).append(safe_kind or "other")
            elif isinstance(node, ast.AugAssign):
                for name in self._target_names(node.target):
                    assignments.setdefault(name, []).append("other")
            elif isinstance(node, (ast.For, ast.comprehension)):
                loop_targets.update(self._target_names(node.target))

        self.callable_names.update(function_names)
        self.callable_names.update(name for name, kinds in assignments.items()
                                   if "agent" in kinds and set(kinds) == {"agent"})
        self.agent_list_names.update(name for name, kinds in assignments.items()
                                     if "agent_list" in kinds and set(kinds) == {"agent_list"})
        # A loop/comprehension target can rebind a previously safe call target.
        self.callable_names.difference_update(loop_targets)
        self.agent_list_names.difference_update(loop_targets)

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _SAFE_AST_NODES):
            raise UnsafeCandidate(f"forbidden syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level != 0 or node.module != "collections" or [alias.name for alias in node.names] != ["Counter"]:
            raise UnsafeCandidate("only `from collections import Counter` is allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or node.attr not in _SAFE_ATTRIBUTES:
            raise UnsafeCandidate(f"forbidden attribute: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_") and node.id != "_":
            raise UnsafeCandidate(f"private or introspection name is forbidden: {node.id}")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id not in self.callable_names:
                raise UnsafeCandidate(f"call target is not allowlisted: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr not in _SAFE_ATTRIBUTES:
                raise UnsafeCandidate(f"forbidden method call: {node.func.attr}")
        elif (isinstance(node.func, ast.Subscript) and isinstance(node.func.value, ast.Name)
              and node.func.value.id in self.agent_list_names
              and isinstance(node.func.slice, (ast.Name, ast.Constant))
              and not (isinstance(node.func.slice, ast.Constant)
                       and not isinstance(node.func.slice.value, int))):
            pass
        else:
            raise UnsafeCandidate("only direct calls to allowlisted functions/methods are allowed")
        self.generic_visit(node)


def compile_candidate(code: str, llm_agent_base: Any) -> Any:
    stripped = code.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```python").removeprefix("```py").removeprefix("```")
        stripped = stripped.removesuffix("```").strip()
    try:
        tree = ast.parse(stripped)
    except SyntaxError as exc:
        raise UnsafeCandidate(f"candidate syntax error: {exc}") from exc
    guard = CandidateGuard()
    guard.functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    guard.defined_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    guard.configure_call_targets(tree)
    guard.generic_visit(tree)
    if len(guard.functions) != 1 or guard.functions[0].name != "forward":
        raise UnsafeCandidate("candidate must define exactly one top-level forward(self, taskInfo)")
    for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
        for call in (node for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            if call.func.id == function.name:
                raise UnsafeCandidate("recursive candidate functions are forbidden")
    # Imports are removed after validating the one benign Counter import; no
    # import function is exposed in the execution namespace.
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)]
    ast.fix_missing_locations(tree)
    safe_builtins = {name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
                     for name in ("abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
                                  "list", "max", "min", "set", "sorted", "str", "sum", "tuple", "zip")}
    safe_builtins["range"] = _bounded_range
    namespace = {"__builtins__": safe_builtins, "Counter": Counter, "LLMAgentBase": llm_agent_base}
    exec(compile(tree, "<ADAS candidate>", "exec"), namespace, namespace)
    return namespace["forward"]


_candidate_budget = threading.local()


def consume_candidate_agent_call_budget(limit: int = 24) -> None:
    calls_used = int(getattr(_candidate_budget, "agent_calls", 0)) + 1
    if calls_used > limit:
        raise UnsafeCandidate(f"candidate exceeded its {limit}-call per-sample agent budget")
    _candidate_budget.agent_calls = calls_used


def _bounded_range(*args: int):
    values = range(*args)
    if len(values) > 12:
        raise UnsafeCandidate("candidate loop exceeds 12 iterations")
    used = getattr(_candidate_budget, "iterations", 0)
    def generate():
        for value in values:
            used_now = getattr(_candidate_budget, "iterations", 0) + 1
            _candidate_budget.iterations = used_now
            if used_now > 120:
                raise UnsafeCandidate("candidate exceeded its per-sample iteration budget")
            yield value
    return generate()


def _native_source(method: str, explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    name = "ADAS" if method == "adas" else "GDesigner"
    env_name = "AIME_ADAS_SOURCE" if method == "adas" else "AIME_GDESIGNER_SOURCE"
    return Path(os.environ.get(
        env_name, str(Path(__file__).resolve().parent / "upstream" / name)
    )).resolve()


def run_adas(dataset: str, rows: dict[str, list[dict[str, Any]]], args: argparse.Namespace,
             run_id: str, runtime: ModelRuntime, source: Path) -> dict[str, Any]:
    """Run the official ADAS archive/reflection/debug loop through safe adapters."""
    source = source.resolve()
    mgsm_dir = source / "_mgsm"
    sys.path.insert(0, str(mgsm_dir))
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    os.environ["OPENAI_BASE_URL"] = args.endpoint.rstrip("/")
    os.environ["OPENAI_TIMEOUT"] = str(runtime.request_timeout_s)
    import importlib
    adas = importlib.import_module("search")
    from openai import OpenAI
    adas.client = OpenAI(base_url=args.endpoint.rstrip("/"), api_key="EMPTY",
                         timeout=runtime.request_timeout_s, max_retries=0)

    info_cls = adas.Info
    active_rows = rows["search"]
    eval_audits: list[dict[str, Any]] = []
    phase_name = "search"

    def native_json_call(messages: list[dict[str, str]], temperature: float) -> dict[str, Any]:
        content = runtime._messages_call(messages, temperature=temperature, json_mode=True, phase=phase_name)
        return _parse_json_object(content)

    def adas_json_response(msg: str, model: str, system_message: str, temperature: float = 0.5):
        return native_json_call([{"role": "system", "content": system_message}, {"role": "user", "content": msg}], temperature)

    def adas_reflect_response(msg_list: list[dict[str, str]], model: str, temperature: float = 0.8):
        return native_json_call(msg_list, temperature)

    # Native ADAS calls these two functions for the meta-agent and its
    # reflection/debug turns. Route them through the same context/token audit.
    adas.get_json_response_from_gpt = adas_json_response
    adas.get_json_response_from_gpt_reflect = adas_reflect_response

    def agent_query(self: Any, input_infos: list[Any], instruction: str, iteration_idx: int = -1):
        consume_candidate_agent_call_budget()
        system_prompt, user_prompt = self.generate_prompt(input_infos, instruction)
        try:
            content = runtime._messages_call(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=(0.0 if dataset == "aime" else float(self.temperature)), json_mode=True, phase=phase_name,
            )
            response_json = _parse_json_object(content)
        except Exception:
            response_json = {}
        values = []
        for key in self.output_fields:
            value = response_json.get(key, "")
            values.append(info_cls(key, self.__repr__(), str(value), iteration_idx))
        return values

    adas.LLMAgentBase.query = agent_query

    original_generate_prompt = adas.LLMAgentBase.generate_prompt
    def task_aware_prompt(self: Any, input_infos: list[Any], instruction: str):
        answer_format = ("Return exactly one final line `FINAL ANSWER: <numeric answer>` as required by the task."
                         if dataset == "masbench" else "Return only the numeric final answer.")
        output_fields = {key: (answer_format if "answer" in key.lower() else f"Your {key}.")
                         for key in self.output_fields}
        system_prompt = adas.ROLE_DESC(self.role) + "\n\n" + adas.FORMAT_INST(output_fields)
        input_text = ""
        for field_name, author, content, iteration_idx in input_infos:
            if field_name == "task":
                input_text += f"# Your Task:\n{content}\n\n"
            elif iteration_idx != -1:
                input_text += f"### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n"
            else:
                input_text += f"### {field_name} by {author}:\n{content}\n\n"
        return system_prompt, input_text + instruction
    adas.LLMAgentBase.generate_prompt = task_aware_prompt

    def evaluate_forward_fn(_native_args: Any, forward_str: str) -> list[float]:
        nonlocal eval_audits
        forward = compile_candidate(forward_str, adas.LLMAgentBase)
        started_usage = dict(runtime.usage["totals"])
        scored_rows: list[dict[str, Any] | None] = [None] * len(active_rows)

        def solve_one(index_row: tuple[int, dict[str, Any]]):
            index, row = index_row
            _candidate_budget.iterations = 0
            _candidate_budget.agent_calls = 0
            calls_before = runtime.usage["totals"]["calls"]
            try:
                task = task_text(dataset, row)
                result = forward(adas.AgentSystem(), info_cls("task", "User", task, -1))
                raw = result.content if isinstance(result, info_cls) else result
                scored = score_row(dataset, raw, row.get("answer", row.get("targets", "")))
                scored_rows[index] = {"id": str(row.get("id", row.get("problem_idx", index))),
                    "prediction": scored["prediction"], "gold": scored["gold"],
                    "parser_valid": bool(scored["parser_valid"]), "correct": bool(scored["correct"]),
                    "score_0_1_2": int(scored["score_0_1_2"]),
                    "sample_failed": not bool(scored["parser_valid"]),
                    "calls": runtime.usage["totals"]["calls"] - calls_before}
            except Exception as exc:
                scored_rows[index] = {"id": str(row.get("id", row.get("problem_idx", index))),
                    "prediction": "", "gold": str(row.get("answer", row.get("targets", ""))),
                    "parser_valid": False, "correct": False, "score_0_1_2": 0,
                    "sample_failed": True, "failure": f"{type(exc).__name__}: {exc}",
                    "calls": runtime.usage["totals"]["calls"] - calls_before}

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            list(pool.map(solve_one, enumerate(active_rows)))
        audit = [row for row in scored_rows if row is not None]
        exact = sum(row["correct"] for row in audit)
        valid = sum(row["parser_valid"] for row in audit)
        failures = sum(row["sample_failed"] for row in audit)
        after_usage = dict(runtime.usage["totals"])
        delta_tokens = after_usage["total_tokens"] - started_usage["total_tokens"]
        delta_calls = after_usage["calls"] - started_usage["calls"]
        delta_failed_calls = after_usage["failed_calls"] - started_usage["failed_calls"]
        delta_truncated = after_usage["completion_truncated"] - started_usage["completion_truncated"]
        summary = {
            "phase": phase_name, "num_examples": len(audit), "correct": exact,
            "accuracy": exact / len(audit) if audit else 0.0,
            "parser_valid_rate": valid / len(audit) if audit else 0.0,
            "mean_score_0_1_2": sum(row["score_0_1_2"] for row in audit) / len(audit) if audit else 0.0,
            "failure_fraction": failures / len(audit) if audit else 1.0,
            "total_tokens": delta_tokens, "calls": delta_calls,
            "failed_calls": delta_failed_calls, "completion_truncated": delta_truncated,
            "request_failure_fraction": delta_failed_calls / max(1, delta_calls),
            "truncation_fraction": delta_truncated / max(1, delta_calls),
            "tokens_per_example": delta_tokens / max(1, len(audit)), "rows": audit,
        }
        eval_audits.append(summary)
        if summary["failure_fraction"] > MAX_SAMPLE_FAILURE_RATE:
            # Keep native ADAS's search signal, but mark the candidate/run
            # invalid later; the optimizer must never hide sample failures.
            pass
        # Preserve the upstream ADAS exact-match utility. The shared 0/1/2
        # ordinal score is reported separately and never changes its search signal.
        return [float(row["correct"]) for row in audit]

    adas.evaluate_forward_fn = evaluate_forward_fn
    adas.get_all_examples = lambda: [
        {"inputs": task_text(dataset, row), "targets": str(row.get("answer", row.get("targets", ""))),
         "id": str(row.get("id", row.get("problem_idx", index)))}
        for index, row in enumerate(active_rows)
    ]
    adas.score_mgsm = lambda target, prediction: float(score_row(dataset, prediction, target)["correct"])

    native_dir = args.run_dir / "native_search"
    native_dir.mkdir(parents=True, exist_ok=True)
    native_args = argparse.Namespace(
        # Native ADAS has a legacy test slice; disable it entirely. Selection
        # and final D_test evaluation are performed explicitly below.
        valid_size=len(rows["search"]), test_size=0, shuffle_seed=args.seed,
        n_repreat=1, multiprocessing=True, max_workers=CONCURRENCY, debug=True,
        save_dir=str(native_dir), expr_name=f"{dataset}_{args.seed}",
        n_generation=args.adas_generations, debug_max=3, model=MODEL,
    )
    phase_name = "search"
    started = time.time()
    adas.search(native_args)
    search_seconds = time.time() - started
    archive_path = native_dir / f"{native_args.expr_name}_run_archive.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))

    # D_select is the only basis for freezing Q/E. Identical code is evaluated
    # once and aliases retain the original ADAS archive identity.
    code_to_candidate: dict[str, dict[str, Any]] = {}
    selection_rows = []
    active_rows = rows["select"]
    phase_name = "select"
    for index, candidate in enumerate(archive):
        code = str(candidate.get("code", ""))
        if not code:
            continue
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        if code_hash in code_to_candidate:
            row = dict(code_to_candidate[code_hash])
            row["archive_index"] = index
            row["duplicate_of"] = row.get("candidate_id")
            selection_rows.append(row)
            continue
        before = len(eval_audits)
        try:
            evaluate_forward_fn(native_args, code)
            measured = eval_audits[-1]
            row = {"candidate_id": f"archive_{index}", "archive_index": index,
                "code_sha256": code_hash, "status": "evaluated",
                "score": measured["accuracy"], "valid": (
                    measured["failure_fraction"] <= MAX_SAMPLE_FAILURE_RATE
                    and measured["request_failure_fraction"] <= MAX_SAMPLE_FAILURE_RATE
                    and measured["truncation_fraction"] <= MAX_SAMPLE_FAILURE_RATE
                ),
                "accuracy": measured["accuracy"], "correct": measured["correct"],
                "num_examples": measured["num_examples"], "parser_valid_rate": measured["parser_valid_rate"],
                "mean_score_0_1_2": measured["mean_score_0_1_2"],
                "failure_fraction": measured["failure_fraction"], "total_tokens": measured["total_tokens"],
                "total_calls": measured["calls"], "failed_calls": measured["failed_calls"],
                "completion_truncated": measured["completion_truncated"],
                "request_failure_fraction": measured["request_failure_fraction"],
                "truncation_fraction": measured["truncation_fraction"], "rows": measured["rows"],
                "native_name": candidate.get("name", ""), "native_fitness": candidate.get("fitness")}
        except Exception as exc:
            row = {"candidate_id": f"archive_{index}", "archive_index": index,
                "code_sha256": code_hash, "status": "invalid", "valid": False,
                "failure": f"{type(exc).__name__}: {exc}"}
        code_to_candidate[code_hash] = row
        selection_rows.append(row)
    # Save every D_select archive candidate (including rejected candidates)
    # before the strict selection gate can abort the formal runner.
    write_json(args.run_dir / "selection_audit.json", {
        "schema": "aime_fourway_selection_audit_v1", "method": "adas",
        "split": "D_select", "dtest_opened": False,
        "candidate_rows": selection_rows,
        "native_search_archive": str(archive_path),
        "failure_limit": MAX_SAMPLE_FAILURE_RATE,
        "telemetry": runtime.summary(),
    })
    valid = [row for row in selection_rows if row.get("valid") is True]
    if not valid:
        raise RuntimeError("ADAS has no valid candidate on D_select (failure rate must be <=5%).")
    from aime_selection import quality_operating_point, efficiency_operating_point
    q = quality_operating_point(valid)
    e = efficiency_operating_point(valid, delta=0.05)
    if q is None or e is None:
        raise RuntimeError("ADAS could not freeze both Q and E from D_select.")

    selected = {"Q": q, "E": e}
    from native_aime_formal import freeze_aime_selection, load_aime_test_rows_after_freeze
    if dataset != "aime":
        raise ValueError("this external runner is packaged for AIME only; use the MASBench-native package separately")
    freeze_aime_selection(
        args.run_dir, "adas", rows["search"], rows["select"],
        {key: {"candidate_id": item["candidate_id"], "code_sha256": item["code_sha256"],
               "score": float(item["score"])} for key, item in selected.items()},
        source_provenance=args.upstream_provenance,
    )
    tests: dict[str, Any] = {}
    for split_name in ("aime_2025", "aime_2026"):
        active_rows = load_aime_test_rows_after_freeze(
            args.data_dir, args.run_dir, "adas", f"{split_name}.jsonl", 30
        )
        phase_name = f"test:{split_name}"
        points: dict[str, Any] = {}
        for operating_point, candidate in selected.items():
            if operating_point == "E" and candidate["candidate_id"] == selected["Q"]["candidate_id"]:
                points["E"] = {**points["Q"], "same_candidate_as": "Q", "operating_point": "E"}
                continue
            measured_before = len(eval_audits)
            evaluate_forward_fn(native_args, next(item["code"] for item in archive
                if hashlib.sha256(str(item.get("code", "")).encode()).hexdigest() == candidate["code_sha256"]))
            measured = eval_audits[-1]
            if (measured["failure_fraction"] > MAX_SAMPLE_FAILURE_RATE
                    or measured["request_failure_fraction"] > MAX_SAMPLE_FAILURE_RATE
                    or measured["truncation_fraction"] > MAX_SAMPLE_FAILURE_RATE):
                raise RuntimeError(f"ADAS {split_name} failure rate exceeds 5%; refusing an invalid test result.")
            points[operating_point] = {key: measured[key] for key in (
                "num_examples", "correct", "accuracy", "parser_valid_rate", "mean_score_0_1_2",
                "failure_fraction", "total_tokens", "calls", "tokens_per_example", "rows")}
            points[operating_point]["failed_calls"] = measured["failed_calls"]
            points[operating_point]["completion_truncated"] = measured["completion_truncated"]
            points[operating_point]["request_failure_fraction"] = measured["request_failure_fraction"]
            points[operating_point]["truncation_fraction"] = measured["truncation_fraction"]
            points[operating_point]["operating_point"] = operating_point
            points[operating_point]["selected_candidate"] = candidate["candidate_id"]
        tests[split_name] = {**points["Q"], "Q": points["Q"], "E": points["E"]}

    return {
        "method": "adas", "search_entrypoint": "ADAS official _mgsm.search.search (archive + reflection + code repair)",
        "native_method_config": {"n_generation": args.adas_generations, "initial_archive_size": 7,
            "debug_max": 3, "per_sample_agent_call_cap": 24, "safe_candidate_executor": "restricted_ast_v1"},
        "search_seconds": search_seconds, "archive_path": str(archive_path),
        "search_audits": [x for x in eval_audits if x["phase"] == "search"],
        "selection_rows": selection_rows, "selected_Q": selected["Q"]["candidate_id"],
        "selected_E": selected["E"]["candidate_id"], "test": tests,
        "telemetry": runtime.summary(),
    }


def _register_gdesigner_prompt() -> dict[str, Any]:
    from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
    from GDesigner.prompt.gsm8k_prompt_set import ROLE_CONNECTION

    roles = ["Algebraic Decomposer", "Independent Solver", "Consistency Checker", "Solution Synthesizer"]
    role_mapping = {
        "Math Solver": "Independent Solver",
        "Mathematical Analyst": "Algebraic Decomposer",
        "Programming Expert": "Solution Synthesizer",
        "Inspector": "Consistency Checker",
    }
    native_edges = [tuple(edge) for edge in ROLE_CONNECTION]
    if (len(native_edges) != 9 or len(set(native_edges)) != 9
            or any(left not in role_mapping or right not in role_mapping for left, right in native_edges)):
        raise RuntimeError("G-Designer pinned GSM8K role connections do not match the audited nine-edge graph")
    mapped_edges = [(role_mapping[left], role_mapping[right]) for left, right in native_edges]
    descriptions = {
        roles[0]: "You decompose the mathematical task into explicit equations and constraints.",
        roles[1]: "You solve the task independently and verify each arithmetic step.",
        roles[2]: "You check proposed calculations for consistency and identify errors.",
        roles[3]: "You synthesize the available work into one reliable answer.",
    }

    @PromptSetRegistry.register("rpas_external")
    class RPASPromptSet:
        @staticmethod
        def get_role():
            return roles[0]
        @staticmethod
        def get_constraint(role):
            return descriptions.get(role, "Solve the task carefully.")
        def get_description(self, role):
            return descriptions.get(role, "Solve the task carefully.")
        @staticmethod
        def get_role_connection():
            # Preserve native role-conditioning connectivity. The earlier
            # AIME complete role graph collapsed GCN features to rank one.
            return mapped_edges
        @staticmethod
        def get_format():
            return "natural language"
        @staticmethod
        def get_answer_prompt(question, role="Independent Solver"):
            return f"Task: {question}\nYour role: {role}. Work carefully and provide a reasoned response."
        @staticmethod
        def get_adversarial_answer_prompt(question):
            return f"Find possible errors in this task and solution: {question}"
        @staticmethod
        def get_query_prompt(question):
            return str(question)
        @staticmethod
        def get_file_analysis_prompt(query, file):
            return f"{query}\n{file}"
        @staticmethod
        def get_websearch_prompt(query):
            return str(query)
        @staticmethod
        def get_distill_websearch_prompt(query, results):
            return f"{query}\n{results}"
        @staticmethod
        def get_reflect_prompt(question, answer):
            return f"Review this response to the task and correct any errors.\nTask: {question}\nResponse: {answer}"
        @staticmethod
        def get_combine_materials(materials):
            return f"Compare the following analyses:\n{materials}"
        @staticmethod
        def get_decision_constraint():
            return "Use the task and other agents' outputs. End with exactly one line `FINAL ANSWER: <numeric answer>`. For MASBench, output the complete required vector in order."
        @staticmethod
        def get_decision_role():
            return "You are the final mathematical decision maker"
        @staticmethod
        def get_decision_few_shot():
            return ""

    return {
        "variant": "aime_native_gsm8k_role_graph_semantic_mapping_v1",
        "role_names": roles,
        "role_mapping": role_mapping,
        "native_role_edges": native_edges,
        "mapped_role_edges": mapped_edges,
        "upstream_repository": UPSTREAM_REPOSITORIES["gdesigner"],
        "upstream_commit": UPSTREAM_COMMITS["gdesigner"],
        "upstream_source": "GDesigner/prompt/gsm8k_prompt_set.py:ROLE_CONNECTION",
        "adapter_change": "replace the author AIME package's complete role graph with mapped native nine-edge role conditioning",
    }


class GDesignerNumericalError(FloatingPointError):
    """A non-finite controller value invalidates the entire native run."""


def require_finite_gdesigner(value: Any, label: str) -> None:
    import torch

    if not bool(torch.isfinite(torch.as_tensor(value)).all()):
        raise GDesignerNumericalError(f"G-Designer non-finite {label}")


def require_finite_gdesigner_model(graph: Any, label: str, *, gradients: bool = False) -> None:
    for module_name in ("gcn", "mlp"):
        for parameter_name, parameter in getattr(graph, module_name).named_parameters():
            require_finite_gdesigner(parameter, f"{label} {module_name}.{parameter_name}")
            if gradients and parameter.grad is not None:
                require_finite_gdesigner(parameter.grad, f"{label} gradient {module_name}.{parameter_name}")


def patch_gdesigner_numerics(graph_module: Any) -> dict[str, Any]:
    """Install an explicit numerical compatibility adapter in this process.

    The pinned Graph.arun normalizes flattened Gram values with an undefined
    zero denominator when every value is equal. Only that exact degenerate
    case receives neutral zero logits; every nonzero span retains the upstream
    formula without an epsilon or altered gradient.
    """
    import torch

    audit = {
        "variant": "gdesigner_exact_zero_span_numerical_compatibility_v1",
        "upstream_function": "GDesigner.graph.graph.min_max_norm",
        "degenerate_policy": "exact zero span maps to zero logits (edge probability 0.5)",
        "nondegenerate_policy": "unchanged upstream min-max formula; no epsilon",
        "zero_span_calls": 0,
        "finite_checks": "normalization, log probability, loss, gradients, parameters; fail closed",
    }

    def finite_min_max_norm(tensor: Any):
        require_finite_gdesigner(tensor, "normalization input")
        min_val, max_val = tensor.min(), tensor.max()
        span = max_val - min_val
        require_finite_gdesigner(span, "normalization span")
        degenerate = span == 0
        if bool(degenerate):
            audit["zero_span_calls"] += 1
        # Avoid evaluating division by zero in either autograd branch. The
        # constant branch remains connected to the input with zero gradient.
        denominator = torch.where(degenerate, torch.ones_like(span), span)
        normalized = (tensor - min_val) / denominator * 2 - 1
        result = torch.where(degenerate, torch.zeros_like(normalized), normalized)
        require_finite_gdesigner(result, "normalized logits")
        return result

    graph_module.min_max_norm = finite_min_max_norm
    return audit


def run_gdesigner(dataset: str, rows: dict[str, list[dict[str, Any]]], args: argparse.Namespace,
                  run_id: str, runtime: ModelRuntime, source: Path) -> dict[str, Any]:
    """Run official G-Designer Graph + GCN/MLP + REINFORCE with a local task promptset."""
    source = source.resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    import GDesigner.prompt.gsm8k_prompt_set  # register native MathSolver/FinalRefer agents
    from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
    from GDesigner.agents.agent_registry import AgentRegistry
    import GDesigner.agents.math_solver
    import GDesigner.agents.final_decision
    from GDesigner.llm.llm_registry import LLMRegistry
    from GDesigner.llm.gpt_chat import GPTChat
    import GDesigner.graph.graph as graph_module
    from GDesigner.graph.graph import Graph
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import torch

    numerical_adapter = patch_gdesigner_numerics(graph_module)
    role_graph_adapter = _register_gdesigner_prompt()
    embedding_dir = Path(os.environ.get(
        "AIME_MINILM_PATH", str(Path(__file__).resolve().parent / "embeddings" / "all-MiniLM-L6-v2")
    ))
    if not embedding_dir.is_dir():
        raise FileNotFoundError(f"local MiniLM embedding model is missing: {embedding_dir}")
    embedding_model = SentenceTransformer(str(embedding_dir), device="cpu")
    graph_module.get_sentence_embedding = lambda text: embedding_model.encode(text, convert_to_numpy=True)

    async def controlled_agen(self: Any, messages: Any, max_tokens: int | None = None,
                              temperature: float | None = None, num_comps: int | None = None):
        if isinstance(messages, str):
            material = [{"role": "user", "content": messages}]
        else:
            material = [{"role": str(item.get("role") if isinstance(item, dict) else item.role),
                         "content": str(item.get("content") if isinstance(item, dict) else item.content)}
                        for item in messages]
        phase = os.environ.get("MASBENCH_PHASE", "unattributed")
        return await runtime.call_async(material, temperature=(0.0 if dataset == "aime" else float(temperature if temperature is not None else 0.2)), phase=phase)
    GPTChat.agen = controlled_agen
    LLMRegistry.register  # explicit reference ensures registry is initialized

    seed_everything(args.seed)
    agent_names = ["MathSolver"] * 4
    role_names = role_graph_adapter["role_names"]
    spatial_masks = [[0 if i == j else 1 for j in range(4)] for i in range(4)]
    temporal_masks = [[0 for _ in range(4)] for _ in range(4)]
    graph = Graph(domain="rpas_external", llm_name="GPTChat", agent_names=agent_names,
        decision_method="FinalRefer", optimized_spatial=True, optimized_temporal=False,
        fixed_spatial_masks=spatial_masks, fixed_temporal_masks=temporal_masks,
        node_kwargs=[{"role": role} for role in role_names])
    graph.gcn.train()
    require_finite_gdesigner_model(graph, "initial parameters")
    optimizer = torch.optim.Adam(graph.gcn.parameters(), lr=0.1)
    controller_path = args.run_dir / "gdesigner_controller.pt"
    controller_sha256 = ""
    frozen_controller_state: dict[str, dict[str, Any]] = {}

    def records(split: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"task": task_text(dataset, row), "gold": str(row.get("answer", "")),
                 "id": str(row.get("id", row.get("problem_idx", index)))}
                for index, row in enumerate(split)]

    async def infer(batch: list[dict[str, Any]], phase: str, train: bool = False) -> tuple[list[dict[str, Any]], list[Any]]:
        require_finite_gdesigner_model(graph, f"{phase} parameters before inference")
        before = dict(runtime.usage["totals"])
        previous = os.environ.get("MASBENCH_PHASE")
        os.environ["MASBENCH_PHASE"] = phase
        try:
            async def one(item: dict[str, Any]):
                realized = copy.deepcopy(graph)
                realized.gcn = graph.gcn
                realized.mlp = graph.mlp
                try:
                    response, log_prob = await realized.arun({"task": item["task"]}, num_rounds=1, max_tries=3, max_time=180)
                    require_finite_gdesigner(log_prob, f"{phase} log probability")
                    raw = response[0] if response else ""
                    score = score_row(dataset, raw, item["gold"])
                    audit = {"id": item["id"], "prediction": score["prediction"], "gold": score["gold"],
                        "parser_valid": bool(score["parser_valid"]), "correct": bool(score["correct"]),
                        "score_0_1_2": int(score["score_0_1_2"]),
                        "sample_failed": not bool(score["parser_valid"]), "raw_output": str(raw)}
                    return audit, log_prob
                except GDesignerNumericalError:
                    raise
                except Exception as exc:
                    return {"id": item["id"], "prediction": "", "gold": item["gold"],
                        "parser_valid": False, "correct": False, "score_0_1_2": 0,
                        "sample_failed": True, "failure": f"{type(exc).__name__}: {exc}"}, torch.tensor(0.0)
            pairs = await asyncio.gather(*(one(item) for item in batch))
            audit, log_probs = zip(*pairs) if pairs else ([], [])
            if train:
                # Match G-Designer's native REINFORCE reward: exact-match
                # correctness is binary. The common ordinal score remains
                # available for reporting/analysis only.
                utilities = [float(bool(row["correct"])) for row in audit]
                losses = [-log_prob * utility for log_prob, utility in zip(log_probs, utilities)]
                losses = [loss for loss in losses if getattr(loss, "requires_grad", False)]
                if losses:
                    loss = torch.mean(torch.stack(losses))
                    require_finite_gdesigner(loss, f"{phase} REINFORCE loss")
                    optimizer.zero_grad()
                    loss.backward()
                    require_finite_gdesigner_model(graph, f"{phase} before Adam update", gradients=True)
                    optimizer.step()
                    require_finite_gdesigner_model(graph, f"{phase} after Adam update", gradients=True)
            after = dict(runtime.usage["totals"])
            phase_usage = runtime.usage["phases"].get(phase, empty_usage())
            return list(audit), list(log_probs)
        finally:
            if previous is None:
                os.environ.pop("MASBENCH_PHASE", None)
            else:
                os.environ["MASBENCH_PHASE"] = previous

    async def training_loop() -> tuple[list[dict[str, Any]], list[list[str]]]:
        search_records = records(rows["search"])
        if not search_records:
            raise ValueError("G-Designer requires a nonempty D_search split")
        # Preserve the native ten optimizer updates using only D_search. The
        # frozen AIME D_search has 60 examples; ten native batches of four are
        # sampled without replacement until a fresh shuffled pass is needed.
        schedule: list[int] = []
        while len(schedule) < 40:
            epoch = list(range(len(search_records)))
            random.shuffle(epoch)
            schedule.extend(epoch)
        schedule = schedule[:40]
        batches = [[search_records[index] for index in schedule[offset:offset + 4]]
                   for offset in range(0, 40, 4)]
        training_row_ids = [[item["id"] for item in batch] for batch in batches]
        for batch in batches:
            await infer(batch, "search:train", train=True)
        # The upstream experiment only changes its local CLI args during eval;
        # Graph.optimized_spatial remains True, so the trained task-conditioned
        # GCN continues to sample the learned spatial topology. Keep that native
        # behavior here; disabling this flag would silently evaluate a fixed
        # fully-connected graph and discard the learned controller.
        graph.gcn.eval()
        post_training, _ = await infer(search_records, "search:post_training", train=False)
        return post_training, training_row_ids

    async def evaluate_split(split_rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
        items = records(split_rows)
        audits: list[dict[str, Any]] = []
        for offset in range(0, len(items), 4):
            batch, _ = await infer(items[offset:offset + 4], phase, train=False)
            audits.extend(batch)
        n = len(audits)
        correct = sum(int(row["correct"]) for row in audits)
        phase_usage = runtime.usage["phases"].get(phase, empty_usage())
        calls = int(phase_usage["calls"])
        failed_calls = int(phase_usage["failed_calls"])
        truncated = int(phase_usage["completion_truncated"])
        request_failure_fraction = failed_calls / max(1, calls)
        truncation_fraction = truncated / max(1, calls)
        return {"num_examples": n, "correct": correct, "accuracy": correct / n if n else 0.0,
            "parser_valid_rate": sum(int(row["parser_valid"]) for row in audits) / n if n else 0.0,
            "mean_score_0_1_2": sum(int(row["score_0_1_2"]) for row in audits) / n if n else 0.0,
            "failure_fraction": sum(int(row["sample_failed"]) for row in audits) / n if n else 1.0,
            "rows": audits, "calls": calls, "failed_calls": failed_calls,
            "completion_truncated": truncated, "request_failure_fraction": request_failure_fraction,
            "truncation_fraction": truncation_fraction,
            "total_tokens": int(phase_usage["total_tokens"])}

    async def full_run() -> tuple[list[dict[str, Any]], list[list[str]], float, dict[str, Any], dict[str, Any]]:
        nonlocal controller_sha256, frozen_controller_state
        runtime.async_semaphore = asyncio.Semaphore(CONCURRENCY)
        started = time.time()
        search_result, training_row_ids = await training_loop()
        elapsed = time.time() - started
        require_finite_gdesigner_model(graph, "controller frozen after D_search", gradients=True)
        frozen_controller_state = {
            name: {key: value.detach().cpu().clone() for key, value in getattr(graph, name).state_dict().items()}
            for name in ("gcn", "mlp")
        }
        with controller_path.open("xb") as stream:
            torch.save({
                "schema": "aime_gdesigner_controller_v1", "seed": args.seed,
                "state_dicts": frozen_controller_state,
                "role_graph_adapter": role_graph_adapter,
                "role_adjacency_edge_index": graph.role_adj_matrix.detach().cpu().clone(),
                "numerical_compatibility_adapter": dict(numerical_adapter),
                "module_training_modes": {name: getattr(graph, name).training for name in ("gcn", "mlp")},
                "frozen_phase": "after D_search and before D_select/D_test",
            }, stream)
        controller_sha256 = sha256_file(controller_path)
        selected = await evaluate_split(rows["select"], "select")
        # Preserve the full 30-item audit and trained controller identity even
        # if the existing 5% quality gate correctly blocks D_test access.
        write_json(args.run_dir / "selection_audit.json", {
            "schema": "aime_fourway_selection_audit_v1", "method": "gdesigner",
            "split": "D_select", "dtest_opened": False,
            "selection_metrics": selected,
            "controller_sha256": controller_sha256,
            "failure_limit": MAX_SAMPLE_FAILURE_RATE,
            "telemetry": runtime.summary(),
        })
        if (selected["failure_fraction"] > MAX_SAMPLE_FAILURE_RATE
                or selected["request_failure_fraction"] > MAX_SAMPLE_FAILURE_RATE
                or selected["truncation_fraction"] > MAX_SAMPLE_FAILURE_RATE):
            raise RuntimeError("G-Designer D_select failure rate exceeds 5%; refusing to open D_test.")
        from native_aime_formal import freeze_aime_selection, load_aime_test_rows_after_freeze
        if dataset != "aime":
            raise ValueError("this external runner is packaged for AIME only; use the MASBench-native package separately")
        freeze_aime_selection(
            args.run_dir, "gdesigner", rows["search"], rows["select"],
            {"candidate_id": "trained_gdesigner_graph",
             "selection_accuracy": float(selected["accuracy"]),
             "selection_rows": int(selected["num_examples"]),
             "controller_path": str(controller_path),
             "controller_sha256": controller_sha256},
            source_provenance=args.upstream_provenance,
        )
        test_results: dict[str, Any] = {}
        for name in ("aime_2025", "aime_2026"):
            test_rows = load_aime_test_rows_after_freeze(
                args.data_dir, args.run_dir, "gdesigner", f"{name}.jsonl", 30
            )
            point = await evaluate_split(test_rows, f"test:{name}")
            if (point["failure_fraction"] > MAX_SAMPLE_FAILURE_RATE
                    or point["request_failure_fraction"] > MAX_SAMPLE_FAILURE_RATE
                    or point["truncation_fraction"] > MAX_SAMPLE_FAILURE_RATE):
                raise RuntimeError(f"G-Designer {name} failure rate exceeds 5%.")
            point["tokens_per_example"] = point["total_tokens"] / max(1, point["num_examples"])
            point["operating_point"] = "Q"
            point["selected_candidate"] = "trained_gdesigner_graph"
            point_e = {**point, "operating_point": "E", "same_candidate_as": "Q"}
            test_results[name] = {**point, "Q": point, "E": point_e}
        require_finite_gdesigner_model(graph, "final inference parameters", gradients=True)
        for module_name, state in frozen_controller_state.items():
            observed = getattr(graph, module_name).state_dict()
            if any(not torch.equal(value, observed[key].detach().cpu()) for key, value in state.items()):
                raise RuntimeError("G-Designer controller changed after its D_search freeze")
        if sha256_file(controller_path) != controller_sha256:
            raise RuntimeError("G-Designer frozen controller artifact changed during evaluation")
        return search_result, training_row_ids, elapsed, selected, test_results

    search_audit, training_row_ids, search_seconds, selection, tests = asyncio.run(full_run())
    return {
        "method": "gdesigner", "search_entrypoint": "G-Designer Graph.arun + task-conditioned GCN/MLP + REINFORCE",
        "native_method_config": {"agent_names": agent_names, "agent_count": 4,
            "decision_method": "FinalRefer", "mode": "FullConnected", "optimized_spatial": True,
            "optimized_temporal": False, "learning_rate": 0.1, "batch_size": 4,
            "training_iterations": 10, "training_batch_size": 4, "training_row_ids": training_row_ids,
            "num_rounds": 1, "post_training_inference": "native task-conditioned GCN with optimized spatial sampling enabled",
            "utility": "native exact-match binary REINFORCE reward; shared 0/1/2 score reported separately",
            "numerical_compatibility_adapter": numerical_adapter,
            "role_graph_adapter": role_graph_adapter,
            "controller_checkpoint": str(controller_path), "controller_sha256": controller_sha256,
            "controller_unchanged_after_search": True,
            "embedding_model": str(embedding_dir)},
        "search_seconds": search_seconds, "search": {"rows": search_audit,
            "accuracy": sum(int(x["correct"]) for x in search_audit) / len(search_audit),
            "unique_d_search_examples": len(rows["search"]), "training_instances": 40,
            "post_training_d_search_examples": len(search_audit)},
        "selection": {**selection, "split": "D_select"}, "selected_Q": "trained_gdesigner_graph",
        "selected_E": "same_candidate_as_Q", "test": tests, "telemetry": runtime.summary(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("aime",), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--axis", choices=MASBENCH_AXES, default="depth")
    parser.add_argument("--adas-generations", type=int, default=30)
    parser.add_argument("--adas-source", type=Path, default=None)
    parser.add_argument("--gdesigner-source", type=Path, default=None)
    parser.add_argument("--smoke-only", action="store_true", help="validate frozen data/source and stop before model calls")
    return parser.parse_args()


def smoke_native_source(method: str, source: Path) -> dict[str, Any]:
    """Exercise upstream imports/configuration without opening the model endpoint."""
    if method == "adas":
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        sys.path.insert(0, str(source.resolve() / "_mgsm"))
        import importlib
        native = importlib.import_module("search")
        accepted, rejected = [], []
        for item in native.get_init_archive():
            try:
                compile_candidate(str(item["code"]), native.LLMAgentBase)
                accepted.append(item["name"])
            except Exception as exc:
                rejected.append({"name": item.get("name"), "error": f"{type(exc).__name__}: {exc}"})
        if rejected:
            raise RuntimeError(f"ADAS native seed archive does not fully pass the safe executor: {rejected}")
        return {"native_import": "passed", "native_seed_architectures": accepted,
                "safe_executor": "restricted_ast_v1"}

    sys.path.insert(0, str(source.resolve()))
    os.chdir(source.resolve())
    import GDesigner.prompt.gsm8k_prompt_set
    from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
    import GDesigner.agents.math_solver
    import GDesigner.agents.final_decision
    from GDesigner.llm.gpt_chat import GPTChat
    import GDesigner.graph.graph as graph_module
    from GDesigner.graph.graph import Graph
    from sentence_transformers import SentenceTransformer

    _register_gdesigner_prompt()
    embedding_dir = Path(os.environ.get(
        "AIME_MINILM_PATH", str(Path(__file__).resolve().parent / "embeddings" / "all-MiniLM-L6-v2")
    ))
    if not embedding_dir.is_dir():
        raise FileNotFoundError(f"local MiniLM embedding model is missing: {embedding_dir}")
    embedding_model = SentenceTransformer(str(embedding_dir), device="cpu")
    graph_module.get_sentence_embedding = lambda text: embedding_model.encode(text, convert_to_numpy=True)
    roles = ["Algebraic Decomposer", "Independent Solver", "Consistency Checker", "Solution Synthesizer"]
    graph = Graph(domain="rpas_external", llm_name="GPTChat", agent_names=["MathSolver"] * 4,
        decision_method="FinalRefer", optimized_spatial=True, optimized_temporal=False,
        fixed_spatial_masks=[[0 if i == j else 1 for j in range(4)] for i in range(4)],
        fixed_temporal_masks=[[0 for _ in range(4)] for _ in range(4)],
        node_kwargs=[{"role": role} for role in roles])
    if graph.num_nodes != 4 or graph.gcn is None or PromptSetRegistry.get("rpas_external") is None or GPTChat is None:
        raise RuntimeError("G-Designer native graph smoke check did not initialize expected components.")
    return {"native_import": "passed", "agent_nodes": graph.num_nodes,
            "decision_method": "FinalRefer", "topology_optimizer": "native GCN/MLP"}


def verify_model_endpoint(endpoint: str) -> None:
    from urllib.request import urlopen
    url = endpoint.rstrip("/") + "/models"
    try:
        with urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot verify model endpoint {url}: {type(exc).__name__}: {exc}") from exc
    served = {str(item.get("id", "")) for item in payload.get("data", [])}
    if MODEL not in served:
        raise RuntimeError(f"endpoint serves {sorted(served)!r}, expected {MODEL!r}")


def main() -> int:
    args = parse_args()
    if args.adas_generations < 0 or args.adas_generations > 30:
        raise ValueError("ADAS generation budget must be between 0 and its native default of 30.")
    if args.run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run artifacts: {args.run_dir}")
    args.data_dir = args.data_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    rows = read_task_rows(args.dataset, args.data_dir, args.seed, None)
    expected = {"search": 60, "select": 30}
    if any(len(rows[key]) != count for key, count in expected.items()):
        raise ValueError(f"expected frozen D_search/D_select sizes {expected}; got { {k: len(rows[k]) for k in expected} }")
    source = _native_source(args.method, args.adas_source if args.method == "adas" else args.gdesigner_source)
    if not source.is_dir():
        raise FileNotFoundError(f"official {args.method} source is unavailable: {source}")
    upstream_provenance = verify_upstream_source(args.method, source)
    args.upstream_provenance = upstream_provenance
    source_hashes = sha256_tree(source)
    if args.smoke_only:
        result = smoke_native_source(args.method, source)
        print(json.dumps({"dataset": args.dataset, "method": args.method, "axis": args.axis,
            "seed": args.seed, "split_counts": {k: len(v) for k, v in rows.items()},
            "official_source": str(source), "source_python_files": len(source_hashes),
            "upstream_provenance": upstream_provenance,
            "source_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
            "preflight": result}, indent=2))
        return 0

    verify_model_endpoint(args.endpoint)

    run_id = f"{args.dataset}:{args.method}:all:seed_{args.seed}:{AIME_PROTOCOL}"
    manifest = {"protocol_version": MASBENCH_PROTOCOL if args.dataset == "masbench" else AIME_PROTOCOL,
        "run_id": run_id, "dataset": args.dataset, "method": args.method,
        "axis": None, "seed": args.seed,
        "data_seed": 2026, "split_counts": {k: len(v) for k, v in rows.items()},
        "model": MODEL, "context_limit": CONTEXT_LIMIT, "output_limit": OUTPUT_LIMIT,
        "concurrency": CONCURRENCY, "selection_split": "D_select", "test_split": "D_test",
        "official_baseline_source": str(source), "upstream_provenance": upstream_provenance,
        "baseline_source_sha256": source_hashes,
        "baseline_source_tree_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        "runner_sha256": sha256_file(Path(__file__)), "status": "running", "started_at": time.time()}
    manifest["split_row_ids"] = {
        name: [str(row.get("id", row.get("problem_idx", ""))) for row in values]
        for name, values in rows.items()
    }
    id_lists = [list(map(str, ids)) for ids in manifest["split_row_ids"].values()]
    for ids in id_lists:
        if any(not item for item in ids) or len(set(ids)) != len(ids):
            raise ValueError("AIME frozen validation partitions must have unique nonempty row IDs")
    if set(id_lists[0]) & set(id_lists[1]):
        raise ValueError("AIME D_search and D_select overlap")
    manifest["pairwise_disjoint"] = None  # completed only after the post-lock test audit
    if args.dataset == "aime":
        from native_aime_formal import (
            FROZEN_AIME_MANIFEST_SHA256, load_frozen_aime_manifest,
        )
        frozen_data = load_frozen_aime_manifest(args.data_dir)
        manifest["data_dir"] = str(args.data_dir)
        manifest["frozen_data_manifest_sha256"] = FROZEN_AIME_MANIFEST_SHA256
        data_files = {
            "aimo-validation-aime.jsonl": frozen_data["validation"]["source"]["sha256_git_content"],
            frozen_data["validation"]["search"]["path"]: frozen_data["validation"]["search"]["sha256_git_content"],
            frozen_data["validation"]["select"]["path"]: frozen_data["validation"]["select"]["sha256_git_content"],
        }
        manifest["data_sha256"] = {name: sha256_file(args.data_dir / name) for name in data_files}
        if manifest["data_sha256"] != data_files:
            raise ValueError("AIME files differ from the pinned frozen data manifest")
    else:
        data_files = ["aimo-validation-aime.jsonl"]
        manifest["data_sha256"] = {name: sha256_file(args.data_dir / name) for name in data_files}
    parser_file = (Path(aime_answer_protocol.__file__).resolve() if args.dataset == "aime"
                   else Path(__file__).resolve().with_name("masbench_answer_protocol.py"))
    manifest["answer_parser"] = parser_file.name
    manifest["answer_parser_sha256"] = sha256_file(parser_file)
    manifest["score_mapping"] = "2=exact match; 1=parseable wrong; 0=unparseable"
    args.run_dir.mkdir(parents=True)
    write_json(args.run_dir / "run_manifest.json", manifest)
    runtime: ModelRuntime | None = None
    try:
        runtime = ModelRuntime(args.endpoint, args.method, run_id)
        result = run_adas(args.dataset, rows, args, run_id, runtime, source) if args.method == "adas" else run_gdesigner(args.dataset, rows, args, run_id, runtime, source)
        access_path = args.run_dir / "dtest_access_manifest.json"
        lock_path = args.run_dir / "selection_frozen.json"
        if not access_path.is_file() or not lock_path.is_file():
            raise RuntimeError("AIME baseline finished without a selection lock and D_test access audit")
        access = json.loads(access_path.read_text(encoding="utf-8"))
        if set(access.get("test_splits", {})) != {"aime_2025.jsonl", "aime_2026.jsonl"}:
            raise RuntimeError("AIME baseline did not open exactly the two frozen test files")
        validation_ids = set(manifest["split_row_ids"]["search"] + manifest["split_row_ids"]["select"])
        for filename, record in access["test_splits"].items():
            ids = list(map(str, record.get("ids", [])))
            if (record.get("row_count") != 30 or len(ids) != 30 or len(set(ids)) != 30
                    or validation_ids.intersection(ids)):
                raise RuntimeError(f"AIME D_test split is invalid or overlaps validation: {filename}")
            if record.get("opened_at_epoch", 0) < access.get("selection_frozen_at_epoch", float("inf")):
                raise RuntimeError(f"AIME D_test opened before the selection lock: {filename}")
            manifest["split_row_ids"][Path(filename).stem] = ids
            manifest["data_sha256"][filename] = record["sha256"]
            manifest["data_sha256"][record["frozen_split_path"]] = record["frozen_split_sha256"]
        manifest["pairwise_disjoint"] = True
        manifest["selection_lock_sha256"] = sha256_file(lock_path)
        manifest["dtest_access_manifest_sha256"] = sha256_file(access_path)
        result["dtest_access_manifest_sha256"] = manifest["dtest_access_manifest_sha256"]
        result["run_id"] = run_id
        result["controls"] = {"model": MODEL, "endpoint": args.endpoint, "max_model_len": CONTEXT_LIMIT,
            "max_tokens": OUTPUT_LIMIT, "temperature_policy": "AIME task executor temperature=0; method-native ADAS meta-search temperature retained",
            "request_timeout_s": runtime.request_timeout_s,
            "top_p": 1.0, "thinking": False, "concurrency": CONCURRENCY, "seed": args.seed,
            "data_seed": 2026, "selection_split": "D_select", "test_split": "D_test",
            "score_mapping": "2=exact match; 1=parseable wrong; 0=unparseable",
            "search_reward_policy": "native exact-match binary utility"}
        write_json(args.run_dir / "native_result.json", result)
        manifest["status"] = "complete"
        manifest["finished_at"] = time.time()
        manifest["native_result_sha256"] = sha256_file(args.run_dir / "native_result.json")
        write_json(args.run_dir / "run_manifest.json", manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        manifest["telemetry"] = runtime.summary() if runtime is not None else None
        write_json(args.run_dir / "run_manifest.json", manifest)
        raise
    finally:
        if runtime is not None:
            runtime.sync.close()
            try:
                asyncio.run(runtime.async_client.close())
            except RuntimeError:
                pass
    print(json.dumps({"run_id": run_id, "status": manifest["status"], "result": str(args.run_dir / "native_result.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
