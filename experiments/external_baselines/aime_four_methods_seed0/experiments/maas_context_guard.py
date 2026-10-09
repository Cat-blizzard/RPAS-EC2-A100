"""Deterministic MaAS prompt repair and 8K-context guard.

The released MaAS MATH workflow can concatenate several complete candidate
solutions before an ensemble call.  On an 8,192-token task context this can
otherwise make a request invalid before the model is called.  This module
keeps the native graph/controller intact, repairs only the released prompt
literal bug, and applies a logged, deterministic input compaction at the
request boundary.
"""

from __future__ import annotations

import ast
import copy
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any


VERSION = "maas_context_guard_v1"
CONTEXT_LIMIT = 16384
# Qwen3.5-9B is being evaluated with an 8K context.  The native AFlow/MaAS
# graphs often need a second code-generation or ensemble response, so leaving
# only ~1.4K output tokens after a 6.8K prompt causes silent truncation and
# malformed Python.  Reserve a balanced 4K/4K window instead.
TARGET_INPUT_TOKENS = 8192
MIN_OUTPUT_RESERVE = 6144
COMPACTION_MARKER = "\n[deterministic context compaction: omitted middle text]\n"


def _source_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _replace_assignments(source: str, values: dict[str, Any]) -> str:
    """Replace top-level string assignment expressions without reformatting code."""
    tree = ast.parse(source)
    offsets = _source_offsets(source)
    replacements: list[tuple[int, int, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        names = [name for name in names if name in values]
        if not names:
            continue
        start = offsets[node.value.lineno - 1] + node.value.col_offset
        end = offsets[node.value.end_lineno - 1] + node.value.end_col_offset
        replacements.append((start, end, repr(values[names[0]])))
    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]
    return source


def patch_template_directory(path: Path, concise: bool = True) -> dict[str, Any]:
    """Repair one native MaAS MATH template directory in place.

    The upstream files contain ordinary Python strings with LaTeX ``\\b``
    escapes and an unescaped demonstration ``{`` in a ``.format`` template.
    We materialize the intended literal values as Python reprs, preserving the
    native prompt names and operator code.
    """
    from maas_prompt_repair import install as repair_operator, install_instructions
    if concise:
        from maas_concise_prompts_v2 import install as install_concise
    else:
        install_concise = None

    prompt_path = path / "prompt.py"
    operator_path = path / "op_prompt.py"
    prompt_source = prompt_path.read_text(encoding="utf-8")
    operator_source = operator_path.read_text(encoding="utf-8")

    # Resume runs reuse the isolated native repo.  The first run has already
    # materialized the repaired strings with repr(), so applying the AST
    # repair a second time would reject an otherwise valid template.
    if concise and "Keep every calculation needed to obtain the answer." in prompt_source and "Return one fenced Python code block" in operator_source:
        return {
            "version": VERSION,
            "path": str(path),
            "concise_prompt_variant": "maas_concise_v2",
            "prompt_sha256": __import__("hashlib").sha256(prompt_path.read_bytes()).hexdigest(),
            "operator_sha256": __import__("hashlib").sha256(operator_path.read_bytes()).hexdigest(),
        }

    prompt_namespace: dict[str, Any] = {}
    operator_namespace: dict[str, Any] = {}
    exec(compile(prompt_source, str(prompt_path), "exec"), prompt_namespace)
    exec(compile(operator_source, str(operator_path), "exec"), operator_namespace)
    prompt_module = SimpleNamespace(**prompt_namespace)
    operator_module = SimpleNamespace(**operator_namespace)

    install_instructions(prompt_module, prompt_source)
    repair_operator(operator_module, operator_source)
    if install_concise is not None:
        install_concise(prompt_module, operator_module)

    prompt_values = {
        name: value
        for name, value in vars(prompt_module).items()
        if name.isupper() and isinstance(value, str)
    }
    operator_values = {
        name: value
        for name, value in vars(operator_module).items()
        if name.isupper() and isinstance(value, str)
    }
    prompt_path.write_text(_replace_assignments(prompt_source, prompt_values), encoding="utf-8")
    operator_path.write_text(_replace_assignments(operator_source, operator_values), encoding="utf-8")
    return {
        "version": VERSION,
        "path": str(path),
        "concise_prompt_variant": "maas_concise_v2" if concise else "native_repaired",
        "prompt_sha256": __import__("hashlib").sha256(prompt_path.read_bytes()).hexdigest(),
        "operator_sha256": __import__("hashlib").sha256(operator_path.read_bytes()).hexdigest(),
    }


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _decode(tokenizer: Any, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def _head_tail(tokenizer: Any, text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    ids = _encode(tokenizer, text)
    if len(ids) <= limit:
        return text
    if limit < 16:
        return _decode(tokenizer, ids[:limit])
    head = max(1, int(limit * 0.58))
    tail = max(1, limit - head - len(_encode(tokenizer, COMPACTION_MARKER)))
    return _decode(tokenizer, ids[:head]) + COMPACTION_MARKER + _decode(tokenizer, ids[-tail:])


def _ensemble_compact(tokenizer: Any, text: str, limit: int) -> str:
    """Retain the problem and the beginning/end of every labeled candidate."""
    marker = re.search(r"Several solutions have been generated", text, flags=re.I)
    labels = list(re.finditer(r"(?m)^\s*([A-Z]):\s*", text))
    if not marker or len(labels) < 2:
        return _head_tail(tokenizer, text, limit)

    prefix = text[: marker.start()]
    chunks: list[str] = []
    for index, match in enumerate(labels):
        start = match.start()
        end = labels[index + 1].start() if index + 1 < len(labels) else len(text)
        chunks.append(text[start:end])
    prefix_budget = min(len(_encode(tokenizer, prefix)), max(256, int(limit * 0.34)))
    remaining = max(128, limit - prefix_budget - len(_encode(tokenizer, COMPACTION_MARKER)))
    per_chunk = max(128, remaining // max(1, len(chunks)))
    result = _head_tail(tokenizer, prefix, prefix_budget)
    result += "\nSeveral solutions have been generated; candidate labels are preserved:\n"
    for chunk in chunks:
        result += _head_tail(tokenizer, chunk, per_chunk) + "\n"
    # A final exact guard handles marker overhead and tokenizer quirks.
    return _head_tail(tokenizer, result, limit)


class PromptGuard:
    """Shared per-process guard used by every MaAS OpenAI-compatible call."""

    def __init__(self, tokenizer: Any, context_limit: int = CONTEXT_LIMIT, target_input: int = TARGET_INPUT_TOKENS):
        self.tokenizer = tokenizer
        self.context_limit = context_limit
        self.target_input = min(target_input, context_limit - MIN_OUTPUT_RESERVE)
        self.calls = 0
        self.compacted_calls = 0
        self.original_input_tokens = 0
        self.final_input_tokens = 0
        self.compacted_input_tokens = 0
        self.max_input_tokens_seen = 0
        self.events: list[dict[str, Any]] = []

    def _count(self, messages: list[dict[str, Any]]) -> int:
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if hasattr(encoded, "keys"):
                encoded = encoded["input_ids"]
            if encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            return len(encoded)
        except Exception:
            return sum(len(_encode(self.tokenizer, str(m.get("content", "")))) for m in messages)

    def _with_public_policy(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            from shared_output_policy import adapt
            return adapt(copy.deepcopy(messages))
        except Exception:
            return copy.deepcopy(messages)

    def prepare(self, messages: list[dict[str, Any]], requested_max_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.calls += 1
        original = copy.deepcopy(messages)
        original_count = self._count(self._with_public_policy(original))
        self.original_input_tokens += original_count
        self.max_input_tokens_seen = max(self.max_input_tokens_seen, original_count)
        if original_count <= self.target_input:
            self.final_input_tokens += original_count
            return original, {
                "compacted": False,
                "original_input_tokens": original_count,
                "final_input_tokens": original_count,
                "requested_max_tokens": requested_max_tokens,
                "effective_max_tokens": max(1, min(requested_max_tokens, self.context_limit - original_count)),
            }

        # Keep the system prompt intact.  The public policy is appended by the
        # bridge, so all local counts include the same suffix before deciding.
        result = copy.deepcopy(original)
        system_count = self._count(self._with_public_policy([m for m in result if m.get("role") == "system"]))
        non_system = [i for i, m in enumerate(result) if m.get("role") != "system"]
        available = max(256, self.target_input - system_count - 8)
        if non_system:
            original_parts = [len(_encode(self.tokenizer, str(result[i].get("content", "")))) for i in non_system]
            total_parts = max(1, sum(original_parts))
            for index, part_tokens in zip(non_system, original_parts):
                budget = max(128, int(available * part_tokens / total_parts))
                content = str(result[index].get("content", ""))
                if re.search(r"Several solutions have been generated", content, flags=re.I):
                    content = _ensemble_compact(self.tokenizer, content, budget)
                else:
                    content = _head_tail(self.tokenizer, content, budget)
                result[index]["content"] = content

        final_count = self._count(self._with_public_policy(result))
        if final_count > self.target_input:
            # Apply a last exact cap to the largest non-system message.
            candidates = sorted(
                non_system,
                key=lambda i: len(_encode(self.tokenizer, str(result[i].get("content", "")))),
                reverse=True,
            )
            for index in candidates:
                current = len(_encode(self.tokenizer, str(result[index].get("content", ""))))
                reduction = max(128, final_count - self.target_input + 32)
                result[index]["content"] = _head_tail(
                    self.tokenizer,
                    str(result[index].get("content", "")),
                    max(128, current - reduction),
                )
                final_count = self._count(self._with_public_policy(result))
                if final_count <= self.target_input:
                    break
        if final_count > self.context_limit - 1:
            raise ValueError(f"MaAS context guard could not fit prompt: {final_count}>{self.context_limit - 1}")

        self.compacted_calls += 1
        self.compacted_input_tokens += max(0, original_count - final_count)
        self.final_input_tokens += final_count
        event = {
            "compacted": True,
            "original_input_tokens": original_count,
            "final_input_tokens": final_count,
            "requested_max_tokens": requested_max_tokens,
            "effective_max_tokens": max(1, min(requested_max_tokens, self.context_limit - final_count)),
        }
        if len(self.events) < 100:
            self.events.append(event)
        return result, event

    def summary(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "target_input_tokens": self.target_input,
            "context_limit": self.context_limit,
            "calls_seen": self.calls,
            "compacted_calls": self.compacted_calls,
            "original_input_tokens": self.original_input_tokens,
            "final_input_tokens": self.final_input_tokens,
            "compacted_input_tokens": self.compacted_input_tokens,
            "max_input_tokens_seen": self.max_input_tokens_seen,
            "events_sample": self.events,
        }
