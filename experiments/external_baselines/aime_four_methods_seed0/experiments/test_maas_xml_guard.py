"""CPU-only tests for narrow MaAS XML repair, without model or upstream install."""
from __future__ import annotations
import asyncio
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("maas_xml_guard", Path(__file__).with_name("maas_xml_guard.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make_class(responses):
    class Node:
        calls = []
        def __init__(self, fields=("solution_letter",)):
            self.fields = fields
        def get_field_names(self):
            return self.fields
        async def xml_fill(self, context, images=None):
            self.calls.append(context)
            return responses.pop(0)
    return Node


def test_good_xml_unchanged():
    Node = make_class([{"solution_letter": "B"}]); usage = {}
    mod.install_scensemble_xml_guard(Node, usage)
    r = asyncio.run(Node().xml_fill("A:... B:..."))
    assert r == {"solution_letter": "B"}
    assert usage["maas_xml_invalid_responses"] == 0
    assert usage["maas_xml_recovered_samples"] == 0


def test_missing_then_valid_retries_without_fabricating():
    Node = make_class([{}, {"solution_letter": "a"}]); usage = {}
    mod.install_scensemble_xml_guard(Node, usage)
    node=Node(); r=asyncio.run(node.xml_fill("Question: find answer"))
    assert r == {"solution_letter": "A"}
    assert len(node.calls) == 2
    assert usage["maas_xml_invalid_responses"] == 1
    assert usage["maas_xml_recovered_samples"] == 1


def test_non_scensemble_uses_native_path_once():
    Node = make_class([{"code": "print(1)"}]); usage={}
    mod.install_scensemble_xml_guard(Node, usage)
    node=Node(("code",))
    assert asyncio.run(node.xml_fill("code")) == {"code": "print(1)"}
    assert len(node.calls) == 1


def test_exhaustion_is_failure_not_a_guessed_choice():
    Node=make_class([{}, {"solution_letter": "not-a-letter"}]); usage={}
    mod.install_scensemble_xml_guard(Node, usage)
    with pytest.raises(ValueError, match="no answer substituted"):
        asyncio.run(Node().xml_fill("question"))
    assert usage["maas_xml_exhausted_samples"] == 1


def test_double_install_rejected():
    Node=make_class([]); usage={}
    mod.install_scensemble_xml_guard(Node, usage)
    with pytest.raises(RuntimeError, match="only be installed once"):
        mod.install_scensemble_xml_guard(Node, usage)
