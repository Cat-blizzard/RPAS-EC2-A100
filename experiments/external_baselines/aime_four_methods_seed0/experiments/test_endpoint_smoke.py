"""Verify disposable endpoint timeout without making model requests."""
from __future__ import annotations

from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest


def endpoint_smoke_code() -> str:
    script = (Path(__file__).resolve().parents[1] / "scripts/run_pair.sh").read_text()
    section = script.split("# A disposable endpoint smoke test;", 1)[1]
    match = re.search(r"<<'PY'\n(.*?)\nPY", section, re.DOTALL)
    assert match is not None
    return match.group(1)


@pytest.mark.parametrize("configured,expected", [(None, 300.0), ("450.5", 450.5)])
def test_endpoint_smoke_timeout_and_request_contract(monkeypatch, capsys, configured, expected) -> None:
    constructors, requests = [], []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])

    def client(**kwargs):
        constructors.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client))
    monkeypatch.setattr(sys, "argv", ["endpoint-smoke", "18181"])
    if configured is None:
        monkeypatch.delenv("AIME_ENDPOINT_SMOKE_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("AIME_ENDPOINT_SMOKE_TIMEOUT_S", configured)
    exec(compile(endpoint_smoke_code(), "run_pair_endpoint_smoke", "exec"), {})
    assert constructors == [{"base_url": "http://127.0.0.1:18181/v1", "api_key": "EMPTY",
                             "timeout": expected, "max_retries": 0}]
    assert requests == [{"model": "Qwen/Qwen3.5-9B", "messages": [{"role": "user", "content": "Return only OK."}],
                         "max_tokens": 16, "temperature": 0,
                         "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}]
    output = capsys.readouterr().out
    assert f"MODEL_ENDPOINT_SMOKE_START timeout_s={expected:g}" in output
    assert "MODEL_ENDPOINT_SMOKE_PASS elapsed_s=" in output


@pytest.mark.parametrize("invalid", ["", "0", "-5", "nan", "inf", "not-a-number"])
def test_endpoint_smoke_rejects_invalid_timeout_before_client_creation(monkeypatch, invalid) -> None:
    def forbidden_client(**kwargs):
        pytest.fail("Invalid timeout must be rejected before client construction")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=forbidden_client))
    monkeypatch.setattr(sys, "argv", ["endpoint-smoke", "18181"])
    monkeypatch.setenv("AIME_ENDPOINT_SMOKE_TIMEOUT_S", invalid)
    with pytest.raises(SystemExit, match="finite positive number"):
        exec(compile(endpoint_smoke_code(), "run_pair_endpoint_smoke", "exec"), {})
