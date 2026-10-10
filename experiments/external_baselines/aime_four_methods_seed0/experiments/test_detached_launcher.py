"""Verify that the worker survives launcher exit without inherited pipes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest


@pytest.mark.parametrize("exit_code", [0, 7])
def test_detached_worker_streams_and_completion(tmp_path: Path, exit_code: int) -> None:
    source = Path(__file__).resolve().parents[1] / "scripts/launch_detached.sh"
    scripts = tmp_path / "package/scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / source.name
    shutil.copyfile(source, launcher)
    (scripts / "run_single48g_all.sh").write_text(
        'set -euo pipefail\n'
        'printf "synthetic worker ready\\n"\n'
        'printf "synthetic stderr ready\\n" >&2\n'
        'sleep 1\n'
        'printf "synthetic worker finished\\n"\n'
        'exit "$SYNTHETIC_EXIT_CODE"\n', encoding="utf-8",
    )
    launch_dir = tmp_path / "launch"
    env = {**os.environ, "AIME_LAUNCH_DIR": str(launch_dir),
           "AIME_OUTPUT_ROOT": str(tmp_path / "fresh_output"),
           "AIME_PYTHON": sys.executable, "AIME_UV_BIN": shutil.which("uv") or "uv",
           "SYNTHETIC_EXIT_CODE": str(exit_code)}
    for name in ("AIME_MODEL_PATH", "AIME_MINILM_PATH", "AIME_MODEL_VERIFICATION_REPORT", "AIME_VLLM_BIN",
                 "AIME_AFLOW_PYTHON", "AIME_MAAS_PYTHON", "AIME_ADAS_PYTHON", "AIME_GDESIGNER_PYTHON"):
        env[name] = "synthetic_fixture"
    worker = None
    try:
        started = subprocess.run(["bash", str(launcher), "all", "--method-order", "maas,adas,gdesigner,aflow"],
                                 env=env, text=True, capture_output=True, timeout=10)
        assert started.returncode == 0, started.stderr
        record = json.loads((launch_dir / "launch.json").read_text())
        worker = record["pid"]
        # The parent launcher has exited and its captured pipes are closed.
        assert os.getsid(worker) == worker
        assert os.readlink(f"/proc/{worker}/fd/0") == "/dev/null"
        for fd in (1, 2):
            assert Path(os.readlink(f"/proc/{worker}/fd/{fd}")).resolve() == launch_dir / "run.log"
        deadline = time.monotonic() + 10
        while not (launch_dir / "completion.json").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        completion = json.loads((launch_dir / "completion.json").read_text())
        assert completion["exit_code"] == exit_code
        assert completion["status"] == ("passed" if exit_code == 0 else "failed")
        assert int((launch_dir / "exit_code.txt").read_text()) == exit_code
        log = (launch_dir / "run.log").read_text()
        assert "synthetic stderr ready" in log
        assert "synthetic worker finished" in log
        # Fresh formal output is only created by the real runner, never this launcher.
        assert not (tmp_path / "fresh_output").exists()
    finally:
        if worker is not None:
            try:
                os.killpg(worker, signal.SIGTERM)
            except ProcessLookupError:
                pass
