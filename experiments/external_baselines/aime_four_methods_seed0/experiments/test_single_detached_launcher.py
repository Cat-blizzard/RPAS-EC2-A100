"""Exercise detachment and single-method gates without GPUs or model requests."""
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


STUB_RUN_PAIR = '''#!/usr/bin/env bash
set -euo pipefail
printf 'synthetic runner stdout\\n'
printf 'synthetic runner stderr\\n' >&2
sleep 1
"$AIME_UV_BIN" run --no-project --python "$AIME_PYTHON" python - "$2" <<'PY'
import hashlib, json, os, pathlib, sys
method = sys.argv[1]
out = pathlib.Path(os.environ['AIME_OUTPUT_DIR'])
out.mkdir(parents=True)
diagnostic = os.environ.get('AIME_DIAGNOSTICS_ONLY') == '1'
mode = os.environ.get('SYNTHETIC_MODE', 'passed')
kind = 'diagnostic' if diagnostic else 'formal'
record = {'method': method, 'kind': kind, 'status': 'passed', 'compute_exit_code': 0}
if diagnostic:
    root = pathlib.Path(os.environ['AIME_DIAGNOSTIC_DIR']) / method
    root.mkdir(parents=True)
    source = pathlib.Path(os.environ['SYNTHETIC_PACKAGE']) / 'scripts/diagnose_native_methods.py'
    report = {'method': method, 'status': 'failed' if mode == 'diagnostic_failed' else 'passed',
              'synthetic_only': True, 'check_only': False, 'max_tokens': 6144, 'concurrency': 2,
              'script_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    if mode == 'wrong_diagnostic_hash':
        report['script_sha256'] = 'stale_source'
    (root / 'report.json').write_text(json.dumps(report))
else:
    root = out / method / 'seed_0'
    root.mkdir(parents=True)
    record.update(preflight_exit_code=0, quality_gate_exit_code=0)
    if mode == 'formal_failed':
        record['quality_gate_exit_code'] = 9
        record['status'] = 'failed'
    manifest = {'method': method, 'status': 'rejected' if mode == 'manifest_rejected' else 'complete'}
    (root / 'run_manifest.json').write_text(json.dumps(manifest))
(out / f'launcher_result_{method}.json').write_text(json.dumps(record))
PY
'''


def fixture_package(tmp_path: Path, mode: str = "passed") -> tuple[Path, dict[str, str]]:
    source = Path(__file__).resolve().parents[1] / "scripts/launch_single_detached.sh"
    package = tmp_path / "package"
    scripts = package / "scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / source.name
    shutil.copyfile(source, launcher)
    (scripts / "run_pair.sh").write_text(STUB_RUN_PAIR, encoding="utf-8")
    (scripts / "diagnose_native_methods.py").write_text("# synthetic diagnostic script\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(package)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(package), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "--allow-empty", "-qm", "Synthetic fixture"], check=True, capture_output=True)
    env = {**os.environ, "AIME_LAUNCH_DIR": str(tmp_path / "launch"),
           "AIME_OUTPUT_ROOT": str(tmp_path / "fresh_output"), "AIME_DIAGNOSTIC_DIR": str(tmp_path / "diagnostic"),
           "AIME_PYTHON": sys.executable, "AIME_UV_BIN": shutil.which("uv") or "uv",
           "SYNTHETIC_MODE": mode, "SYNTHETIC_PACKAGE": str(package), "CUDA_VISIBLE_DEVICES": "9", "AIME_PORT": "18182"}
    for name in ("AIME_MODEL_PATH", "AIME_MINILM_PATH", "AIME_MODEL_VERIFICATION_REPORT", "AIME_VLLM_BIN",
                 "AIME_DATA_DIR", "AIME_AFLOW_PYTHON", "AIME_ADAS_PYTHON", "AIME_GDESIGNER_PYTHON"):
        env[name] = "synthetic_fixture"
    return launcher, env


def wait_completion(launch_dir: Path) -> dict:
    deadline = time.monotonic() + 15
    while not (launch_dir / "completion.json").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (launch_dir / "completion.json").exists(), (launch_dir / "run.log").read_text()
    return json.loads((launch_dir / "completion.json").read_text())


@pytest.mark.parametrize("method", ["adas", "aflow", "gdesigner"])
def test_single_worker_survives_launcher_and_passes_gates(tmp_path: Path, method: str) -> None:
    launcher, env = fixture_package(tmp_path)
    launch_dir = Path(env["AIME_LAUNCH_DIR"])
    worker = None
    try:
        parent = subprocess.run(["bash", str(launcher), "--method", method], env=env, text=True,
                                capture_output=True, timeout=10)
        assert parent.returncode == 0, parent.stderr
        launch = json.loads((launch_dir / "launch.json").read_text())
        worker = launch["pid"]
        assert launch["method"] == method and launch["status"] == "started"
        assert launch["source_commit"] and launch["launcher_sha256"]
        assert launch["gpu"] == "9" and launch["port"] == "18182"
        assert launch["protocol"]["max_num_seqs"] == 8
        assert os.getsid(worker) == worker
        stat = Path(f"/proc/{worker}/stat").read_text()
        assert launch["starttime_ticks"] == int(stat.rsplit(")", 1)[1].split()[19])
        command = Path(f"/proc/{worker}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        assert launch["cmd_contains"] in command
        assert not (launch_dir / "launch.json.tmp").exists()
        assert os.readlink(f"/proc/{worker}/fd/0") == "/dev/null"
        for fd in (1, 2):
            assert Path(os.readlink(f"/proc/{worker}/fd/{fd}")).resolve() == launch_dir / "run.log"
        completion = wait_completion(launch_dir)
        assert completion["method"] == method
        assert completion["status"] == "passed" and completion["stage"] == "complete"
        assert completion["exit_code"] == 0 and (launch_dir / "exit_code.txt").read_text().strip() == "0"
        pair = "aflow_maas" if method == "aflow" else "adas_gdesigner"
        assert (Path(env["AIME_OUTPUT_ROOT"]) / pair / method / "seed_0/run_manifest.json").exists()
        log = (launch_dir / "run.log").read_text()
        assert "synthetic runner stderr" in log and "AIME_SINGLE_FORMAL_VERIFIED" in log
    finally:
        if worker:
            try:
                os.killpg(worker, signal.SIGTERM)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("mode", ["diagnostic_failed", "wrong_diagnostic_hash", "formal_failed", "manifest_rejected"])
def test_single_worker_cannot_report_pass_on_failed_evidence(tmp_path: Path, mode: str) -> None:
    launcher, env = fixture_package(tmp_path, mode)
    parent = subprocess.run(["bash", str(launcher), "--method", "adas"], env=env, text=True,
                            capture_output=True, timeout=10)
    assert parent.returncode == 0, parent.stderr
    completion = wait_completion(Path(env["AIME_LAUNCH_DIR"]))
    assert completion["status"] == "failed" and completion["exit_code"] != 0
    if mode in {"diagnostic_failed", "wrong_diagnostic_hash"}:
        assert completion["stage"] == "diagnostic_verification"
        assert not (Path(env["AIME_OUTPUT_ROOT"]) / "adas_gdesigner").exists()
    else:
        assert completion["stage"] == "formal_verification"


@pytest.mark.parametrize("existing", ["AIME_OUTPUT_ROOT", "AIME_LAUNCH_DIR", "AIME_DIAGNOSTIC_DIR"])
def test_single_launcher_refuses_existing_directory(tmp_path: Path, existing: str) -> None:
    launcher, env = fixture_package(tmp_path)
    Path(env[existing]).mkdir()
    parent = subprocess.run(["bash", str(launcher), "--method", "adas"], env=env, text=True,
                            capture_output=True, timeout=10)
    assert parent.returncode == 3
    assert not (Path(env["AIME_LAUNCH_DIR"]) / "worker.pid").exists()
