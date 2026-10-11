"""CPU checks for recovery orchestration; none starts a model or native test."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import shlex
import socket
import subprocess
import time

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_recovery_worker.py"
spec = importlib.util.spec_from_file_location("aime_recovery_worker", SCRIPT)
assert spec and spec.loader
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
build_runtime_environment = launcher.runtime_environment


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "package"
    root.mkdir()
    monkeypatch.setattr(launcher, "ROOT", root)
    monkeypatch.setattr(launcher, "LOCK_ROOT", tmp_path / "locks")
    env = dict(os.environ)
    env.update({"AIME_UV_BIN": "/home/ZZD/.local/bin/uv", "AIME_PYTHON": os.environ.get("AIME_TEST_PYTHON", "/data/jxc/envs/pbfm_isaaclab/bin/python"),
                "AIME_VLLM_BIN": "/unused/vllm", "AIME_AFLOW_PYTHON": "/unused/aflow/python",
                "AIME_MAAS_PYTHON": "/unused/maas/python", "AIME_GDESIGNER_PYTHON": "/unused/gdesigner/python",
                "AIME_MODEL_PATH": str(tmp_path / "model"),
                "AIME_MINILM_PATH": str(tmp_path / "embedding"), "AIME_DATA_DIR": str(tmp_path / "data"),
                "AIME_MODEL_VERIFICATION_REPORT": str(tmp_path / "verification.json"),
                "AIME_TEMP_DIR": str(tmp_path / "temporary"), "AIME_CACHE_DIR": str(tmp_path / "cache")})
    for name in ("HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "TRITON_CACHE_DIR",
                 "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT", "CUDA_CACHE_PATH", "TMPDIR"):
        env[name] = str(tmp_path / name)
    env.update({"CUDA_VISIBLE_DEVICES": "9", "AIME_MAX_MODEL_LEN": "16384", "AIME_MAX_NUM_SEQS": "8",
                "AIME_MAX_NUM_BATCHED_TOKENS": "16384", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    env_file = tmp_path / "env.sh"
    env_file.write_text("export EXAMPLE=1\n")
    sources = []
    for method in ("aflow", "maas", "gdesigner"):
        source = tmp_path / ("source_" + method)
        source.mkdir()
        sources.append(source)
    output = tmp_path / "output"
    launch_dir = tmp_path / "launch"
    plan = {"package_root": str(root), "env_file": str(env_file), "gpu": "9", "port": 18182,
            "output_root": str(output), "launch_dir": str(launch_dir),
            "runs": [{"method": "aflow", "source_run": str(sources[0]), "run_dir": str(output / "aflow")}]}
    plan_file = tmp_path / "plan.json"

    def save():
        plan_file.write_text(json.dumps(plan))
        return plan_file

    monkeypatch.setattr(launcher, "runtime_environment", lambda _plan: env)
    return root, env, plan, save


def test_plan_rejects_duplicate_methods_existing_source_overlap_and_unauthorized_maas(fixture):
    _, _, plan, save = fixture
    assert launcher.load_plan(save())["runs"][0]["method"] == "aflow"
    aflow = plan["runs"][0]
    plan["runs"] = [aflow, dict(aflow)]
    with pytest.raises(ValueError, match="ordered unique"):
        launcher.load_plan(save())
    plan["runs"] = [dict(aflow, method="maas")]
    with pytest.raises(ValueError, match="allow_stage_reseed"):
        launcher.load_plan(save())
    plan["runs"][0]["allow_stage_reseed"] = True
    assert launcher.load_plan(save())["runs"][0]["allow_stage_reseed"] is True
    plan["runs"][0]["run_dir"] = str(Path(aflow["source_run"]) / "recovery")
    with pytest.raises(ValueError, match="within new output_root"):
        launcher.load_plan(save())


def test_launch_refuses_existing_directory_before_spawn(fixture, monkeypatch):
    _, _, plan, save = fixture
    Path(plan["output_root"]).mkdir()
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not spawn"))
    with pytest.raises(FileExistsError):
        launcher.launch(save())


def test_gdesigner_plan_requires_explicit_boolean_waiver_and_cannot_mix_methods(fixture):
    _, _, plan, save = fixture
    aflow = plan["runs"][0]
    gd = dict(aflow, method="gdesigner", run_dir=str(Path(plan["output_root"]) / "gdesigner"))
    plan["runs"] = [gd]
    for value in (None, False):
        if value is None:
            gd.pop("allow_selection_gate_waiver", None)
        else:
            gd["allow_selection_gate_waiver"] = value
        with pytest.raises(ValueError, match="allow_selection_gate_waiver"):
            launcher.load_plan(save())
    gd["allow_selection_gate_waiver"] = "true"
    with pytest.raises(ValueError, match="explicit boolean"):
        launcher.load_plan(save())
    gd["allow_selection_gate_waiver"] = True
    assert launcher.load_plan(save())["runs"][0]["allow_selection_gate_waiver"] is True
    for runs in ([gd, aflow], [gd, dict(gd)]):
        plan["runs"] = runs
        with pytest.raises(ValueError, match="ordered unique"):
            launcher.load_plan(save())
    plan["runs"] = [dict(aflow, allow_selection_gate_waiver=True)]
    with pytest.raises(ValueError, match="only for gdesigner"):
        launcher.load_plan(save())


def test_server_and_core_commands_preserve_protocol_and_only_recover_2026(fixture):
    _, env, plan, _ = fixture
    server = launcher.server_command(plan, env)
    expected = {"--dtype": "bfloat16", "--max-model-len": "16384", "--max-num-seqs": "8",
                "--tensor-parallel-size": "1", "--gpu-memory-utilization": "0.92"}
    for flag, value in expected.items():
        assert server[server.index(flag) + 1] == value
    core = launcher.core_command(plan, env, plan["runs"][0], inspect=False)
    assert core[:6] == [env["AIME_UV_BIN"], "run", "--no-project", "--python", env["AIME_AFLOW_PYTHON"], "python"]
    assert "recover_aime_tests.py" in " ".join(core)
    assert "--inspect-only" not in core and "--allow-stage-reseed" not in core
    assert "native_aime_formal.py" not in " ".join(core)
    assert "--inspect-only" in launcher.core_command(plan, env, plan["runs"][0], inspect=True)


def test_gdesigner_core_dispatches_only_its_authorized_supplementary_entry(fixture):
    root, env, plan, _ = fixture
    entry = dict(plan["runs"][0], method="gdesigner", allow_selection_gate_waiver=True)
    for inspect in (False, True):
        core = launcher.core_command(plan, env, entry, inspect=inspect)
        assert core[:6] == [env["AIME_UV_BIN"], "run", "--no-project", "--python", env["AIME_GDESIGNER_PYTHON"], "python"]
        assert str(root / "experiments/recover_gdesigner_tests.py") in core
        assert str(root / "experiments/recover_aime_tests.py") not in core
        assert "--allow-selection-gate-waiver" in core
        assert "--allow-stage-reseed" not in core
        assert ("--inspect-only" in core) is inspect


def test_gdesigner_runtime_requires_method_python_and_preserves_request_timeout(fixture):
    root, env, plan, _ = fixture
    plan["runs"][0].update(method="gdesigner", allow_selection_gate_waiver=True)

    def save_env(values):
        Path(plan["env_file"]).write_text("\n".join(f"export {key}={shlex.quote(value)}" for key, value in values.items()) + "\n")

    save_env(env)
    runtime = build_runtime_environment(plan)
    assert runtime["AIME_GDESIGNER_PYTHON"] == env["AIME_GDESIGNER_PYTHON"]
    assert runtime["AIME_GDESIGNER_SOURCE"] == str(root / "upstream/GDesigner")
    assert runtime["AIME_EXTERNAL_REQUEST_TIMEOUT_S"] == "600"
    changed = dict(env, AIME_EXTERNAL_REQUEST_TIMEOUT_S="180")
    save_env(changed)
    with pytest.raises(ValueError, match="AIME_EXTERNAL_REQUEST_TIMEOUT_S=600"):
        build_runtime_environment(plan)
    missing = dict(env)
    missing.pop("AIME_GDESIGNER_PYTHON")
    save_env(missing)
    with pytest.raises(ValueError, match="AIME_GDESIGNER_PYTHON"):
        build_runtime_environment(plan)


def test_gpu_guard_rejects_foreign_process_and_busy_port(fixture, monkeypatch):
    _, env, plan, _ = fixture
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        bound.listen()
        plan["port"] = bound.getsockname()[1]
        with pytest.raises(OSError):
            launcher.guard_gpu_port(plan, env)
    plan["port"] = bound.getsockname()[1] if bound.fileno() >= 0 else 18182
    responses = iter(["9, GPU-test, 0, 49152\n", "22222\n"])
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, next(responses), ""))
    with pytest.raises(ValueError, match="occupied"):
        launcher.guard_gpu_port(plan, env)


def test_model_report_rejects_pinned_hash_mismatch(fixture):
    root, env, _, _ = fixture
    (root / "deployment").mkdir()
    manifest = {kind: {"revision": kind + "revision", "files": {"weights": [1, kind + "hash"]}}
                for kind in ("qwen", "embedding")}
    manifest_path = root / "deployment/model_revisions.json"
    manifest_path.write_text(json.dumps(manifest))
    report = {"manifest_sha256": launcher.sha256(manifest_path), "models": {}}
    for kind, key in (("qwen", "AIME_MODEL_PATH"), ("embedding", "AIME_MINILM_PATH")):
        Path(env[key]).mkdir()
        (Path(env[key]) / "weights").write_bytes(b"x")
        report["models"][kind] = {"path": env[key], "revision": kind + "revision", "sha256_verified": {"weights": kind + "hash"}}
    Path(env["AIME_MODEL_VERIFICATION_REPORT"]).write_text(json.dumps(report))
    assert launcher.validate_model_report(env)["manifest_sha256"] == report["manifest_sha256"]
    report["models"]["qwen"]["sha256_verified"]["weights"] = "wrong"
    Path(env["AIME_MODEL_VERIFICATION_REPORT"]).write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not verified"):
        launcher.validate_model_report(env)


def configure_fake_worker(fixture, monkeypatch, codes=None):
    _, _, plan, save = fixture
    Path(plan["output_root"]).mkdir()
    Path(plan["launch_dir"]).mkdir()
    events = []

    class FakeProcess:
        def __init__(self, command, env, log):
            self.command = command
            self.identity = {"pid": 999999, "starttime_ticks": 1, "session": 999999}
            self.process = type("P", (), {"pid": 999999})()
            log.write_text("CPU fixture, no model requests\n")
            if any("record_run_session.py" in part for part in command):
                Path(command[command.index("--output") + 1]).write_text('{"schema":"CPU fixture"}\n')
            events.append(("start", command))

        def wait(self):
            is_core = any("recover_aime_tests.py" in part or "recover_gdesigner_tests.py" in part for part in self.command)
            if is_core and "--inspect-only" not in self.command:
                method = self.command[self.command.index("--method") + 1]
                return (codes or {}).get(method, 0)
            return 0

        def poll(self):
            return None

        def close(self):
            events.append(("close", self.command))

    monkeypatch.setattr(launcher, "OwnedProcess", FakeProcess)
    monkeypatch.setattr(launcher, "guard_gpu_port", lambda p, e: {"compute_pids": []})
    monkeypatch.setattr(launcher, "validate_model_report", lambda e: {"verified": True})
    monkeypatch.setattr(launcher, "wait_ready", lambda *a: None)
    monkeypatch.setattr(launcher, "warmup", lambda e: {"status": "passed", "max_tokens": 16, "timeout_s": 300})
    monkeypatch.setattr(launcher.signal, "signal", lambda *a: None)
    return plan, save, events


def test_aflow_only_worker_completes_supplemental_and_closes_own_service(fixture, monkeypatch):
    plan, save, events = configure_fake_worker(fixture, monkeypatch)
    assert launcher.worker(save()) == 0
    completion = json.loads((Path(plan["launch_dir"]) / "completion.json").read_text())
    assert completion["status"] == "supplemental_completed"
    assert completion["formal_status"] == "NOT_FORMAL_PASS"
    assert list(completion["methods"]) == ["aflow"]
    assert completion["run_session_record_sha256"] == launcher.sha256(Path(completion["run_session_record"]))
    computes = [command for event, command in events if event == "start" and any("recover_aime_tests.py" in x for x in command) and "--inspect-only" not in command]
    assert len(computes) == 1
    assert any(event == "close" and command[0] == "/unused/vllm" for event, command in events)
    assert not any("aime_quality_gate" in " ".join(command) for _, command in events)


def test_failure_is_preserved_and_next_independent_method_runs_once(fixture, monkeypatch):
    _, _, initial, _ = fixture
    second = dict(initial["runs"][0], method="maas", run_dir=str(Path(initial["output_root"]) / "maas"), allow_stage_reseed=True)
    initial["runs"].append(second)
    plan, save, events = configure_fake_worker(fixture, monkeypatch, {"aflow": 1})
    assert launcher.worker(save()) == 1
    completion = json.loads((Path(plan["launch_dir"]) / "completion.json").read_text())
    assert completion["methods"]["aflow"]["status"] == "recovery_failed"
    assert completion["methods"]["maas"]["status"] == "supplemental_completed"
    computes = [command for event, command in events if event == "start" and any("recover_aime_tests.py" in x for x in command) and "--inspect-only" not in command]
    assert [cmd[cmd.index("--method") + 1] for cmd in computes] == ["aflow", "maas"]


def test_gdesigner_only_worker_evaluates_once_and_stays_not_formal_pass(fixture, monkeypatch):
    _, _, initial, _ = fixture
    initial["runs"][0].update(method="gdesigner", allow_selection_gate_waiver=True)
    plan, save, events = configure_fake_worker(fixture, monkeypatch)
    assert launcher.worker(save()) == 0
    completion = json.loads((Path(plan["launch_dir"]) / "completion.json").read_text())
    assert completion["status"] == "supplemental_completed"
    assert completion["formal_status"] == "NOT_FORMAL_PASS"
    assert list(completion["methods"]) == ["gdesigner"]
    cores = [command for event, command in events if event == "start" and any("recover_gdesigner_tests.py" in part for part in command)]
    assert len(cores) == 2
    assert "--inspect-only" in cores[0] and "--inspect-only" not in cores[1]
    assert all("--allow-selection-gate-waiver" in command for command in cores)
    assert not any("recover_aime_tests.py" in " ".join(command) for _, command in events)
    assert not any("aime_quality_gate" in " ".join(command) for _, command in events)
    assert any(event == "close" and command[0] == "/unused/vllm" for event, command in events)


def test_startup_failure_records_failure_and_cleanup_without_native_compute(fixture, monkeypatch):
    plan, save, events = configure_fake_worker(fixture, monkeypatch)
    monkeypatch.setattr(launcher, "warmup", lambda e: (_ for _ in ()).throw(TimeoutError("warmup timeout")))
    assert launcher.worker(save()) == 1
    completion = json.loads((Path(plan["launch_dir"]) / "completion.json").read_text())
    assert completion["status"] == "failed" and "warmup timeout" in completion["error"]
    assert not any(event == "start" and any("recover_aime_tests.py" in x for x in command) and "--inspect-only" not in command for event, command in events)
    assert sum(event == "close" for event, _ in events) == sum(event == "start" for event, _ in events)


def test_cleanup_does_not_signal_reused_or_foreign_pid(monkeypatch):
    process = object.__new__(launcher.OwnedProcess)
    process.identity = {"pid": 101, "starttime_ticks": 11}
    process.members = {101: {"pid": 101, "starttime_ticks": 11}, 102: {"pid": 102, "starttime_ticks": 12}}
    process.capture = lambda: None
    process.process = type("P", (), {"pid": 101, "wait": lambda s, **kw: 0})()
    process.log_stream = type("L", (), {"close": lambda s: None})()
    monkeypatch.setattr(launcher, "proc_identity", lambda pid: {"pid": pid, "starttime_ticks": 999, "session": 101, "state": "Z"})
    monkeypatch.setattr(launcher.os, "kill", lambda *args: pytest.fail("must not signal reused pid"))
    process.close()


def test_owned_process_is_detached_and_uses_closed_stdin(tmp_path):
    env = dict(os.environ)
    env["AIME_UV_BIN"] = "/home/ZZD/.local/bin/uv"
    python = os.environ.get("AIME_TEST_PYTHON", "/data/jxc/envs/pbfm_isaaclab/bin/python")
    command = launcher.uv_python(env, python, "-u", "-c", "import json,os,sys; print(json.dumps({'sid':os.getsid(0),'stdin':sys.stdin.read()}))")
    process = launcher.OwnedProcess(command, env, tmp_path / "child.log")
    pid = process.process.pid
    assert process.identity["session"] == pid
    assert process.wait() == 0
    process.close()
    payload = json.loads((tmp_path / "child.log").read_text())
    assert payload == {"sid": pid, "stdin": ""}


def test_launch_worker_survives_parent_exit_with_file_log(fixture):
    root, env, plan, save = fixture
    (root / "scripts").mkdir()
    (root / "scripts/launch_recovery_detached.sh").write_text("CPU fixture\n")
    (root / "experiments").mkdir()
    (root / "experiments/recover_aime_tests.py").write_text("CPU fixture\n")
    Path(plan["env_file"]).write_text("\n".join(f"export {key}={shlex.quote(value)}" for key, value in env.items()) + "\n")
    plan_path = save()
    child_code = "import json,os,pathlib,sys,time; pathlib.Path(sys.argv[1]).joinpath('alive.json').write_text(json.dumps({'sid':os.getsid(0),'stdin':sys.stdin.read()})); print('DETACHED_CPU_WORKER',flush=True); time.sleep(1)"
    parent_code = """
import importlib.util,pathlib,sys
spec=importlib.util.spec_from_file_location('detach_fixture',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ROOT=pathlib.Path(sys.argv[2])
module.subprocess.check_output=lambda *args,**kwargs:'cpu-fixture-commit'
module.uv_python=lambda env,python,*args:[env['AIME_UV_BIN'],'run','--no-project','--python',python,'python','-u','-c',sys.argv[4],sys.argv[5]]
raise SystemExit(module.launch(pathlib.Path(sys.argv[3])))
"""
    parent = subprocess.run([env["AIME_UV_BIN"], "run", "--no-project", "--python", env["AIME_PYTHON"], "python",
                             "-c", parent_code, str(SCRIPT), str(root), str(plan_path), child_code, plan["launch_dir"]],
                            check=True, capture_output=True, text=True, timeout=10)
    record = json.loads(parent.stdout)
    child_pid = record["pid"]
    deadline = time.monotonic() + 5
    alive = Path(plan["launch_dir"]) / "alive.json"
    while not alive.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert alive.exists(), parent.stderr
    assert launcher.proc_identity(child_pid)["session"] == child_pid
    assert json.loads(alive.read_text()) == {"sid": child_pid, "stdin": ""}
    assert "DETACHED_CPU_WORKER" in (Path(plan["launch_dir"]) / "run.log").read_text()
    assert json.loads((Path(plan["launch_dir"]) / "launch.json").read_text())["formal_status"] == "NOT_FORMAL_PASS"
