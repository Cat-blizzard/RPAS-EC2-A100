"""CPU tests for dispatcher retirement and completion-only aggregation."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/supervise_parallel.py"
spec = importlib.util.spec_from_file_location("aime_parallel_supervisor", SCRIPT)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def identity(process):
    current = supervisor.proc_snapshot(process.pid)
    return {"pid": process.pid, "starttime_ticks": current["starttime_ticks"], "cmd_contains": "bash"}


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="Linux /proc required")
def test_stopped_dispatcher_retirement_preserves_child_and_never_runs_next_method(tmp_path):
    child_done = tmp_path / "child_done"
    after = tmp_path / "should_never_start"
    script = tmp_path / "queue.sh"
    script.write_text('trap "exit 143" TERM\nbash -c \'sleep 0.3; touch "$1"\' child "$1" &\nwait\ntouch "$2"\n')
    parent = subprocess.Popen(["bash", str(script), str(child_done), str(after)], start_new_session=True,
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ident = identity(parent)
    try:
        time.sleep(0.08)
        os.kill(parent.pid, signal.SIGSTOP)
        deadline = time.monotonic() + 3
        while not child_done.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_done.exists()
        assert not after.exists()
        assert supervisor.proc_snapshot(parent.pid)["state"] == "T"
        supervisor.retire_queue(ident, grace_seconds=3)
        assert parent.wait(timeout=2) == 143
        assert child_done.exists()
        assert not after.exists()
    finally:
        if parent.poll() is None:
            os.kill(parent.pid, signal.SIGTERM)
            os.kill(parent.pid, signal.SIGCONT)
            parent.wait(timeout=3)


def test_retirement_rejects_pid_reuse_before_any_signal(monkeypatch):
    monkeypatch.setattr(supervisor, "proc_snapshot", lambda pid: {"pid": pid, "starttime_ticks": 999, "state": "T", "cmdline": "bash queue.sh"})
    calls = []
    monkeypatch.setattr(supervisor.os, "kill", lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="identity"):
        supervisor.retire_queue({"pid": 100, "starttime_ticks": 111, "cmd_contains": "bash"})
    assert calls == []


def test_retirement_rejects_wrong_command_or_nonstopped_process(monkeypatch):
    monkeypatch.setattr(supervisor, "proc_snapshot", lambda pid: {"pid": pid, "starttime_ticks": 111, "state": "S", "cmdline": "bash queue.sh"})
    calls = []
    monkeypatch.setattr(supervisor.os, "kill", lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="identity"):
        supervisor.retire_queue({"pid": 100, "starttime_ticks": 111, "cmd_contains": "unrelated.py"})
    with pytest.raises(RuntimeError, match="not stopped"):
        supervisor.retire_queue({"pid": 100, "starttime_ticks": 111, "cmd_contains": "bash"})
    assert calls == []


def passed_run(tmp_path, method):
    root = tmp_path / method
    run = root / "seed_0"
    run.mkdir(parents=True)
    write_json(run / "run_manifest.json", {"method": method, "status": "complete"})
    write_json(run / "native_result.json", {"synthetic_fixture": True})
    launcher = root / "launcher_result.json"
    write_json(launcher, {"method": method, "kind": "formal", "status": "passed", "preflight_exit_code": 0,
                          "compute_exit_code": 0, "quality_gate_exit_code": 0})
    return {"run_dir": str(run), "launcher_result": str(launcher), "launch_dir": str(root / "launch")}


def test_completion_required_even_if_formal_record_passed(tmp_path):
    spec = passed_run(tmp_path, "adas")
    assert supervisor.independent_status("adas", spec)["status"] == "not_started"
    write_json(Path(spec["launch_dir"]) / "completion.json", {"status": "passed", "exit_code": 0})
    assert supervisor.independent_status("adas", spec)["status"] == "passed"
    write_json(Path(spec["run_dir"]) / "run_manifest.json", {"method": "adas", "status": "running"})
    assert supervisor.independent_status("adas", spec)["status"] == "failed"


def test_quality_gate_failure_blocks_aggregate_copy_and_preserves_sources(tmp_path):
    spec = passed_run(tmp_path, "aflow")
    record = supervisor.read_json(spec["launcher_result"])
    record["quality_gate_exit_code"] = 1
    write_json(Path(spec["launcher_result"]), record)
    assert supervisor.formal_status("aflow", spec)["status"] == "failed"
    aggregate = tmp_path / "aggregate"
    with pytest.raises(ValueError, match="incomplete"):
        supervisor.copy_completed_runs({"aggregate_root": str(aggregate)}, {"aflow": {"status": "failed"}})
    assert not aggregate.exists()
    assert Path(spec["run_dir"]).is_dir()


def test_completed_copy_layout_and_sha_mapping(tmp_path):
    statuses = {}
    for method in supervisor.METHODS:
        spec = passed_run(tmp_path, method)
        statuses[method] = supervisor.formal_status(method, spec)
    plan = {"aggregate_root": str(tmp_path / "aggregate")}
    root = supervisor.copy_completed_runs(plan, statuses)
    mapping = supervisor.read_json(root / "parallel_copy_provenance.json")
    for method in supervisor.METHODS:
        assert (root / supervisor.LAYOUT[method] / "native_result.json").is_file()
        assert Path(statuses[method]["run_dir"]).is_dir()
        entry = mapping["methods"][method]["files"]["native_result.json"]
        assert len(entry["sha256"]) == 64
        assert entry["original"] != entry["copy"]
    with pytest.raises(FileExistsError):
        supervisor.copy_completed_runs(plan, statuses)


def test_dead_worker_without_completion_is_failure(tmp_path, monkeypatch):
    spec = passed_run(tmp_path, "adas")
    write_json(Path(spec["launch_dir"]) / "launch.json", {"pid": 555, "starttime_ticks": 123, "cmd_contains": "single_detached"})
    monkeypatch.setattr(supervisor, "proc_snapshot", lambda pid: None)
    assert supervisor.independent_status("adas", spec)["status"] == "failed"


def test_worker_gone_before_identity_capture_does_not_crash_supervisor(tmp_path):
    spec = passed_run(tmp_path, "adas")
    write_json(Path(spec["launch_dir"]) / "launch.json", {"pid": 555, "starttime_ticks": None,
                                                           "cmd_contains": "single_detached"})
    assert supervisor.independent_status("adas", spec)["status"] == "failed"
    assert supervisor.record_writer_finished(spec) is True
    write_json(Path(spec["launch_dir"]) / "completion.json", {"status": "passed", "exit_code": 0})
    assert supervisor.independent_status("adas", spec)["status"] == "passed"


def test_maas_missing_formal_record_is_rejected(tmp_path):
    assert supervisor.formal_status("maas", {"launcher_result": str(tmp_path / "absent.json"),
                                             "run_dir": str(tmp_path / "absent")})["status"] == "failed"


def test_failed_maas_still_hands_gpu_to_gdesigner_and_blocks_aggregate(tmp_path, monkeypatch):
    maas = passed_run(tmp_path, "maas")
    record = supervisor.read_json(maas["launcher_result"])
    record.update(status="failed", quality_gate_exit_code=1)
    write_json(Path(maas["launcher_result"]), record)
    methods = {method: passed_run(tmp_path, method) for method in ("adas", "aflow", "gdesigner")}
    for method in ("adas", "aflow"):
        write_json(Path(methods[method]["launch_dir"]) / "completion.json", {"status": "passed", "exit_code": 0})
    plan = {"maas": maas, "methods": methods, "old_queue": {"pid": 11}, "old_maas_run_pair": {"pid": 12},
            "old_maas_server": {"pid": 13}, "audit_dir": str(tmp_path / "audit"),
            "aggregate_root": str(tmp_path / "aggregate"), "poll_seconds": 0.001}
    actions = []
    monkeypatch.setattr(supervisor, "validate_plan", lambda plan: None)
    monkeypatch.setattr(supervisor, "require_queue", lambda *args, **kwargs: {})
    monkeypatch.setattr(supervisor, "process_finished", lambda identity: True)
    monkeypatch.setattr(supervisor, "retire_queue", lambda identity: actions.append("retire"))

    def start_gdesigner(plan):
        actions.append("gdesigner")
        write_json(Path(methods["gdesigner"]["launch_dir"]) / "completion.json", {"status": "passed", "exit_code": 0})

    monkeypatch.setattr(supervisor, "launch_gdesigner", start_gdesigner)
    monkeypatch.setattr(supervisor, "aggregate", lambda plan: pytest.fail("Incomplete set must never aggregate"))
    assert supervisor.supervise(plan) == 1
    assert actions == ["retire", "gdesigner"]
    state = supervisor.read_json(Path(plan["audit_dir"]) / "status.json")
    assert state["aggregate_status"] == "blocked_incomplete"
    assert state["methods"]["maas"]["status"] == "failed"
    assert state["methods"]["gdesigner"]["status"] == "passed"
    assert not Path(plan["aggregate_root"]).exists()
