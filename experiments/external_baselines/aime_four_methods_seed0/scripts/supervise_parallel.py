#!/usr/bin/env python3
"""Observe independent AIME runs, retire the paused dispatcher, and aggregate only verified completions."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


METHODS = ("maas", "adas", "aflow", "gdesigner")
LAYOUT = {"maas": "aflow_maas/maas/seed_0", "aflow": "aflow_maas/aflow/seed_0",
          "adas": "adas_gdesigner/adas/seed_0", "gdesigner": "adas_gdesigner/gdesigner/seed_0"}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def proc_snapshot(pid: int) -> dict | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rindex(")") + 2:].split()
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        return {"pid": pid, "state": fields[0], "starttime_ticks": int(fields[19]), "cmdline": cmd}
    except FileNotFoundError:
        return None


def matches(identity: dict, current: dict | None) -> bool:
    return bool(current and int(identity["starttime_ticks"]) == current["starttime_ticks"]
                and identity["cmd_contains"] in current["cmdline"])


def process_finished(identity: dict) -> bool:
    current = proc_snapshot(int(identity["pid"]))
    # A reused PID is not this original process and must never receive a signal.
    return current is None or current["starttime_ticks"] != int(identity["starttime_ticks"]) or current["state"] == "Z"


def require_queue(identity: dict, *, stopped: bool) -> dict:
    current = proc_snapshot(int(identity["pid"]))
    if not matches(identity, current):
        raise RuntimeError("Old dispatcher identity changed or vanished; refusing to signal it")
    if stopped and current["state"] not in ("T", "t"):
        raise RuntimeError("Old dispatcher is not stopped; refusing retirement signals")
    return current


def retire_queue(identity: dict, *, grace_seconds: float = 30) -> None:
    require_queue(identity, stopped=True)
    os.kill(int(identity["pid"]), signal.SIGTERM)
    # SIGTERM is pending for a stopped bash; SIGCONT permits its exit trap.
    # Signal only this dispatcher PID, never its child or process group.
    require_queue(identity, stopped=True)
    os.kill(int(identity["pid"]), signal.SIGCONT)
    deadline = time.monotonic() + grace_seconds
    while not process_finished(identity):
        if time.monotonic() >= deadline:
            raise RuntimeError("Old dispatcher did not exit after TERM/CONT; no forceful cleanup performed")
        time.sleep(0.1)


def validate_plan(plan: dict) -> None:
    for key in ("old_queue", "old_maas_run_pair", "old_maas_server"):
        identity = plan[key]
        if int(identity["pid"]) <= 1 or int(identity["starttime_ticks"]) <= 0 or not identity["cmd_contains"]:
            raise ValueError(f"Invalid process identity: {key}")
    if len({int(plan[key]["pid"]) for key in ("old_queue", "old_maas_run_pair", "old_maas_server")}) != 3:
        raise ValueError("Dispatcher, MaAS launcher, and server PIDs must be distinct")
    require_queue(plan["old_queue"], stopped=False)
    if not Path(plan["package_root"]).is_dir() or not Path(plan["env_file"]).is_file():
        raise ValueError("Package root and common environment file must exist")
    if set(plan["methods"]) != {"adas", "aflow", "gdesigner"}:
        raise ValueError("Independent methods must be ADAS, AFlow, and G-Designer")
    for method, spec in plan["methods"].items():
        for key in ("launch_dir", "output_root", "run_dir", "launcher_result", "cache_dir", "temp_dir", "diagnostic_dir"):
            if not Path(spec[key]).is_absolute():
                raise ValueError(f"{method}.{key} must be absolute")
        if not 1 <= int(spec["port"]) <= 65535 or str(spec["gpu"]) == "":
            raise ValueError(f"Invalid GPU/port for {method}")
    for field in ("port", "gpu", "launch_dir", "output_root", "cache_dir", "temp_dir", "diagnostic_dir"):
        if len({str(spec[field]) for spec in plan["methods"].values()}) != 3:
            raise ValueError(f"New methods must have distinct {field}")
    roots = [Path(plan["aggregate_root"]), Path(plan["audit_dir"])]
    if any(not path.is_absolute() for path in roots) or roots[0] == roots[1]:
        raise ValueError("Aggregate and audit roots must be distinct absolute paths")
    if roots[0].exists():
        raise FileExistsError("Aggregate root must be fresh")
    if not Path(plan["maas"]["run_dir"]).is_dir():
        raise ValueError("Original MaAS run directory must exist")
    if float(plan.get("poll_seconds", 15)) <= 0:
        raise ValueError("poll_seconds must be positive")
    if "old_maas_gpu_monitor" in plan:
        monitor = plan["old_maas_gpu_monitor"]
        if int(monitor["pid"]) <= 1 or int(monitor["starttime_ticks"]) <= 0 or not monitor["cmd_contains"]:
            raise ValueError("Invalid old GPU monitor identity")


def formal_status(method: str, spec: dict) -> dict:
    record_path = Path(spec["launcher_result"])
    if not record_path.is_file():
        return {"status": "failed", "reason": "completed launcher has no formal result"}
    record = read_json(record_path)
    passed = (record.get("method") == method and record.get("kind") == "formal"
              and record.get("status") == "passed"
              and all(record.get(key) == 0 for key in ("preflight_exit_code", "compute_exit_code", "quality_gate_exit_code")))
    run_dir = Path(spec["run_dir"])
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    passed = passed and manifest.get("method") == method and manifest.get("status") == "complete"
    passed = passed and (run_dir / "native_result.json").is_file()
    return {"status": "passed" if passed else "failed", "launcher_result": str(record_path),
            "run_dir": str(run_dir), "launcher_status": record.get("status"), "manifest_status": manifest.get("status")}


def independent_status(method: str, spec: dict) -> dict:
    completion = Path(spec["launch_dir"]) / "completion.json"
    if not completion.is_file():
        launch = Path(spec["launch_dir"]) / "launch.json"
        if launch.is_file():
            record = read_json(launch)
            if record.get("starttime_ticks") is None:
                return {"status": "failed", "reason": "worker ended before identity capture and has no completion", "launch_record": str(launch)}
            if all(key in record for key in ("pid", "starttime_ticks", "cmd_contains")) and process_finished(record):
                return {"status": "failed", "reason": "worker exited without a completion record", "launch_record": str(launch)}
        return {"status": "running" if Path(spec["launch_dir"]).exists() else "not_started"}
    record = read_json(completion)
    if record.get("status") != "passed" or record.get("exit_code") != 0:
        return {"status": "failed", "reason": "independent worker failed", "completion": str(completion)}
    result = formal_status(method, spec)
    result["completion"] = str(completion)
    return result


def record_writer_finished(spec: dict) -> bool:
    """A malformed final record is retryable while its writer can still finish it."""
    launch = Path(spec["launch_dir"]) / "launch.json"
    if not launch.is_file():
        return False
    try:
        record = read_json(launch)
    except json.JSONDecodeError:
        return False
    if record.get("starttime_ticks") is None:
        return True
    return all(key in record for key in ("pid", "starttime_ticks", "cmd_contains")) and process_finished(record)


def launch_gdesigner(plan: dict) -> None:
    spec = plan["methods"]["gdesigner"]
    if Path(spec["launch_dir"]).exists() or Path(spec["output_root"]).exists():
        raise FileExistsError("G-Designer launch/output must be fresh; refusing duplicate launch")
    overrides = {
        "CUDA_VISIBLE_DEVICES": str(spec["gpu"]), "AIME_PORT": str(spec["port"]),
        "AIME_CACHE_DIR": spec["cache_dir"], "AIME_TEMP_DIR": spec["temp_dir"],
        "AIME_OUTPUT_ROOT": spec["output_root"], "AIME_LAUNCH_DIR": spec["launch_dir"],
        "AIME_DIAGNOSTIC_DIR": spec["diagnostic_dir"],
        "AIME_SUPERVISOR_WRAPPER": str(Path(plan["package_root"]) / "scripts/launch_single_detached.sh"),
    }
    command = ["bash", "-c", 'set -e; source "$1"; shift; export "$@"; exec bash "$AIME_SUPERVISOR_WRAPPER" --method gdesigner',
               "aime-supervisor", plan["env_file"], *[f"{key}={value}" for key, value in overrides.items()]]
    with (Path(plan["audit_dir"]) / "gdesigner_launch.log").open("x") as log:
        result = subprocess.run(command, cwd=plan["package_root"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"G-Designer detached launch failed with exit {result.returncode}")


def copy_completed_runs(plan: dict, statuses: dict) -> Path:
    if any(statuses.get(method, {}).get("status") != "passed" for method in METHODS):
        raise ValueError("Refusing aggregate copy of incomplete or failed runs")
    root = Path(plan["aggregate_root"])
    root.mkdir(parents=True, exist_ok=False)
    mapping = {"schema": "aime_parallel_copy_provenance_v1", "created_at_utc": utc(), "methods": {}}
    for method in METHODS:
        source = Path(statuses[method]["run_dir"])
        destination = root / LAYOUT[method]
        if any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError(f"Unexpected symlink in completed {method} run; refusing mutable external copy")
        shutil.copytree(source, destination)
        files = {}
        for path in sorted(source.rglob("*")):
            if path.is_file():
                relative = path.relative_to(source)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                copied_digest = hashlib.sha256((destination / relative).read_bytes()).hexdigest()
                if digest != copied_digest:
                    raise RuntimeError(f"Copy checksum mismatch: {method}/{relative}")
                files[str(relative)] = {"sha256": digest, "original": str(path), "copy": str(destination / relative)}
        mapping["methods"][method] = {"original_run_dir": str(source), "copied_run_dir": str(destination), "files": files}
    mapping["note"] = "Hashes describe the preserved source and initial copies. Fresh quality checks may rewrite audit manifests/gate reports in the copies; original run directories remain unchanged."
    atomic_json(root / "parallel_copy_provenance.json", mapping)
    return root


def aggregate(plan: dict) -> int:
    command = ["bash", "-c", 'set -e; source "$1"; "$AIME_UV_BIN" run --no-project --python "$AIME_PYTHON" python "$2" --root "$3"',
               "aime-aggregate", plan["env_file"], str(Path(plan["package_root"]) / "scripts/aggregate_aime_seed0.py"), plan["aggregate_root"]]
    with (Path(plan["audit_dir"]) / "aggregate.log").open("x") as log:
        return subprocess.run(command, cwd=plan["package_root"], stdin=subprocess.DEVNULL,
                              stdout=log, stderr=subprocess.STDOUT).returncode


def supervise(plan: dict) -> int:
    validate_plan(plan)
    require_queue(plan["old_queue"], stopped=True)
    audit = Path(plan["audit_dir"])
    audit.mkdir(parents=True, exist_ok=False)
    atomic_json(audit / "plan.json", plan)
    state = {"schema": "aime_parallel_supervisor_v1", "started_at_utc": utc(), "status": "running",
             "maas_handoff": "waiting", "aggregate_status": "not_run", "methods": {}}

    def event(kind: str, **data) -> None:
        with (audit / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at_utc": utc(), "event": kind, **data}, ensure_ascii=False) + "\n")

    def save() -> None:
        state["updated_at_utc"] = utc()
        atomic_json(audit / "status.json", state)

    event("supervisor_started")
    maas_missing_record_since = None
    maas_invalid_record_since = None
    invalid_record_since = {}
    try:
        while True:
            if state["maas_handoff"] == "waiting":
                require_queue(plan["old_queue"], stopped=True)
                maas_stopped = (process_finished(plan["old_maas_run_pair"]) and process_finished(plan["old_maas_server"])
                                and ("old_maas_gpu_monitor" not in plan or process_finished(plan["old_maas_gpu_monitor"])))
                if maas_stopped and not Path(plan["maas"]["launcher_result"]).is_file():
                    maas_missing_record_since = maas_missing_record_since or time.monotonic()
                ready = maas_stopped and (Path(plan["maas"]["launcher_result"]).is_file()
                                         or (maas_missing_record_since is not None
                                             and time.monotonic() - maas_missing_record_since >= 30))
                if ready:
                    try:
                        observed_maas = formal_status("maas", plan["maas"])
                    except json.JSONDecodeError:
                        maas_invalid_record_since = maas_invalid_record_since or time.monotonic()
                        if time.monotonic() - maas_invalid_record_since < 30:
                            event("record_write_pending", method="maas")
                            save()
                            time.sleep(float(plan.get("poll_seconds", 15)))
                            continue
                        observed_maas = {"status": "failed", "reason": "invalid final record after all MaAS writers exited"}
                    state["methods"]["maas"] = observed_maas
                    event("maas_completed", status=state["methods"]["maas"]["status"])
                    retire_queue(plan["old_queue"])
                    state["maas_handoff"] = "retired"
                    event("old_dispatcher_retired", pid=plan["old_queue"]["pid"])
                    try:
                        launch_gdesigner(plan)
                        event("gdesigner_detached_started")
                    except Exception as error:
                        state["methods"]["gdesigner"] = {"status": "failed", "reason": str(error)}
                        event("gdesigner_launch_failed", error=str(error))
                else:
                    state["methods"]["maas"] = {"status": "running", "run_dir": plan["maas"]["run_dir"]}
            for method in ("adas", "aflow", "gdesigner"):
                if state["methods"].get(method, {}).get("status") in ("passed", "failed"):
                    continue
                try:
                    observed = independent_status(method, plan["methods"][method])
                    invalid_record_since.pop(method, None)
                except json.JSONDecodeError:
                    observed = {"status": "running", "reason": "record write pending"}
                    if record_writer_finished(plan["methods"][method]):
                        invalid_record_since.setdefault(method, time.monotonic())
                        if time.monotonic() - invalid_record_since[method] >= 30:
                            observed = {"status": "failed", "reason": "invalid final record after worker exited"}
                if observed != state["methods"].get(method):
                    event("method_status", method=method, status=observed["status"])
                state["methods"][method] = observed
            save()
            if state["maas_handoff"] == "retired" and all(state["methods"].get(m, {}).get("status") in ("passed", "failed") for m in METHODS):
                break
            time.sleep(float(plan.get("poll_seconds", 15)))
        if not all(state["methods"][m]["status"] == "passed" for m in METHODS):
            state.update(status="partial", aggregate_status="blocked_incomplete", finished_at_utc=utc())
            event("aggregate_blocked_incomplete")
            save()
            return 1
        copy_completed_runs(plan, state["methods"])
        code = aggregate(plan)
        state.update(status="passed" if code == 0 else "failed", aggregate_status="passed" if code == 0 else "failed",
                     aggregate_exit_code=code, finished_at_utc=utc())
        event("aggregate_finished", exit_code=code)
        save()
        return 0 if code == 0 else 1
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at_utc=utc())
        event("supervisor_failed", error=state["error"])
        save()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--check-plan", action="store_true")
    args = parser.parse_args()
    plan = read_json(args.plan)
    if args.check_plan:
        validate_plan(plan)
        print("PARALLEL_PLAN_VALID", flush=True)
        return 0
    return supervise(plan)


if __name__ == "__main__":
    raise SystemExit(main())
