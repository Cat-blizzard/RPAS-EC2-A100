#!/usr/bin/env python3
"""Detach a guarded server and recover previously unopened frozen AIME tests.

Contract: a fresh output and launch directory, one local GPU, one local port,
and an ordered nonempty subset of [aflow, maas], or G-Designer alone with an
explicit selection-gate waiver. The frozen source runs are
read-only; recover_aime_tests owns durable source claims and test provenance.
Completion is supplemental/NOT_FORMAL_PASS even when all recoveries succeed.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
LOCK_ROOT = Path("/data/jxc/tmp/rpas/aime/recovery_locks")
MODEL = "Qwen/Qwen3.5-9B"
PROTOCOL = {"context": 16384, "max_tokens": 6144, "concurrency": 8,
            "max_num_seqs": 8, "tensor_parallel_size": 1, "dtype": "bfloat16",
            "max_num_batched_tokens": 16384, "gpu_memory_utilization": 0.92,
            "search_seed": 0, "data_seed": 2026, "D_test_each_year": 30}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> None:
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def proc_identity(pid: int) -> dict[str, Any] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "starttime_ticks": int(fields[19]),
                "session": int(fields[3]), "state": fields[0],
                "cmdline": Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()}
    except (OSError, ValueError, IndexError):
        return None


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    for key in ("package_root", "env_file", "output_root", "launch_dir"):
        value = Path(plan[key]).expanduser()
        if not value.is_absolute():
            raise ValueError(f"{key} must be absolute")
        plan[key] = str(value.resolve())
    if Path(plan["package_root"]) != ROOT:
        raise ValueError("package_root must identify this launcher's package")
    if not Path(plan["env_file"]).is_file():
        raise ValueError("env_file must exist")
    gpu = str(plan["gpu"])
    if not gpu.isdigit():
        raise ValueError("gpu must be one physical index, not a list")
    plan["gpu"] = gpu
    port = plan["port"]
    if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535:
        raise ValueError("port must be an integer between 1024 and 65535")
    methods = [item["method"] for item in plan["runs"]]
    if methods not in (["aflow"], ["maas"], ["aflow", "maas"], ["gdesigner"]):
        raise ValueError("runs must be a nonempty ordered unique subset of aflow, maas, or gdesigner alone")
    output, launch = Path(plan["output_root"]), Path(plan["launch_dir"])
    if output == launch or output in launch.parents or launch in output.parents:
        raise ValueError("output_root and launch_dir must be independent fresh directories")
    seen: set[str] = set()
    for item in plan["runs"]:
        for key in ("source_run", "run_dir"):
            value = Path(item[key]).expanduser()
            if not value.is_absolute():
                raise ValueError(f"{key} must be absolute")
            item[key] = str(value.resolve())
        source, target = Path(item["source_run"]), Path(item["run_dir"])
        if not source.is_dir():
            raise ValueError(f"source_run missing: {source}")
        if output not in target.parents or source == target or source in target.parents:
            raise ValueError("run_dir must be strictly within new output_root and outside source_run")
        if str(target) in seen:
            raise ValueError("duplicate recovery run_dir")
        seen.add(str(target))
        if "allow_stage_reseed" in item and not isinstance(item["allow_stage_reseed"], bool):
            raise ValueError("allow_stage_reseed must be an explicit boolean")
        if item["method"] == "maas" and item.get("allow_stage_reseed") is not True:
            raise ValueError("MaAS requires explicit allow_stage_reseed=true authorization")
        if "allow_selection_gate_waiver" in item and not isinstance(item["allow_selection_gate_waiver"], bool):
            raise ValueError("allow_selection_gate_waiver must be an explicit boolean")
        if item["method"] == "gdesigner" and item.get("allow_selection_gate_waiver") is not True:
            raise ValueError("G-Designer requires explicit allow_selection_gate_waiver=true authorization")
        if item["method"] != "gdesigner" and item.get("allow_selection_gate_waiver") is True:
            raise ValueError("allow_selection_gate_waiver is authorized only for gdesigner")
    return plan


def runtime_environment(plan: dict[str, Any]) -> dict[str, str]:
    completed = subprocess.run(
        ["bash", "-c", 'set -ae; source "$1"; env -0', "bash", plan["env_file"]],
        check=True, capture_output=True)
    env = {entry.split(b"=", 1)[0].decode(): entry.split(b"=", 1)[1].decode()
           for entry in completed.stdout.split(b"\0") if b"=" in entry}
    for name in ("AIME_UV_BIN", "AIME_PYTHON", "AIME_VLLM_BIN", "AIME_AFLOW_PYTHON",
                 "AIME_MAAS_PYTHON", "AIME_MODEL_PATH", "AIME_MINILM_PATH",
                 "AIME_MODEL_VERIFICATION_REPORT", "AIME_DATA_DIR"):
        if not env.get(name):
            raise ValueError(f"env_file must provide {name}")
    if any(entry["method"] == "gdesigner" for entry in plan["runs"]) and not env.get("AIME_GDESIGNER_PYTHON"):
        raise ValueError("env_file must provide AIME_GDESIGNER_PYTHON")
    env.update({"CUDA_VISIBLE_DEVICES": plan["gpu"], "AIME_PORT": str(plan["port"]),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "OPENAI_API_KEY": env.get("OPENAI_API_KEY", "EMPTY"),
                "AIME_TOKENIZER_PATH": env["AIME_MODEL_PATH"],
                "AIME_MAAS_EMBEDDING_PATH": env["AIME_MINILM_PATH"],
                "AIME_MAX_MODEL_LEN": "16384", "AIME_MAX_NUM_SEQS": "8", "AIME_TP_SIZE": "1",
                "AIME_MAX_NUM_BATCHED_TOKENS": "16384", "AIME_PAIR": "supplemental_recovery",
                "RPAS_SERVE_PYTHON": env["AIME_PYTHON"],
                "RPAS_AFLOW_PYTHON": env["AIME_AFLOW_PYTHON"],
                "RPAS_MAAS_PYTHON": env["AIME_MAAS_PYTHON"],
                "RPAS_VLLM": env["AIME_VLLM_BIN"], "RPAS_MODEL_PATH": env["AIME_MODEL_PATH"],
                "RPAS_MAAS_EMBEDDING_PATH": env["AIME_MINILM_PATH"],
                "RPAS_MODEL_VERIFICATION_REPORT": env["AIME_MODEL_VERIFICATION_REPORT"],
                "RPAS_MODEL_REVISION": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                "RPAS_MAAS_EMBEDDING_REVISION": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"})
    for method, folder in (("AFLOW", "AFlow"), ("MAAS", "MaAS"), ("GDESIGNER", "GDesigner")):
        env[f"AIME_{method}_SOURCE"] = str(ROOT / "upstream" / folder)
    defaults = {"AIME_AFLOW_REQUEST_TIMEOUT_S": "600", "AIME_AFLOW_SAMPLE_TIMEOUT_S": "1200",
                "AIME_AFLOW_CODE_TIMEOUT_S": "30", "AIME_MAAS_REQUEST_TIMEOUT_S": "600",
                "AIME_MAAS_SAMPLE_TIMEOUT_S": "1500", "AIME_MAAS_SAMPLE_RETRIES": "3",
                "AIME_MAAS_CODE_TIMEOUT_S": "60", "AIME_EXTERNAL_REQUEST_TIMEOUT_S": "600"}
    for key, expected in defaults.items():
        if env.get(key, expected) != expected:
            raise ValueError(f"recovery must preserve {key}={expected}")
        env[key] = expected
    env["PYTHONPATH"] = str(ROOT / "experiments") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cache = Path(env.get("AIME_CACHE_DIR", "/data/jxc/cache/rpas"))
    for name, suffix in (("HF_HOME", "huggingface"), ("XDG_CACHE_HOME", "xdg"), ("TORCH_HOME", "torch"),
                         ("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
                         ("VLLM_CACHE_ROOT", "vllm"), ("CUDA_CACHE_PATH", "cuda")):
        env[name] = str(cache / suffix)
    env["TMPDIR"] = env.get("AIME_TEMP_DIR", "/data/jxc/tmp/rpas/aime")
    proxy_exclusions = env.get("NO_PROXY", env.get("no_proxy", ""))
    env["NO_PROXY"] = (proxy_exclusions + "," if proxy_exclusions else "") + "127.0.0.1,localhost"
    env["no_proxy"] = env["NO_PROXY"]
    return env


def validate_model_report(env: dict[str, str]) -> dict[str, Any]:
    manifest_path = ROOT / "deployment/model_revisions.json"
    manifest = json.loads(manifest_path.read_text())
    report_path = Path(env["AIME_MODEL_VERIFICATION_REPORT"])
    report = json.loads(report_path.read_text())
    if report.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("model verification refers to a different manifest")
    for kind, name in (("qwen", "AIME_MODEL_PATH"), ("embedding", "AIME_MINILM_PATH")):
        checked = report["models"][kind]
        expected = {file: entry[1] for file, entry in manifest[kind]["files"].items()}
        if checked.get("sha256_verified") != expected or checked.get("revision") != manifest[kind]["revision"]:
            raise ValueError(f"pinned {kind} weights are not verified")
        if Path(checked["path"]).resolve() != Path(env[name]).resolve():
            raise ValueError(f"verified {kind} model path differs")
        for file, (size, _) in manifest[kind]["files"].items():
            if (Path(env[name]) / file).stat().st_size != size:
                raise ValueError(f"verified {kind} asset missing or resized: {file}")
    return {"verification_report": str(report_path.resolve()), "verification_report_sha256": sha256(report_path),
            "manifest_sha256": sha256(manifest_path), "note": "Uses the pinned existing SHA verification report; file existence and sizes rechecked."}


def uv_python(env: dict[str, str], python: str, *args: str) -> list[str]:
    return [env["AIME_UV_BIN"], "run", "--no-project", "--python", python, "python", *args]


def core_script(method: str) -> Path:
    return ROOT / "experiments" / ("recover_gdesigner_tests.py" if method == "gdesigner" else "recover_aime_tests.py")


def core_command(plan: dict[str, Any], env: dict[str, str], entry: dict[str, Any], *, inspect: bool) -> list[str]:
    args = ["-u", str(core_script(entry["method"])), "--method", entry["method"],
            "--source-run", entry["source_run"], "--run-dir", entry["run_dir"],
            "--data-dir", env["AIME_DATA_DIR"], "--baseline-root", str(ROOT / "upstream"),
            "--endpoint", f"http://127.0.0.1:{plan['port']}/v1", "--seed", "0"]
    if inspect:
        args.append("--inspect-only")
    if entry.get("allow_stage_reseed") is True:
        args.append("--allow-stage-reseed")
    if entry.get("allow_selection_gate_waiver") is True:
        args.append("--allow-selection-gate-waiver")
    return uv_python(env, env[f"AIME_{entry['method'].upper()}_PYTHON"], *args)


def server_command(plan: dict[str, Any], env: dict[str, str]) -> list[str]:
    return [env["AIME_VLLM_BIN"], "serve", env["AIME_MODEL_PATH"], "--served-model-name", MODEL,
            "--host", "127.0.0.1", "--port", str(plan["port"]), "--tensor-parallel-size", "1",
            "--dtype", "bfloat16", "--max-model-len", "16384", "--max-num-seqs", "8",
            "--max-num-batched-tokens", "16384", "--gpu-memory-utilization", "0.92",
            "--enable-prefix-caching", "--reasoning-parser", "qwen3", "--language-model-only"]


def guard_gpu_port(plan: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", plan["port"]))
    gpu = subprocess.run(["nvidia-smi", "--id", plan["gpu"],
                          "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                         check=True, capture_output=True, text=True, env=env, timeout=15)
    rows = [row for row in gpu.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError("GPU guard did not identify exactly one physical GPU")
    index, uuid, used, total = [cell.strip() for cell in rows[0].split(",")]
    processes = subprocess.run(["nvidia-smi", "--id", plan["gpu"],
                                "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                               check=True, capture_output=True, text=True, env=env, timeout=15)
    pids = [row.strip() for row in processes.stdout.splitlines() if row.strip()]
    if index != plan["gpu"] or pids or int(used) >= 1024:
        raise ValueError(f"GPU {plan['gpu']} is occupied: used={used} MiB, compute_pids={pids}")
    return {"gpu_index": index, "gpu_uuid": uuid, "memory_used_mib": int(used),
            "memory_total_mib": int(total), "compute_pids": [], "checked_at_utc": now()}


class OwnedProcess:
    """Signal only captured identities in our newly created local session."""

    def __init__(self, command: list[str], env: dict[str, str], log: Path):
        self.log_stream = log.open("x", encoding="utf-8")
        try:
            self.process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                            stdout=self.log_stream, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            self.log_stream.close()
            raise
        self.identity = proc_identity(self.process.pid)
        if self.identity is None:
            self.process.wait()
            self.log_stream.close()
            raise RuntimeError("child exited before its process identity was captured")
        self.members: dict[int, dict[str, Any]] = {self.process.pid: self.identity}

    def capture(self) -> None:
        root = proc_identity(self.process.pid)
        if root is None or root["starttime_ticks"] != self.identity["starttime_ticks"]:
            return
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                candidate = proc_identity(int(entry.name))
                if candidate and candidate["session"] == self.process.pid and candidate["starttime_ticks"] >= self.identity["starttime_ticks"]:
                    self.members[candidate["pid"]] = candidate

    def poll(self) -> int | None:
        self.capture()
        return self.process.poll()

    def wait(self) -> int:
        while self.poll() is None:
            time.sleep(1)
        return int(self.process.returncode)

    def close(self) -> None:
        self.capture()
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid, saved in reversed(list(self.members.items())):
                current = proc_identity(pid)
                if current and current["starttime_ticks"] == saved["starttime_ticks"] and current["session"] == self.process.pid:
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
            if sig == signal.SIGTERM:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    remaining = [proc_identity(pid) for pid in self.members]
                    if not any(item and item["state"] != "Z" for item in remaining):
                        break
                    time.sleep(0.1)
        try:
            self.process.wait(timeout=5)
        finally:
            self.log_stream.close()


def wait_ready(server: OwnedProcess, endpoint: str, timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("vLLM exited before readiness; inspect server.log")
        try:
            with urlopen(endpoint + "/models", timeout=3) as response:
                payload = json.load(response)
            if any(row.get("id") == MODEL for row in payload.get("data", [])):
                return
        except (OSError, URLError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError("vLLM readiness timed out after 900 seconds")


def warmup(endpoint: str) -> dict[str, Any]:
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "Return only OK."}],
               "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    request = Request(endpoint + "/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    started = time.monotonic()
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    if not result.get("choices") or not (result["choices"][0]["message"].get("content") or "").strip():
        raise ValueError("endpoint warmup returned an empty completion")
    return {"status": "passed", "timeout_s": 300, "max_tokens": 16, "max_retries": 0,
            "temperature": 0, "enable_thinking": False, "elapsed_s": time.monotonic() - started,
            "usage": result.get("usage"), "disposable_not_test_metric": True}


def launch(plan_path: Path) -> int:
    plan = load_plan(plan_path)
    env = runtime_environment(plan)
    for key in ("output_root", "launch_dir"):
        if Path(plan[key]).exists():
            raise FileExistsError(f"fresh {key} required: {plan[key]}")
    for item in plan["runs"]:
        if Path(item["run_dir"]).exists():
            raise FileExistsError(f"recovery run_dir already exists: {item['run_dir']}")
    output, directory = Path(plan["output_root"]), Path(plan["launch_dir"])
    output.parent.mkdir(parents=True, exist_ok=True)
    directory.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    directory.mkdir()
    saved_plan = directory / "plan.json"
    write_json(saved_plan, plan, exclusive=True)
    record = {"schema": "aime_supplemental_recovery_launch_v1", "started_at_utc": now(),
              "status": "started", "classification": "supplemental", "formal_status": "NOT_FORMAL_PASS",
              "plan_sha256": sha256(saved_plan), "source_root": str(ROOT),
              "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "launcher_sha256": sha256(ROOT / "scripts/launch_recovery_detached.sh"),
              "worker_sha256": sha256(Path(__file__)),
              "recovery_core": str(core_script(plan["runs"][0]["method"])),
              "recovery_core_sha256": sha256(core_script(plan["runs"][0]["method"])), "protocol": PROTOCOL,
              "stdin": "/dev/null", "stdout_stderr": str(directory / "run.log")}
    write_json(output / "launch_record.json", record, exclusive=True)
    command = uv_python(env, env["AIME_PYTHON"], "-u", str(Path(__file__).resolve()), "--worker", "--plan", str(saved_plan))
    with (directory / "run.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record["worker_identity"] = proc_identity(child.pid)
    record["pid"] = child.pid
    write_json(directory / "launch.json", record, exclusive=True)
    print(json.dumps({"status": "started", "pid": child.pid, "log": str(directory / "run.log"),
                      "output": str(output), "formal_status": "NOT_FORMAL_PASS"}))
    return 0


def worker(plan_path: Path) -> int:
    plan = load_plan(plan_path)
    env = runtime_environment(plan)
    output, directory = Path(plan["output_root"]), Path(plan["launch_dir"])
    if not directory.is_dir() or not output.is_dir():
        raise ValueError("worker must use directories reserved by --launch")
    completion: dict[str, Any] = {"schema": "aime_supplemental_recovery_completion_v1", "status": "running",
                                  "classification": "supplemental", "formal_status": "NOT_FORMAL_PASS",
                                  "started_at_utc": now(), "methods": {}, "stage": "preflight"}
    owned: list[OwnedProcess] = []
    exit_code = 1

    def stop(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"worker interrupted by signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    try:
        with ExitStack() as locks:
            lock_root = LOCK_ROOT
            lock_root.mkdir(parents=True, exist_ok=True)
            for name in (f"gpu{plan['gpu']}", f"port{plan['port']}"):
                stream = locks.enter_context((lock_root / (name + ".lock")).open("a+"))
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            completion["gpu_guard"] = guard_gpu_port(plan, env)
            completion["model_verification"] = validate_model_report(env)
            for name in ("HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "TRITON_CACHE_DIR",
                         "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT", "CUDA_CACHE_PATH", "TMPDIR"):
                Path(env[name]).mkdir(parents=True, exist_ok=True)
            valid: list[dict[str, Any]] = []
            for entry in plan["runs"]:
                method = entry["method"]
                process = OwnedProcess(core_command(plan, env, entry, inspect=True), env, directory / f"inspect_{method}.log")
                owned.append(process)
                code = process.wait()
                process.close()
                owned.remove(process)
                completion["methods"][method] = {"source_run": entry["source_run"], "run_dir": entry["run_dir"],
                                                   "inspect_exit_code": code, "status": "ready" if code == 0 else "inspect_failed"}
                write_json(directory / "status.json", completion)
                if code == 0:
                    valid.append(entry)
            if not valid:
                raise RuntimeError("No recovery passed offline inspection; no server started")
            preflight = OwnedProcess(uv_python(env, env["AIME_PYTHON"], str(ROOT / "scripts/preflight_gpu.py"), "--role", "serve"),
                                     env, directory / "serve_preflight.log")
            owned.append(preflight)
            preflight_code = preflight.wait()
            preflight.close()
            owned.remove(preflight)
            if preflight_code != 0:
                raise RuntimeError(f"serving environment preflight failed exit={preflight_code}")
            completion["gpu_guard_before_start"] = guard_gpu_port(plan, env)
            server = OwnedProcess(server_command(plan, env), env, directory / "server.log")
            owned.append(server)
            monitor = OwnedProcess(["nvidia-smi", "--id", plan["gpu"], "--query-gpu=timestamp,index,memory.used,utilization.gpu",
                                    "--format=csv", "-lms", "1000"], env, directory / "gpu.csv")
            owned.append(monitor)
            completion["server_identity"] = server.identity
            completion["gpu_monitor_identity"] = monitor.identity
            completion["server_command"] = server_command(plan, env)
            completion["service_environment"] = {name: env[name] for name in (
                "CUDA_VISIBLE_DEVICES", "AIME_MAX_MODEL_LEN", "AIME_MAX_NUM_SEQS", "AIME_MAX_NUM_BATCHED_TOKENS",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME", "VLLM_CACHE_ROOT", "TMPDIR")}
            completion["stage"] = "server_startup"
            write_json(directory / "status.json", completion)
            endpoint = f"http://127.0.0.1:{plan['port']}/v1"
            wait_ready(server, endpoint)
            session_path = directory / "run_session.json"
            session_command = uv_python(
                env, env["AIME_PYTHON"], str(ROOT / "scripts/record_run_session.py"),
                "--root", str(ROOT), "--output", str(session_path),
                "--server-pid", str(server.process.pid),
                "--gpu-monitor-pid", str(monitor.process.pid),
                "--server-log", str(directory / "server.log"),
                "--gpu-log", str(directory / "gpu.csv"), "--port", str(plan["port"]),
            )
            session_recorder = OwnedProcess(session_command, env, directory / "session_recorder.log")
            owned.append(session_recorder)
            session_code = session_recorder.wait()
            session_recorder.close()
            owned.remove(session_recorder)
            if session_code != 0:
                raise RuntimeError(f"service session provenance failed exit={session_code}")
            env["AIME_RUN_SESSION_RECORD"] = str(session_path)
            completion["run_session_record"] = str(session_path)
            completion["run_session_record_sha256"] = sha256(session_path)
            completion["endpoint_warmup"] = warmup(endpoint)
            for entry in valid:
                if server.poll() is not None:
                    raise RuntimeError("vLLM exited; no additional recovery started")
                method = entry["method"]
                completion["stage"] = f"recover_{method}_2025_2026" if method == "gdesigner" else f"recover_{method}_2026"
                result = completion["methods"][method]
                result.update({"status": "running", "started_at_utc": now(), "formal_status": "NOT_FORMAL_PASS"})
                write_json(directory / "status.json", completion)
                process = OwnedProcess(core_command(plan, env, entry, inspect=False), env, directory / f"recover_{method}.log")
                owned.append(process)
                result["worker_identity"] = process.identity
                write_json(directory / "status.json", completion)
                code = process.wait()
                process.close()
                owned.remove(process)
                result.update({"status": "supplemental_completed" if code == 0 else "recovery_failed",
                               "compute_exit_code": code, "finished_at_utc": now()})
                write_json(directory / "status.json", completion)
            exit_code = 0 if all(item.get("status") == "supplemental_completed" for item in completion["methods"].values()) else 1
            completion["status"] = "supplemental_completed" if exit_code == 0 else "partial_or_failed"
            completion["stage"] = "complete"
    except Exception as exc:
        completion["status"] = "failed"
        completion["error"] = f"{type(exc).__name__}: {exc}"
        print(completion["error"], file=sys.stderr, flush=True)
    finally:
        # Interrupted cleanup must finish even if another TERM arrives.
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, signal.SIG_IGN)
        cleanup_errors = []
        for process in reversed(owned):
            try:
                process.close()
            except Exception as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        if cleanup_errors:
            completion["cleanup_errors"] = cleanup_errors
            completion["status"] = "failed"
            exit_code = 1
        completion.update({"exit_code": exit_code, "finished_at_utc": now()})
        write_json(directory / "status.json", completion)
        write_json(directory / "completion.json", completion, exclusive=True)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--launch", action="store_true")
    operation.add_argument("--worker", action="store_true")
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    try:
        return launch(args.plan.resolve()) if args.launch else worker(args.plan.resolve())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"RECOVERY_LAUNCH_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
