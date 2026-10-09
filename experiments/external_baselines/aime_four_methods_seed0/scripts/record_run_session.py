#!/usr/bin/env python3
"""Record the immutable launch context for one seed-0 server session."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


PACKAGE_NAMES = ("torch", "transformers", "vllm", "openai", "sentence-transformers", "numpy", "pandas")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def versions(python: str) -> dict[str, str]:
    script = f"""import json
from importlib import metadata
result = {{}}
for name in {PACKAGE_NAMES!r}:
    try:
        result[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
print(json.dumps(result))
"""
    completed = subprocess.run([python, "-c", script], check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def positive_int_env(name: str, default: int) -> int:
    """Read a positive integer runtime setting, preserving the 8K fallback."""
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def server_settings() -> dict[str, object]:
    """Return the actual launcher settings, including the selected context profile."""
    return {
        "served_model_name": "Qwen/Qwen3.5-9B", "dtype": "bfloat16", "language_model_only": True,
        "tensor_parallel_size": 1,
        "max_model_len": positive_int_env("AIME_MAX_MODEL_LEN", 8192),
        "max_num_seqs": positive_int_env("AIME_MAX_NUM_SEQS", 24),
        "max_num_batched_tokens": positive_int_env("AIME_MAX_NUM_BATCHED_TOKENS", 16384),
        "gpu_memory_utilization": 0.92,
        "reasoning_parser": "qwen3", "enable_prefix_caching": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--gpu-monitor-pid", type=int, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    git_commit = subprocess.run(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    python_paths = {
        "serve": os.environ["RPAS_SERVE_PYTHON"],
        "aflow": os.environ["RPAS_AFLOW_PYTHON"],
        "maas": os.environ["RPAS_MAAS_PYTHON"],
    }
    for role in ("adas", "gdesigner"):
        if os.environ.get(f"AIME_{role.upper()}_PYTHON"):
            python_paths[role] = os.environ[f"AIME_{role.upper()}_PYTHON"]
    verification_path = os.environ.get("RPAS_MODEL_VERIFICATION_REPORT", "")
    if verification_path and not Path(verification_path).is_file():
        raise FileNotFoundError(f"RPAS_MODEL_VERIFICATION_REPORT does not exist: {verification_path}")
    verification = json.loads(Path(verification_path).read_text(encoding="utf-8")) if verification_path else None
    if verification:
        model_manifest_path = args.root / "deployment/model_revisions.json"
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        if verification.get("manifest_sha256") != sha256_file(model_manifest_path):
            raise ValueError("model verification report refers to a different asset manifest")
        for kind in ("qwen", "embedding"):
            expected_files = {name: expected_sha for name, (_, expected_sha) in model_manifest[kind]["files"].items()}
            if verification["models"][kind].get("sha256_verified") != expected_files:
                raise ValueError(f"model verification report lacks the pinned {kind} file checks")
        expected_paths = {
            "qwen": os.environ["RPAS_MODEL_PATH"],
            "embedding": os.environ.get("RPAS_MAAS_EMBEDDING_PATH", ""),
        }
        for kind, expected_path in expected_paths.items():
            if expected_path and Path(verification["models"][kind]["path"]).resolve() != Path(expected_path).resolve():
                raise ValueError(f"model verification report {kind} path differs from the run path")
        expected_revisions = {
            "qwen": os.environ.get("RPAS_MODEL_REVISION", ""),
            "embedding": os.environ.get("RPAS_MAAS_EMBEDDING_REVISION", ""),
        }
        for kind, expected_revision in expected_revisions.items():
            if expected_revision and verification["models"][kind]["revision"] != expected_revision:
                raise ValueError(f"model verification report {kind} revision differs from the run revision")
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "project_root": str(args.root.resolve()),
        "project_git_commit": git_commit,
        "project_sha256sums_sha256": sha256_file(args.root / "SHA256SUMS"),
        "server_pid": args.server_pid,
        "gpu_monitor_pid": args.gpu_monitor_pid,
        "server_log": str(args.server_log.resolve()),
        "gpu_log": str(args.gpu_log.resolve()),
        "port": args.port,
        "server_bind": "127.0.0.1",
        "server_binary": os.environ["RPAS_VLLM"],
        "model_path": os.environ["RPAS_MODEL_PATH"],
        "model_revision": os.environ.get("RPAS_MODEL_REVISION", "unspecified"),
        "embedding_path": os.environ.get("RPAS_MAAS_EMBEDDING_PATH", ""),
        "embedding_revision": os.environ.get("RPAS_MAAS_EMBEDDING_REVISION", "unspecified"),
        "model_verification_report": verification_path,
        "model_verification_report_sha256": sha256_file(Path(verification_path)) if verification_path else None,
        "server_settings": server_settings(),
        "python_paths": python_paths,
        "package_versions": {role: versions(path) for role, path in python_paths.items()},
        "pair": os.environ.get("AIME_PAIR", ""),
        "runtime_overrides": {name: os.environ[name] for name in (
            "AIME_AFLOW_REQUEST_TIMEOUT_S", "AIME_AFLOW_SAMPLE_TIMEOUT_S", "AIME_AFLOW_CODE_TIMEOUT_S",
            "AIME_MAAS_REQUEST_TIMEOUT_S", "AIME_MAAS_SAMPLE_TIMEOUT_S", "AIME_MAAS_SAMPLE_RETRIES",
            "AIME_MAAS_CODE_TIMEOUT_S", "AIME_EXTERNAL_REQUEST_TIMEOUT_S",
        ) if name in os.environ},
        "gpu": subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version",
             "--format=csv,noheader"], check=True, capture_output=True, text=True,
        ).stdout.splitlines(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(f"RUN_SESSION_RECORD {args.output}")


if __name__ == "__main__":
    main()
