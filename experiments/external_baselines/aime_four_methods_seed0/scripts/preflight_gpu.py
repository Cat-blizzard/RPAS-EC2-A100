#!/usr/bin/env python3
"""Local-only checks for the isolated serving and method environments."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


REQUIRED_SERVE_FLAGS = (
    "--served-model-name", "--max-model-len", "--max-num-seqs",
    "--tensor-parallel-size", "--gpu-memory-utilization",
    "--max-num-batched-tokens", "--enable-prefix-caching",
    "--reasoning-parser", "--language-model-only", "--dtype",
)


def fail(message: str) -> None:
    raise SystemExit(f"PREFLIGHT_FAIL: {message}")


def local_directory(variable: str, description: str) -> Path:
    value = os.environ.get(variable, "").strip()
    if not value:
        fail(f"set {variable} to a local {description} directory (no implicit download)")
    path = Path(value).expanduser()
    if not path.is_dir():
        fail(f"{variable} does not point to a local {description} directory: {path}")
    return path.resolve()


def import_packages(*names: str) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            fail(f"{name} import failed in {sys.executable}: {type(exc).__name__}: {exc}")
        versions[name] = str(getattr(module, "__version__", "installed"))
    return versions


def check_tokenizer(path: Path) -> None:
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
    except Exception as exc:
        fail(f"local tokenizer load failed from {path}: {type(exc).__name__}: {exc}")


def check_gpu(torch: object, *, require_bf16: bool) -> dict[str, object]:
    if not torch.cuda.is_available():
        fail("PyTorch cannot access a CUDA GPU in this environment")
    count = torch.cuda.device_count()
    if count != 1:
        fail(f"expected exactly one visible GPU for TP=1, found {count}; set CUDA_VISIBLE_DEVICES")
    device = torch.cuda.get_device_properties(0)
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    if require_bf16 and not bf16_supported:
        fail("the visible GPU does not support the required BF16 serving dtype")
    return {
        "gpu": device.name,
        "compute_capability": f"{device.major}.{device.minor}",
        "gpu_memory_gib": round(device.total_memory / (1024**3), 2),
        "visible_gpu_count": count,
        "bf16_supported": bf16_supported,
    }


def check_serve() -> dict[str, object]:
    model_path = local_directory("RPAS_MODEL_PATH", "Qwen3.5-9B model")
    packages = import_packages("torch", "transformers", "vllm")
    import torch

    gpu = check_gpu(torch, require_bf16=True)
    try:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"local model config read failed from {model_path}: {type(exc).__name__}: {exc}")
    if config.get("model_type") != "qwen3_5" or "Qwen3_5ForConditionalGeneration" not in config.get("architectures", []):
        fail("local config is not the expected Qwen3.5-9B conditional-generation architecture")
    check_tokenizer(model_path)
    text_config = config.get("text_config", {})
    max_positions = text_config.get("max_position_embeddings")
    if not isinstance(max_positions, int) or max_positions < 16384:
        fail(f"model config supports only {max_positions} positions, below required context 16384")

    vllm_bin = os.environ.get("RPAS_VLLM", "vllm")
    try:
        help_result = subprocess.run(
            [vllm_bin, "serve", "--help=all"], check=True, capture_output=True, text=True, timeout=90
        )
    except Exception as exc:
        fail(f"could not inspect vLLM serve options via {vllm_bin!r}: {type(exc).__name__}: {exc}")
    help_text = help_result.stdout + help_result.stderr
    missing = [flag for flag in REQUIRED_SERVE_FLAGS if flag not in help_text]
    if missing:
        fail("installed vLLM lacks required server flags: " + ", ".join(missing))

    return {
        "role": "serve",
        "python": sys.executable,
        "packages": packages,
        "torch_cuda_build": torch.version.cuda,
        **gpu,
        "model_path": str(model_path),
        "model_context_capacity": max_positions,
        "required_context": 16384,
        "required_output": 6144,
        "required_dtype": "bfloat16",
        "vllm_bin": vllm_bin,
        "note": "Environment check only; a model-load and request smoke test remain required.",
    }


def check_method(role: str) -> dict[str, object]:
    tokenizer_variable = (
        "RPAS_TOKENIZER_PATH" if os.environ.get("RPAS_TOKENIZER_PATH", "").strip()
        else "RPAS_MODEL_PATH"
    )
    tokenizer_path = local_directory(tokenizer_variable, "Qwen3.5-9B tokenizer")
    common = ("openai", "transformers", "huggingface_hub", "tqdm.asyncio", "numpy", "yaml")
    packages = import_packages(*common)
    check_tokenizer(tokenizer_path)
    report: dict[str, object] = {
        "role": role,
        "python": sys.executable,
        "packages": packages,
        "tokenizer_path": str(tokenizer_path),
        "note": "Method dependency check only; the native method still needs a real model-call diagnostic.",
    }
    if role == "maas":
        embedding = local_directory("RPAS_MAAS_EMBEDDING_PATH", "all-MiniLM-L6-v2 embedding model")
        if not any((embedding / name).is_file() for name in (
            "modules.json", "config.json", "sentence_bert_config.json", "tokenizer.json"
        )):
            fail(f"embedding directory has no recognizable local model configuration: {embedding}")
        packages.update(import_packages("torch", "sentence_transformers", "aiohttp"))
        import torch

        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["visible_gpu_count"] = torch.cuda.device_count() if report["cuda_available"] else 0
        report["embedding_path"] = str(embedding)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("serve", "aflow", "maas"),
                        help="check only the selected process environment")
    parser.add_argument("--method", choices=("aflow", "maas", "all"),
                        help="legacy combined-environment check")
    args = parser.parse_args()
    if args.role and args.method:
        parser.error("--role and --method cannot be combined")

    if args.role:
        roles = (args.role,)
    elif args.method == "aflow":
        roles = ("serve", "aflow")
    elif args.method == "maas":
        roles = ("serve", "maas")
    else:
        roles = ("serve", "aflow", "maas")

    for role in roles:
        report = check_serve() if role == "serve" else check_method(role)
        print(json.dumps(report, indent=2))
        print(f"{role.upper()}_PREFLIGHT_PASS")
    if not args.role:
        print("GPU_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
