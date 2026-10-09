#!/usr/bin/env python3
"""Read-only report of AIME four-way attempt status, including failed D_select.

Usage: python scripts/audit_fourway_attempts.py --outputs OUTPUT_ROOT
Does not read test questions and never changes method scores or gates.
"""
import argparse
import json
from pathlib import Path

METHOD_DIRS = {
    "aflow": "aflow_maas/aflow/seed_0",
    "maas": "aflow_maas/maas/seed_0",
    "adas": "adas_gdesigner/adas/seed_0",
    "gdesigner": "adas_gdesigner/gdesigner/seed_0",
}

def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None

def audit(root):
    result = {}
    for method, rel in METHOD_DIRS.items():
        run_dir = root / rel
        manifest = load(run_dir / "run_manifest.json") or {}
        native = load(run_dir / "native_result.json") or {}
        gate = load(run_dir / "quality_gate.json") or {}
        selection = load(run_dir / "selection_audit.json") or {}
        access = load(run_dir / "dtest_access_manifest.json") or {}
        telemetry = native.get("telemetry") or manifest.get("telemetry") or {}
        totals = telemetry.get("totals") if isinstance(telemetry.get("totals"), dict) else telemetry
        # Some native baselines expose token usage as a telemetry summary.
        if not totals:
            totals = selection.get("telemetry") or {}
            totals = totals.get("totals", totals)
        tests = native.get("test") or native.get("tests") or {}
        test_counts = {}
        for year in ("aime_2025", "aime_2026"):
            point = tests.get(year, {})
            point = point.get("Q", point) if isinstance(point, dict) else {}
            if isinstance(point, dict) and point.get("num_examples"):
                test_counts[year] = {"correct": point.get("correct"), "n": point.get("num_examples")}
        result[method] = {
            "status": manifest.get("status", "no_run"),
            "quality_gate": gate.get("status", "not_available"),
            "failure": manifest.get("failure"),
            "selection_audit_saved": bool(selection),
            "selection_score": ((selection.get("selection_metrics") or {}).get("score")
                                if method == "maas" else
                                (selection.get("selection_metrics") or {}).get("accuracy")
                                if method == "gdesigner" else None),
            "dtest_opened": bool(access.get("test_splits")),
            "raw_test_counts_not_necessarily_formal": test_counts,
            "calls": totals.get("calls"),
            "tokens": totals.get("total_tokens"),
            "completion_truncated": totals.get("completion_truncated"),
        }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = {"schema": "aime_fourway_attempt_audit_v1",
              "outputs": str(args.outputs.resolve()),
              "methods": audit(args.outputs)}
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(encoded, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded, encoding="utf-8")

if __name__ == "__main__":
    main()
