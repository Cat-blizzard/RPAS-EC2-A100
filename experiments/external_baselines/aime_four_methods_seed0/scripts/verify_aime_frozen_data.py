#!/usr/bin/env python3
"""Frozen AIME consistency audit: prelock mode never parses test questions.

Default: verify hashes and counts for all partitions, parse ONLY validation
questions and verify its 60/30 content split. D_test is treated as opaque bytes.
After candidate freeze one may additionally use the native run's access ledger
for content-level disjointness; that is part of the formal runner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jsonlines(data: bytes):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def problem_key(row):
    return " ".join(str(row.get("problem", row.get("input", ""))).casefold().split())


def audit(data_dir: Path):
    manifest = json.loads((data_dir / "frozen_aime_manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("data_seed") == 2026
    validation = manifest["validation"]
    expected = {
        "validation/source": (validation["source"], 90),
        "validation/search": (validation["search"], 60),
        "validation/select": (validation["select"], 30),
    }
    test_specs=[]
    for year, spec in sorted(manifest["test"].items()):
        assert spec["rows"] == 30 and spec["source_rows"] == 30
        test_specs.append((year,spec["source_sha256_git_content"],30))
        test_specs.append((spec["split_path"],spec["split_sha256_git_content"],30))
    for relative,expected_hash,count in test_specs:
        data=(data_dir / relative).read_bytes()  # hashing raw bytes is not parsing model-visible tasks
        assert sha256(data)==expected_hash, f"D_test checksum failed: {relative}"
        assert len([line for line in data.splitlines() if line.strip()])==count, f"D_test count failed: {relative}"
    parsed={}
    for name,(spec,n) in expected.items():
        data=(data_dir/spec["path"]).read_bytes()
        assert sha256(data)==spec["sha256_git_content"], f"validation checksum failed: {name}"
        parsed[name]=jsonlines(data)
        assert len(parsed[name])==n
    keys=[set(map(problem_key,parsed[name])) for name in ("validation/source","validation/search","validation/select")]
    assert all("" not in s for s in keys), "empty problem text"
    assert len(keys[0])==90 and len(keys[1])==60 and len(keys[2])==30
    assert keys[1].isdisjoint(keys[2]), "D_search/D_select overlap"
    assert keys[1]|keys[2]==keys[0], "validation split differs from source"
    source_answers={problem_key(row):str(row["answer"]).strip() for row in parsed["validation/source"]}
    for key in ("validation/search","validation/select"):
        for row in parsed[key]:
            assert source_answers[problem_key(row)]==str(row["answer"]).strip(), "validation answer mismatch"
    return {"status":"PASS","D_search":60,"D_select":30,
            "D_test_each_year":30,"validation_disjoint":True,"test_contents_parsed":False,
            "test_model_access":False,"test_hash_and_counts_verified":True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,default=Path(__file__).resolve().parents[1]/"data")
    args=parser.parse_args()
    print(json.dumps(audit(args.data_dir.resolve()),ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
