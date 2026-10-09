#!/usr/bin/env python3
"""Offline four-way AIME protocol + native-search fingerprint audit.

Checks repository source and optional pinned upstream clones without running an
LLM and WITHOUT reading D_test answers. Does not certify GPU behavior.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PINNED = {
    'aflow': ('AFlow', '3f457218fc716093fe53f6df8a5d5e6379d66346'),
    'maas': ('MaAS', '987f3c1bc9a96e844fe090db3791446e3ef0f5c7'),
    'adas': ('ADAS', '2702bee8fefda42255efc5be9f60e3bd3db96ae4'),
    'gdesigner': ('GDesigner', 'a6efcfa3b40bb4d9cbf46f883a95d62020bd8251'),
}

METHOD_CONTRACTS = {
    'aflow': ('native_aime_formal.py', ['optimizer.optimize("Graph")', 'aflow_search_rounds(', 'evaluate_aflow_graph(', 'freeze_aime_selection(']),
    'maas': ('native_aime_formal.py', ['optimizer.optimize("Graph")', 'controller_path =', 'optimizer.test()', 'freeze_aime_selection(']),
    'adas': ('native_external_methods.py', ['adas.search(native_args)', 'native_args = argparse.Namespace(', 'evaluate_forward_fn(native_args, code)', 'freeze_aime_selection(']),
    'gdesigner': ('native_external_methods.py', ['graph.gcn.train()', 'optimizer = torch.optim.Adam(graph.gcn.parameters()', 'await infer(batch, "search:train", train=True)', 'controller_sha256 = sha256_file(controller_path)']),
}


def run(root: Path, profile: str, require_upstream: bool):
    e = root / 'experiments'
    path = root / 'data'
    manifest_file = path / 'frozen_aime_manifest.json'
    frozen = json.loads(manifest_file.read_text(encoding='utf8'))
    methods = {}
    for method, (source_name, anchors) in METHOD_CONTRACTS.items():
        src = (e / source_name).read_text(encoding='utf8')
        ast.parse(src)
        found = [x for x in anchors if x in src]
        methods[method] = {'search_core_anchors':len(found), 'expected':len(anchors), 'all_present':len(found)==len(anchors), 'missing':[x for x in anchors if x not in src]}
    required_count = {'search': 60, 'select': 30, 'aime_2025': 30, 'aime_2026': 30}
    rowspec = {'search':frozen['validation']['search'], 'select':frozen['validation']['select'],
        **{y: {'path':frozen['test'][y+'.jsonl']['split_path'], 'rows':frozen['test'][y+'.jsonl']['rows']}
           for y in ('aime_2025','aime_2026')}}
    counts = {k:v['rows'] for k,v in rowspec.items()}
    assert counts == required_count, f'frozen manifest sizes: {counts}'
    # Test content is intentionally never opened by this program.
    tests_present = {k:(path / rowspec[k]['path']).is_file() for k in ('aime_2025','aime_2026')}
    assert all(tests_present.values()), 'frozen test files absent (contents remain unread)'
    for k in ('search','select'):
        spec=rowspec[k]; f=path/spec['path']; data=f.read_bytes();
        assert hashlib.sha256(data).hexdigest()==spec['sha256_git_content'], f'{k} hash mismatch'
    expected_context=16384 if profile=='16k' else 8192
    for key in ('native_aime_formal.py','native_external_methods.py','maas_context_guard.py'):
        s=(e/key).read_text(encoding='utf8')
        assert f'CONTEXT_LIMIT = {expected_context}' in s, f'{key} context mismatch'
    if profile=='16k':
        assert 'TARGET_INPUT_TOKENS = 8192' in (e/'maas_context_guard.py').read_text(encoding='utf8')
        assert 'MIN_OUTPUT_RESERVE = 6144' in (e/'maas_context_guard.py').read_text(encoding='utf8')
    sources={}
    import subprocess
    for method,(folder, sha) in PINNED.items():
        upstream=root/'upstream'/folder
        if upstream.is_dir():
            try:
                head=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
                clean=subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True,stderr=subprocess.DEVNULL)==''
            except Exception as err:
                head='unavailable';clean=False
            sources[method]={'expected':sha,'actual':head,'match':sha==head,'clean':clean}
        else:
            sources[method]={'expected':sha,'actual':None,'match':False,'clean':False}
    errors=[f'{m}: native search calls missing {d["missing"]}' for m,d in methods.items() if not d['all_present']]
    if require_upstream:
        errors.extend(f'{m}: missing/dirty/nonmatching upstream checkout' for m,x in sources.items() if not x['match'] or not x['clean'])
    return {'status':'PASS' if not errors else 'FAIL','schema':'aime_fourway_native_fidelity_preflight_v2',
        'profile':profile,'context':expected_context,'requested_output_tokens':6144,
        'D_search':60,'D_select':30,'D_test_per_year':30,
        'test_contents_read':False,'test_files_present':tests_present,
        'native_core':methods,'upstream':sources,'errors':errors,
        'honest_scope':'native optimization cores with benchmark-specific compatibility adaptations; no CUDA execution verified'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--context-profile',choices=['8k','16k'],default='16k')
    p.add_argument('--require-upstream',action='store_true')
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    result=run(a.root.resolve(),a.context_profile,a.require_upstream)
    blob=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    print(blob,end='')
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(blob,encoding='utf8')
    raise SystemExit(0 if result['status']=='PASS' else 2)
if __name__=='__main__': main()
