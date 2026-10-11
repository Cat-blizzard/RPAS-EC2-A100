"""Offline frozen-controller recovery checks; every model call is synthetic."""

from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import native_external_methods as native

torch = pytest.importorskip("torch")
PACKAGE = Path(__file__).resolve().parents[1]


def _controller_fixture(tmp_path):
    graph = SimpleNamespace(
        gcn=torch.nn.Linear(2, 3), mlp=torch.nn.Linear(3, 2),
        role_adj_matrix=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        optimized_spatial=True, optimized_temporal=False,
    )
    graph.gcn.eval()
    graph.mlp.train()
    numerics = native.patch_gdesigner_numerics(SimpleNamespace())
    roles = {"variant": "synthetic_role_fixture", "mapped_role_edges": [("a", "b")]}
    checkpoint = {
        "schema": "aime_gdesigner_controller_v1", "seed": 0,
        "state_dicts": {name: copy.deepcopy(getattr(graph, name).state_dict()) for name in ("gcn", "mlp")},
        "role_graph_adapter": roles,
        "role_adjacency_edge_index": graph.role_adj_matrix.clone(),
        "numerical_compatibility_adapter": {**numerics, "zero_span_calls": 7},
        "module_training_modes": {"gcn": False, "mlp": True},
        "frozen_phase": "after D_search and before D_select/D_test",
    }
    path = tmp_path / "controller.pt"
    torch.save(checkpoint, path)
    return graph, checkpoint, path, roles, numerics


def test_checkpoint_restore_keeps_native_eval_modes_and_exact_state(tmp_path):
    graph, checkpoint, path, roles, numerics = _controller_fixture(tmp_path)
    with torch.no_grad():
        graph.gcn.weight.add_(100)
    graph.gcn.train()
    state, modes, edges = native._restore_gdesigner_controller(
        graph, path, native.sha256_file(path), 0, roles, numerics,
    )
    assert graph.gcn.training is False and graph.mlp.training is True
    assert torch.equal(graph.gcn.weight, checkpoint["state_dicts"]["gcn"]["weight"])
    native._require_gdesigner_frozen_controller(graph, state, modes, edges, path, native.sha256_file(path))


@pytest.mark.parametrize("mutation", ["schema", "seed", "adapter", "role_edges", "mode", "keys", "shape", "nan"])
def test_invalid_checkpoint_fails_closed(tmp_path, mutation):
    graph, checkpoint, path, roles, numerics = _controller_fixture(tmp_path)
    if mutation == "schema":
        checkpoint["schema"] = "unknown"
    elif mutation == "seed":
        checkpoint["seed"] = 1
    elif mutation == "adapter":
        checkpoint["role_graph_adapter"]["variant"] = "changed"
        roles = {**roles, "variant": "synthetic_role_fixture"}
    elif mutation == "role_edges":
        checkpoint["role_adjacency_edge_index"][0, 0] = 1
    elif mutation == "mode":
        checkpoint["module_training_modes"]["mlp"] = False
    elif mutation == "keys":
        checkpoint["state_dicts"]["gcn"]["unexpected"] = torch.tensor(1)
    elif mutation == "shape":
        checkpoint["state_dicts"]["gcn"]["weight"] = torch.ones(1, 1)
    elif mutation == "nan":
        checkpoint["state_dicts"]["gcn"]["weight"][0, 0] = float("nan")
    torch.save(checkpoint, path)
    with pytest.raises((RuntimeError, native.GDesignerNumericalError)):
        native._restore_gdesigner_controller(graph, path, native.sha256_file(path), 0, roles, numerics)


@pytest.mark.parametrize("mutation", ["weight", "mode", "edges", "artifact", "optimized_spatial"])
def test_inference_identity_mutation_fails_closed(tmp_path, mutation):
    graph, _, path, roles, numerics = _controller_fixture(tmp_path)
    digest = native.sha256_file(path)
    state, modes, edges = native._restore_gdesigner_controller(graph, path, digest, 0, roles, numerics)
    if mutation == "weight":
        with torch.no_grad():
            graph.gcn.weight.add_(1)
    elif mutation == "mode":
        graph.mlp.eval()
    elif mutation == "edges":
        graph.role_adj_matrix[0, 0] = 1
    elif mutation == "artifact":
        with path.open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "optimized_spatial":
        graph.optimized_spatial = False
    with pytest.raises(RuntimeError):
        native._require_gdesigner_frozen_controller(graph, state, modes, edges, path, digest)


class _SyntheticRuntime:
    def __init__(self):
        self.usage = {"totals": native.empty_usage(), "phases": {}}
        self.async_semaphore = None
        self.active = self.max_active = 0

    async def call_async(self, messages, *, temperature, phase):
        assert phase.startswith("test:") and temperature == 0.0
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        part = self.usage["phases"].setdefault(phase, native.empty_usage())
        first = part["calls"] == 0
        for target in (self.usage["totals"], part):
            target["calls"] += 1
            target["prompt_tokens"] += 2
            target["completion_tokens"] += 5
            target["total_tokens"] += 7
            target["completion_truncated"] += 1
        return "malformed output" if first else "FINAL ANSWER: 42"

    def summary(self):
        return copy.deepcopy(self.usage)


@pytest.fixture
def native_graph_modules(monkeypatch):
    source = PACKAGE / "upstream" / "GDesigner"
    if not source.is_dir():
        pytest.skip("pinned GDesigner source is not installed")
    pytest.importorskip("torch_geometric")
    sentence_transformers = pytest.importorskip("sentence_transformers")
    import numpy as np

    class SyntheticEmbedding:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, text, **kwargs):
            return np.linspace(0, 1, 384, dtype=np.float32)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", SyntheticEmbedding)
    monkeypatch.syspath_prepend(str(source))
    import GDesigner.prompt.gsm8k_prompt_set
    import GDesigner.agents.math_solver
    import GDesigner.agents.final_decision
    import GDesigner.graph.graph as graph_module
    from GDesigner.graph.graph import Graph

    monkeypatch.setattr(graph_module, "get_sentence_embedding", lambda _: SyntheticEmbedding().encode(""))
    monkeypatch.setenv("AIME_MINILM_PATH", str(PACKAGE / "data"))
    monkeypatch.chdir(PACKAGE)
    return source, graph_module, Graph


def _native_checkpoint(tmp_path, modules):
    _, graph_module, Graph = modules
    roles = native._register_gdesigner_prompt()
    numerics = native.patch_gdesigner_numerics(graph_module)
    graph = Graph(domain="rpas_external", llm_name="GPTChat", agent_names=["MathSolver"] * 4,
                  decision_method="FinalRefer", optimized_spatial=True, optimized_temporal=False,
                  fixed_spatial_masks=[[int(i != j) for j in range(4)] for i in range(4)],
                  fixed_temporal_masks=[[0] * 4 for _ in range(4)],
                  node_kwargs=[{"role": role} for role in roles["role_names"]])
    graph.gcn.eval()
    with torch.no_grad():
        for parameter in graph.gcn.parameters():
            parameter.fill_(0.25)
    checkpoint = {
        "schema": "aime_gdesigner_controller_v1", "seed": 0,
        "state_dicts": {name: copy.deepcopy(getattr(graph, name).state_dict()) for name in ("gcn", "mlp")},
        "role_graph_adapter": roles, "role_adjacency_edge_index": graph.role_adj_matrix.clone(),
        "numerical_compatibility_adapter": numerics,
        "module_training_modes": {"gcn": False, "mlp": True},
        "frozen_phase": "after D_search and before D_select/D_test",
    }
    path = tmp_path / "original.pt"
    torch.save(checkpoint, path)
    return path


def test_native_recovery_only_tests_batches_and_waives_original_gates(tmp_path, monkeypatch, native_graph_modules):
    import native_aime_formal as formal

    source, _, Graph = native_graph_modules
    original = _native_checkpoint(tmp_path, native_graph_modules)
    digest = native.sha256_file(original)
    output = tmp_path / "recovered"
    output.mkdir()
    args = SimpleNamespace(run_dir=output, data_dir=PACKAGE / "data", seed=0, upstream_provenance={})
    rows = {split: [{"id": split, "problem": split, "answer": "42"}] for split in ("search", "select")}
    runtime = _SyntheticRuntime()
    phases, samples, completed, before = [], [], [], []

    async def synthetic_arun(graph, task, **kwargs):
        assert kwargs == {"num_rounds": 1, "max_tries": 3, "max_time": 180}
        assert graph.gcn.training is False and graph.mlp.training is True
        assert all(torch.equal(p, torch.full_like(p, 0.25)) for p in graph.gcn.parameters())
        phase = os.environ["MASBENCH_PHASE"]
        phases.append(phase)
        response = await graph.decision_node.llm.agen("synthetic model input")
        return [response], torch.tensor(0.0)

    monkeypatch.setattr(Graph, "arun", synthetic_arun)
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: pytest.fail("recovery constructed a training optimizer"))
    monkeypatch.setattr(formal, "freeze_aime_selection", lambda *a, **k: phases.append("freeze"))

    def load_rows(data, run, method, filename, count):
        assert phases[0] == "freeze" and before and count == 30
        return [{"id": f"{filename}:{index}", "problem": f"synthetic {index}", "answer": "42"}
                for index in range(5)]

    monkeypatch.setattr(formal, "load_aime_test_rows_after_freeze", load_rows)
    selected = {"num_examples": 30, "accuracy": 22 / 30, "failure_fraction": 0.1,
                "request_failure_fraction": 0.0, "truncation_fraction": 0.28, "rows": []}
    result = native.run_gdesigner("aime", rows, args, "offline-synthetic", runtime, source,
        test_recovery={"controller_path": original, "checkpoint_sha256": digest,
            "selection_metrics": selected, "before_tests": before.append,
            "sample_completed": lambda phase, audit: samples.append((phase, audit)),
            "checkpoint_completed": lambda name, point, run: completed.append((name, copy.deepcopy(point)))})
    assert phases == ["freeze"] + ["test:aime_2025"] * 5 + ["test:aime_2026"] * 5
    assert len(samples) == 10 and len(completed) == 2
    assert runtime.max_active == 4 and runtime.async_semaphore._value == 8
    assert runtime.usage["totals"]["calls"] == 10
    assert runtime.usage["totals"]["total_tokens"] == 70
    assert native.sha256_file(original) == digest == native.sha256_file(output / "gdesigner_controller.pt")
    assert result["search_seconds"] is None and result["search"]["rows"] is None
    assert result["search"]["reused_search"] and result["native_method_config"]["training_row_ids"] is None
    assert result["selection"]["accuracy"] == selected["accuracy"]
    assert before[0] == result["recovery_metadata"]
    assert len(before[0]["nodes"]) == 4 and before[0]["module_training_modes"] == {"gcn": False, "mlp": True}
    for name, point in completed:
        assert point["phase"] == f"test:{name}" and point["calls"] == 5 and point["total_tokens"] == 35
        assert point["gate_audit"]["waived"] and not point["gate_audit"]["original_gate_passed"]
        assert point["Q"]["accuracy"] == point["E"]["accuracy"] == 0.8
    waiver = json.loads((output / "gate_waiver_audit.json").read_text())
    assert waiver["selection"]["original_failure_limit"] == 0.05
    assert set(waiver["test"]) == {"aime_2025", "aime_2026"}


def test_sample_persistence_error_is_not_swallowed(tmp_path, monkeypatch, native_graph_modules):
    import native_aime_formal as formal

    source, _, Graph = native_graph_modules
    original = _native_checkpoint(tmp_path, native_graph_modules)
    output = tmp_path / "recovered"
    output.mkdir()
    args = SimpleNamespace(run_dir=output, data_dir=PACKAGE / "data", seed=0, upstream_provenance={})
    rows = {split: [{"id": split, "problem": split, "answer": "42"}] for split in ("search", "select")}

    async def synthetic_arun(graph, task, **kwargs):
        return ["FINAL ANSWER: 42"], torch.tensor(0.0)

    monkeypatch.setattr(Graph, "arun", synthetic_arun)
    monkeypatch.setattr(formal, "freeze_aime_selection", lambda *a, **k: None)
    monkeypatch.setattr(formal, "load_aime_test_rows_after_freeze", lambda *a: rows["search"])

    def broken_persistence(phase, audit):
        raise OSError("synthetic disk full")

    with pytest.raises(OSError, match="synthetic disk full"):
        native.run_gdesigner("aime", rows, args, "offline", _SyntheticRuntime(), source,
            test_recovery={"controller_path": original, "checkpoint_sha256": native.sha256_file(original),
                "selection_metrics": {"num_examples": 30, "accuracy": 0.7, "failure_fraction": 0.1,
                    "request_failure_fraction": 0.0, "truncation_fraction": 0.28},
                "checkpoint_completed": lambda *a: pytest.fail("unfinished test persisted"),
                "sample_completed": broken_persistence})


def test_default_native_run_still_blocks_test_after_selection_failure(tmp_path, monkeypatch, native_graph_modules):
    import native_aime_formal as formal

    source, _, Graph = native_graph_modules
    output = tmp_path / "original-default-run"
    output.mkdir()
    args = SimpleNamespace(run_dir=output, data_dir=PACKAGE / "data", seed=0, upstream_provenance={})
    rows = {split: [{"id": split, "problem": split, "answer": "42"}] for split in ("search", "select")}
    phases = []

    async def synthetic_arun(graph, task, **kwargs):
        assert kwargs == {"num_rounds": 1, "max_tries": 3, "max_time": 180}
        phases.append(os.environ["MASBENCH_PHASE"])
        return ["malformed output"], torch.tensor(0.0)

    monkeypatch.setattr(Graph, "arun", synthetic_arun)
    monkeypatch.setattr(formal, "freeze_aime_selection", lambda *a, **k: pytest.fail("default gate waived"))
    monkeypatch.setattr(formal, "load_aime_test_rows_after_freeze", lambda *a: pytest.fail("default D_test opened"))
    with pytest.raises(RuntimeError, match="D_select failure rate exceeds 5%"):
        native.run_gdesigner("aime", rows, args, "offline-default", _SyntheticRuntime(), source)
    assert phases == ["search:train"] * 40 + ["search:post_training", "select"]
    assert (output / "gdesigner_controller.pt").exists()
    assert (output / "selection_audit.json").exists()
    assert not (output / "gate_waiver_audit.json").exists()


@pytest.mark.parametrize("phase_callback_fails", [False, True])
def test_async_client_closes_on_inference_loop_even_when_phase_callback_fails(
    tmp_path, monkeypatch, native_graph_modules, phase_callback_fails,
):
    import native_aime_formal as formal

    source, _, Graph = native_graph_modules
    original = _native_checkpoint(tmp_path, native_graph_modules)
    output = tmp_path / "recovered"
    output.mkdir()
    args = SimpleNamespace(run_dir=output, data_dir=PACKAGE / "data", seed=0, upstream_provenance={})
    rows = {split: [{"id": split, "problem": split, "answer": "42"}] for split in ("search", "select")}
    inference_loops, close_loops = [], []

    async def synthetic_arun(graph, task, **kwargs):
        inference_loops.append(asyncio.get_running_loop())
        return ["FINAL ANSWER: 42"], torch.tensor(0.0)

    async def close_async_client():
        close_loops.append(asyncio.get_running_loop())
        assert inference_loops and all(loop is close_loops[0] for loop in inference_loops)
        assert not close_loops[0].is_closed()

    def checkpoint_completed(name, point, runtime):
        if phase_callback_fails:
            raise OSError("synthetic phase audit persistence failure")

    monkeypatch.setattr(Graph, "arun", synthetic_arun)
    monkeypatch.setattr(formal, "freeze_aime_selection", lambda *a, **k: None)
    monkeypatch.setattr(formal, "load_aime_test_rows_after_freeze", lambda *a: rows["search"])
    recovery = {"controller_path": original, "checkpoint_sha256": native.sha256_file(original),
                "selection_metrics": {"num_examples": 30, "accuracy": 0.7, "failure_fraction": 0.1,
                    "request_failure_fraction": 0.0, "truncation_fraction": 0.28},
                "checkpoint_completed": checkpoint_completed, "close_async_client": close_async_client}
    if phase_callback_fails:
        with pytest.raises(OSError, match="synthetic phase audit persistence failure"):
            native.run_gdesigner("aime", rows, args, "offline-close", _SyntheticRuntime(), source,
                                 test_recovery=recovery)
    else:
        native.run_gdesigner("aime", rows, args, "offline-close", _SyntheticRuntime(), source,
                             test_recovery=recovery)
    assert len(close_loops) == 1 and close_loops[0].is_closed()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
