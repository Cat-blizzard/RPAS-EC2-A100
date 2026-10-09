from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from native_aime_formal import namespace_aime_rows, validate_aime_partitions, native_math_rows, format_aime_task
from native_external_methods import (
    MASBENCH_AXES,
    ModelRuntime,
    GDesignerNumericalError,
    UnsafeCandidate,
    _register_gdesigner_prompt,
    _sha256_json,
    compile_candidate,
    consume_candidate_agent_call_budget,
    patch_gdesigner_numerics,
    require_finite_gdesigner,
    require_finite_gdesigner_model,
    score_row,
    validate_masbench_splits,
)


class NativeExternalAdapterTests(unittest.TestCase):
    def test_aime_task_input_identical_for_all_four_methods(self):
        from native_external_methods import task_text
        row = {"id": "sample", "problem": "Find the value of 7 + 9.", "answer": "16"}
        via_math = native_math_rows([row])[0]["problem"]
        via_external = task_text("aime", row)
        self.assertEqual(via_math, via_external)
        self.assertEqual(via_math, format_aime_task(row["problem"]))
        self.assertTrue(via_math.endswith(row["problem"]))
        self.assertNotIn("\\boxed{16}", via_math)

    def test_aime_role_graph_maps_native_nine_edges_without_complete_graph_collapse(self):
        native_edges = [
            ("Mathematical Analyst", "Math Solver"),
            ("Mathematical Analyst", "Programming Expert"),
            ("Mathematical Analyst", "Inspector"),
            ("Math Solver", "Programming Expert"),
            ("Programming Expert", "Math Solver"),
            ("Programming Expert", "Inspector"),
            ("Inspector", "Math Solver"),
            ("Inspector", "Programming Expert"),
            ("Inspector", "Mathematical Analyst"),
        ]
        registered = {}
        class Registry:
            @staticmethod
            def register(name):
                def decorate(template):
                    registered[name] = template
                    return template
                return decorate
        with patch.dict("sys.modules", {
            "GDesigner.prompt.prompt_set_registry": SimpleNamespace(PromptSetRegistry=Registry),
            "GDesigner.prompt.gsm8k_prompt_set": SimpleNamespace(ROLE_CONNECTION=native_edges),
        }):
            audit = _register_gdesigner_prompt()
        edges = registered["rpas_external"].get_role_connection()
        self.assertEqual(len(set(edges)), 9)
        self.assertEqual(audit["mapped_role_edges"], edges)
        self.assertEqual(audit["native_role_edges"], native_edges)
        self.assertIn(("Algebraic Decomposer", "Independent Solver"), edges)
        self.assertNotIn(("Independent Solver", "Algebraic Decomposer"), edges)
        self.assertEqual(audit["role_names"], ["Algebraic Decomposer", "Independent Solver",
                                             "Consistency Checker", "Solution Synthesizer"])

    def _fake_model_runtime(self, environment):
        sync_factory = Mock(return_value=SimpleNamespace())
        async_factory = Mock(return_value=SimpleNamespace())
        tokenizer_loader = Mock(return_value=object())
        modules = {
            "huggingface_hub": SimpleNamespace(snapshot_download=Mock()),
            "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader)),
            "openai": SimpleNamespace(OpenAI=sync_factory, AsyncOpenAI=async_factory),
        }
        with patch.dict("os.environ", {"AIME_TOKENIZER_PATH": "/fixture/tokenizer", **environment}, clear=True), \
                patch.dict("sys.modules", modules):
            runtime = ModelRuntime("http://fixture/v1/", "adas", "fixture-run")
        return runtime, sync_factory, async_factory

    def test_external_clients_keep_default_request_timeout(self):
        runtime, sync_factory, async_factory = self._fake_model_runtime({})
        self.assertEqual(runtime.request_timeout_s, 180.0)
        self.assertEqual(runtime.summary()["request_timeout_s"], 180.0)
        for factory in (sync_factory, async_factory):
            self.assertEqual(factory.call_args.kwargs["timeout"], 180.0)
            self.assertEqual(factory.call_args.kwargs["max_retries"], 0)

    def test_external_clients_share_configured_finite_request_timeout(self):
        runtime, sync_factory, async_factory = self._fake_model_runtime({"AIME_EXTERNAL_REQUEST_TIMEOUT_S": "600.5"})
        self.assertEqual(runtime.request_timeout_s, 600.5)
        self.assertEqual(runtime.summary()["request_timeout_s"], 600.5)
        for factory in (sync_factory, async_factory):
            self.assertEqual(factory.call_args.kwargs["timeout"], 600.5)

    def test_external_request_timeout_rejects_invalid_values_before_model_loading(self):
        for value in ("", "0", "-1", "nan", "inf", "-inf", "not-a-number"):
            with self.subTest(value=value), \
                    patch.dict("os.environ", {"AIME_EXTERNAL_REQUEST_TIMEOUT_S": value}, clear=True), \
                    self.assertRaisesRegex(ValueError, "AIME_EXTERNAL_REQUEST_TIMEOUT_S.*positive finite"):
                ModelRuntime("http://fixture/v1", "gdesigner", "fixture-run")

    def test_aime_source_local_ids_are_qualified_and_content_is_checked(self):
        validation = namespace_aime_rows(
            [{"id": 1, "problem": "validation problem", "answer": "2"}], "validation"
        )
        aime_2025 = namespace_aime_rows(
            [{"problem_idx": 1, "problem": "2025 problem", "answer": "3"}], "aime_2025"
        )
        aime_2026 = namespace_aime_rows(
            [{"problem_idx": 1, "problem": "2026 problem", "answer": "4"}], "aime_2026"
        )
        self.assertEqual([validation[0]["id"], aime_2025[0]["id"], aime_2026[0]["id"]],
                         ["validation:1", "aime_2025:1", "aime_2026:1"])
        validate_aime_partitions({"D_search": validation, "D_select": aime_2025,
                                  "D_test": aime_2026})
        duplicate = namespace_aime_rows(
            [{"problem_idx": 9, "problem": "  VALIDATION   PROBLEM ", "answer": "2"}], "aime_2025"
        )
        with self.assertRaisesRegex(ValueError, "content overlaps"):
            validate_aime_partitions({"D_search": validation, "D_test": duplicate})

    def test_adas_native_code_patterns_are_supported(self):
        source = '''from collections import Counter
def forward(self, taskInfo):
    values = []
    for i in range(5):
        values.append(i)
        if i == 2:
            break
    return Counter(values).most_common(1)[0][0]
'''
        fn = compile_candidate(source, object())
        self.assertEqual(fn(None, None), 0)

    def test_adas_agent_class_is_in_namespace(self):
        class Stub:
            def __init__(self, output_fields, *args, **kwargs):
                self.output_fields = output_fields
        fn = compile_candidate(
            "def forward(self, taskInfo):\n    agent = LLMAgentBase(['answer'], 'solver')\n    return agent\n",
            Stub,
        )
        self.assertIsInstance(fn(None, None), Stub)

    def test_adas_native_indexed_agent_dispatch_and_underscore_loop_are_supported(self):
        class Stub:
            def __init__(self, output_fields, *args, **kwargs):
                self.output_fields = output_fields

            def __call__(self, *_args, **_kwargs):
                return ("reasoning", "answer")

        source = '''def forward(self, taskInfo):
    agents = [LLMAgentBase(['thinking', 'answer'], 'expert', role=role)
              for role in ['math professor', 'teacher']]
    expert_id = 1
    for _ in range(1):
        thinking, answer = agents[expert_id]([taskInfo], 'solve')
    return answer
'''
        fn = compile_candidate(source, Stub)
        self.assertEqual(fn(None, None), "answer")

    def test_guard_rejects_untrusted_callable_alias_and_non_agent_subscript(self):
        bad = (
            "def forward(self, taskInfo):\n    invoke = taskInfo\n    return invoke()\n",
            "def forward(self, taskInfo):\n    agents = [taskInfo]\n    return agents[0]()\n",
        )
        for source in bad:
            with self.subTest(source=source), self.assertRaises(UnsafeCandidate):
                compile_candidate(source, object())

    def test_guard_rejects_imports_and_introspection(self):
        bad = (
            "import os\ndef forward(self, taskInfo):\n    return 1\n",
            "def forward(self, taskInfo):\n    return __builtins__['open']('/tmp/x')\n",
            "def forward(self, taskInfo):\n    return taskInfo.__class__\n",
            "def forward(self, taskInfo):\n    while True:\n        pass\n",
        )
        for source in bad:
            with self.subTest(source=source), self.assertRaises(UnsafeCandidate):
                compile_candidate(source, object())

    def test_guard_bounds_candidate_loops(self):
        fn = compile_candidate("def forward(self, taskInfo):\n    return sum(range(13))\n", object())
        with self.assertRaises(UnsafeCandidate):
            fn(None, None)

    def test_shared_masbench_ordinal_scoring(self):
        exact = score_row("masbench", "FINAL ANSWER: 11", "11")
        wrong = score_row("masbench", "FINAL ANSWER: 12", "11")
        invalid = score_row("masbench", "I think the answer is 11", "11")
        self.assertEqual(exact["score_0_1_2"], 2)
        self.assertTrue(exact["correct"])
        self.assertEqual(wrong["score_0_1_2"], 1)
        self.assertFalse(wrong["correct"])
        self.assertEqual(invalid["score_0_1_2"], 0)

    def test_repository_frozen_masbench_splits_validate_for_every_axis(self):
        counts = {"search": 24, "select": 24, "test": 60}
        for axis in MASBENCH_AXES:
            rows = {}
            for split, size in counts.items():
                official_split = "test" if split == "test" else "train"
                rows[split] = [
                    {"id": f"{axis}:{split}:{i}", "input": f"task {i}", "answer": "1",
                     "dataset": "masbench", "axis": axis, "axis_value": "1", "attachment": "",
                     "official_split": official_split}
                    for i in range(size)
                ]
            normalized = lambda values: [{key: str(row.get(key, "")) for key in
                ("id", "input", "answer", "dataset", "axis", "axis_value", "attachment")} for row in values]
            manifest = {
                "schema": "rpas_frozen_splits_v1", "group": axis, "seed": 2026,
                "source": {"dataset": "Salesforce/MASBench", "axis": axis},
                "counts": counts,
                "files": {"search": f"{axis}/search_24.jsonl", "select": f"{axis}/select_24.jsonl",
                          "test": f"{axis}/test_60.jsonl"},
                "id_sha256": {name: _sha256_json([str(row["id"]) for row in values])
                              for name, values in rows.items()},
                "content_sha256": {name: _sha256_json(normalized(values)) for name, values in rows.items()},
            }
            validate_masbench_splits(rows, manifest, axis)

    def test_masbench_manifest_rejects_mismatched_axis_metadata(self):
        axis = "depth"
        rows = {name: [{"id": f"{name}:{i}", "input": "x", "answer": "1", "dataset": "masbench",
                        "axis": axis, "axis_value": "1", "attachment": "",
                        "official_split": "test" if name == "test" else "train"}
                       for i in range(size)]
                for name, size in (("search", 24), ("select", 24), ("test", 60))}
        normalized = lambda values: [{key: str(row.get(key, "")) for key in
            ("id", "input", "answer", "dataset", "axis", "axis_value", "attachment")} for row in values]
        manifest = {"schema": "rpas_frozen_splits_v1", "group": axis, "seed": 2026,
            "source": {"dataset": "Salesforce/MASBench", "axis": axis},
            "counts": {"search": 24, "select": 24, "test": 60},
            "files": {"search": "depth/search_24.jsonl", "select": "depth/select_24.jsonl",
                      "test": "depth/test_60.jsonl"},
            "id_sha256": {name: _sha256_json([str(row["id"]) for row in values]) for name, values in rows.items()},
            "content_sha256": {name: _sha256_json(normalized(values)) for name, values in rows.items()}}
        rows["search"][0]["axis"] = "breadth"
        with self.assertRaisesRegex(ValueError, "axis"):
            validate_masbench_splits(rows, manifest, axis)

    def test_shared_aime_exact_and_valid_wrong_scores(self):
        exact = score_row("aime", "FINAL ANSWER: 7", "7")
        wrong = score_row("aime", "FINAL ANSWER: 8", "7")
        invalid = score_row("aime", "The answer might be 7", "7")
        self.assertEqual(exact["score_0_1_2"], 2)
        self.assertEqual(wrong["score_0_1_2"], 1)
        self.assertEqual(invalid["score_0_1_2"], 0)

    def test_adas_agent_call_budget_is_24_per_sample(self):
        from native_external_methods import _candidate_budget

        _candidate_budget.agent_calls = 0
        for _ in range(24):
            consume_candidate_agent_call_budget()
        with self.assertRaisesRegex(UnsafeCandidate, "24-call"):
            consume_candidate_agent_call_budget()


class GDesignerNumericalAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is required for controller numerical regression tests")
        cls.torch = torch

    def test_constant_gram_is_neutral_and_keeps_finite_zero_gradient(self):
        torch = self.torch
        module = SimpleNamespace()
        audit = patch_gdesigner_numerics(module)
        gram = torch.full((16,), 12.498273849487305, requires_grad=True)
        logits = module.min_max_norm(gram)
        self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))
        self.assertTrue(logits.requires_grad)
        loss = -torch.log(torch.sigmoid(logits)).sum()
        require_finite_gdesigner(loss, "fixture loss")
        loss.backward()
        require_finite_gdesigner(gram.grad, "fixture gradient")
        self.assertTrue(torch.equal(gram.grad, torch.zeros_like(gram)))
        self.assertEqual(audit["zero_span_calls"], 1)

    def test_nonzero_spans_keep_exact_upstream_values_and_gradients(self):
        torch = self.torch
        module = SimpleNamespace()
        audit = patch_gdesigner_numerics(module)
        for values in ([1.0, 2.0, 4.0], [1e-12, 2e-12, 4e-12]):
            with self.subTest(values=values):
                native = torch.tensor(values, dtype=torch.float64, requires_grad=True)
                adapted = native.detach().clone().requires_grad_(True)
                expected = (native - native.min()) / (native.max() - native.min()) * 2 - 1
                actual = module.min_max_norm(adapted)
                self.assertTrue(torch.equal(actual, expected))
                expected.square().sum().backward()
                actual.square().sum().backward()
                self.assertTrue(torch.equal(adapted.grad, native.grad))
        self.assertEqual(audit["zero_span_calls"], 0)

    def test_nonfinite_normalization_and_training_values_fail_closed(self):
        torch = self.torch
        module = SimpleNamespace()
        patch_gdesigner_numerics(module)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(GDesignerNumericalError, "normalization input"):
                    module.min_max_norm(torch.tensor([0.0, value]))
                with self.assertRaisesRegex(GDesignerNumericalError, "log probability"):
                    require_finite_gdesigner(torch.tensor(value), "log probability")

    def test_nonfinite_controller_parameters_and_gradients_fail_closed(self):
        torch = self.torch
        graph = SimpleNamespace(gcn=torch.nn.Linear(1, 1), mlp=torch.nn.Linear(1, 1))
        require_finite_gdesigner_model(graph, "fixture", gradients=True)
        graph.gcn.weight.grad = torch.full_like(graph.gcn.weight, float("nan"))
        with self.assertRaisesRegex(GDesignerNumericalError, "gradient gcn.weight"):
            require_finite_gdesigner_model(graph, "fixture", gradients=True)
        graph.gcn.weight.grad = None
        with torch.no_grad():
            graph.mlp.weight.fill_(float("inf"))
        with self.assertRaisesRegex(GDesignerNumericalError, "mlp.weight"):
            require_finite_gdesigner_model(graph, "fixture")


if __name__ == "__main__":
    unittest.main()
