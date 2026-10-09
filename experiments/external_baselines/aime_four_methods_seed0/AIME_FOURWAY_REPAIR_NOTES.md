# RPAS AIME 四方法统一实验 · Final V3

**目标：**基于 `Simplification555/RPAS-EC2-A100` 的 `aime-scir-runner-seed0` 分支，构建可审计、四方法公平控制、在单张 A100 上执行的 AIME 2025/2026 复现实验。此包是**一次应用的补丁**，不是完整仓库，也不包含模型权重。不要先运行 V1 / V2 / smoke-fixed。

**冻结基点：** `4e5791c4c56c8ddb098f656589116674d8364217`。请从该提交/分支的**干净工作副本**应用，不覆盖旧实验产物。4 个上游优化器仍需在该仓库的 `upstream/` 内按固定 commit 安装。

## V3 修正了哪些具体问题？

1. **真正执行 16K 协议。** 修复了 V2 的严重遗漏：V2 虽定义 `PROFILE_16K`，`compute_changes()` 却没有应用。现在选 `16k` 后确实修改 vLLM 启动器、AFlow/MaAS runner、ADAS/G-Designer runner、PromptGuard、quality gate、外部聚合器、GPU preflight 和相关测试；应用前逐个断言，并对结果源代码进行内存语法编译。
2. **四方法一致的 AIME 任务输入。** AFlow/MaAS 原本直接使用 `problem`，ADAS/G-Designer 原本额外加入 AIME 作答指令；修复后由相同 task formatter 生成题目文本。答案和 reference solution 仅在评分容器，不进入模型输入。新协议 ID 明确记录该修正。
3. **受控 task-agent 解码。** 同一模型 Qwen3.5-9B、同一 revision、temperature=0、top_p=1、thinking=false、最多 6144 请求输出；ADAS 的*元优化器*保留原生探索温度，不能声称全部请求同温。四方法保持各自的原生搜索轨迹，不能把它们当成等 search-budget 比较。
4. **MaAS ScEnsemble XML 容错。** `ActionNode.xml_fill` 仅在 `solution_letter` 字段为空/格式错误时追加一次明确格式约束并重试，仍使用原生 LLM 与 ScEnsemble 算子。不会编造默认候选、不吞掉失败；重试实际消耗计入调用/Token，另记录恢复与耗尽计数。受限兼容 adapter，论文需披露。
5. **失败也要留下证据。** AFlow、MaAS、ADAS、G-Designer 都尽量在 D_select 门禁抛异常前写入 `selection_audit.json`，收集真实已有的候选记录、错误和统计。已有 MaAS 在 optimizer.test() 本身抛异常时可能没有完整逐题记录，不能假称具备全阶段无损断点恢复。
6. **正确的 prelock 数据预检。** `scripts/verify_aime_frozen_data.py` 对 AIME 2025/2026 仅核对原始 bytes SHA256 与行数，**不解析 D_test JSONL 题目**；验证集 60/30 才读取题目，检查分区和答案。选定并锁定候选后，原 runner 才能打开 D_test 题目并做内容重叠审计。
7. **补丁原子性与回滚。** 所有源码改动先在内存中构建、校验 16K 配置、编译 Python；再从仓库外备份原始字节，原子写入，校验整个 `SHA256SUMS`，有异常恢复改动前的文件。现有历史发布结果与冻结数据完全不改。

## 严格统一的协议

| 条目 | 固定设定 |
|---|---|
| 模型 | Qwen/Qwen3.5-9B，固定模型/tokenizer revision，BF16，TP=1 |
| 算力 | 1×A100，四方法顺序执行，绝不同时占卡 |
| D_search / D_select | 冻结的 60 / 30 道验证题，data_seed=2026 |
| D_test | AIME 2025 和 AIME 2026 各 30 道，selection lock 之后读取 |
| Task executor | temperature=0，top_p=1，enable_thinking=false |
| ADAS 元优化器 | 保留原生候选生成/反思的 temperature |
| Context | **16,384 tokens**（先过真实 A100 服务 smoke） |
| Output cap | 每请求 6,144 tokens；实际 effective cap 必须记录 |
| Batch/并发 | vLLM max_num_seqs=8；method concurrency=8 |
| 选择 | D_select；Q/E 按原 Q/E 规则，质量带宽 delta=0.05 |
| 统一评分 | `answer_protocol_v4`，整数 exact match，同时保留 0/1/2 映射 |
| 质量门禁 | 原有 5% workflow/截断/请求失败相关门槛不放宽 |
| 上游 search 核心 | AFlow optimizer / MaAS controller / ADAS evolution / G-Designer Graph+GCN |

原生性仍存在如下**必须披露的适配**：AFlow/MaAS 的 MATH→AIME 协议与 bounded code-execution；MaAS 的 native prompt bug 修复和此处 XML compat 重试；ADAS `restricted_ast_v1` 候选执行器；G-Designer AIME role-graph、有限值及 zero-span numerical guard。上游核心不等于原仓库逐字运行。

## 1. 只运行一次补丁

```bash
# 在工作机上：
git clone --branch aime-scir-runner-seed0 https://github.com/Simplification555/RPAS-EC2-A100.git
cd RPAS-EC2-A100
git rev-parse HEAD
# 应当为 4e5791c4c56c8ddb098f656589116674d8364217

# 将此 ZIP 解压到仓库外，用本包的脚本
python /path/to/FinalV3/apply_patch.py --repo . --context-profile 16k --check
python /path/to/FinalV3/apply_patch.py --repo . --context-profile 16k --apply
```

**前提：**该仓库完整、上游源文件与指定基点匹配。遇到 missing/duplicate anchor 时脚本会拒绝替换；不要手工跳过失败断言。补丁会修订原目录的 `SHA256SUMS`。`--apply` 是有写入操作的，不是仅生成 diff。

## 2. 四方法运行前的 CPU gate

```bash
cd experiments/external_baselines/aime_four_methods_seed0
python scripts/verify_aime_frozen_data.py
bash scripts/setup_upstreams.sh   # 需具备网络和固定上游版本
python scripts/audit_native_fidelity.py --context-profile 16k --require-upstream
sha256sum -c SHA256SUMS
python -m pytest -q \
  experiments/test_answer_protocol.py \
  experiments/test_dtest_blind_freeze.py \
  experiments/test_aime_quality_gate.py \
  experiments/test_aime_runtime_safety.py \
  experiments/test_native_external_methods.py \
  experiments/test_aggregate_external_methods.py \
  experiments/test_maas_xml_guard.py
```

所有门禁通过才允许消耗正式 GPU 时间。此处 native-fidelity 检查是**源代码指纹 + pinned checkout**审计，并不等同于四个方法训练及解题表现已通过。

## 3. 单卡 A100 运行

将下列变量设置成你机器的实际绝对路径（注意 `AIME_MINILM_PATH`）：

```bash
export CUDA_VISIBLE_DEVICES=0
export AIME_MODEL_PATH=/absolute/path/Qwen3.5-9B
export AIME_MINILM_PATH=/absolute/path/all-MiniLM-L6-v2
export AIME_MODEL_VERIFICATION_REPORT=/absolute/path/model_verification.json
export AIME_PYTHON=/absolute/path/serve/bin/python
export AIME_VLLM_BIN=/absolute/path/serve/bin/vllm
export AIME_AFLOW_PYTHON=/absolute/path/aflow/bin/python
export AIME_MAAS_PYTHON=/absolute/path/maas/bin/python
export AIME_ADAS_PYTHON=/absolute/path/adas/bin/python
export AIME_GDESIGNER_PYTHON=/absolute/path/gdesigner/bin/python

# 原仓库生成模型报告：
# python scripts/verify_local_models.py --help

# 用真实模型进行 synthetic preflight，不读正式 test：
bash scripts/run_single48g_all.sh diagnostics
```

**上机正式运行前的最后闸门：**除了 synthetic PASS，还要在 **D_search** 中检查真实长请求的 model/tokenizer 是否一致、`finish_reason=length`、`effective_max_tokens`、MaAS `solution_letter` XML 恢复率、ADAS AST 拒绝率、G-Designer 数值梯度，以及 GPU OOM/请求错误。某种方法的这些缺陷若未消除，不要一次烧完整四方法正式预算。这个包没有假装拥有完整的四方法 GPU 真实 canary 结果。

```bash
# 全新的输出根目录；禁止覆盖以前的 failure evidence：
export AIME_OUTPUT_ROOT="$PWD/outputs/aime_finalv3_seed0_$(date +%Y%m%d_%H%M%S)"
bash scripts/run_single48g_all.sh all --method-order maas,adas,gdesigner,aflow

# 四方法结果统一诊断，不将 FAIL 自动变成正式通过：
python scripts/audit_fourway_attempts.py --outputs "$AIME_OUTPUT_ROOT" \
  --json-out "$AIME_OUTPUT_ROOT/diagnosis.json"
```

如果脚本失败，先保留整个输出目录，读取 `selection_audit.json`/`run_manifest.json`/`quality_gate.json`，只在 D_search/synthetic 范围内针对**客观工程 bug**做版本化修复。质量门禁仍锁住 D_test，不能人为将失败标成 PASS。

## 4. GPU 内存与实验公平性

A100 40GB/80GB、vLLM 版本、KV cache 及同机运行的 PyTorch 控制器会影响 16K 的可运行性。即便请求上限为 6144，模型仍可能发生 `length` 截断；**没有代码可以保证四方法 100% 通过 5% 门禁**。16K 失败时，停止全部方法，在全新干净 clone 使用包中明确的 `8k` 预定义 profile 统一重跑，并将性能限制在论文中披露。绝不允许 AFlow 16K、MaAS 8K、其他方法不同预算拼入一张表。

四方法的**原生搜索预算不匹配**：AFlow 8 轮 mutation、MaAS 1 round controller、ADAS 30 generations、G-Designer 10×4 REINFORCE。报告每方法真实 `search+select` token/calls/wall-time，另报告 final inference token/calls；不能宣称 matched search tokens。

更重要的是：公开 AIME 2025/2026 在你之前 2026-10-02～04 的运行中已有 D_test 暴露。这版应描述为**修复后复现实验**，不能包装成此前完全未触碰的盲测；若需要新的 confirmatory claim，应有独立、未用于调参的新测试集。

## 5. 本包检测的边界

仓库不在当前执行容器内，且该容器不能直接从 GitHub 下载完整仓库。我们依据 GitHub 上该分支的代码审核了关键运行路径，并对补丁器、16K profile 实际 materialize、MaAS XML 重试、fail-closed、D_test prelock 校验做了本地 CPU 离线测试。**这些测试不等于“完整仓库原生环境 + 单 A100 运行已通过”**。在你的机器上请先执行 `--check`、`--apply`、`sha256sum -c` 与上述 CPU gate，随后再做 GPU gate。

包内文件：`apply_patch.py`（唯一入口）、`maas_xml_guard.py`、`test_maas_xml_guard_repo.py`（复制进仓库）、`audit_native_fidelity.py`、`verify_aime_frozen_data.py`、`test_patch_logic.py`、`test_final_profile_integration.py`、`test_data_prelock.py`、`test_maas_xml_guard.py` 和此文档。
