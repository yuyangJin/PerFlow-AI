# PADOC 论文实验交接文档

> 这份文档给**集群侧**的 Cursor 看。本地 Cursor 主要负责核心代码、单测；
> 集群侧 Cursor 负责真实 trace 上的所有实验、调脚本、出图。
> 同步靠 `git`：哪边改完就 `git push origin padoc`，另一边 `git pull --rebase`。
> **不要**两边同时改同一个文件。

---

## 1. 这个项目在做什么（一句话）

`perflowai/padoc/` 是一套面向 AI profiler trace 的**结构化模板压缩 + in-situ
分析**系统。论文目标：在 dense-70B / MoE-671B / ViT / LeWorldModel 这类
千卡训练 trace 上，证明 PADOC 在压缩比和分析速度两个维度都超过
gzip / ScalaTrace / TraceZip。

## 2. 已经做完的工作（本地）

* **三个 baseline 重写完毕**（`perflowai/padoc/baselines/`）：
  - `RawJsonCompressor` / `RawMsgpackCompressor` (无压缩)
  - `GzipJsonCompressor` / `GzipMsgpackCompressor`
  - `TracezipCompressor` (SRT + dictionary + key tokenization, 论文 [arxiv 2502.06318](https://arxiv.org/pdf/2502.06318))
  - `ScalaTraceCompressor` (RSD + PRSD greedy folding，仅事件相似性，无结构压缩)
  - `PADOCCompressor` (本系统的 adapter)
  所有 baseline 共享 `BaselineCompressor` 抽象类，roundtrip 测试都过。
* **bench harness**（`perflowai/padoc/bench/`）：
  - `run_compression_matrix` / `run_analysis_matrix` 主跑表
  - `run_gpu_sweep / run_layer_sweep / run_iteration_sweep` 模型可扩展性
  - `run_parallel_compression(backend='process'|'thread')` 并行可扩展性
  - `python -m perflowai.padoc.bench` CLI（`compress / analyze / scalability / parallel / list`）
* **CompressorConfig 消融开关**（`perflowai/padoc/config.py`）：8 个预设
  (`default / no_structural / no_anchor / no_slp / no_args_dedup /
  no_kernel_links / no_name_pattern / minimal`)，本地都通过 roundtrip。
* **存储/树形 profile**（`perflowai/padoc/storage_breakdown.py`,
  `perflowai/padoc/tree_stats.py`）：模板/结构/软链接边的字节占比 +
  depth/branching/SameCPUNode multiplier 直方图 + unique_subtree_shapes。
* **合成 trace 生成器**（`perflowai/padoc/synthetic.py`）：可参数化
  gpus/tp/dp/pp/ep/layers/iterations/micro_batches，本地 sweep 用。
* **5 个分析任务**（`perflowai/padoc/bench/tasks.py`）：
  - `operator_hotspot` ✅ 有 PADOC in-situ 实现 (10–34× 加速)
  - `stream_load_balance` ✅ 有 PADOC in-situ 实现
  - `gpu_kernel_breakdown / comm_comp_overlap / temporal_breakdown` (走 HTA)
  - `layer_operator_balance` (按 `transformer.layers.<N>` 命名探测层间 imbalance)
  - `parallel_group` (按 NCCL 算子 + rank ID 算术启发式标 TP/DP/PP/EP)
* **89 个 pytest 全过**（`tests/padoc_test/`）。

## 3. 集群侧待做的事（按优先级排序）

### P0 — 数据采集
集群上准备好真实 trace 目录。期望布局：
```
/scratch/ai-trace/
  dense_70b_1024gpu/   <- 一个目录里 N 个 rank<i>.json 文件
  moe_671b_1024gpu/
  vit_512gpu/
  leworldmodel_256/
```
> 单 rank 文件大小 PADOC 已经能 stream，多卡走 `TemplateCompressor.inter_compress_dir`。

### P1 — 写 manifest 跑主表
```jsonc
// /scratch/ai-trace/datasets.json
{
  "datasets": [
    {"name": "dense_70b_1024gpu", "path": "/scratch/ai-trace/dense_70b_1024gpu/", "is_directory": true,
     "gpus": 1024, "layers": 80, "iterations": 10},
    {"name": "moe_671b_1024gpu",  "path": "/scratch/ai-trace/moe_671b_1024gpu/",  "is_directory": true,
     "gpus": 1024, "layers": 64, "iterations": 8},
    {"name": "vit_512gpu",        "path": "/scratch/ai-trace/vit_512gpu/",        "is_directory": true,
     "gpus": 512,  "layers": 24, "iterations": 32},
    {"name": "leworldmodel_256",  "path": "/scratch/ai-trace/leworldmodel_256/",  "is_directory": true,
     "gpus": 256,  "layers": 36, "iterations": 16}
  ]
}
```

```bash
mkdir -p /scratch/ai-trace/report
python -m perflowai.padoc.bench compress --manifest /scratch/ai-trace/datasets.json \
    --out-md  /scratch/ai-trace/report/compress.md \
    --out-csv /scratch/ai-trace/report/compress.csv \
    --out-json /scratch/ai-trace/report/compress.json

python -m perflowai.padoc.bench analyze --manifest /scratch/ai-trace/datasets.json \
    --tasks operator_hotspot stream_load_balance gpu_kernel_breakdown \
            comm_comp_overlap temporal_breakdown layer_operator_balance parallel_group \
    --out-md  /scratch/ai-trace/report/analyze.md \
    --out-csv /scratch/ai-trace/report/analyze.csv \
    --out-json /scratch/ai-trace/report/analyze.json
```

### P2 — Ablation 表
对每个真实 trace × 8 个 CompressorConfig 预设跑一次 compress（暂时
没有 CLI flag，需要写一个小 driver 脚本，5 行）：

```python
# scripts/run_ablation.py
import json
from perflowai.padoc import all_ablation_presets, PADOCCompressor
from perflowai.padoc.bench import load_dataset_manifest, run_compression_matrix

datasets = load_dataset_manifest("/scratch/ai-trace/datasets.json")
out = []
for label, cfg in all_ablation_presets().items():
    # PADOCCompressor with custom config
    compressor_options = {"padoc": {"config": cfg}}  # NOTE: BaselineCompressor 注册表当前不直接接受 config
    # 改用直接构造的方式：
    ...
```

> ⚠️ 当前 bench runner 通过 `get_compressor(name)` 查注册表，**不接受
> CompressorConfig 参数**。集群侧需要给 runner 加一个
> `compressor_factory_overrides` 参数，或者写脚本直接调 `PADOCCompressor(config=cfg)`。
> 这个改动可以集群侧做，本地我也可以加，看你方便。

### P3 — 模型可扩展性扫掠（合成 trace 即可，但用集群多核）
```bash
python -m perflowai.padoc.bench scalability --axis gpus       --values 8 16 32 64 128 256 --out-md /scratch/.../sweep_gpus.md
python -m perflowai.padoc.bench scalability --axis layers     --values 4 8 16 32 64       --out-md /scratch/.../sweep_layers.md
python -m perflowai.padoc.bench scalability --axis iterations --values 1 2 4 8 16 32      --out-md /scratch/.../sweep_iters.md
```

### P4 — 并行可扩展性（重点：在集群上才能看到真实 speedup）
```bash
python -m perflowai.padoc.bench parallel \
    --trace-dir /scratch/ai-trace/dense_70b_1024gpu/ \
    --workers 1 2 4 8 16 32 64 \
    --backend process \
    --out-md /scratch/ai-trace/report/parallel.md
```

### P5 — 存储 breakdown / 树形态 表
没现成 CLI，写个 driver：
```python
from perflowai.padoc import (
    TemplateCompressor, measure_storage, measure_tree_statistics,
)
from perflowai.padoc.trace import Trace
trace = Trace.from_dir("/scratch/ai-trace/dense_70b_1024gpu/")
ct = TemplateCompressor().inter_compress(trace)
print(measure_storage(ct).render_markdown())
print(measure_tree_statistics(ct).as_dict())
```

## 4. 集群侧可能踩的坑

| 现象 | 原因 / 解决 |
| --- | --- |
| `ModuleNotFoundError: drawsvg` | `perflowai/__init__.py` 会导入 `visualizer`，集群没装 drawsvg。要么 `pip install drawsvg`，要么直接 `from perflowai.padoc import ...` 不走 `perflowai` 顶包 |
| `ModuleNotFoundError: hta` | HTA 是 Meta 的 holistic-trace-analysis，部分分析任务依赖。如果集群没装：装了最干净 (`pip install holistic-trace-analysis`)；不装的话只能跑 `operator_hotspot / stream_load_balance / layer_operator_balance / parallel_group` 这 4 个任务 |
| `MemoryError` 在大 trace 上 | 一次性把整个目录 `Trace.from_dir(...)` 进内存了。改用 `iter_dir_data` + 逐文件 `intra_compress` 然后再合并；或者 `inter_compress_dir(path)` 自己已经 stream 了 |
| 分析时 PADOC 显示 `decompress=0ms` | 这是预期 — `operator_hotspot` 和 `stream_load_balance` 走 in-situ，根本不解压 |
| 分析时 PADOC 比 raw 慢 | 大概率是分析任务**没有** PADOC in-situ 实现，所以 PADOC 多了一步解压。看 `--in_situ` 列：`-` 就是没 in-situ |
| `Permission denied (publickey)` | 跟代码无关，是 ssh 配置问题，自查 `~/.ssh/config` 里是否给目标 host 指定了 IdentityFile |

## 5. 与本地 Cursor 协作约定

* **本地负责**：`perflowai/padoc/{compressor,event,node,trace,slp,analysis,
  baselines/*,config,storage_breakdown,tree_stats,synthetic}.py` 和 `tests/`。
* **集群侧负责**：
  - `scripts/`（论文实验 driver、SLURM 提交脚本、画图代码 — 这个目录现在还没有，集群侧建）
  - `perflowai/padoc/bench/datasets.py` 里的 `builtin_datasets` （如果想加集群上的具体路径）
  - `perflowai/padoc/bench/manifest.json` 这种本地敏感的文件
* **共享**：`bench/{runner,scalability,parallel,tasks,report}.py` —
  谁要改先 ping 一下，避免冲突。
* **新功能**：用 `[Feat] #112 ...` 的 commit message 风格保持和现有历史一致。
* **测试**：本地 `python -m pytest tests/padoc_test/` 必须通过；集群侧
  额外的测试可以放 `tests/padoc_cluster_test/`，标 `@pytest.mark.cluster` 跳过本地。

## 6. 当前仓库状态

```
分支:    padoc
HEAD:    8b7c199 [Feat] #112 Add ablation switches, storage profile, scalability sweeps
本地:    89 passed, 2 skipped
```

> 第 6 节里那 2 个 skipped 测试是因为 `tests/example_trace/merge_small/`
> 目录在仓库里没有，无视。

## 7. 集群侧第一件事

```bash
# 1. clone + checkout padoc 分支
git clone <repo-url> ~/PerFlow-AI && cd ~/PerFlow-AI && git checkout padoc

# 2. 装依赖
pip install -e .  # 或者 pip install pytest zstandard pympler msgpack numpy

# 3. 确认本地测试也过（无 trace 依赖的部分）
python -m pytest tests/padoc_test/test_baselines.py tests/padoc_test/test_config_ablation.py -x

# 4. 准备 manifest 然后跑 P1 主表
```

如果有任何机器特定的环境配置（conda env / module load / SLURM 脚本），
集群侧 Cursor 就在仓库里建 `cluster/` 或 `scripts/` 目录把这些固化下来。
