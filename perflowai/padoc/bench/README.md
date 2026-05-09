# PADOC bench harness

Unified evaluation system for the PADOC paper.  All experiments
(compression ratio, analysis speed, scalability, ablation) drive the
same set of compressors through a common
:class:`perflowai.padoc.baselines.BaselineCompressor` interface so
adding a new baseline / new analysis task / new dataset is one file each.

## Layout

```
perflowai/padoc/
  baselines/                # paper-level compressors
    base.py                 #   - BaselineCompressor abstract class
    raw.py                  #   - RawJsonCompressor / RawMsgpackCompressor
    gzip_baseline.py        #   - GzipJsonCompressor / GzipMsgpackCompressor
    tracezip.py             #   - TraceZip (SRT + dictionary + key tokenization)
    scalatrace.py           #   - ScalaTrace (RSD + PRSD greedy folding)
    padoc_adapter.py        #   - PADOC behind the same interface
  bench/                    # the harness itself
    datasets.py             #   - TraceDataset descriptors + manifest loader
    metrics.py              #   - CompressionRecord / AnalysisRecord
    tasks.py                #   - AnalysisTask wrappers
    runner.py               #   - run_compression_matrix / run_analysis_matrix
    report.py               #   - markdown / csv / json renderers
    __main__.py             #   - python -m perflowai.padoc.bench CLI
```

## Quick start

```bash
# List available compressors / tasks / built-in datasets.
python -m perflowai.padoc.bench list compressors
python -m perflowai.padoc.bench list tasks
python -m perflowai.padoc.bench list traces

# Compression-ratio matrix (default: raw_msgpack, gzip_msgpack, tracezip, scalatrace, padoc).
python -m perflowai.padoc.bench compress \
    --traces example_small \
    --out-md /tmp/compress.md --out-csv /tmp/compress.csv --out-json /tmp/compress.json

# Analysis-time matrix (operator hotspot + stream load balance, by default).
python -m perflowai.padoc.bench analyze \
    --traces example_profiler_585 \
    --tasks operator_hotspot stream_load_balance \
    --out-md /tmp/analyze.md
```

`--traces` accepts either built-in dataset names (`example_small`,
`example_profiler_585`) or arbitrary file/directory paths.

## Cluster runs

For paper-scale runs, write a manifest of trace inputs and pass it via
`--manifest`.

```jsonc
// /scratch/ai-trace/datasets.json
{
  "datasets": [
    {"name": "dense_70b_1024gpu", "path": "/scratch/ai-trace/dense_70b/", "gpus": 1024, "layers": 80, "iterations": 10},
    {"name": "moe_671b_1024gpu", "path": "/scratch/ai-trace/moe_671b/",  "gpus": 1024, "layers": 64, "iterations": 8},
    {"name": "vit_512gpu",       "path": "/scratch/ai-trace/vit/",        "gpus": 512,  "layers": 24, "iterations": 32},
    {"name": "leworldmodel_256", "path": "/scratch/ai-trace/leworldmodel/", "gpus": 256, "layers": 36, "iterations": 16}
  ]
}
```

```bash
python -m perflowai.padoc.bench compress --manifest /scratch/ai-trace/datasets.json \
    --out-md  $REPORT/compress.md \
    --out-csv $REPORT/compress.csv \
    --out-json $REPORT/compress.json

python -m perflowai.padoc.bench analyze --manifest /scratch/ai-trace/datasets.json \
    --tasks operator_hotspot stream_load_balance gpu_kernel_breakdown comm_comp_overlap temporal_breakdown \
    --out-md  $REPORT/analyze.md \
    --out-csv $REPORT/analyze.csv \
    --out-json $REPORT/analyze.json
```

## Adding a new compressor

```python
from perflowai.padoc.baselines.base import BaselineCompressor, register_compressor, CompressArtifact

@register_compressor
class MyCompressor(BaselineCompressor):
    name = "mine"

    def compress_trace(self, trace):
        ...
        return CompressArtifact(blob=..., metadata=..., compress_seconds=...)

    def decompress_to_trace(self, blob):
        ...
```

`register_compressor` plugs the new class into the bench CLI
automatically; ``python -m perflowai.padoc.bench list compressors`` will
include it.

## Adding a new analysis task

```python
from perflowai.padoc.bench.tasks import AnalysisTask, _TASKS

class ParallelGroupTask(AnalysisTask):
    name = "parallel_group"

    def run_on_raw(self, trace):
        ...

_TASKS[ParallelGroupTask.name] = ParallelGroupTask
```

If your task can run in-situ on a particular compressor (e.g. PADOC),
override ``has_in_situ_for`` and ``run_in_situ``.  See
``OperatorHotspotTask`` for the canonical example.

## Adding a new ablation switch

`BaselineCompressor` subclasses accept arbitrary ``__init__`` kwargs and
the runner forwards them via the ``--compressor-options`` mechanism (or
direct construction).  PADOC-side ablation switches should live as
``CompressorConfig`` flags on :class:`TemplateCompressor` so they can be
toggled per run; the adapter ``PADOCCompressor`` simply forwards them.

## Verification semantics

`run_compression_matrix(verify=True)` decompresses the artifact and
compares **per-rank, per-event-name, count + total duration**.  It does
*not* require bit-exact event ordering across streams (different
compressors return events in implementation-specific order).  This is
the same level of fidelity the paper claims: "lossless w.r.t. analysis
inputs", not "lossless w.r.t. raw byte stream".
