#!/usr/bin/env python3
"""PADOC 论文消融：等价于 ``python -m perflowai.padoc.bench ablation ...``。

用法示例::

    python scripts/run_ablation.py --manifest /scratch/ai-trace/datasets.json \\
        --mode compress --out-md report/ablation_compress.md

    python scripts/run_ablation.py --traces /path/to/trace.json \\
        --mode both --tasks operator_hotspot --out-json report/ablation.json
"""

from __future__ import annotations

import sys


def main() -> int:
    from perflowai.padoc.bench.__main__ import main as bench_main

    return bench_main(["ablation", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
