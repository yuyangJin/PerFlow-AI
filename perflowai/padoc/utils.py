"""
This module contains utility functions for PADoc.
"""

import logging
from typing import Any
import numpy as np
from pympler import asizeof
from collections import defaultdict, Counter

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

def to_json_safe(obj: Any) -> Any:
    """
    Recursively convert numpy objects into JSON-serializable types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]

    return obj

def log_memory_breakdown(logger, memory_info):
    total = memory_info.get("total", sum(memory_info.values()))

    logger.info("Memory breakdown of templates:")
    for k, v in sorted(memory_info.items()):
        if k == "total":
            continue
        mb = v / 1024 / 1024
        pct = (v / total) * 100 if total > 0 else 0.0
        logger.info("    %-12s: %8.2f MB (%5.1f%%)", k, mb, pct)

    logger.info("    %-12s: %8.2f MB (100.0%%)", "total", total / 1024 / 1024)

def log_memory_diff(logger, before, after):
    total_before = before.get("total", sum(before.values()))
    total_after = after.get("total", sum(after.values()))

    logger.info("Template value compression summary:")
    logger.info(
        "    %-12s: %8.2f MB -> %8.2f MB  (%s%.1f%%)",
        "total",
        total_before / 1024 / 1024,
        total_after / 1024 / 1024,
        "↓" if total_after < total_before else "↑",
        abs(total_after - total_before) / total_before * 100
        if total_before > 0 else 0.0,
    )

    for k in sorted(before.keys()):
        if k == "total":
            continue

        b = before.get(k, 0)
        a = after.get(k, 0)

        # if b == a:
        #     continue  # 可选：只显示有变化的项

        mb_b = b / 1024 / 1024
        mb_a = a / 1024 / 1024
        delta = mb_a - mb_b

        pct = abs(delta) / mb_b * 100 if mb_b > 0 else 0.0
        arrow = "↓" if delta < 0 else "↑"

        logger.info(
            "    %-12s: %8.2f MB -> %8.2f MB  (%s%.1f%%)",
            k, mb_b, mb_a, arrow, pct
        )


def log_memory_reduction_summary(logger, before, after, title: str = "Template value compression summary"):
    total_before = before.get("total", sum(before.values()))
    total_after = after.get("total", sum(after.values()))

    delta = total_after - total_before
    pct = abs(delta) / total_before * 100 if total_before > 0 else 0.0
    arrow = "↓" if delta < 0 else "↑"

    logger.info(
        "%s: %.2f MB -> %.2f MB (%s%.1f%%, %.2f MB)",
        title,
        total_before / 1024 / 1024,
        total_after / 1024 / 1024,
        arrow,
        pct,
        abs(delta) / 1024 / 1024,
    )


def analyze_node_dict(node_dict):
    """
    Analyze nested dict of nodes without double-counting memory
    - Only consider unique objects
    - Only count children for structure, not add their memory to parent
    - Outputs per-type stats and total memory summary
    """
    type_counts = defaultdict(int)
    type_total_size = defaultdict(int)
    type_field_sizes = defaultdict(lambda: Counter())
    seen_nodes = set()

    def walk(node):
        oid = id(node)
        if oid in seen_nodes:
            return
        seen_nodes.add(oid)

        cls_name = node.__class__.__name__
        type_counts[cls_name] += 1

        # 计算当前 node 的大小（不递归 children）
        size = asizeof.asized(node, detail=1).size
        type_total_size[cls_name] += size

        # 字段分析（不递归 children 内存）
        if hasattr(node, "__slots__"):
            for s in node.__slots__:
                try:
                    v = getattr(node, s)
                except AttributeError:
                    continue
                type_field_sizes[cls_name][s] += asizeof.asized(v, detail=1).size

                # 递归 children 用于统计数量
                if isinstance(v, list):
                    for child in v:
                        if hasattr(child, "__class__") and child.__class__.__name__.endswith("Node"):
                            walk(child)
                elif hasattr(v, "__class__") and v.__class__.__name__.endswith("Node"):
                    walk(v)

    def walk_dict(d):
        for v in d.values():
            if isinstance(v, dict):
                walk_dict(v)
            elif hasattr(v, "__class__") and v.__class__.__name__.endswith("Node"):
                walk(v)

    walk_dict(node_dict)

    # 输出 per-type 结果
    print("===== Node Dict Memory Analysis (Unique) =====\n")
    for t in type_counts:
        count = type_counts[t]
        total_mb = type_total_size[t] / 1024 / 1024
        avg_kb = (type_total_size[t]/count)/1024
        print(f"Type: {t}")
        print(f"  Count: {count}")
        print(f"  Total size: {total_mb:.2f} MB")
        print(f"  Avg size per node: {avg_kb:.2f} KB")
        print("  Field breakdown:")
        for field, sz in type_field_sizes[t].most_common():
            print(f"    {field:20s}: {sz/1024/1024:.2f} MB")
        print("")

    # 最后汇总
    total_node_size = sum(type_total_size.values())
    total_node_mb = total_node_size / 1024 / 1024
    dict_size = asizeof.asizeof(node_dict) / 1024 / 1024
    print("===== Summary =====")
    print(f"Total node memory (sum of unique nodes): {total_node_mb:.2f} MB")
    print(f"Total dict memory (actual dict including all references): {dict_size:.2f} MB")
    print(f"Difference (nodes vs dict): {total_node_mb - dict_size:.2f} MB")
    print("===================")

    return {
        "counts": type_counts,
        "total_sizes": type_total_size,
        "field_sizes": type_field_sizes,
        "total_node_size": total_node_size,
        "dict_size": asizeof.asizeof(node_dict),
    }
