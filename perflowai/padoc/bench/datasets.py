"""Lightweight trace-dataset descriptors for the bench harness.

A :class:`TraceDataset` records:

* a human-readable name (used in the report tables),
* the on-disk path (file or directory),
* optional metadata about scale (rank count, layer count, iteration count)
  used for the scalability sweeps.

Example usage::

    >>> ds = TraceDataset(
    ...     name="dense_70b_1024gpu",
    ...     path="/scratch/ai-trace/dense_70b/",
    ...     gpus=1024, layers=80, iterations=10,
    ... )
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceDataset:
    """One trace input for the bench harness."""

    name: str
    path: str
    description: str = ""
    is_directory: bool = False

    # Scale knobs used by run_scalability.
    gpus: Optional[int] = None
    layers: Optional[int] = None
    iterations: Optional[int] = None

    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.is_directory:
            self.is_directory = os.path.isdir(self.path)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Built-in registrations (only the local sample for now)
# ----------------------------------------------------------------------


def builtin_datasets() -> List[TraceDataset]:
    """Datasets that ship with the repo for smoke testing."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    sample_small = os.path.join(repo_root, "tests", "example_trace", "out-1024.json")
    sample_big = os.path.join(repo_root, "tests", "example_trace", "profiler_585.json")
    out: List[TraceDataset] = []
    if os.path.exists(sample_small):
        out.append(
            TraceDataset(
                name="example_small",
                path=sample_small,
                description="Local 1024-event smoke trace (out-1024.json).",
            )
        )
    if os.path.exists(sample_big):
        out.append(
            TraceDataset(
                name="example_profiler_585",
                path=sample_big,
                description="Local profiler_585.json -- exercise larger trace path.",
            )
        )
    return out


def load_dataset(path_or_name: str) -> TraceDataset:
    """Load a dataset by built-in name OR by path on disk.

    If ``path_or_name`` ends with ``.json`` and contains a top-level
    ``"datasets"`` array, every entry is parsed as a :class:`TraceDataset`.
    """
    for dataset in builtin_datasets():
        if dataset.name == path_or_name:
            return dataset

    if os.path.isfile(path_or_name) and path_or_name.endswith(".json"):
        try:
            with open(path_or_name, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "name" in payload and "path" in payload:
                return TraceDataset(**payload)
        except (json.JSONDecodeError, OSError):
            pass

    if not os.path.exists(path_or_name):
        raise FileNotFoundError(f"Dataset not found: {path_or_name}")
    name = os.path.splitext(os.path.basename(path_or_name.rstrip("/")))[0] or path_or_name
    return TraceDataset(name=name, path=path_or_name)


def load_dataset_manifest(manifest_path: str) -> List[TraceDataset]:
    """Load multiple datasets from a JSON manifest file.

    Manifest format::

        {
          "datasets": [
            {"name": "dense_70b", "path": "/data/dense_70b", "gpus": 1024, "layers": 80},
            ...
          ]
        }
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    out: List[TraceDataset] = []
    for entry in payload.get("datasets", []):
        out.append(TraceDataset(**entry))
    return out
