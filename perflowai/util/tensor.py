from __future__ import annotations

from perflowai.workflow.flow import Parameter

from .checks import _require


def _normalize_dtype_name(dtype: str) -> str:
    return str(dtype).strip().lower().replace("_", "")


def is_floating_dtype(dtype: str) -> bool:
    key = _normalize_dtype_name(dtype)
    return key in {
        "float16",
        "fp16",
        "half",
        "bfloat16",
        "bf16",
        "float32",
        "fp32",
        "float",
        "float64",
        "fp64",
        "double",
    }


def dtype_bytes(dtype: str) -> int:
    """Map common dtype strings to bytes/element.

    This project stores dtype as a string in Parameter. We keep the mapping small
    and explicit to avoid silently mis-estimating memory.
    """

    key = _normalize_dtype_name(dtype)
    mapping = {
        "float16": 2,
        "fp16": 2,
        "half": 2,
        "bfloat16": 2,
        "bf16": 2,
        "float32": 4,
        "fp32": 4,
        "float": 4,
        "float64": 8,
        "fp64": 8,
        "double": 8,
        "int8": 1,
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "int32": 4,
        "uint32": 4,
        "int64": 8,
        "uint64": 8,
        "bool": 1,
    }
    _require(key in mapping, f"Unsupported dtype '{dtype}'. Supported: {sorted(mapping.keys())}")
    return mapping[key]


def numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        _require(isinstance(d, int), f"Shape dims must be int, got {d} ({type(d)})")
        _require(d > 0, f"Shape dims must be > 0, got {shape}")
        n *= d
    return n


def tensor_bytes(p: Parameter) -> int:
    _require(p.shape is not None, f"Parameter '{p.name}' must have shape")
    _require(isinstance(p.shape, tuple), f"Parameter '{p.name}' shape must be tuple, got {type(p.shape)}")
    return numel(p.shape) * dtype_bytes(p.dtype)
