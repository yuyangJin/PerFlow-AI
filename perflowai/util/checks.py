from __future__ import annotations


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


# Internal alias expected by other modules
# Keep both names to preserve a clear public API and internal usage
_require = require

__all__ = ["require", "_require"]
