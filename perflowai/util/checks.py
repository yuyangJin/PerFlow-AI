from __future__ import annotations


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)
