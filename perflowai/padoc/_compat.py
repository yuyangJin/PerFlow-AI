from __future__ import annotations

from types import SimpleNamespace

try:
    from pympler import asizeof as _asizeof_module
except ModuleNotFoundError:
    from sys import getsizeof

    class _AsizeofFallback:
        @staticmethod
        def asizeof(obj):
            return getsizeof(obj)

        @staticmethod
        def asized(obj, detail=1):
            return SimpleNamespace(size=getsizeof(obj))

    asizeof = _AsizeofFallback()
else:
    asizeof = _asizeof_module
