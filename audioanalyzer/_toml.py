"""TOML reader with a Python 3.10 fallback.

``tomllib`` is standard library from 3.11, but this package declares ``>=3.10`` — so a bare
``import tomllib`` makes the *whole tool* unimportable on 3.10 (config.py is in every import
chain), not just the TOML feature.  On 3.10 the backport ``tomli`` provides the same API and is
declared in ``pyproject.toml`` under a ``python_version`` marker.
"""
from __future__ import annotations

try:                                    # 3.11+
    import tomllib                      # type: ignore[import-not-found]
except ModuleNotFoundError:             # pragma: no cover - depends on interpreter version
    import tomli as tomllib             # type: ignore[import-not-found,no-redef]

__all__ = ["tomllib"]
