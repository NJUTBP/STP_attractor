"""Run manifests: command line, versions and task parameters.  Never overwrites.

v3 (author, 2026-10-12): no file hashing; frozen-file SHA checks are optional (sha_check.py) and not recorded here.
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import sys
from pathlib import Path

import numpy as np


def write_manifest(out_dir: str | Path, *, task: str, argv, extra: dict | None = None, code_root: str | Path = ".") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:  # pragma: no cover
        scipy_version = None
    m = {
        "task": task,
        "utc": stamp,
        "argv": list(argv),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy_version,
        "platform": platform.platform(),
        "code_root": str(Path(code_root).resolve()),
    }
    if extra:
        m.update(extra)
    path = out_dir / "run_manifest.json"
    if path.exists():
        path = out_dir / f"run_manifest_{stamp}.json"
    path.write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
