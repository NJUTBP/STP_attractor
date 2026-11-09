from __future__ import annotations

import subprocess
from pathlib import Path


def git_revision(root: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None
