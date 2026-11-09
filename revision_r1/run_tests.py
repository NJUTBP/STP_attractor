"""Run every revision_r1 unit test without pytest:  python revision_r1/run_tests.py"""
import importlib
import pkgutil
import sys
import traceback
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
fails = passes = 0
for mod in pkgutil.walk_packages([str(CODE_ROOT / "revision_r1")], prefix="revision_r1."):
    if not mod.name.split(".")[-1].startswith("test_"):
        continue
    m = importlib.import_module(mod.name)
    for name in sorted(n for n in dir(m) if n.startswith("test_")):
        try:
            getattr(m, name)(); passes += 1; print("PASS", mod.name, name)
        except Exception:
            fails += 1; print("FAIL", mod.name, name); traceback.print_exc()
print(f"{passes} passed, {fails} failed")
sys.exit(1 if fails else 0)
