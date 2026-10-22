"""Rerun the study's saved-output analyses and compare their outputs with the shipped files, byte for byte.

    python tools/replay_analyses.py          # after tools/unpack.py; NumPy and SciPy only; exit status 1 on a difference

Reruns `study/code/analyse.py` for the development and confirmation runs of GPT-2 Small (from the per-task records)
and for arm d on the companion study's development and final networks (from the arm's alignment tables and the
companion study's candidate tables), each in a temporary copy of the release from which its saved outputs were
removed first (the analysis refuses to overwrite them), and compares the outputs with the shipped files.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
A = "study/results/analysis"
STEPS = [  # (analyse.py argument, outputs)
    ("development", [f"{A}/development/summary.json", f"{A}/development/alignments.csv"]),
    ("confirmation", [f"{A}/confirmation/summary.json", f"{A}/confirmation/alignments.csv"]),
    ("overlap-development", [f"{A}/overlap-development/summary.json"]),
    ("overlap-final", [f"{A}/overlap-final/summary.json"]),
]
IGNORE = shutil.ignore_patterns("__pycache__", "runs")     # the companion study's networks are not needed here


def main():
    missing = [o for _, outs in STEPS for o in outs if not (RELEASE / o).exists()]
    if missing:
        sys.exit("saved outputs missing; run tools/unpack.py (make unpack) first")
    failed = 0
    with tempfile.TemporaryDirectory(prefix="replay-") as tmp:
        work = Path(tmp)
        for rel in ("Makefile", "release-changes.json", "tools", "study", "reference"):
            src = RELEASE / rel
            if src.is_dir():
                shutil.copytree(src, work / rel, symlinks=True,
                                ignore=IGNORE if rel == "reference" else shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(src, work / rel)
        for phase, outs in STEPS:
            for o in outs:
                (work / o).unlink()
            run = subprocess.run([sys.executable, "-B", "analyse.py", phase], cwd=work / "study/code",
                                 capture_output=True, text=True)
            if run.returncode != 0:
                failed += 1
                print(f"FAIL  analyse.py {phase} exited with {run.returncode}")
                print("      " + (run.stdout + run.stderr).strip().replace("\n", "\n      ")[-2000:])
                continue
            for o in outs:
                same = (work / o).read_bytes() == (RELEASE / o).read_bytes()
                failed += not same
                print(f"{'identical' if same else 'DIFFERENT'}  {o}  (analyse.py {phase})")
    print(f"{len(STEPS)} analyses rerun from the saved records: "
          + ("every output identical to the shipped file" if not failed else f"{failed} problem(s)"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
