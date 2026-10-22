"""Rerun the studies' saved-output analyses and compare their outputs with the shipped files, byte for byte.

    python tools/replay_analyses.py            # after tools/unpack.py; NumPy and SciPy only
    python tools/replay_analyses.py --all      # also the analyses that need Tracr and PyTorch (make setup)

Every analysis here reads saved records only: the compiled networks of Study 1, the per-network censuses of
Studies 2 and 3, their candidate tables, and Study 2's saved held-out refits. Each runs in a temporary copy of the
release, so the release itself is never written. Exit status 1 if any output differs.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
S1, S2, S3 = "studies/1-compiled-programs", "studies/2-trained-networks", "studies/3-multi-site"
STEPS = [  # (command run from the release root, outputs, modules it needs beyond NumPy and SciPy)
    ([f"{S1}/posthoc/summary.py"], [f"{S1}/posthoc/results/summary.json"], []),
    (["analysis/constructed_example.py"], ["analysis/results/constructed-example.json"], []),
    ([f"{S2}/code/analyze.py", "final"], [f"{S2}/results/final/analysis.json"], []),
    ([f"{S2}/posthoc/joint_only.py"], [f"{S2}/posthoc/results/joint-only.json"], []),
    ([f"{S2}/posthoc/detection_sensitivity.py"], [f"{S2}/posthoc/results/detection-sensitivity.json"], []),
    ([f"{S3}/code/analyze.py", "final"], [f"{S3}/results/final/analysis.json"], []),
    ([f"{S3}/posthoc/mechanism.py"], [f"{S3}/posthoc/results/mechanism.json"], []),
    ([f"{S1}/posthoc/worked_example.py"], [f"{S1}/posthoc/results/worked-example.json"], ["tracr"]),
    ([f"{S1}/posthoc/panel_simulation.py"], [f"{S1}/posthoc/results/panel-simulation.json"], ["tracr"]),
    ([f"{S2}/posthoc/heldout_split.py"], [f"{S2}/posthoc/results/heldout-split.json"], ["torch"]),
]
COPY = ["studies", "analysis", "tools", "release-changes.json"]


def main():
    run_all = "--all" in sys.argv
    steps = [s for s in STEPS if run_all or not s[2]]
    missing = [o for _, outs, _ in steps for o in outs if not (RELEASE / o).exists()]
    if missing:
        sys.exit("saved outputs missing; run tools/unpack.py (make unpack) first")
    absent = sorted({m for _, _, mods in steps for m in mods if importlib.util.find_spec(m) is None})
    if absent:
        sys.exit(f"--all needs {', '.join(absent)} in this Python environment (make setup)")
    failed = 0
    with tempfile.TemporaryDirectory(prefix="replay-") as tmp:
        work = Path(tmp)
        for rel in COPY:
            src, target = RELEASE / rel, work / rel
            if src.is_dir():
                shutil.copytree(src, target, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(src, target)
        for cmd, outs, _ in steps:
            run = subprocess.run([sys.executable, "-B", *cmd], cwd=work, capture_output=True, text=True)
            if run.returncode != 0:
                failed += 1
                print(f"FAIL  {' '.join(cmd)} exited with {run.returncode}")
                print("      " + (run.stdout + run.stderr).strip().replace("\n", "\n      ")[-2000:])
                continue
            for o in outs:
                same = (work / o).read_bytes() == (RELEASE / o).read_bytes()
                failed += not same
                print(f"{'identical' if same else 'DIFFERENT'}  {o}  ({' '.join(cmd)})")
    print(f"{len(steps)} analyses rerun from the saved records: "
          + ("every output identical to the shipped file" if not failed else f"{failed} problem(s)"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
