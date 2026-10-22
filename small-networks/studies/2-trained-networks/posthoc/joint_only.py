"""Post-freeze description of Study 2's joint-only failures (not part of the frozen analysis).

For every alignment that passes all 1,458 single interchanges but fails a joint one: where its joint failures
lie, how far its single edits are from natural states, and how likely sampled joint and single panels are to
detect it at the frozen budgets. Also checks, for all irreducible joint failures of the full-domain DAS
alignments, that both single edits involved are off the natural states (Corollary 1).
Reads runs/final and results/final/candidates.csv; writes posthoc/results/joint-only.json.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np  # noqa: E402

STUDY = Path(__file__).resolve().parents[1]
X = np.array([(a, b, n) for a in range(3) for b in range(3) for n in range(3)])
INDEX = {tuple(x): i for i, x in enumerate(X.tolist())}
TR, T1, T2 = np.array([(r, a, b) for r in range(27) for a in range(27) for b in range(27)]).T
BOTH = (X[T1, 0] != X[TR, 0]) & (X[T2, 1] != X[TR, 1])
RA = np.array([INDEX[(X[r, 0], X[t, 1], X[r, 2])] for r, t in zip(TR, T2)])
RB = np.array([INDEX[(X[s, 0], X[r, 1], X[r, 2])] for r, s in zip(TR, T1)])


def main():
    rows = list(csv.DictReader(open(STUDY / "results/final/candidates.csv")))
    perfect = [r for r in rows if int(r["single_fail"]) == 0 and int(r["single_tie"]) == 0]
    cases = []
    for r in perfect:
        if int(r["joint_fail_all"]) == 0:
            continue
        c = np.load(STUDY / "runs/final" / r["network"] / "census.npz")
        i = list(c["names"]).index(r["candidate"])
        joint, off = c["joint"][i], c["off_natural"][i]
        p_joint = float(((joint == 1) & BOTH).sum() / BOTH.sum())
        cases.append({"network": r["network"], "candidate": r["candidate"],
                      "joint_failures": int((joint == 1).sum()), "joint_failures_in_panel": int(((joint == 1) & BOTH).sum()),
                      "joint_queries": int(joint.size), "joint_panel_queries": int(BOTH.sum()),
                      "median_off_natural": float(np.median(off)), "max_off_natural": float(off.max()),
                      "detect_joint_panel": {str(b): 1 - (1 - p_joint) ** (b // 4) for b in (12, 24, 60, 120)}})
    others = []
    for r in perfect:
        if int(r["joint_fail_all"]) == 0:
            c = np.load(STUDY / "runs/final" / r["network"] / "census.npz")
            others.append(float(np.median(c["off_natural"][list(c["names"]).index(r["candidate"])])))
    # Corollary 1 on every irreducible joint failure of the full-domain DAS alignments.
    smallest, count, natural_edits = np.inf, 0, 0
    for r in rows:
        if r["candidate"] not in ("das_r1", "das_r2", "das_r4"):
            continue
        c = np.load(STUDY / "runs/final" / r["network"] / "census.npz")
        i = list(c["names"]).index(r["candidate"])
        sa, sb = c["single"][i][0].reshape(27, 27), c["single"][i][1].reshape(27, 27)
        oa, ob = c["off_natural"][i][0].reshape(27, 27), c["off_natural"][i][1].reshape(27, 27)
        irr = (c["joint"][i] == 1) & BOTH & (sa[RA, T1] == 0) & (sb[RB, T2] == 0)
        if irr.any():
            both = np.minimum(oa[TR, T1], ob[TR, T2])[irr]
            smallest = min(smallest, float(both.min()))
            count += int(irr.sum())
            natural_edits += int((both <= 1e-9).sum())
    # Landing check: does any variable-changing single edit of a single-perfect alignment hit its target state?
    changed = [np.array([X[r, v] != X[d, v] for r in range(27) for d in range(27)]) for v in (0, 1)]
    land_min, landing_edits = np.inf, 0
    for r in perfect:
        c = np.load(STUDY / "runs/final" / r["network"] / "census.npz")
        off = c["off_natural"][list(c["names"]).index(r["candidate"])]
        for v in (0, 1):
            land_min = min(land_min, float(off[v][changed[v]].min()))
            landing_edits += int((off[v][changed[v]] <= 1e-6).sum())
    # The same check over every alignment of every network.
    all_min, all_landing = np.inf, 0
    for folder in sorted((STUDY / "runs/final").iterdir()):
        c = np.load(folder / "census.npz")
        for i in range(len(c["names"])):
            for v in (0, 1):
                o = c["off_natural"][i][v][changed[v]]
                all_min, all_landing = min(all_min, float(o.min())), all_landing + int((o <= 1e-6).sum())
    out = {"single_perfect": len(perfect), "joint_only": len(cases), "cases": cases,
           "all_alignments_min_off_natural_changed_edits": all_min, "all_alignments_landing_edits": all_landing,
           "single_perfect_min_off_natural_changed_edits": land_min,
           "single_perfect_landing_edits": landing_edits,
           "joint_only_with_panel_failures": sum(c["joint_failures_in_panel"] > 0 for c in cases),
           "joint_only_max_panel_share": max(c["joint_failures_in_panel"] / c["joint_panel_queries"] for c in cases),
           "irreducible_in_single_perfect_das": sum(c["joint_failures_in_panel"] for c in cases if c["candidate"].startswith("das_r")),
           "single_perfect_without_joint_failure_median_off_natural": {
               "median": float(np.median(others)), "min": float(min(others)), "max": float(max(others))},
           "mean_detect_joint_panel_120": float(np.mean([c["detect_joint_panel"]["120"] for c in cases])) if cases else None,
           "max_detect_joint_panel_120": float(max(c["detect_joint_panel"]["120"] for c in cases)) if cases else None,
           "irreducible_das_failures_checked": count,
           "irreducible_with_a_natural_edit": natural_edits,
           "smallest_off_natural_of_both_edits": smallest}
    dest = STUDY / "posthoc/results"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "joint-only.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()
