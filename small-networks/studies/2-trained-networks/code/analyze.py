"""Study 2 analysis from the saved censuses (no model needed).

    python code/analyze.py development|final

Writes results/<split>/analysis.json and results/<split>/candidates.csv. Definitions are in PROTOCOL.md:
query spaces of the four panels, irreducible joint failures, detection probabilities at matched budgets,
and the program-cluster bootstrap.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np  # noqa: E402

STUDY = Path(__file__).resolve().parents[1]
X = np.array([(a, b, n) for a in range(3) for b in range(3) for n in range(3)])
INDEX = {tuple(x): i for i, x in enumerate(X.tolist())}
R, D = np.array([(r, d) for r in range(27) for d in range(27)]).T
TR, T1, T2 = np.array([(r, a, b) for r in range(27) for a in range(27) for b in range(27)]).T
RESTRICTED = (X[:, :2] < 2).all(axis=1)
BUDGETS = [12, 24, 60, 120]
COST = {"full_single": 3, "minimal_single": 3, "restricted_single": 3, "joint": 4}
DAS = ["das_r1", "das_r2", "das_r4"]
BOOT, BOOT_SEED = 10000, 20261006


def nat(a, b, n):
    return np.array([INDEX[t] for t in zip(a.tolist(), b.tolist(), n.tolist())])


def panel_masks():
    out = {}
    for v in (0, 1):
        changed = X[R, v] != X[D, v]
        o = 1 - v
        out[("full_single", v)] = changed
        out[("minimal_single", v)] = changed & (X[R, o] == X[D, o]) & (X[R, 2] == X[D, 2])
        out[("restricted_single", v)] = changed & RESTRICTED[R] & RESTRICTED[D]
    out["joint"] = (X[T1, 0] != X[TR, 0]) & (X[T2, 1] != X[TR, 1])
    return out


MASKS = panel_masks()
RA = nat(X[TR, 0], X[T2, 1], X[TR, 2])        # natural recipient the joint query reduces to after the B-edit
RB = nat(X[T1, 0], X[TR, 1], X[TR, 2])        # ... after the A-edit


def detect(p_a, p_b, n, single):
    """Probability that n draws from a panel contain at least one robust failure."""
    if single:
        return 1 - (1 - p_a) ** ((n + 1) // 2) * (1 - p_b) ** (n // 2)
    return 1 - (1 - p_a) ** n


def candidate_rows(folder: Path):
    rec = json.loads((folder / "record.json").read_text())
    if not rec["competent"]:
        return rec, []
    c = np.load(folder / "census.npz")
    rows = []
    for i, name in enumerate(c["names"].tolist()):
        single, joint, off = c["single"][i], c["joint"][i], c["off_natural"][i]
        row = {"network": folder.name, "family": rec["family"], "seed": rec["seed"], "arch": rec["arch"],
               "candidate": name}
        row["single_fail"] = int((single == 1).sum())
        row["single_tie"] = int((single == 2).sum())
        row["joint_fail_all"] = int((joint == 1).sum())
        row["joint_tie_all"] = int((joint == 2).sum())
        for key in ("full_single", "minimal_single", "restricted_single"):
            fa = (single[0][MASKS[(key, 0)]] == 1).mean()
            fb = (single[1][MASKS[(key, 1)]] == 1).mean()
            row[f"p_{key}_A"], row[f"p_{key}_B"] = float(fa), float(fb)
        jm = MASKS["joint"]
        row["p_joint"] = float((joint[jm] == 1).mean())
        sa, sb = single[0].reshape(27, 27), single[1].reshape(27, 27)
        reduce_fail = (sa[RA, T1] == 1) | (sb[RB, T2] == 1)
        both_pass = (sa[RA, T1] == 0) & (sb[RB, T2] == 0)
        jf = (joint == 1) & jm
        row["joint_fail"] = int(jf.sum())
        row["joint_reducible_fail"] = int((jf & reduce_fail).sum())
        row["joint_irreducible_fail"] = int((jf & both_pass).sum())
        row["joint_queries"] = int(jm.sum())
        row["joint_both_reduced_pass"] = int((jm & both_pass).sum())
        row["joint_pass_despite_reduced_fail"] = int(((joint == 0) & jm & reduce_fail).sum())
        oa, ob = off[0].reshape(27, 27)[TR, T1], off[1].reshape(27, 27)[TR, T2]
        sel = jm & both_pass
        row["_irr"] = (joint[sel] == 1)
        row["_min_off"], row["_max_off"] = np.minimum(oa, ob)[sel], np.maximum(oa, ob)[sel]
        row["median_off_natural"] = float(np.median(off))
        rows.append(row)
    return rec, rows


def auc(scores_pos, scores_neg):
    """Probability that a random positive outranks a random negative (ties count half): Mann-Whitney U / (n1 n2)."""
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return None
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(scores_pos, scores_neg).statistic / (len(scores_pos) * len(scores_neg)))


def summarize(rows, networks):
    out = {}
    das = [r for r in rows if r["candidate"] in DAS]
    perfect = [r for r in rows if r["single_fail"] == 0 and r["single_tie"] == 0]
    out["single_perfect"] = {"candidates": len(perfect), "with_joint_failure": sum(r["joint_fail_all"] > 0 for r in perfect),
                             "by_candidate": {c: sum(r["candidate"] == c for r in perfect) for c in sorted({r["candidate"] for r in rows})}}
    def pooled(rs):
        jf = sum(r["joint_fail"] for r in rs)
        irr = sum(r["joint_irreducible_fail"] for r in rs)
        return {"candidates": len(rs), "joint_failures": jf, "irreducible": irr,
                "irreducible_share": irr / jf if jf else None,
                "mean_single_rate": float(np.mean([(r["p_full_single_A"] + r["p_full_single_B"]) / 2 for r in rs])) if rs else None,
                "mean_joint_rate": float(np.mean([r["p_joint"] for r in rs])) if rs else None,
                "mean_irreducible_rate": float(np.mean([r["joint_irreducible_fail"] / r["joint_queries"] for r in rs])) if rs else None,
                "candidates_with_irreducible": sum(r["joint_irreducible_fail"] > 0 for r in rs)}
    out["decomposition"] = {"das": pooled(das)}
    for c in sorted({r["candidate"] for r in rows}):
        out["decomposition"][c] = pooled([r for r in rows if r["candidate"] == c])
    cells = sorted({(r["family"], r["arch"]) for r in rows})
    out["decomposition_by_cell"] = {f"{f}/{a}": pooled([r for r in das if r["family"] == f and r["arch"] == a]) for f, a in cells}
    # Detection at matched budgets (expected fraction of candidates with at least one failure in the sample).
    def det(rs, panel, b):
        n = b // COST[panel]
        if panel == "joint":
            return float(np.mean([detect(r["p_joint"], None, n, False) for r in rs]))
        return float(np.mean([detect(r[f"p_{panel}_A"], r[f"p_{panel}_B"], n, True) for r in rs]))
    out["detection"] = {}
    for label, rs in [("das", das)] + [(c, [r for r in rows if r["candidate"] == c]) for c in sorted({r["candidate"] for r in rows})]:
        out["detection"][label] = {panel: {str(b): det(rs, panel, b) for b in BUDGETS} for panel in COST}
    # Primary contrast with a program-cluster bootstrap: joint minus full-domain singles, DAS candidates.
    rng = np.random.default_rng(BOOT_SEED)
    keys = sorted({r["network"] for r in das})
    by_net = defaultdict(list)
    for r in das:
        by_net[r["network"]].append(r)
    cell_of = {k: (by_net[k][0]["family"], by_net[k][0]["arch"]) for k in keys}
    contrast = {}
    for b in BUDGETS:
        per_net = {k: np.array([detect(r["p_joint"], None, b // 4, False) -
                                detect(r["p_full_single_A"], r["p_full_single_B"], b // 3, True) for r in by_net[k]]) for k in keys}
        point = float(np.mean(np.concatenate([per_net[k] for k in keys])))
        groups = defaultdict(list)
        for k in keys:
            groups[cell_of[k]].append(k)
        draws = []
        for _ in range(BOOT):
            sample = []
            for g in groups.values():
                pick = rng.integers(len(g), size=len(g))
                sample.extend(per_net[g[i]] for i in pick)
            draws.append(np.mean(np.concatenate(sample)))
        contrast[str(b)] = {"difference": point, "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}
    out["primary_joint_minus_single"] = contrast
    # Coverage: alignments fitted on the restricted support.
    rs = [r for r in rows if r["candidate"] == "das_restricted_r2"]
    out["coverage"] = {"candidates": len(rs),
                       "restricted_rate": float(np.mean([(r["p_restricted_single_A"] + r["p_restricted_single_B"]) / 2 for r in rs])) if rs else None,
                       "full_rate": float(np.mean([(r["p_full_single_A"] + r["p_full_single_B"]) / 2 for r in rs])) if rs else None,
                       "pass_all_restricted": sum(r["p_restricted_single_A"] == 0 and r["p_restricted_single_B"] == 0 for r in rs),
                       "fail_some_full": sum(r["p_full_single_A"] > 0 or r["p_full_single_B"] > 0 for r in rs)}
    # Exploratory: does the smaller or the larger of the two single-edit distances predict irreducible failures?
    irr = np.concatenate([r["_irr"] for r in das]) if das else np.array([])
    mn = np.concatenate([r["_min_off"] for r in das]) if das else np.array([])
    mx = np.concatenate([r["_max_off"] for r in das]) if das else np.array([])
    out["exploratory_distance_auc"] = {"queries": int(len(irr)), "irreducible": int(irr.sum()),
                                       "auc_min_distance": auc(mn[irr], mn[~irr]), "auc_max_distance": auc(mx[irr], mx[~irr])}
    out["networks"] = networks
    return out


def main():
    split = sys.argv[1]
    rows, networks = [], {"trained": 0, "competent": 0, "attempts_needed": defaultdict(int), "failed": []}
    for folder in sorted((STUDY / "runs" / split).iterdir()):
        rec, rs = candidate_rows(folder)
        networks["trained"] += 1
        networks["attempts_needed"][len(rec["training_attempts"])] += 1
        if rec["competent"]:
            networks["competent"] += 1
        else:
            networks["failed"].append(folder.name)
        rows.extend(rs)
    networks["attempts_needed"] = dict(networks["attempts_needed"])
    out = summarize(rows, networks)
    dest = STUDY / "results" / split
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    fields = [k for k in rows[0] if not k.startswith("_")]
    with open(dest / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(json.dumps({k: v for k, v in out.items() if k not in ("detection",)}, indent=1))


if __name__ == "__main__":
    main()
