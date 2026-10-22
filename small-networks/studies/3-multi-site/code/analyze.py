"""Study 3 analysis from the saved censuses (no model needed).

    python code/analyze.py development|final

Writes results/<split>/analysis.json and results/<split>/candidates.csv. Definitions are in PROTOCOL.md: panels and
costs as in Study 2, explained and unexplained joint failures, the two primary quantities with their intervals, and
the secondary quantities.
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
from scipy.stats import beta  # noqa: E402

STUDY = Path(__file__).resolve().parents[1]
X = np.array([(a, b, n) for a in range(3) for b in range(3) for n in range(3)])
INDEX = {tuple(x): i for i, x in enumerate(X.tolist())}
R, D = np.array([(r, d) for r in range(27) for d in range(27)]).T
TR, T1, T2 = np.array([(r, a, b) for r in range(27) for a in range(27) for b in range(27)]).T
RESTRICTED = (X[:, :2] < 2).all(axis=1)
BUDGETS = [12, 24, 60, 120]
COST = {"full_single": 3, "minimal_single": 3, "restricted_single": 3, "joint": 4}
DAS = ["das_r1", "das_r2", "das_r4"]
BOOT, BOOT_SEED = 10000, 20261007
LANDS = 1e-6
STUDY2 = {"joint_only": (7, 127), "contrast_120": -0.028116135974152533}     # Study 2's frozen results


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
RB = nat(X[T1, 0], X[TR, 1], X[TR, 2])        # r'' = (a_s, b_r, n_r): the B-query a joint query equals if its A-edit lands


def detect(p_a, p_b, n, single):
    """Probability that n uniform draws from a panel contain at least one robust failure."""
    if single:
        return 1 - (1 - p_a) ** ((n + 1) // 2) * (1 - p_b) ** (n // 2)
    return 1 - (1 - p_a) ** n


def clopper_pearson(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return [float(lo), float(hi)]


def candidate_rows(folder: Path):
    rec = json.loads((folder / "record.json").read_text())
    if not rec["competent"]:
        return rec, []
    c = np.load(folder / "census.npz")
    rows = []
    for i, name in enumerate(c["names"].tolist()):
        single, joint, off = c["single"][i], c["joint"][i], c["off_natural"][i]
        row = {"network": folder.name, "family": rec["family"], "seed": rec["seed"], "arch": rec["arch"],
               "kind": rec["kind"], "candidate": name}
        row["single_fail"] = int((single == 1).sum())
        row["single_tie"] = int((single == 2).sum())
        row["single_fail_A"], row["single_fail_B"] = int((single[0] == 1).sum()), int((single[1] == 1).sum())
        row["joint_fail_all"] = int((joint == 1).sum())
        row["joint_tie_all"] = int((joint == 2).sum())
        for key in ("full_single", "minimal_single", "restricted_single"):
            row[f"p_{key}_A"] = float((single[0][MASKS[(key, 0)]] == 1).mean())
            row[f"p_{key}_B"] = float((single[1][MASKS[(key, 1)]] == 1).mean())
        jm = MASKS["joint"]
        row["p_joint"] = float((joint[jm] == 1).mean())
        sa, sb = single[0].reshape(27, 27), single[1].reshape(27, 27)
        explained = (sa[TR, T1] == 1) | (sb[RB, T2] == 1)
        both_pass = (sa[TR, T1] == 0) & (sb[RB, T2] == 0)
        jf = (joint == 1) & jm
        row["joint_fail"] = int(jf.sum())
        row["joint_explained_fail"] = int((jf & explained).sum())
        row["joint_unexplained_fail"] = int((jf & both_pass).sum())
        row["joint_queries"] = int(jm.sum())
        changed_a, changed_b = X[R, 0] != X[D, 0], X[R, 1] != X[D, 1]
        row["a_edits_landing"] = float((off[2][changed_a] < LANDS).mean())
        row["median_off_a"] = float(np.median(off[0][changed_a]))
        row["median_off_b"] = float(np.median(off[1][changed_b]))
        row["median_off_a_at_b"] = float(np.median(off[2][changed_a]))
        rows.append(row)
    return rec, rows


def joint_only(rows):
    perfect = [r for r in rows if r["single_fail"] == 0 and r["single_tie"] == 0]
    k = sum(r["joint_fail_all"] > 0 for r in perfect)
    return {"single_perfect": len(perfect), "with_joint_failure": k,
            "share": k / len(perfect) if perfect else None,
            "ci95": clopper_pearson(k, len(perfect)) if perfect else None,
            "alignments": [f"{r['network']}/{r['candidate']}" for r in perfect if r["joint_fail_all"] > 0]}


def single_det(r, n):
    return detect(r["p_full_single_A"], r["p_full_single_B"], n, True)


def joint_det(r, n):
    return detect(r["p_joint"], None, n, False)


def costings(rs):
    """Joint minus single detection under the three costings, and the add-on comparison (as in Study 2's post hoc)."""
    if not rs:
        return None
    m = lambda xs: float(np.mean(xs))  # noqa: E731
    return {"alignments": len(rs),
            "calls_120": m([joint_det(r, 30) - single_det(r, 40) for r in rs]),
            "equal_queries_30": m([joint_det(r, 30) - single_det(r, 30) for r in rs]),
            "cached_states_120": m([joint_det(r, 120) - single_det(r, 120) for r in rs]),
            "addon_joint_after_single_120": m([(1 - single_det(r, 40)) * joint_det(r, 30) for r in rs]),
            "addon_single_after_single_120": m([single_det(r, 80) - single_det(r, 40) for r in rs]),
            "fail_no_panel_query": sum(r["p_full_single_A"] == 0 and r["p_full_single_B"] == 0 and r["p_joint"] == 0
                                       for r in rs)}


def pooled(rs):
    jf = sum(r["joint_fail"] for r in rs)
    un = sum(r["joint_unexplained_fail"] for r in rs)
    return {"alignments": len(rs), "joint_failures": jf, "explained": sum(r["joint_explained_fail"] for r in rs),
            "unexplained": un, "unexplained_share": un / jf if jf else None,
            "mean_single_rate_A": float(np.mean([r["p_full_single_A"] for r in rs])) if rs else None,
            "mean_single_rate_B": float(np.mean([r["p_full_single_B"] for r in rs])) if rs else None,
            "mean_joint_rate": float(np.mean([r["p_joint"] for r in rs])) if rs else None,
            "mean_a_edits_landing": float(np.mean([r["a_edits_landing"] for r in rs])) if rs else None,
            "all_a_edits_land": sum(r["a_edits_landing"] == 1.0 for r in rs)}


def contrast_ci(das):
    rng = np.random.default_rng(BOOT_SEED)
    by_net = defaultdict(list)
    for r in das:
        by_net[r["network"]].append(r)
    keys = sorted(by_net)
    groups = defaultdict(list)
    for k in keys:
        groups[(by_net[k][0]["family"], by_net[k][0]["arch"])].append(k)
    out = {}
    for b in BUDGETS:
        per_net = {k: np.array([joint_det(r, b // 4) - single_det(r, b // 3) for r in by_net[k]]) for k in keys}
        point = float(np.mean(np.concatenate([per_net[k] for k in keys])))
        draws = []
        for _ in range(BOOT):
            sample = []
            for g in groups.values():
                pick = rng.integers(len(g), size=len(g))
                sample.extend(per_net[g[i]] for i in pick)
            draws.append(np.mean(np.concatenate(sample)))
        out[str(b)] = {"difference": point, "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}
    return out


def summarize(rows, networks):
    out = {}
    das = [r for r in rows if r["candidate"] in DAS]
    cands = sorted({r["candidate"] for r in rows})
    cells = sorted({(r["family"], r["arch"]) for r in rows})
    # Primary 1: joint-only failures among alignments with no single failure or tie.
    out["joint_only"] = joint_only(rows)
    out["joint_only"]["by_candidate"] = {c: joint_only([r for r in rows if r["candidate"] == c]) for c in cands}
    out["joint_only"]["by_kind"] = {k: joint_only([r for r in rows if r["kind"] == k]) for k in ("transformer", "mlp")}
    out["joint_only"]["by_cell"] = {f"{f}/{a}": joint_only([r for r in rows if (r["family"], r["arch"]) == (f, a)])
                                    for f, a in cells}
    k, n = STUDY2["joint_only"]
    jo = out["joint_only"]
    out["prediction_1"] = {"study2_share": k / n, "observed_share": jo["share"],
                           "holds": jo["share"] is not None and jo["share"] > k / n,
                           "holds_beyond_sampling_error": jo["ci95"] is not None and jo["ci95"][0] > k / n}
    # Primary 2: joint minus full-domain single detection at matched budgets, DAS alignments.
    out["primary_joint_minus_single"] = contrast_ci(das)
    c120 = out["primary_joint_minus_single"]["120"]
    out["prediction_2"] = {"study2_contrast": STUDY2["contrast_120"], "observed": c120["difference"],
                           "ci95": c120["ci95"], "holds": c120["difference"] > STUDY2["contrast_120"],
                           "holds_beyond_sampling_error": c120["ci95"][0] > STUDY2["contrast_120"]}
    # Secondary.
    out["costings"] = {"das": costings(das),
                       "by_kind": {k: costings([r for r in das if r["kind"] == k]) for k in ("transformer", "mlp")},
                       "by_cell": {f"{f}/{a}": costings([r for r in das if (r["family"], r["arch"]) == (f, a)])
                                   for f, a in cells}}
    out["decomposition"] = {"das": pooled(das), **{c: pooled([r for r in rows if r["candidate"] == c]) for c in cands}}
    out["decomposition_by_kind"] = {k: pooled([r for r in das if r["kind"] == k]) for k in ("transformer", "mlp")}
    out["decomposition_by_cell"] = {f"{f}/{a}": pooled([r for r in das if (r["family"], r["arch"]) == (f, a)])
                                    for f, a in cells}

    def det(rs, panel, b):
        m = b // COST[panel]
        if panel == "joint":
            return float(np.mean([detect(r["p_joint"], None, m, False) for r in rs]))
        return float(np.mean([detect(r[f"p_{panel}_A"], r[f"p_{panel}_B"], m, True) for r in rs]))
    out["detection"] = {label: {panel: {str(b): det(rs, panel, b) for b in BUDGETS} for panel in COST}
                        for label, rs in [("das", das)] + [(c, [r for r in rows if r["candidate"] == c]) for c in cands]}
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
    with open(dest / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    brief = {k: out[k] for k in ("prediction_1", "prediction_2", "networks")}
    brief["joint_only"] = {k: v for k, v in out["joint_only"].items() if k in ("single_perfect", "with_joint_failure", "share", "ci95")}
    brief["joint_only_by_kind"] = {k: (v["single_perfect"], v["with_joint_failure"]) for k, v in out["joint_only"]["by_kind"].items()}
    brief["joint_only_by_candidate"] = {k: (v["single_perfect"], v["with_joint_failure"]) for k, v in out["joint_only"]["by_candidate"].items()}
    brief["costings_das"] = out["costings"]["das"]
    brief["decomposition_das"] = out["decomposition"]["das"]
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
