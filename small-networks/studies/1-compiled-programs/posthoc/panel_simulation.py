"""Sampling distribution of the frozen panel decisions of Study 1 (post hoc).

For every panel and budget, replays the frozen sampling rule 10,000 times on the exact census: queries are drawn
uniformly with replacement from the panel's query space, shared by all alignments of a program (as in the frozen
run), singles alternate between the two variables, and an alignment is accepted if none of its sampled queries
fails. Ties are counted once as passes and once as failures. Reports the expected number of accepted invalid
alignments, a 95% range (2.5% quantile with ties as failures to 97.5% quantile with ties as passes), and whether
each frozen observation lies inside it. Output: posthoc/results/panel-simulation.json.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "02-Code"))
sys.path.insert(0, str(STUDY / "posthoc"))
import numpy as np  # noqa: E402
from census import INPUTS, PAIRS, RESTRICTED, TRIPLES, decide, natural  # noqa: E402
from v2_model import Network  # noqa: E402

REPS, SEED = 10000, 20261006
BUDGETS = [12, 24, 60, 120]
COST = {"ordinary": 3, "full_single": 3, "heldout_single": 3, "faithfulness": 3, "joint": 4, "raw_latent": 4, "transport": 6}
INVALID = {"lookup": ["mean_restricted", "das_restricted", "random"], "additive": ["random"]}


def program_spaces(folder: Path):
    """Per-query decisions (0 pass, 1 fail, 2 tie) of the program's invalid alignments in every panel's space."""
    net = Network.load(folder)
    _, layers = net.native_numpy(INPUTS)
    raw = layers[net.site][:, -1]
    z = net.encode_site(raw)
    table = np.asarray(net.meta["table"])
    oracle = table[INPUTS[:, 0], INPUTS[:, 1]]
    cand = np.load(folder / "candidates.npz")
    r, d = PAIRS.T
    tr, t1, t2 = TRIPLES.T
    stream = (INPUTS[t1, 0] != INPUTS[tr, 0]) & (INPUTS[t2, 1] != INPUTS[tr, 1])
    jt = natural(INPUTS[t1, 0], INPUTS[t2, 1], INPUTS[tr, 2])
    spaces = defaultdict(list)
    for name in INVALID[net.meta["family"]]:
        pa, pb = cand[name + "_a"], cand[name + "_b"]
        for v, p, key in ((0, pa, "A"), (1, pb, "B")):
            tgt = INPUTS[r].copy()
            tgt[:, v] = INPUTS[d, v]
            dec = decide(net, raw[r], z[r] + (z[d] - z[r]) @ p, table[tgt[:, 0], tgt[:, 1]])
            changed = INPUTS[r, v] != INPUTS[d, v]
            spaces[f"ordinary_{key}"].append(dec[changed & RESTRICTED[r] & RESTRICTED[d]])
            spaces[f"full_single_{key}"].append(dec[changed])
            spaces[f"heldout_single_{key}"].append(dec[changed & (tgt[:, :2] == 2).any(axis=1)])
        fz = z[r] + (z[d] - z[r]) @ (np.eye(net.d) - pa - pb)
        spaces["faithfulness"].append(decide(net, raw[r], fz, oracle[r]))
        jz = (z[tr] + (z[t1] - z[tr]) @ pa + (z[t2] - z[tr]) @ pb)[stream]
        spaces["joint"].append(decide(net, raw[tr][stream], jz, table[INPUTS[t1, 0], INPUTS[t2, 1]][stream]))
        delta = jz - z[jt[stream]]
        spaces["raw_latent"].append((np.linalg.norm(delta, axis=1) > 1e-4).astype(np.int8))
        tdec = [decide(net, np.repeat(raw[[a]], len(delta), axis=0), z[a] + delta, np.full(len(delta), oracle[a]))
                for a in range(27)]
        spaces["transport"].append(np.concatenate(tdec))
    return {k: np.stack(v) for k, v in spaces.items()}            # candidates x queries


def accepted(fails: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """fails: candidates x queries (bool); idx: reps x n. Returns reps vector of accepted counts."""
    return (~fails[:, idx].any(axis=2)).sum(axis=0)


def main():
    rng = np.random.default_rng(SEED)
    totals = {(p, b, t): np.zeros(REPS, dtype=np.int64) for p in COST for b in BUDGETS for t in ("fail", "pass")}
    for folder in sorted((STUDY / "03-Data/v2/final").iterdir()):
        sp = program_spaces(folder)
        for panel in COST:
            for b in BUDGETS:
                n = b // COST[panel]
                if panel in ("ordinary", "full_single", "heldout_single"):
                    na, nb = (n + 1) // 2, n // 2
                    a, bb = sp[f"{panel}_A"], sp[f"{panel}_B"]
                    ia = rng.integers(a.shape[1], size=(REPS, na))
                    ib = rng.integers(bb.shape[1], size=(REPS, nb))
                    for tie, val in (("fail", True), ("pass", False)):
                        fa = (a == 1) | ((a == 2) & val)
                        fb = (bb == 1) | ((bb == 2) & val)
                        both = ~(fa[:, ia].any(axis=2) | fb[:, ib].any(axis=2))
                        totals[(panel, b, tie)] += both.sum(axis=0)
                else:
                    s = sp[panel]
                    idx = rng.integers(s.shape[1], size=(REPS, n))
                    for tie, val in (("fail", True), ("pass", False)):
                        totals[(panel, b, tie)] += accepted((s == 1) | ((s == 2) & val), idx)
    observed = defaultdict(int)
    with open(STUDY / "04-Results/v2/candidate-panel-results.csv") as f:
        for row in csv.DictReader(f):
            if row["valid"] == "False" and row["accepted"] == "True":
                observed[(row["method"], int(row["budget"]))] += 1
    out = []
    for panel in COST:
        for b in BUDGETS:
            lo, hi = totals[(panel, b, "fail")], totals[(panel, b, "pass")]
            low, high = float(np.quantile(lo, 0.025)), float(np.quantile(hi, 0.975))
            obs = observed[(panel, b)]
            out.append({"panel": panel, "budget": b, "mean_ties_fail": float(lo.mean()), "mean_ties_pass": float(hi.mean()),
                        "range95_low": low, "range95_high": high, "observed": obs, "inside": low <= obs <= high})
    dest = STUDY / "posthoc/results"
    (dest / "panel-simulation.json").write_text(json.dumps({"replicates": REPS, "seed": SEED, "rows": out}, indent=1) + "\n")
    for row in out:
        print(f"{row['panel']:15s}{row['budget']:5d}  mean {row['mean_ties_fail']:6.1f}-{row['mean_ties_pass']:6.1f}  "
              f"95% [{row['range95_low']:5.0f}, {row['range95_high']:5.0f}]  observed {row['observed']:4d}  {'ok' if row['inside'] else 'OUTSIDE'}")


if __name__ == "__main__":
    main()
