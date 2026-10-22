"""Single-test and joint-test strategies at small budgets, Studies 2 and 3 (post hoc, exploratory).

Written after the frozen results of both studies were known. For every full-domain DAS alignment, the expected share
of alignments detected (at least one robust failure among the queries run) at 12, 24, 60 and 120 model calls, with 3
calls per single query, 4 per joint query and 1 per gradient:

  alternating   single queries alternating between A and B (the frozen comparison), exact;
  joint         joint-panel queries, exact;
  oracle        all single queries on whichever variable fails more often, which a tester would have to know, exact;
  adaptive      alternate for the first k = 4 single queries, then spend the rest on the variable with more observed
                failures, ties alternating, exact. It can switch only after observing a failure, when the alignment is
                already detected, so it equals the alternating test at every budget;
  adaptive_margin  the same, but after the first k = 4 queries (two per variable), if none failed, spend the rest on the
                variable whose two queries had the smaller mean margin (ties alternating), exact by enumerating all
                pairs of draws;
  guided        margin-guided joint test: draw n joint-panel queries; run the two single queries each one reduces to
                (Study 2: (r', s) and (r'', t); Study 3: (r, s) and (r'', t)) with the gradients the signed score
                needs (two in Study 2, one in Study 3); score every joint-panel query that the drawn single queries
                compose (the n drawn ones and any others) by the signed score sigma; run the m top-scored ones.
                n = budget // (c + 2) with c the cost of a candidate (8 calls in Study 2, 7 in Study 3), and m spends
                what is left at 4 calls each. Monte Carlo, 4,000 tests per alignment and budget (seed 20261009);
  guided_random the same test with the candidates run in random order instead of by score.

Output: analysis/results/test-strategies.json: means over the DAS alignments, overall, by kind, and over the
joint-only ones (no single failure or tie, at least one joint failure), with the Monte Carlo standard errors.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
PROJECT = Path(__file__).resolve().parents[1]
S2 = PROJECT / "studies/2-trained-networks"
S3 = PROJECT / "studies/3-multi-site"
sys.path.insert(0, str(S3 / "code"))          # sites.py (Study 3) puts Study 2's code on the path itself
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sites import INPUTS, PAIRS, TRIPLES, a_to_b, class_of, natural_index, readout  # noqa: E402   (frozen)
from common import build  # noqa: E402   (Study 2's frozen code)

BUDGETS = (12, 24, 60, 120)
K = 4
R = 4000
SEED = 20261009
DAS = ("das_r1", "das_r2", "das_r4")
TR, T1, T2 = TRIPLES.T
JOINT_PANEL = np.flatnonzero((INPUTS[T1, 0] != INPUTS[TR, 0]) & (INPUTS[T2, 1] != INPUTS[TR, 1]))
RA = natural_index(INPUTS[TR, 0], INPUTS[T2, 1], INPUTS[TR, 2])     # (a_r, b_t, n_r)
RB = natural_index(INPUTS[T1, 0], INPUTS[TR, 1], INPUTS[TR, 2])     # (a_s, b_r, n_r)
CHANGED = np.stack([INPUTS[PAIRS[:, 0], v] != INPUTS[PAIRS[:, 1], v] for v in (0, 1)])
TOL = 1e-4


def margin_and_grad(f, states, cls):
    w = torch.tensor(states, requires_grad=True)
    logits = f(w)
    hot = torch.nn.functional.one_hot(torch.tensor(cls), logits.shape[1]).bool()
    mu = logits[torch.arange(len(cls)), torch.tensor(cls)] - logits.masked_fill(hot, float("-inf")).max(1).values
    mu.sum().backward()
    return mu.detach().numpy(), w.grad.numpy()


def load(folder):
    rec = json.loads((folder / "record.json").read_text())
    task = json.loads((folder / "task.json").read_text())
    model = build(rec["arch"], len(task["classes"]))
    model.load_state_dict({k: torch.tensor(v) for k, v in np.load(folder / "weights.npz").items()})
    model.eval()
    model.requires_grad_(False)
    return rec, task, model


def study2(folder, name):
    """Single-query margins and the two-term signed score of every joint query (as in posthoc/margin_bound.py)."""
    rec, task, model = load(folder)
    c, cand = np.load(folder / "census.npz"), np.load(folder / "candidates.npz")
    i = c["names"].tolist().index(name)
    z, pa, pb = c["natural_states"], cand[name + "_a"], cand[name + "_b"]
    r, d = PAIRS.T
    ma, ga = margin_and_grad(model.suffix, z[r] + (z[d] - z[r]) @ pa, class_of(task, INPUTS[d, 0], INPUTS[r, 1]))
    mb, gb = margin_and_grad(model.suffix, z[r] + (z[d] - z[r]) @ pb, class_of(task, INPUTS[r, 0], INPUTS[d, 1]))
    ia, ib = RA * 27 + T1, RB * 27 + T2
    eye = np.eye(z.shape[1])
    ua = ((z[TR] + (z[T2] - z[TR]) @ pb) - z[RA]) @ (eye - pa)
    ub = ((z[TR] + (z[T1] - z[TR]) @ pa) - z[RB]) @ (eye - pb)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.minimum(-(ga[ia] * ua).sum(1) / ma[ia], -(gb[ib] * ub).sum(1) / mb[ib])
    sigma[~np.isfinite(sigma) | (ma[ia] <= TOL) | (mb[ib] <= TOL)] = np.inf
    return {"kind": rec["kind"], "single": c["single"][i], "joint": c["joint"][i], "ma": ma, "mb": mb,
            "sigma": sigma, "ia": ia, "ib": ib, "gradients": 2}


def study3(folder, name):
    """Single-query margins and the one-term signed score of every joint query (as in posthoc/signed_rule.py)."""
    rec, task, model = load(folder)
    kind = rec["kind"]
    c, cand = np.load(folder / "census.npz"), np.load(folder / "candidates.npz")
    i = c["names"].tolist().index(name)
    za, zb, pa, pb = c["states_a"], c["states_b"], cand[name + "_a"], cand[name + "_b"]
    r, d = PAIRS.T
    with torch.no_grad():
        wa = a_to_b(model, kind, torch.tensor(za[r] + (za[d] - za[r]) @ pa), r)
        logits = readout(model, kind, wa)
        cls = torch.tensor(class_of(task, INPUTS[d, 0], INPUTS[r, 1]))
        hot = torch.nn.functional.one_hot(cls, logits.shape[1]).bool()
        ma = (logits[torch.arange(len(cls)), cls] - logits.masked_fill(hot, float("-inf")).max(1).values).numpy()
    wa = wa.numpy()
    mb, gb = margin_and_grad(lambda w: readout(model, kind, w), zb[r] + (zb[d] - zb[r]) @ pb,
                             class_of(task, INPUTS[r, 0], INPUTS[d, 1]))
    ia, ib = TR * 27 + T1, RB * 27 + T2
    u = (wa[ia] - zb[RB]) @ (np.eye(zb.shape[1]) - pb)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = -(gb[ib] * u).sum(1) / mb[ib]
    sigma[~np.isfinite(sigma) | (mb[ib] <= TOL)] = np.inf
    return {"kind": kind, "single": c["single"][i], "joint": c["joint"][i], "ma": ma, "mb": mb,
            "sigma": sigma, "ia": ia, "ib": ib, "gradients": 1}


def alternating(pa, pb, m):
    return 1 - (1 - pa) ** ((m + 1) // 2) * (1 - pb) ** (m // 2)


def choice_probabilities(ma_ok, mb_ok):
    """P(mean margin of two A draws < / = / > that of two B draws), draws uniform over non-failing queries."""
    if len(ma_ok) == 0 or len(mb_ok) == 0:
        return 0.0, 1.0, 0.0
    sa = np.sort((ma_ok[:, None] + ma_ok[None, :]).ravel())
    sb = np.sort((mb_ok[:, None] + mb_ok[None, :]).ravel())
    less = np.searchsorted(sa, sb, side="left").sum()           # pairs with sum_A < sum_B
    leq = np.searchsorted(sa, sb, side="right").sum()
    total = float(len(sa)) * float(len(sb))
    q_a, q_t = less / total, (leq - less) / total
    return q_a, q_t, 1 - q_a - q_t


def compose(study, a_idx, b_idx):
    """Joint queries composed by drawn A-queries (rows) and B-queries (columns); -1 where they do not compose."""
    ra, s = a_idx // 27, a_idx % 27
    rb, t = b_idx // 27, b_idx % 27
    if study == 2:      # A-query (r', s), B-query (r'', t): a_{r''} = a_s, b_{r'} = b_t, n_{r'} = n_{r''}
        ok = ((INPUTS[rb, 0][:, None, :] == INPUTS[s, 0][:, :, None])
              & (INPUTS[ra, 1][:, :, None] == INPUTS[t, 1][:, None, :])
              & (INPUTS[ra, 2][:, :, None] == INPUTS[rb, 2][:, None, :]))
        r = (INPUTS[ra, 0][:, :, None] * 9 + INPUTS[rb, 1][:, None, :] * 3 + INPUTS[ra, 2][:, :, None])
    else:               # A-query (r, s), B-query (r'', t): r'' = (a_s, b_r, n_r)
        target = INPUTS[s, 0] * 9 + INPUTS[ra, 1] * 3 + INPUTS[ra, 2]
        ok = target[:, :, None] == rb[:, None, :]
        r = np.broadcast_to(ra[:, :, None], ok.shape)
    joint = r * 729 + s[:, :, None] * 27 + t[:, None, :]
    return np.where(ok, joint, -1)


def guided(study, x, budget, rng, ranked=True):
    cost = 6 + x["gradients"]
    n = budget // (cost + 2)
    if n == 0:
        return 0.0, 0.0
    m = (budget - n * cost) // 4
    draws = JOINT_PANEL[rng.integers(len(JOINT_PANEL), size=(R, n))]
    a_idx, b_idx = x["ia"][draws], x["ib"][draws]
    detected = (x["single"][0][a_idx] == 1).any(1) | (x["single"][1][b_idx] == 1).any(1)
    if m > 0:
        cand = compose(study, a_idx, b_idx).reshape(R, -1)
        valid = cand >= 0
        score = np.where(valid, x["sigma"][np.where(valid, cand, 0)], -np.inf)
        if not ranked:
            score = np.where(valid, rng.random(score.shape), -np.inf)
        order = np.argsort(cand, axis=1, kind="stable")             # drop repeated candidates
        sorted_cand = np.take_along_axis(cand, order, 1)
        repeat = np.zeros_like(valid)
        repeat[:, 1:] = sorted_cand[:, 1:] == sorted_cand[:, :-1]
        np.put_along_axis(score, order, np.where(repeat, -np.inf, np.take_along_axis(score, order, 1)), 1)
        top = np.argsort(-score, axis=1, kind="stable")[:, :m]
        chosen = np.take_along_axis(cand, top, 1)
        run = np.take_along_axis(score, top, 1) > -np.inf
        detected |= (run & (x["joint"][np.where(chosen >= 0, chosen, 0)] == 1)).any(1)
    return float(detected.mean()), float(detected.std(ddof=1) / np.sqrt(R))


def evaluate(study, x, rng):
    s, j = x["single"], x["joint"]
    pa, pb = float((s[0][CHANGED[0]] == 1).mean()), float((s[1][CHANGED[1]] == 1).mean())
    pj = float((j[JOINT_PANEL] == 1).mean())
    ma_ok = x["ma"][CHANGED[0] & (s[0] != 1)]
    mb_ok = x["mb"][CHANGED[1] & (s[1] != 1)]
    q_a, q_t, q_b = choice_probabilities(ma_ok, mb_ok)
    out = {}
    for budget in BUDGETS:
        m, n = budget // 3, budget // 4
        alt = alternating(pa, pb, m)
        if m > K:
            rest = m - K
            none_first = (1 - pa) ** (K // 2) * (1 - pb) ** (K // 2)
            adaptive = 1 - none_first * (1 - pa) ** ((rest + 1) // 2) * (1 - pb) ** (rest // 2)
            adaptive_margin = 1 - none_first * (q_a * (1 - pa) ** rest + q_b * (1 - pb) ** rest
                                                + q_t * (1 - pa) ** ((rest + 1) // 2) * (1 - pb) ** (rest // 2))
        else:
            adaptive = adaptive_margin = alt
        g, g_se = guided(study, x, budget, rng)
        gr, gr_se = guided(study, x, budget, rng, ranked=False)
        out[str(budget)] = {"alternating": alt, "joint": 1 - (1 - pj) ** n, "oracle": 1 - (1 - max(pa, pb)) ** m,
                            "adaptive": adaptive, "adaptive_margin": adaptive_margin,
                            "guided": g, "guided_se": g_se, "guided_random": gr, "guided_random_se": gr_se}
    return out


def summarize(results):
    keys = ("alternating", "joint", "oracle", "adaptive", "adaptive_margin", "guided", "guided_random")
    def mean(rs):
        return {b: {k: float(np.mean([r["by_budget"][b][k] for r in rs])) for k in keys} |
                   {"guided_se": float(np.sqrt(np.sum([r["by_budget"][b]["guided_se"] ** 2 for r in rs])) / len(rs)),
                    "guided_random_se": float(np.sqrt(np.sum([r["by_budget"][b]["guided_random_se"] ** 2 for r in rs]))
                                              / len(rs))}
                for b in map(str, BUDGETS)}
    joint_only = [r for r in results if r["joint_only"]]
    return {"alignments": len(results), "overall": mean(results),
            "by_kind": {k: mean([r for r in results if r["kind"] == k]) for k in ("transformer", "mlp")},
            "joint_only": {"alignments": [r["alignment"] for r in joint_only], **mean(joint_only)},
            "adaptive_equals_alternating_max_abs": float(max(abs(r["by_budget"][b]["adaptive"] - r["by_budget"][b]["alternating"])
                                                             for r in results for b in map(str, BUDGETS)))}


def main():
    import csv
    rng = np.random.default_rng(SEED)
    out = {}
    for study, root, fn in ((2, S2, study2), (3, S3, study3)):
        rows = [r for r in csv.DictReader(open(root / "results/final/candidates.csv")) if r["candidate"] in DAS]
        results = []
        for k, r in enumerate(rows):
            x = fn(root / "runs/final" / r["network"], r["candidate"])
            results.append({"alignment": f"{r['network']}/{r['candidate']}", "kind": x["kind"],
                            "joint_only": int(r["single_fail"]) == 0 and int(r["single_tie"]) == 0
                                          and int(r["joint_fail_all"]) > 0,
                            "by_budget": evaluate(study, x, rng)})
            if k % 100 == 0:
                print(f"study {study}: {k}/{len(rows)}", flush=True)
        out[f"study{study}"] = summarize(results)
    dest = PROJECT / "analysis/results"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "test-strategies.json").write_text(json.dumps(out, indent=1) + "\n")
    for study in ("study2", "study3"):
        print(study, "adaptive = alternating to", out[study]["adaptive_equals_alternating_max_abs"])
        for b, v in out[study]["overall"].items():
            print(f"  {b:>4} calls  " + "  ".join(f"{k} {v[k]:.3f}" for k in ("alternating", "adaptive", "adaptive_margin",
                                                                                  "oracle", "joint", "guided", "guided_random")))


if __name__ == "__main__":
    main()
