"""Held-out single split for Study 2 (exploratory, post hoc; the networks and their results were inspected before).

The frozen full-domain DAS alignments were fitted on every single interchange that changes the swapped variable, so the
full-domain single panel tests them on their own training data while the joint panel does not. Here, for every Study 2
network, the full-domain DAS alignments (ranks 1, 2 and 4) are refitted with Study 2's settings on a random half of
those interchanges (243 of 486 per variable, drawn separately for each network), and every single and joint query is
decided. Detection by the held-out half of the single queries is then compared with detection by the joint panel at
equal budgets: 3 calls per single and 4 per joint query, equal numbers of queries, and one call per query; with the
program-cluster bootstrap of Study 2 for the first, and the add-on comparison.

    python posthoc/heldout_split.py [workers]

Output: posthoc/results/heldout-split.json, and one file per network in posthoc/heldout-split/ (training split,
projectors and census decisions).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.dont_write_bytecode = True
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "code"))
import numpy as np  # noqa: E402

DEST = STUDY / "posthoc/heldout-split"
SPLIT_SEED = 20261008
RANKS = (1, 2, 4)
BOOT, BOOT_SEED = 10000, 20261006


def fit_on(model, z, task, r, ri, di, vi, seed):
    """Study 2's fit_das (code/common.py) with an explicit training set; settings and initialisation unchanged."""
    import torch
    import torch.nn.functional as F
    from common import DAS, INPUTS, class_of, contrasts
    d = z.shape[1]
    tgt = INPUTS[ri].copy()
    tgt[np.arange(len(ri)), vi] = INPUTS[di, vi]
    y = torch.tensor(class_of(task, tgt[:, 0], tgt[:, 1]))
    ua, ub = contrasts(z, INPUTS, 0), contrasts(z, INPUTS, 1)
    rng = np.random.default_rng(seed)
    cols = [ua[:, :r], rng.normal(size=(d, r - min(r, ua.shape[1]))),
            ub[:, :r], rng.normal(size=(d, r - min(r, ub.shape[1]))), rng.normal(size=(d, d))]
    q0, _ = np.linalg.qr(np.column_stack(cols))
    q0 = torch.tensor(q0[:, :d])
    theta = torch.nn.Parameter(torch.zeros(d, d))
    eye = torch.eye(d)
    zt, dz = torch.tensor(z[ri]), torch.tensor(z[di] - z[ri])
    sel = torch.tensor(vi == 0)[:, None]
    opt = torch.optim.Adam([theta], lr=DAS["lr"])
    best, best_loss = None, float("inf")
    for _ in range(DAS["steps"]):
        s = (theta - theta.T) / 2
        u = q0 @ torch.linalg.solve(eye - s, eye + s)
        pa, pb = u[:, :r] @ u[:, :r].T, u[:, r:2 * r] @ u[:, r:2 * r].T
        loss = F.cross_entropy(model.suffix(zt + torch.where(sel, dz @ pa, dz @ pb)), y)
        if float(loss) < best_loss:
            best_loss, best = float(loss), (pa.detach().numpy().copy(), pb.detach().numpy().copy())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return best, best_loss


def one(job):
    import torch
    from common import ARCHS, INPUTS, PAIRS, build, census, natural_scale
    index, net = job
    out = DEST / f"{net}.npz"
    if out.exists():
        return f"{net}: done earlier"
    folder = STUDY / "runs/final" / net
    rec, task = json.loads((folder / "record.json").read_text()), json.loads((folder / "task.json").read_text())
    model = build(rec["arch"], len(task["classes"]))
    model.load_state_dict({k: torch.tensor(v) for k, v in np.load(folder / "weights.npz").items()})
    model.eval()
    model.requires_grad_(False)
    z = np.load(folder / "census.npz")["natural_states"]
    rng = np.random.default_rng(SPLIT_SEED + index)
    train = np.zeros((2, 729), dtype=bool)
    for v in (0, 1):
        changed = np.flatnonzero(INPUTS[PAIRS[:, 0], v] != INPUTS[PAIRS[:, 1], v])
        train[v, rng.choice(changed, size=len(changed) // 2, replace=False)] = True
    ri = np.concatenate([PAIRS[train[v], 0] for v in (0, 1)])
    di = np.concatenate([PAIRS[train[v], 1] for v in (0, 1)])
    vi = np.concatenate([np.full(train[v].sum(), v) for v in (0, 1)])
    seed = rec["seed"] * 1000 + list(ARCHS).index(rec["arch"])
    saved, losses = {}, {}
    scale = natural_scale(z)
    for r in RANKS:
        (pa, pb), losses[r] = fit_on(model, z, task, r, ri, di, vi, seed + 11 * r)
        c = census(model, z, task, pa, pb, scale)
        saved.update({f"r{r}_a": pa, f"r{r}_b": pb, f"r{r}_single": c["single"], f"r{r}_joint": c["joint"]})
    DEST.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, train=train, **saved, best_loss=np.array([losses[r] for r in RANKS]))
    return f"{net}: done"


def detect(p, n):
    return 1 - (1 - p) ** n


def single_det(pa, pb, n):
    return 1 - (1 - pa) ** ((n + 1) // 2) * (1 - pb) ** (n // 2)


def summarize(nets):
    from common import INPUTS, PAIRS, TRIPLES
    tr, t1, t2 = TRIPLES.T
    jmask = (INPUTS[t1, 0] != INPUTS[tr, 0]) & (INPUTS[t2, 1] != INPUTS[tr, 1])
    changed = np.stack([INPUTS[PAIRS[:, 0], v] != INPUTS[PAIRS[:, 1], v] for v in (0, 1)])
    rows = []
    for net in nets:
        f = np.load(DEST / f"{net}.npz")
        rec = json.loads((STUDY / "runs/final" / net / "record.json").read_text())
        train = f["train"]
        held = changed & ~train
        for r in RANKS:
            s, j = f[f"r{r}_single"], f[f"r{r}_joint"]
            row = {"network": net, "family": rec["family"], "arch": rec["arch"], "rank": r}
            for v, name in ((0, "A"), (1, "B")):
                row[f"train_{name}"] = float((s[v][train[v]] == 1).mean())
                row[f"held_{name}"] = float((s[v][held[v]] == 1).mean())
            row["joint"] = float((j[jmask] == 1).mean())
            row["fails_no_training_query"] = bool(not (s[0][train[0]] == 1).any() and not (s[1][train[1]] == 1).any())
            row["fails_no_heldout_query"] = bool(not (s[0][held[0]] == 1).any() and not (s[1][held[1]] == 1).any())
            row["fails_joint"] = bool((j[jmask] == 1).any())
            rows.append(row)

    def contrasts(rs, panel):
        m = lambda xs: float(np.mean(xs))  # noqa: E731
        sd = lambda r, n: single_det(r[f"{panel}_A"], r[f"{panel}_B"], n)  # noqa: E731
        return {"calls_120": m([detect(r["joint"], 30) - sd(r, 40) for r in rs]),
                "equal_queries_30": m([detect(r["joint"], 30) - sd(r, 30) for r in rs]),
                "cached_states_120": m([detect(r["joint"], 120) - sd(r, 120) for r in rs]),
                "addon_joint_after_single_120": m([(1 - sd(r, 40)) * detect(r["joint"], 30) for r in rs]),
                "addon_single_after_single_120": m([sd(r, 80) - sd(r, 40) for r in rs]),
                "single_detection_120": m([sd(r, 40) for r in rs]), "joint_detection_120": m([detect(r["joint"], 30) for r in rs])}

    out = {"alignments": len(rows), "networks": len(nets),
           "mean_failure_rate": {k: float(np.mean([r[k] for r in rows])) for k in
                                 ("train_A", "train_B", "held_A", "held_B", "joint")},
           "heldout_vs_joint": contrasts(rows, "held"), "training_vs_joint": contrasts(rows, "train"),
           "pass_all_training": sum(r["fails_no_training_query"] for r in rows),
           "pass_all_training_fail_heldout": sum(r["fails_no_training_query"] and not r["fails_no_heldout_query"] for r in rows),
           "pass_all_training_fail_joint": sum(r["fails_no_training_query"] and r["fails_joint"] for r in rows),
           "pass_all_training_and_heldout_fail_joint": sum(r["fails_no_training_query"] and r["fails_no_heldout_query"]
                                                           and r["fails_joint"] for r in rows)}
    cells = defaultdict(list)
    for r in rows:
        cells[f"{r['family']}/{r['arch']}"].append(r)
    out["by_cell"] = {k: contrasts(v, "held") for k, v in sorted(cells.items())}
    out["by_kind"] = {k: contrasts([r for r in rows if r["arch"].startswith(k[0].upper())], "held")
                      for k in ("transformer", "mlp")}
    # Program-cluster bootstrap of the held-out contrast at 120 calls (networks resampled within cells).
    rng = np.random.default_rng(BOOT_SEED)
    by_net = defaultdict(list)
    for r in rows:
        by_net[r["network"]].append(detect(r["joint"], 30) - single_det(r["held_A"], r["held_B"], 40))
    groups = defaultdict(list)
    for r in rows:
        if r["network"] not in groups[(r["family"], r["arch"])]:
            groups[(r["family"], r["arch"])].append(r["network"])
    draws = []
    for _ in range(BOOT):
        sample = []
        for g in groups.values():
            sample.extend(v for i in rng.integers(len(g), size=len(g)) for v in by_net[g[i]])
        draws.append(np.mean(sample))
    out["heldout_vs_joint"]["calls_120_ci95"] = np.quantile(draws, [0.025, 0.975]).tolist()
    return out, rows


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    nets = sorted(p.name for p in (STUDY / "runs/final").iterdir()
                  if json.loads((p / "record.json").read_text())["competent"])
    with Pool(workers) as pool:
        for message in pool.imap_unordered(one, list(enumerate(nets))):
            print(message, flush=True)
    out, _ = summarize(nets)
    dest = STUDY / "posthoc/results"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "heldout-split.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "by_cell"}, indent=1))


if __name__ == "__main__":
    main()
