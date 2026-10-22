"""Arm d (no GPT-2): separately fitted subspaces on AI-05 Study 2's saved networks. PROTOCOL.md Section 7.

AI-05 Study 2's saved alignments are all orthogonal by construction (largest ||P_A P_B|| over its 1,200 is 1.4e-12,
checked on 17 October 2026), so the overlap cannot be read off them. This arm fits, on each saved network and at
Study 2's site, P_A and P_B separately (AI-05 Study 3's procedure: Cayley rotation, contrasts plus seeded random
completion, Adam at 0.02, 1,500 steps, lowest loss kept, each trained on the 486 single interchanges of its own
variable), plus the raw contrast subspaces without orthogonalisation; takes Study 2's census of every alignment with
Study 2's frozen code; and computes the margin rule with u written for non-orthogonal subspaces.

    python -B overlap_ai05.py development    AI-05's 32 development networks (checks; no decisions)
    python -B overlap_ai05.py final          AI-05's 200 final networks (refused before the AI-10 freeze)

CPU, float64 (Study 2's settings). AI-05 is read only; outputs go to study/results/overlap-ai05/<split>/.
Status (17 October 2026): written, not yet run (it runs AI-05's small networks, so it waits for approval).
"""
from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np

import ai10_common as common

sys.path.insert(0, str(common.AI05 / "studies/2-trained-networks/code"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from common import (ARCHS, INPUTS, PAIRS, TRIPLES, build, census, class_of, contrasts,  # noqa: E402
                    natural_scale)                       # AI-05 Study 2's frozen code, imported read-only

STUDY2 = common.AI05 / "studies/2-trained-networks"
INDEX = {tuple(x): i for i, x in enumerate(INPUTS.tolist())}
TR, T1, T2 = TRIPLES.T
RA = np.array([INDEX[(INPUTS[r, 0], INPUTS[t, 1], INPUTS[r, 2])] for r, t in zip(TR, T2)])
RB = np.array([INDEX[(INPUTS[s, 0], INPUTS[r, 1], INPUTS[r, 2])] for r, s in zip(TR, T1)])
BOTH = (INPUTS[T1, 0] != INPUTS[TR, 0]) & (INPUTS[T2, 1] != INPUTS[TR, 1])
TOL = 1e-4


def fit_one(model, z, task, variable: int, r: int, seed: int, steps=1500, lr=0.02):
    """AI-05 Study 3's fit_das at Study 2's site: rank-r projector for one variable, its 486 changed single queries."""
    d = z.shape[1]
    keep = INPUTS[PAIRS[:, 0], variable] != INPUTS[PAIRS[:, 1], variable]
    ri, di = PAIRS[keep].T
    tgt = INPUTS[ri].copy()
    tgt[:, variable] = INPUTS[di, variable]
    y = torch.tensor(class_of(task, tgt[:, 0], tgt[:, 1]))
    u0 = contrasts(z, INPUTS, variable)
    rng = np.random.default_rng(seed)
    q0, _ = np.linalg.qr(np.column_stack([u0[:, :r], rng.normal(size=(d, r - min(r, u0.shape[1]))),
                                          rng.normal(size=(d, d))]))
    q0 = torch.tensor(q0[:, :d])
    theta = torch.nn.Parameter(torch.zeros(d, d))
    eye = torch.eye(d)
    zt, dz = torch.tensor(z[ri]), torch.tensor(z[di] - z[ri])
    opt = torch.optim.Adam([theta], lr=lr)
    best, best_loss = None, float("inf")
    for _ in range(steps):
        s = (theta - theta.T) / 2
        u = q0 @ torch.linalg.solve(eye - s, eye + s)
        p = u[:, :r] @ u[:, :r].T
        loss = F.cross_entropy(model.suffix(zt + dz @ p), y)
        if float(loss) < best_loss:
            best_loss, best = float(loss), p.detach().numpy().copy()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return best, best_loss


def margin_and_grad(model, states, cls):
    w = torch.tensor(states, requires_grad=True)
    logits = model.suffix(w)
    other = logits.masked_fill(F.one_hot(torch.tensor(cls), logits.shape[1]).bool(), float("-inf")).max(1).values
    mu = logits[torch.arange(len(cls)), cls] - other
    mu.sum().backward()
    return mu.detach().numpy(), w.grad.numpy()


def sigma_scores(model, z, task, pa, pb):
    """AI-05's signed rule with u_A = J - S_A(z_r', s), u_B = J - S_B(z_r'', t) (equal to AI-05's u when P_A P_B = 0)."""
    r, s = PAIRS.T
    wa, wb = z[r] + (z[s] - z[r]) @ pa, z[r] + (z[s] - z[r]) @ pb
    ma, ga = margin_and_grad(model, wa, class_of(task, INPUTS[s, 0], INPUTS[r, 1]))
    mb, gb = margin_and_grad(model, wb, class_of(task, INPUTS[r, 0], INPUTS[s, 1]))
    j = z[TR] + (z[T1] - z[TR]) @ pa + (z[T2] - z[TR]) @ pb
    ia, ib = RA * 27 + T1, RB * 27 + T2
    ua, ub = j - wa[ia], j - wb[ib]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.minimum(-(ga[ia] * ua).sum(1) / ma[ia], -(gb[ib] * ub).sum(1) / mb[ib])
    reduced_pass = (ma[ia] > TOL) & (mb[ib] > TOL)
    q = sigma[reduced_pass]
    q = q[np.isfinite(q)]
    return {"max_sigma": float(q.max()) if len(q) else float("-inf"), "flagged_queries": int((q > 1).sum())}


def run_network(folder) -> list[dict]:
    rec = json.loads((folder / "record.json").read_text())
    if not rec["competent"]:
        return []
    task = json.loads((folder / "task.json").read_text())
    model = build(rec["arch"], len(task["classes"]))
    model.load_state_dict({k: torch.tensor(v) for k, v in np.load(folder / "weights.npz").items()})
    model.eval()
    model.requires_grad_(False)
    z = np.load(folder / "census.npz")["natural_states"]
    base = rec["seed"] * 1000 + list(ARCHS).index(rec["arch"])
    cands = {}
    for r in (1, 2, 4):
        pa, la = fit_one(model, z, task, 0, r, base + 11 * r)
        pb, lb = fit_one(model, z, task, 1, r, base + 11 * r + 5)
        cands[f"sep_r{r}"] = (pa, pb, {"loss_a": la, "loss_b": lb})
    ua, ub = contrasts(z, INPUTS, 0), contrasts(z, INPUTS, 1)
    cands["mean_raw"] = (ua @ ua.T, ub @ ub.T, {})
    scale = natural_scale(z)
    rows = []
    for name, (pa, pb, info) in cands.items():
        c = census(model, z, task, pa, pb, scale)
        rows.append({"network": folder.name, "family": rec["family"], "arch": rec["arch"], "candidate": name,
                     "overlap": float(np.linalg.norm(pa @ pb, 2)), "rank_a": int(round(np.trace(pa))),
                     "rank_b": int(round(np.trace(pb))), "single_fail": int((c["single"] == 1).sum()),
                     "single_tie": int((c["single"] == 2).sum()), "joint_fail_all": int((c["joint"] == 1).sum()),
                     "p_joint": float((c["joint"][BOTH] == 1).mean()), **sigma_scores(model, z, task, pa, pb), **info})
    return rows


def main() -> None:
    split = sys.argv[1]
    if split == "final":
        common.require_freeze()
    elif split != "development":
        raise SystemExit(__doc__)
    common.require_predictions()
    torch.set_num_threads(C_THREADS)
    dest = common.RESULTS / "overlap-ai05" / split
    dest.mkdir(parents=True, exist_ok=True)
    started, rows = time.perf_counter(), []
    for folder in sorted((STUDY2 / "runs" / split).iterdir()):
        rows.extend(run_network(folder))
        print(folder.name, f"{time.perf_counter() - started:.0f} s", flush=True)
    with open(dest / "alignments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    common.save_json(dest / "record.json", {"written_at_utc": common.utc_now(), "networks": len({r["network"] for r in rows}),
                                            "alignments": len(rows), "seconds": time.perf_counter() - started,
                                            "study2_frozen_manifest_sha256": common.sha256(STUDY2 / "FROZEN-MANIFEST.json")})


C_THREADS = common.CONFIG["cpu_threads"]

if __name__ == "__main__":
    main()
