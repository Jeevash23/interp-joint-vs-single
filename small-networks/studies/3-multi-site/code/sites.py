"""Study 3: alignments that put A and B at different sites, on networks trained by Study 2's frozen code.

Sites. MLP: A at the first hidden layer after its ReLU, B at the second hidden layer after its ReLU. Transformer: A at
the residual stream of position 1 (the token a) before attention, B at the residual stream of the last position after
attention. A single A-query edits A's site and runs the rest of the network; a single B-query edits B's site; a joint
query applies the A-edit, propagates it to B's site and then replaces the P_B component there with the B-donor's.
Everything runs on CPU in float64 (set by Study 2's common.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
sys.path.append(str(STUDY.parent / "2-trained-networks" / "code"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from common import (ARCHS, DAS, INPUTS, PAIRS, TOL, TRIPLES, class_of, contrasts, encode,  # noqa: E402,F401
                    margins, natural_index, natural_scale)          # Study 2's frozen code, imported unchanged

CANDIDATES = ["das_r1", "das_r2", "das_r4", "mean_full", "random_r2"]
TOKENS = encode("transformer")                                    # [27, 4]: BOS, a, b, n


def natural_states(model, kind: str):
    """Natural states of the 27 inputs at A's site and at B's site."""
    with torch.no_grad():
        if kind == "mlp":
            za = model.site(encode("mlp"))
            zb = torch.relu(model.l2(za))
        else:
            za = model.tok(TOKENS[:, 1]) + model.pos.weight[1]
            zb = model.site(TOKENS)
    return za.numpy(), zb.numpy()


def a_to_b(model, kind: str, ea: torch.Tensor, recipients: np.ndarray) -> torch.Tensor:
    """States at B's site when A's site holds ea and the rest of the input is the recipient's."""
    if kind == "mlp":
        return torch.relu(model.l2(ea))
    x = model.tok(TOKENS[recipients]) + model.pos(torch.arange(4))
    x = torch.cat([x[:, :1], ea[:, None], x[:, 2:]], dim=1)
    q, k, v = model.qkv(model.ln1(x)).split(model.d, dim=-1)
    n, dh = len(x), model.d // model.h
    q, k, v = [t.reshape(n, 4, model.h, dh).transpose(1, 2) for t in (q, k, v)]
    att = ((q @ k.transpose(-1, -2)) / dh ** 0.5).masked_fill(model.mask, float("-inf")).softmax(-1)
    x = x + model.out((att @ v).transpose(1, 2).reshape(n, 4, model.d))
    return x[:, -1]


def readout(model, kind: str, zb: torch.Tensor) -> torch.Tensor:
    """Logits from states at B's site."""
    return model.l3(zb) if kind == "mlp" else model.suffix(zb)


# ------------------------------------------------------------------------------------ alignments
def changed_pairs(variable: int):
    """The 486 (recipient, donor) pairs in which the variable changes, and the expected input of each."""
    keep = INPUTS[PAIRS[:, 0], variable] != INPUTS[PAIRS[:, 1], variable]
    ri, di = PAIRS[keep].T
    tgt = INPUTS[ri].copy()
    tgt[:, variable] = INPUTS[di, variable]
    return ri, di, tgt


def fit_das(model, kind: str, z: np.ndarray, task: dict, variable: int, r: int, seed: int):
    """Rank-r projector at the variable's site (Cayley rotation), trained on its 486 single interchanges."""
    d = z.shape[1]
    ri, di, tgt = changed_pairs(variable)
    y = torch.tensor(class_of(task, tgt[:, 0], tgt[:, 1]))
    u0 = contrasts(z, INPUTS, variable)
    rng = np.random.default_rng(seed)
    q0, _ = np.linalg.qr(np.column_stack([u0[:, :r], rng.normal(size=(d, r - min(r, u0.shape[1]))),
                                          rng.normal(size=(d, d))]))
    q0 = torch.tensor(q0[:, :d])
    theta = torch.nn.Parameter(torch.zeros(d, d))
    eye = torch.eye(d)
    zt, dz = torch.tensor(z[ri]), torch.tensor(z[di] - z[ri])
    if variable == 0:
        forward = lambda e: readout(model, kind, a_to_b(model, kind, e, ri))  # noqa: E731
    else:
        forward = lambda e: readout(model, kind, e)  # noqa: E731
    opt = torch.optim.Adam([theta], lr=DAS["lr"])
    best, best_loss = None, float("inf")
    for _ in range(DAS["steps"]):
        s = (theta - theta.T) / 2
        u = q0 @ torch.linalg.solve(eye - s, eye + s)
        p = u[:, :r] @ u[:, :r].T
        loss = F.cross_entropy(forward(zt + dz @ p), y)
        if float(loss) < best_loss:
            best_loss, best = float(loss), p.detach().numpy().copy()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return best, {"training_pairs": int(len(ri)), "best_loss": best_loss}


def contrast_projector(z: np.ndarray, variable: int) -> np.ndarray:
    u = contrasts(z, INPUTS, variable)
    return u @ u.T


def random_projector(d: int, r: int, seed: int) -> np.ndarray:
    u, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    return u[:, :r] @ u[:, :r].T


def make_candidates(model, kind: str, za: np.ndarray, zb: np.ndarray, task: dict, seed: int):
    model.requires_grad_(False)
    out, info = {}, {}
    for r in (1, 2, 4):
        pa, ia = fit_das(model, kind, za, task, 0, r, seed + 11 * r)
        pb, ib = fit_das(model, kind, zb, task, 1, r, seed + 11 * r + 5)
        out[f"das_r{r}"], info[f"das_r{r}"] = (pa, pb), {"a": ia, "b": ib}
    out["mean_full"] = (contrast_projector(za, 0), contrast_projector(zb, 1))
    out["random_r2"] = (random_projector(za.shape[1], 2, seed + 131), random_projector(zb.shape[1], 2, seed + 137))
    for name, ps in out.items():
        for p in ps:
            assert np.abs(p @ p - p).max() < 1e-8 and np.abs(p - p.T).max() < 1e-8, name
    return out, info


# --------------------------------------------------------------------------------------- census
def decide(model, kind: str, states: np.ndarray, cls: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """0 = pass, 1 = robust failure (expected class behind by > 1e-4), 2 = tie (within 1e-4)."""
    out = []
    with torch.no_grad():
        for s in range(0, len(states), chunk):
            gap = margins(readout(model, kind, torch.tensor(states[s:s + chunk])), torch.tensor(cls[s:s + chunk])).numpy()
            out.append(np.where(gap > TOL, 0, np.where(gap < -TOL, 1, 2)).astype(np.int8))
    return np.concatenate(out)


def propagate_a_edits(model, kind: str, za: np.ndarray, pa: np.ndarray):
    """All 729 A-edits (index recipient * 27 + donor) and their states at B's site."""
    r, d = PAIRS.T
    ea = za[r] + (za[d] - za[r]) @ pa
    with torch.no_grad():
        wa = a_to_b(model, kind, torch.tensor(ea), r).numpy()
    return ea, wa


def census(model, kind: str, za: np.ndarray, zb: np.ndarray, task: dict, pa: np.ndarray, pb: np.ndarray,
           scale_a: float, scale_b: float) -> dict:
    """Every single (729 per variable) and joint (19,683) query, with distances from natural states."""
    r, d = PAIRS.T
    ea, wa = propagate_a_edits(model, kind, za, pa)
    eb = zb[r] + (zb[d] - zb[r]) @ pb
    tgt_a = natural_index(INPUTS[d, 0], INPUTS[r, 1], INPUTS[r, 2])
    tgt_b = natural_index(INPUTS[r, 0], INPUTS[d, 1], INPUTS[r, 2])
    single = np.stack([decide(model, kind, wa, class_of(task, INPUTS[d, 0], INPUTS[r, 1])),
                       decide(model, kind, eb, class_of(task, INPUTS[r, 0], INPUTS[d, 1]))])
    off = np.stack([np.linalg.norm(ea - za[tgt_a], axis=1) / scale_a,      # A-edit at A's site
                    np.linalg.norm(eb - zb[tgt_b], axis=1) / scale_b,      # B-edit at B's site
                    np.linalg.norm(wa - zb[tgt_a], axis=1) / scale_b])     # A-edit propagated to B's site
    tr, t1, t2 = TRIPLES.T
    w = wa[tr * 27 + t1]
    joint = decide(model, kind, w + (zb[t2] - w) @ pb, class_of(task, INPUTS[t1, 0], INPUTS[t2, 1]))
    return {"single": single, "joint": joint, "off_natural": off.astype(np.float32)}
