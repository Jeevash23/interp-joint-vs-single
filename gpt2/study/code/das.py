"""Alignments for AI-10: distributed alignment search (DAS) and its baselines, for any differentiable suffix.

AI-05 parameterised DAS by a full d x d Cayley rotation (d = 16 or 64). At AI-10's one-site site d is about 10,000, so a
full rotation is too large; we optimise an unconstrained D x k matrix and map it to an orthonormal frame by a QR
retraction (columns' signs fixed by diag(R) > 0). This spans the same family of subspaces. Everything else follows
AI-05: initialisation from the conditional-mean contrasts plus a seeded random completion, cross-entropy on the class
of the counterfactual input, Adam, a fixed number of steps, and the lowest-loss iterate is kept.

  fit_joint     one frame [D, r_A + r_B]: P_A and P_B orthogonal by construction (AI-05 Study 2), trained on the
                single queries of both variables;
  fit_separate  one frame per variable, each trained on its own variable's single queries only (AI-05 Study 3's
                procedure), so P_A P_B need not vanish (AI-10 arm c);
  mean_frames   conditional-mean contrasts of each variable, Loewdin-orthogonalised (AI-05) or left as they are;
  random_frames seeded random orthogonal subspaces.

suffix(states [B, D] tensor, recipients [B] array) -> logits [B, classes] tensor.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from interchange import Domain


def contrast_basis(z: np.ndarray, x: np.ndarray, v: int) -> np.ndarray:
    """Orthonormal basis of the centred conditional means of variable v (relative cutoff 1e-6, as AI-05)."""
    values = np.unique(x[:, v])
    means = np.array([z[x[:, v] == val].mean(0) for val in values])
    _, s, vt = np.linalg.svd(means - means.mean(0), full_matrices=False)
    return vt[s > 1e-6 * max(s.max(), 1e-12)].T


def mean_frames(z: np.ndarray, x: np.ndarray, orthogonalise: bool):
    ua, ub = contrast_basis(z, x, 0), contrast_basis(z, x, 1)
    if not orthogonalise:
        return ua, ub
    m = np.column_stack([ua, ub])
    w, v = np.linalg.eigh(m.T @ m)
    q = np.linalg.qr(m)[0] if w.min() < 1e-10 else m @ v @ np.diag(w ** -0.5) @ v.T
    return q[:, :ua.shape[1]], q[:, ua.shape[1]:]


def random_frames(d: int, ra: int, rb: int, seed: int):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, ra + rb)))
    return q[:, :ra], q[:, ra:]


def orthonormal(frame: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(frame)
    return q * np.sign(np.diag(r))[None, :]


def retract(m: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(m)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def _initial(z, x, ranks: list[tuple[int, int]], seed: int) -> np.ndarray:
    """Columns: for each (variable, rank), that variable's contrasts then random completion; then QR."""
    rng = np.random.default_rng(seed)
    cols = []
    for v, r in ranks:
        u = contrast_basis(z, x, v)[:, :r]
        cols += [u, rng.normal(size=(z.shape[1], r - u.shape[1]))]
    q, _ = np.linalg.qr(np.column_stack(cols))
    return q


def training_set(dom: Domain, v: int, mask: np.ndarray | None = None):
    """Single queries of variable v in which v changes (the full-domain single panel), optionally a subset."""
    keep = dom.panels()[("full_single", v)].copy()
    if mask is not None:
        keep &= mask
    rows = np.flatnonzero(keep)
    return dom.pairs[rows, 0], dom.pairs[rows, 1], dom.cls[dom.single_target(v)[rows]]


def _optimise(suffix, z, blocks, m0, lr, steps, batch, device, checkpoints):
    """blocks: list of (column slice, recipients, donors, classes). Returns the best frame and the loss record."""
    zt = torch.tensor(z, dtype=torch.float32, device=device)
    m = torch.nn.Parameter(torch.tensor(m0, dtype=torch.float32, device=device))
    opt = torch.optim.Adam([m], lr=lr)
    total = sum(len(b[1]) for b in blocks)
    best, best_loss, record = None, float("inf"), {}
    for step in range(steps):
        frame = retract(m)
        leaf = frame.detach().requires_grad_()          # chunks backpropagate to the frame, then once through the QR
        opt.zero_grad()
        loss_value = 0.0
        for cols, rec, don, cls in blocks:
            f = leaf[:, cols]
            for i in range(0, len(rec), batch):
                r, d = rec[i:i + batch], don[i:i + batch]
                delta = zt[d] - zt[r]
                states = zt[r] + (delta @ f) @ f.T
                y = torch.tensor(cls[i:i + batch], device=device)
                loss = F.cross_entropy(suffix(states, r), y, reduction="sum") / total
                loss.backward()
                loss_value += float(loss.detach())
        frame.backward(leaf.grad)
        if loss_value < best_loss:
            best_loss, best = loss_value, leaf.detach().cpu().double().numpy()
        if step + 1 in checkpoints:
            record[str(step + 1)] = best_loss
        opt.step()
    return best, {"best_loss": best_loss, "best_loss_at_step": record, "training_queries": total}


def fit_joint(suffix, z, dom: Domain, rank: int, seed: int, lr: float, steps: int, batch: int, device,
              checkpoints=(250, 500, 1000, 1500), mask: np.ndarray | None = None):
    m0 = _initial(z, dom.x, [(0, rank), (1, rank)], seed)
    blocks = [(slice(0, rank), *training_set(dom, 0, mask)), (slice(rank, 2 * rank), *training_set(dom, 1, mask))]
    frame, info = _optimise(suffix, z, blocks, m0, lr, steps, batch, device, set(checkpoints))
    frame = orthonormal(frame)                          # float32 training; exactly orthonormal in float64 afterwards
    return (frame[:, :rank], frame[:, rank:]), info


def fit_separate(suffixes, zs, dom: Domain, rank: int, seed: int, lr: float, steps: int, batch: int, device,
                 checkpoints=(250, 500, 1000, 1500)):
    """One frame per variable, each trained on its own single queries. suffixes and zs are (A, B) pairs: the same
    site twice for arm c, the two head sites for arm b."""
    frames, info = [], {}
    for v, name in ((0, "a"), (1, "b")):
        m0 = _initial(zs[v], dom.x, [(v, rank)], seed + 7 * v)
        f, i = _optimise(suffixes[v], zs[v], [(slice(0, rank), *training_set(dom, v))], m0, lr, steps, batch, device,
                         set(checkpoints))
        frames.append(orthonormal(f))
        info[name] = i
    return tuple(frames), info


def overlap(fa: np.ndarray, fb: np.ndarray) -> float:
    """Spectral norm of P_A P_B (= largest cosine of a principal angle between the two subspaces)."""
    if fa.shape[1] == 0 or fb.shape[1] == 0:
        return 0.0
    return float(np.linalg.norm(fa.T @ fb, 2))
