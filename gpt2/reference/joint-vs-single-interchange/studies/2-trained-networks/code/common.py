"""Study 2: tasks, trained networks, alignment procedures and exhaustive interchange censuses.

Everything runs on CPU in float64. Inputs are (a, b, n) in {0,1,2}^3 in the same order and with the
same table generators as Study 1 (studies/1-compiled-programs/02-Code/v2_model.py).
"""
from __future__ import annotations

import itertools
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)

INPUTS = np.array(list(itertools.product(range(3), repeat=3)))
RESTRICTED = (INPUTS[:, :2] < 2).all(axis=1)
INDEX = {tuple(x): i for i, x in enumerate(INPUTS.tolist())}
PAIRS = np.array(list(itertools.product(range(27), repeat=2)))
TRIPLES = np.array(list(itertools.product(range(27), repeat=3)))
TOL = 1e-4
BOS = 3

# Settings fixed during development (see PROTOCOL.md).
ARCHS = {"T16": ("transformer", 16), "T64": ("transformer", 64), "M16": ("mlp", 16), "M64": ("mlp", 64)}
TRAIN = {"transformer": {"steps": 3000, "lr": 3e-3}, "mlp": {"steps": 3000, "lr": 1e-2}}
GATE_MARGIN = 1.0
RETRIES = 3
DAS = {"steps": 1500, "lr": 0.02}
CANDIDATES = ["das_r1", "das_r2", "das_r4", "das_restricted_r2", "mean_full", "random_r2"]


# ----------------------------------------------------------------------------------------- tasks
def make_task(family: str, seed: int) -> dict:
    """Same random draws as Study 1's generator, so a seed gives the same table in both studies."""
    rng = np.random.default_rng(seed)
    if family == "lookup":
        while True:
            table = rng.integers(0, 5, size=(3, 3))
            if len(set(map(tuple, table))) == 3 and len(set(map(tuple, table.T))) == 3:
                break
    elif family == "additive":
        av = rng.choice(np.arange(1, 11), size=3, replace=False)
        bv = rng.choice(np.arange(1, 11), size=3, replace=False)
        table = av[:, None] + bv[None, :]
    else:
        raise ValueError(family)
    classes = sorted(set(table.ravel().tolist()))
    return {"family": family, "seed": seed, "table": table.tolist(), "classes": classes}


def class_of(task: dict, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    table = np.asarray(task["table"])
    lookup = {v: i for i, v in enumerate(task["classes"])}
    return np.array([lookup[int(v)] for v in table[a, b]])


def encode(arch: str) -> torch.Tensor:
    if arch == "mlp":
        return torch.tensor(np.concatenate([np.eye(3)[INPUTS[:, i]] for i in range(3)], axis=1))
    return torch.tensor(np.column_stack([np.full(27, BOS), INPUTS]), dtype=torch.long)


# ---------------------------------------------------------------------------------------- models
class MLP(torch.nn.Module):
    """One-hot (a, b, n) -> W -> ReLU [site] -> W -> ReLU -> classes."""

    def __init__(self, width: int, k: int):
        super().__init__()
        self.l1, self.l2, self.l3 = torch.nn.Linear(9, width), torch.nn.Linear(width, width), torch.nn.Linear(width, k)

    def site(self, x):
        return torch.relu(self.l1(x))

    def suffix(self, z):
        return self.l3(torch.relu(self.l2(z)))

    def forward(self, x):
        return self.suffix(self.site(x))


class Transformer(torch.nn.Module):
    """[BOS, a, b, n] -> embeddings -> causal 2-head attention (pre-LN) [site: last position] -> MLP -> LN -> classes."""

    def __init__(self, d: int, k: int, heads: int = 2):
        super().__init__()
        self.d, self.h = d, heads
        self.tok, self.pos = torch.nn.Embedding(4, d), torch.nn.Embedding(4, d)
        self.ln1, self.ln2, self.lnf = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.qkv, self.out = torch.nn.Linear(d, 3 * d), torch.nn.Linear(d, d)
        self.fc1, self.fc2 = torch.nn.Linear(d, 4 * d), torch.nn.Linear(4 * d, d)
        self.unembed = torch.nn.Linear(d, k)
        self.register_buffer("mask", torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1))

    def site(self, tokens):
        x = self.tok(tokens) + self.pos(torch.arange(4))
        q, k, v = self.qkv(self.ln1(x)).split(self.d, dim=-1)
        n, dh = len(tokens), self.d // self.h
        q, k, v = [t.reshape(n, 4, self.h, dh).transpose(1, 2) for t in (q, k, v)]
        att = (q @ k.transpose(-1, -2)) / dh ** 0.5
        att = att.masked_fill(self.mask, float("-inf")).softmax(-1)
        x = x + self.out((att @ v).transpose(1, 2).reshape(n, 4, self.d))
        return x[:, -1]

    def suffix(self, z):
        z = z + self.fc2(torch.relu(self.fc1(self.ln2(z))))
        return self.unembed(self.lnf(z))

    def forward(self, tokens):
        return self.suffix(self.site(tokens))


def build(arch_name: str, k: int) -> torch.nn.Module:
    kind, width = ARCHS[arch_name]
    return MLP(width, k) if kind == "mlp" else Transformer(width, k)


def margins(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    other = logits.masked_fill(F.one_hot(y, logits.shape[1]).bool(), float("-inf")).max(1).values
    return logits[torch.arange(len(y)), y] - other


def train(task: dict, arch_name: str, init_seed: int):
    """Full-batch AdamW on the 27 inputs; the competence gate needs every input correct by margin >= 1."""
    kind = ARCHS[arch_name][0]
    torch.manual_seed(init_seed)
    model = build(arch_name, len(task["classes"]))
    x, y = encode(kind), torch.tensor(class_of(task, INPUTS[:, 0], INPUTS[:, 1]))
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN[kind]["lr"], weight_decay=0.0)
    for _ in range(TRAIN[kind]["steps"]):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        m = margins(model(x), y)
    return model, {"init_seed": init_seed, "final_loss": float(loss), "min_margin": float(m.min()),
                   "passed": bool(m.min() >= GATE_MARGIN)}


# ------------------------------------------------------------------------------------ alignments
def contrasts(z: np.ndarray, x: np.ndarray, variable: int) -> np.ndarray:
    """Orthonormal basis of the conditional-mean contrasts of one variable (relative cutoff 1e-6)."""
    values = np.unique(x[:, variable])
    means = np.array([z[x[:, variable] == v].mean(0) for v in values])
    _, s, vt = np.linalg.svd(means - means.mean(0), full_matrices=False)
    return vt[s > 1e-6 * max(s.max(), 1e-12)].T


def projectors(u: np.ndarray, r: int):
    return u[:, :r] @ u[:, :r].T, u[:, r:2 * r] @ u[:, r:2 * r].T


def mean_alignment(z: np.ndarray):
    """Contrast subspaces of A and B made orthogonal by symmetric (Loewdin) orthogonalisation."""
    ua, ub = contrasts(z, INPUTS, 0), contrasts(z, INPUTS, 1)
    m = np.column_stack([ua, ub])
    if m.shape[1] == 0:
        return np.zeros((z.shape[1], z.shape[1])), np.zeros((z.shape[1], z.shape[1]))
    w, v = np.linalg.eigh(m.T @ m)
    if w.min() < 1e-10:                      # A and B contrasts not jointly independent: orthogonalise B against A
        q, _ = np.linalg.qr(m)
    else:
        q = m @ v @ np.diag(w ** -0.5) @ v.T
    qa, qb = q[:, :ua.shape[1]], q[:, ua.shape[1]:]
    return qa @ qa.T, qb @ qb.T


def random_alignment(d: int, r: int, seed: int):
    u, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    return projectors(u, r)


def interchange_pairs(mask: np.ndarray):
    """All (recipient, donor, variable) with both states in the mask and the variable changed."""
    idx = np.flatnonzero(mask)
    rows = [(i, j, v) for i in idx for j in idx for v in (0, 1) if INPUTS[i, v] != INPUTS[j, v]]
    return np.array(rows).T


def fit_das(model, z: np.ndarray, task: dict, r: int, support: np.ndarray, seed: int):
    """Orthogonal rotation (Cayley) with rank-r subspaces for A and B, trained on single interchanges."""
    d = z.shape[1]
    ri, di, vi = interchange_pairs(support)
    tgt = INPUTS[ri].copy()
    tgt[np.arange(len(ri)), vi] = INPUTS[di, vi]
    y = torch.tensor(class_of(task, tgt[:, 0], tgt[:, 1]))
    ua, ub = contrasts(z[support], INPUTS[support], 0), contrasts(z[support], INPUTS[support], 1)
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
    return best, {"training_pairs": len(ri), "best_loss": best_loss}


def make_candidates(model, z: np.ndarray, task: dict, seed: int):
    out, info = {}, {}
    full = np.ones(27, dtype=bool)
    for r in (1, 2, 4):
        out[f"das_r{r}"], info[f"das_r{r}"] = fit_das(model, z, task, r, full, seed + 11 * r)
    out["das_restricted_r2"], info["das_restricted_r2"] = fit_das(model, z, task, 2, RESTRICTED, seed + 97)
    out["mean_full"] = mean_alignment(z)
    out["random_r2"] = random_alignment(z.shape[1], 2, seed + 131)
    for name, (pa, pb) in out.items():
        assert np.abs(pa @ pb).max() < 1e-8, (name, np.abs(pa @ pb).max())
        assert np.abs(pa @ pa - pa).max() < 1e-8 and np.abs(pb @ pb - pb).max() < 1e-8
    return out, info


# --------------------------------------------------------------------------------------- census
def decisions(model, states: np.ndarray, target_class: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """0 = pass, 1 = robust failure (expected class behind by > 1e-4), 2 = tie (within 1e-4)."""
    out = []
    with torch.no_grad():
        for s in range(0, len(states), chunk):
            logits = model.suffix(torch.tensor(states[s:s + chunk]))
            gap = margins(logits, torch.tensor(target_class[s:s + chunk])).numpy()
            out.append(np.where(gap > TOL, 0, np.where(gap < -TOL, 1, 2)).astype(np.int8))
    return np.concatenate(out)


def natural_index(a, b, n):
    return np.array([INDEX[t] for t in zip(a.tolist(), b.tolist(), n.tolist())])


def census(model, z: np.ndarray, task: dict, pa: np.ndarray, pb: np.ndarray, scale: float) -> dict:
    """Every single (729 per variable) and joint (19,683) interchange, with distances from natural states."""
    r, d = PAIRS.T
    single, off = [], []
    for v, p in ((0, pa), (1, pb)):
        tgt = INPUTS[r].copy()
        tgt[:, v] = INPUTS[d, v]
        edit = z[r] + (z[d] - z[r]) @ p
        single.append(decisions(model, edit, class_of(task, tgt[:, 0], tgt[:, 1])))
        off.append(np.linalg.norm(edit - z[natural_index(tgt[:, 0], tgt[:, 1], tgt[:, 2])], axis=1) / scale)
    tr, t1, t2 = TRIPLES.T
    joint_states = z[tr] + (z[t1] - z[tr]) @ pa + (z[t2] - z[tr]) @ pb
    joint = decisions(model, joint_states, class_of(task, INPUTS[t1, 0], INPUTS[t2, 1]))
    return {"single": np.stack(single), "off_natural": np.stack(off).astype(np.float32), "joint": joint}


def natural_scale(z: np.ndarray) -> float:
    """Root-mean-square distance between distinct natural site states (the unit for off-natural distances)."""
    diff = z[:, None, :] - z[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    return float(np.sqrt((dist[np.triu_indices(27, 1)] ** 2).mean()))
