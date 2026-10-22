"""Single and joint interchange queries on a finite factorial domain, with AI-05's definitions (model-agnostic).

Inputs x = (a, b, n), indexed 0..N-1 in the order of prompts.inputs(). Natural site states z [N, D]. A subspace is
given by an orthonormal frame F [D, r], so P = F F^T is never formed (D is about 10,000 at the one-site site).

  single A-query (r, s):   z_r + (z_s - z_r) P_A                  predicts the class of (a_s, b_r, n_r)
  single B-query (r, t):   z_r + (z_t - z_r) P_B                  predicts the class of (a_r, b_t, n_r)
  joint query (r, s, t):   z_r + (z_s - z_r) P_A + (z_t - z_r) P_B  predicts the class of (a_s, b_t, n_r)

Single queries are indexed r * N + d and joint queries (r * N + s) * N + t, as in AI-05. A query passes if the
predicted class's margin exceeds the tie band eps, fails if it is below -eps, and is a tie otherwise (codes 0, 1, 2).
The margin is supplied by the caller: margin_fn(states, recipients, classes) -> margins, so the same code serves
GPT-2 and the toy networks of the unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Domain:
    x: np.ndarray                      # [N, 3] columns a, b, n
    cls: np.ndarray                    # [N] class of each input

    def __post_init__(self):
        n = len(self.x)
        self.n = n
        self.index = {tuple(v): i for i, v in enumerate(self.x.tolist())}
        self.pairs = np.array([(r, d) for r in range(n) for d in range(n)])
        self.triples = np.array([(r, s, t) for r in range(n) for s in range(n) for t in range(n)])

    def lookup(self, a, b, n) -> np.ndarray:
        return np.array([self.index[k] for k in zip(np.asarray(a).tolist(), np.asarray(b).tolist(),
                                                    np.asarray(n).tolist())])

    # Counterfactual inputs.
    def single_target(self, v: int) -> np.ndarray:
        """Index of the input each single query of variable v simulates."""
        r, d = self.pairs.T
        t = self.x[r].copy()
        t[:, v] = self.x[d, v]
        return self.lookup(t[:, 0], t[:, 1], t[:, 2])

    def joint_target(self) -> np.ndarray:
        r, s, t = self.triples.T
        return self.lookup(self.x[s, 0], self.x[t, 1], self.x[r, 2])

    def reduced(self) -> tuple[np.ndarray, np.ndarray]:
        """For each joint query, the single queries it reduces to (AI-05 Lemma 1): the A-query with recipient
        r' = (a_r, b_t, n_r) and donor s, and the B-query with recipient r'' = (a_s, b_r, n_r) and donor t."""
        r, s, t = self.triples.T
        r1 = self.lookup(self.x[r, 0], self.x[t, 1], self.x[r, 2])
        r2 = self.lookup(self.x[s, 0], self.x[r, 1], self.x[r, 2])
        return r1 * self.n + s, r2 * self.n + t

    # Panels (AI-05 Section 2): query spaces from which a finite test samples uniformly.
    def panels(self) -> dict:
        r, d = self.pairs.T
        out = {}
        for v in (0, 1):
            o = 1 - v
            changed = self.x[r, v] != self.x[d, v]
            out[("full_single", v)] = changed
            out[("minimal_single", v)] = changed & (self.x[r, o] == self.x[d, o]) & (self.x[r, 2] == self.x[d, 2])
        tr, ts, tt = self.triples.T
        out["joint"] = (self.x[ts, 0] != self.x[tr, 0]) & (self.x[tt, 1] != self.x[tr, 1])
        return out


def project(delta: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return (delta @ frame) @ frame.T


def single_states(z: np.ndarray, frame: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    r, d = pairs.T
    return z[r] + project(z[d] - z[r], frame)


def joint_states(z: np.ndarray, fa: np.ndarray, fb: np.ndarray, triples: np.ndarray) -> np.ndarray:
    r, s, t = triples.T
    return z[r] + project(z[s] - z[r], fa) + project(z[t] - z[r], fb)


def decide(margins: np.ndarray, eps: float) -> np.ndarray:
    return np.where(margins > eps, 0, np.where(margins < -eps, 1, 2)).astype(np.int8)


def census(z, fa, fb, dom: Domain, margin_fn, eps: float, chunk: int = 4096) -> dict:
    """Every single (N^2 per variable) and joint (N^3) query of a one-site alignment: margins and decisions."""
    single_m = []
    for v, frame in ((0, fa), (1, fb)):
        tgt = dom.single_target(v)
        m = [margin_fn(single_states(z, frame, dom.pairs[i:i + chunk]), dom.pairs[i:i + chunk, 0],
                       dom.cls[tgt[i:i + chunk]]) for i in range(0, len(dom.pairs), chunk)]
        single_m.append(np.concatenate(m))
    tgt = dom.joint_target()
    joint_m = np.concatenate([margin_fn(joint_states(z, fa, fb, dom.triples[i:i + chunk]), dom.triples[i:i + chunk, 0],
                                        dom.cls[tgt[i:i + chunk]]) for i in range(0, len(dom.triples), chunk)])
    single_m = np.stack(single_m).astype(np.float32)
    return {"single_margin": single_m, "joint_margin": joint_m.astype(np.float32),
            "single": decide(single_m, eps), "joint": decide(joint_m, eps)}


def classify_one_site(single: np.ndarray, joint: np.ndarray, dom: Domain) -> dict:
    """Joint-panel failures split by whether a reduced single query fails (reducible) or both pass (irreducible)."""
    ia, ib = dom.reduced()
    jp = dom.panels()["joint"]
    both_pass = (single[0][ia] == 0) & (single[1][ib] == 0)
    reduce_fail = (single[0][ia] == 1) | (single[1][ib] == 1)
    jf = (joint == 1) & jp
    return {"joint_panel_failures": int(jf.sum()), "reducible": int((jf & reduce_fail).sum()),
            "irreducible": int((jf & both_pass).sum()), "both_reduced_pass": both_pass}


def classify_two_site(single: np.ndarray, joint: np.ndarray, dom: Domain) -> dict:
    """AI-05 Study 3: a joint-panel failure is unexplained if the B-query (r'', t) and the A-query (r, s) both pass."""
    r, s, t = dom.triples.T
    _, ib = dom.reduced()
    jp = dom.panels()["joint"]
    both_pass = (single[1][ib] == 0) & (single[0][r * dom.n + s] == 0)
    jf = (joint == 1) & jp
    return {"joint_panel_failures": int(jf.sum()), "explained": int((jf & ~both_pass).sum()),
            "unexplained": int((jf & both_pass).sum()), "both_reduced_pass": both_pass}


def detect(p_a: float, p_b: float | None, n: int, single: bool) -> float:
    """Probability that n uniform draws contain a failure; single tests alternate A and B (AI-05 Section 2)."""
    if single:
        return 1 - (1 - p_a) ** ((n + 1) // 2) * (1 - p_b) ** (n // 2)
    return 1 - (1 - p_a) ** n


def panel_rates(single: np.ndarray, joint: np.ndarray, dom: Domain) -> dict:
    masks = dom.panels()
    out = {}
    for key in ("full_single", "minimal_single"):
        for v, name in ((0, "A"), (1, "B")):
            out[f"p_{key}_{name}"] = float((single[v][masks[(key, v)]] == 1).mean())
    out["p_joint"] = float((joint[masks["joint"]] == 1).mean())
    # Development-informed addendum (PROTOCOL Section 17): every single query of each variable, including those in
    # which the swapped variable keeps its value, and those alone.
    for v, name in ((0, "A"), (1, "B")):
        out[f"p_all_single_{name}"] = float((single[v] == 1).mean())
        out[f"p_unchanged_single_{name}"] = float((single[v][~masks[("full_single", v)]] == 1).mean())
    return out
