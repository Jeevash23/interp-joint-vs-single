"""Independent check of Study 3 from saved files (NumPy only; imports nothing from sites.py, analyze.py or Study 2).

    python code/verify.py development|final

1. Re-implements both architectures from the saved weights and recomputes the natural states at both sites; every
   competent network must classify all 27 inputs correctly.
2. Recomputes every single (1,458) and joint (19,683) cross-site decision for every alignment and compares it with
   the saved census; a disagreement is allowed only when the recomputed score gap is within 1e-4.
3. Recounts, per alignment, the single failures, the joint-panel failures and the unexplained joint failures, and
   compares them with results/<split>/candidates.csv.
4. Checks the frozen file hashes for the final split.
Writes results/<split>/verification.json; exits nonzero on any failure.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np  # noqa: E402

STUDY = Path(__file__).resolve().parents[1]
GRID = np.array(list(itertools.product(range(3), repeat=3)))
TOKENS = np.column_stack([np.full(27, 3), GRID])
TOL = 1e-4


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def lin(x, w, prefix):
    return x @ w[prefix + ".weight"].T + w[prefix + ".bias"]


class Net:
    def __init__(self, folder, kind):
        self.w = dict(np.load(folder / "weights.npz"))
        self.kind = kind

    def a_states(self):
        w = self.w
        if self.kind == "mlp":
            onehot = np.concatenate([np.eye(3)[GRID[:, i]] for i in range(3)], axis=1)
            return np.maximum(lin(onehot, w, "l1"), 0)
        return w["tok.weight"][GRID[:, 0]] + w["pos.weight"][1]

    def b_states(self, ea, recipients):
        """States at B's site when A's site holds ea and the rest of the input is the recipient's."""
        w = self.w
        if self.kind == "mlp":
            return np.maximum(lin(ea, w, "l2"), 0)
        x = w["tok.weight"][TOKENS[recipients]] + w["pos.weight"][np.arange(4)][None]
        x[:, 1] = ea
        d = x.shape[-1]
        q, k, v = np.split(lin(layer_norm(x, w["ln1.weight"], w["ln1.bias"]), w, "qkv"), 3, axis=-1)
        dh = d // 2
        out = np.zeros_like(x)
        for h in range(2):
            sl = slice(h * dh, (h + 1) * dh)
            s = q[..., sl] @ np.swapaxes(k[..., sl], 1, 2) / np.sqrt(dh)
            s = np.where(np.triu(np.ones((4, 4), bool), 1)[None], -np.inf, s)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            out[..., sl] = s @ v[..., sl]
        return (x + lin(out, w, "out"))[:, -1]

    def logits(self, z):
        w = self.w
        if self.kind == "mlp":
            return lin(z, w, "l3")
        z = z + lin(np.maximum(lin(layer_norm(z, w["ln2.weight"], w["ln2.bias"]), w, "fc1"), 0), w, "fc2")
        return lin(layer_norm(z, w["lnf.weight"], w["lnf.bias"]), w, "unembed")


def gaps(net, states, cls):
    out = []
    for s in range(0, len(states), 20000):
        lg = net.logits(states[s:s + 20000])
        c = cls[s:s + 20000]
        rows = np.arange(len(c))
        exp = lg[rows, c].copy()
        lg[rows, c] = -np.inf
        out.append(exp - lg.max(1))
    return np.concatenate(out)


def decide(g):
    return np.where(g > TOL, 0, np.where(g < -TOL, 1, 2))


def main():
    split = sys.argv[1]
    problems, checked, decisions, disagreements = [], 0, 0, 0
    if split == "final":
        m = json.loads((STUDY / "FROZEN-MANIFEST.json").read_text())
        for path, digest in m["sha256"].items():
            if hashlib.sha256((STUDY / path).read_bytes()).hexdigest() != digest:
                problems.append(f"frozen file changed: {path}")
    saved_rows = {}
    with open(STUDY / "results" / split / "candidates.csv") as f:
        for row in csv.DictReader(f):
            saved_rows[(row["network"], row["candidate"])] = row
    idx = {tuple(x): i for i, x in enumerate(GRID.tolist())}
    r, d = np.array(list(itertools.product(range(27), repeat=2))).T
    tr, t1, t2 = np.array(list(itertools.product(range(27), repeat=3))).T
    both = (GRID[t1, 0] != GRID[tr, 0]) & (GRID[t2, 1] != GRID[tr, 1])
    rb = np.array([idx[(GRID[s, 0], GRID[q, 1], GRID[q, 2])] for q, s in zip(tr, t1)])     # (a_s, b_r, n_r)
    for folder in sorted((STUDY / "runs" / split).iterdir()):
        rec = json.loads((folder / "record.json").read_text())
        if not rec["competent"]:
            continue
        task = json.loads((folder / "task.json").read_text())
        table, classes = np.array(task["table"]), task["classes"]
        cls_of = {v: i for i, v in enumerate(classes)}
        target_class = lambda a, b: np.array([cls_of[int(v)] for v in table[a, b]])  # noqa: E731
        net = Net(folder, rec["kind"])
        za = net.a_states()
        zb = net.b_states(za, np.arange(27))
        census = np.load(folder / "census.npz")
        if np.abs(za - census["states_a"]).max() > 1e-9 or np.abs(zb - census["states_b"]).max() > 1e-9:
            problems.append(f"{folder.name}: natural site states differ")
        if not (gaps(net, zb, target_class(GRID[:, 0], GRID[:, 1])) > 0).all():
            problems.append(f"{folder.name}: network misclassifies a natural input")
        cand = np.load(folder / "candidates.npz")
        for i, name in enumerate(census["names"].tolist()):
            pa, pb = cand[name + "_a"], cand[name + "_b"]
            wa = net.b_states(za[r] + (za[d] - za[r]) @ pa, r)
            mine = []
            for v, states, cls in ((0, wa, target_class(GRID[d, 0], GRID[r, 1])),
                                   (1, zb[r] + (zb[d] - zb[r]) @ pb, target_class(GRID[r, 0], GRID[d, 1]))):
                g = gaps(net, states, cls)
                dec = decide(g)
                saved = census["single"][i][v]
                disagreements += int((dec != saved).sum())
                if ((dec != saved) & (np.abs(g) > TOL)).any():
                    problems.append(f"{folder.name}/{name}: single decisions differ beyond the margin")
                decisions += len(dec)
                mine.append(dec)
            w = wa[tr * 27 + t1]
            g = gaps(net, w + (zb[t2] - w) @ pb, target_class(GRID[t1, 0], GRID[t2, 1]))
            joint = decide(g)
            disagreements += int((joint != census["joint"][i]).sum())
            decisions += len(joint)
            if ((joint != census["joint"][i]) & (np.abs(g) > TOL)).any():
                problems.append(f"{folder.name}/{name}: joint decisions differ beyond the margin")
            row = saved_rows[(folder.name, name)]
            sa, sb = mine[0].reshape(27, 27), mine[1].reshape(27, 27)
            unexplained = (joint == 1) & both & (sa[tr, t1] == 0) & (sb[rb, t2] == 0)
            counts = {"single_fail": int(sum((s == 1).sum() for s in mine)),
                      "joint_fail": int(((joint == 1) & both).sum()),
                      "joint_unexplained_fail": int(unexplained.sum())}
            for key, value in counts.items():
                if int(row[key]) != value:
                    problems.append(f"{folder.name}/{name}: {key} saved {row[key]} recomputed {value}")
            checked += 1
    out = {"split": split, "alignments_checked": checked, "decisions_recomputed": decisions,
           "decisions_differing_within_margin": disagreements, "problems": problems, "all_pass": not problems}
    (STUDY / "results" / split / "verification.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "problems"}, indent=1))
    for p in problems[:20]:
        print("FAIL", p)
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
