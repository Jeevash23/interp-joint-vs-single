"""Exploratory quantity of Study 3: the first-order margin predictor with its single term (needs the saved models).

    python code/margins.py development|final

For a joint query (r, s, t), let w be the A-edit of (r, s) propagated to B's site and r'' = (a_s, b_r, n_r). The joint
state equals the single B-query state S_B(z_{r''}, t) plus u = (w - z_{r''})(I - P_B). The predictor is
|grad mu . u| / mu at the B-query state, mu being the B-query's margin. Population: joint-panel queries of the DAS
alignments whose single queries (r'', t) and (r, s) both pass; positives are the joint queries that fail. Writes
results/<split>/margins.json with areas under the ROC curve, by kind, beside that of the A-edit's distance at A's site.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "code"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
from sites import INPUTS, PAIRS, TRIPLES, a_to_b, class_of, natural_index, readout  # noqa: E402
from common import build  # noqa: E402   (Study 2's frozen code)

DAS = ("das_r1", "das_r2", "das_r4")
TR, T1, T2 = TRIPLES.T
BOTH = (INPUTS[T1, 0] != INPUTS[TR, 0]) & (INPUTS[T2, 1] != INPUTS[TR, 1])
RB = natural_index(INPUTS[T1, 0], INPUTS[TR, 1], INPUTS[TR, 2])


def auroc(score, positive):
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def margin_and_grad(model, kind, states, cls):
    w = torch.tensor(states, requires_grad=True)
    logits = readout(model, kind, w)
    hot = torch.nn.functional.one_hot(torch.tensor(cls), logits.shape[1]).bool()
    mu = logits[torch.arange(len(cls)), torch.tensor(cls)] - logits.masked_fill(hot, float("-inf")).max(1).values
    mu.sum().backward()
    return mu.detach().numpy(), w.grad.numpy()


def main():
    split = sys.argv[1]
    pooled = {k: {"pred": [], "signed": [], "dist": [], "label": []} for k in ("transformer", "mlp")}
    identity = 0.0
    for folder in sorted((STUDY / "runs" / split).iterdir()):
        rec = json.loads((folder / "record.json").read_text())
        if not rec["competent"]:
            continue
        task, kind = json.loads((folder / "task.json").read_text()), rec["kind"]
        model = build(rec["arch"], len(task["classes"]))
        model.load_state_dict({k: torch.tensor(v) for k, v in np.load(folder / "weights.npz").items()})
        model.eval()
        model.requires_grad_(False)
        c, cand = np.load(folder / "census.npz"), np.load(folder / "candidates.npz")
        za, zb = c["states_a"], c["states_b"]
        names = c["names"].tolist()
        r, d = PAIRS.T
        for name in DAS:
            i = names.index(name)
            pa, pb = cand[name + "_a"], cand[name + "_b"]
            with torch.no_grad():
                wa = a_to_b(model, kind, torch.tensor(za[r] + (za[d] - za[r]) @ pa), r).numpy()
            wb = zb[r] + (zb[d] - zb[r]) @ pb                                    # single B-queries
            mb, gb = margin_and_grad(model, kind, wb, class_of(task, INPUTS[r, 0], INPUTS[d, 1]))
            ib = RB * 27 + T2
            w = wa[TR * 27 + T1]
            u = (w - zb[RB]) @ (np.eye(zb.shape[1]) - pb)
            joint_state = w + (zb[T2] - w) @ pb
            identity = max(identity, float(np.abs(wb[ib] + u - joint_state).max()))
            dhat = (gb[ib] * u).sum(1)
            single, joint = c["single"][i], c["joint"][i]
            sel = BOTH & (single[0][TR * 27 + T1] == 0) & (single[1][ib] == 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                pooled[kind]["pred"].append((np.abs(dhat) / mb[ib])[sel])
                pooled[kind]["signed"].append((-dhat / mb[ib])[sel])
            pooled[kind]["dist"].append(c["off_natural"][i][0][TR * 27 + T1][sel])
            pooled[kind]["label"].append((joint == 1)[sel])
    out = {"identity_max_abs": identity}
    for kind, v in pooled.items():
        if not v["label"]:
            continue
        lab = np.concatenate(v["label"])
        out[kind] = {"queries": int(len(lab)), "unexplained_failures": int(lab.sum()),
                     "auroc_predictor": auroc(np.concatenate(v["pred"]), lab),
                     "auroc_predictor_signed": auroc(np.concatenate(v["signed"]), lab),
                     "auroc_a_edit_distance": auroc(np.concatenate(v["dist"]), lab)}
    dest = STUDY / "results" / split
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "margins.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
