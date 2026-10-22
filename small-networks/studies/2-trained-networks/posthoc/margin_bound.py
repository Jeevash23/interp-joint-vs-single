"""Quantitative version of Proposition 1 on the saved Study 2 networks (post hoc).

For a joint query (r, s, t), Lemma 1 gives J = S_A(z_r', s) + u_A with r' = (a_r, b_t, n_r) and
u_A = (S_B(z_r, t) - z_r') (I - P_A), and symmetrically J = S_B(z_r'', t) + u_B with r'' = (a_s, b_r, n_r).
With margin mu (target logit minus best other logit), the joint margin is the reduced single query's margin plus
the change caused by u. The first-order change is grad mu . u at the single-query state. The predictor of a
joint-only failure is min over the two orderings of |first-order change| / single-query margin.

For every joint query of the full-domain DAS alignments (ranks 1, 2, 4) and of every alignment that passes all
single queries, this script computes margins, first-order changes, the predictor, its area under the ROC curve for
irreducible failures (joint fails, both reduced single queries pass) against passing joint queries whose reduced
single queries pass, how often the first-order change has the right sign, and, for MLPs, the share of joint
queries certified by a global Lipschitz bound. Output: posthoc/results/margin-bound.json.
"""
from __future__ import annotations

import csv
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
from common import ARCHS, INPUTS, build, class_of  # noqa: E402   (frozen code, imported read-only)

TOL = 1e-4
INDEX = {tuple(x): i for i, x in enumerate(INPUTS.tolist())}
TR, T1, T2 = np.array([(r, s, t) for r in range(27) for s in range(27) for t in range(27)]).T
BOTH = (INPUTS[T1, 0] != INPUTS[TR, 0]) & (INPUTS[T2, 1] != INPUTS[TR, 1])
RA = np.array([INDEX[(INPUTS[r, 0], INPUTS[t, 1], INPUTS[r, 2])] for r, t in zip(TR, T2)])   # (a_r, b_t, n_r)
RB = np.array([INDEX[(INPUTS[s, 0], INPUTS[r, 1], INPUTS[r, 2])] for r, s in zip(TR, T1)])   # (a_s, b_r, n_r)
DAS = ("das_r1", "das_r2", "das_r4")


def margin_and_grad(model, states, cls):
    w = torch.tensor(states, requires_grad=True)
    logits = model.suffix(w)
    rows = torch.arange(len(cls))
    target = logits[rows, cls]
    other = logits.masked_fill(torch.nn.functional.one_hot(torch.tensor(cls), logits.shape[1]).bool(), float("-inf"))
    mu = target - other.max(1).values
    mu.sum().backward()
    return mu.detach().numpy(), w.grad.numpy()


def margins(model, states, cls):
    with torch.no_grad():
        logits = model.suffix(torch.tensor(states))
        rows = torch.arange(len(cls))
        other = logits.masked_fill(torch.nn.functional.one_hot(torch.tensor(cls), logits.shape[1]).bool(), float("-inf"))
        return (logits[rows, cls] - other.max(1).values).numpy()


def lipschitz_mlp(state):
    """Global Lipschitz constant of every margin of an MLP suffix w -> W3 relu(W2 w + b2) + b3."""
    w2, w3 = state["l2.weight"], state["l3.weight"]
    spec = np.linalg.norm(w2, 2)
    diff = max(np.linalg.norm(w3[i] - w3[j]) for i in range(len(w3)) for j in range(len(w3)) if i != j)
    return float(spec * diff)


def analyse(folder: Path, name: str, rec: dict):
    task = json.loads((folder / "task.json").read_text())
    kind = ARCHS[rec["arch"]][0]
    model = build(rec["arch"], len(task["classes"]))
    state = dict(np.load(folder / "weights.npz"))
    model.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
    model.eval()
    c = np.load(folder / "census.npz")
    z = c["natural_states"]
    cand = np.load(folder / "candidates.npz")
    pa, pb = cand[name + "_a"], cand[name + "_b"]
    eye = np.eye(z.shape[1])
    r, s = np.array([(i, j) for i in range(27) for j in range(27)]).T
    # Single A- and B-queries: states, margins and margin gradients (index i*27 + j for recipient i, donor j).
    wa = z[r] + (z[s] - z[r]) @ pa
    wb = z[r] + (z[s] - z[r]) @ pb
    ma, ga = margin_and_grad(model, wa, class_of(task, INPUTS[s, 0], INPUTS[r, 1]))
    mb, gb = margin_and_grad(model, wb, class_of(task, INPUTS[r, 0], INPUTS[s, 1]))
    # Joint queries.
    j = z[TR] + (z[T1] - z[TR]) @ pa + (z[T2] - z[TR]) @ pb
    mj = margins(model, j, class_of(task, INPUTS[T1, 0], INPUTS[T2, 1]))
    ia, ib = RA * 27 + T1, RB * 27 + T2                       # reduced single queries
    ua = ((z[TR] + (z[T2] - z[TR]) @ pb) - z[RA]) @ (eye - pa)  # J = wa[ia] + ua
    ub = ((z[TR] + (z[T1] - z[TR]) @ pa) - z[RB]) @ (eye - pb)  # J = wb[ib] + ub
    identity = float(max(np.abs(wa[ia] + ua - j).max(), np.abs(wb[ib] + ub - j).max()))
    da_hat, db_hat = (ga[ia] * ua).sum(1), (gb[ib] * ub).sum(1)
    da, db = mj - ma[ia], mj - mb[ib]
    reduced_pass = (ma[ia] > TOL) & (mb[ib] > TOL)
    with np.errstate(divide="ignore", invalid="ignore"):
        pred = np.minimum(np.abs(da_hat) / ma[ia], np.abs(db_hat) / mb[ib])
        pred_signed = np.minimum(-da_hat / ma[ia], -db_hat / mb[ib])
    out = {"identity_max_abs": identity, "reduced_pass": reduced_pass, "joint_fail": mj < -TOL,
           "pred": pred, "pred_signed": pred_signed, "both": BOTH,
           "sign_right_a": np.sign(da_hat) == np.sign(da), "sign_right_b": np.sign(db_hat) == np.sign(db)}
    if kind == "mlp":
        lip = lipschitz_mlp(state)
        out["lipschitz"] = lip
        out["certified"] = (lip * np.linalg.norm(ua, axis=1) < ma[ia] - TOL) | (lip * np.linalg.norm(ub, axis=1) < mb[ib] - TOL)
    return out


def auroc(score, positive):
    pos, neg = score[positive], score[~positive]
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def main():
    rows = list(csv.DictReader(open(STUDY / "results/final/candidates.csv")))
    perfect = {(r["network"], r["candidate"]) for r in rows if int(r["single_fail"]) == 0 and int(r["single_tie"]) == 0}
    joint_only = {(r["network"], r["candidate"]) for r in rows
                  if (r["network"], r["candidate"]) in perfect and int(r["joint_fail_all"]) > 0}
    todo = sorted({(r["network"], r["candidate"]) for r in rows if r["candidate"] in DAS} | perfect)
    recs = {r["network"]: r for r in rows}
    das_scores, das_labels, das_signed, sign_right, sign_irr = [], [], [], [], []
    certified, cert_fail, mlp_queries, identity, per_alignment = 0, 0, 0, 0.0, []
    for net, name in todo:
        folder = STUDY / "runs/final" / net
        rec = json.loads((folder / "record.json").read_text())
        o = analyse(folder, name, rec)
        identity = max(identity, o["identity_max_abs"])
        sel = o["both"] & o["reduced_pass"]
        if name in DAS:
            das_scores.append(o["pred"][sel])
            das_signed.append(o["pred_signed"][sel])
            das_labels.append(o["joint_fail"][sel])
            sign_right.append(np.concatenate([o["sign_right_a"][sel], o["sign_right_b"][sel]]))
            irr = sel & o["joint_fail"]
            sign_irr.append(np.concatenate([o["sign_right_a"][irr], o["sign_right_b"][irr]]))
            if "certified" in o:
                mlp_queries += int(o["both"].sum())
                certified += int((o["certified"] & o["both"]).sum())
                cert_fail += int((o["certified"] & o["both"] & o["joint_fail"]).sum())
        if (net, name) in perfect:
            allq = o["reduced_pass"]
            per_alignment.append({"network": net, "candidate": name, "arch": recs[net]["arch"],
                                  "joint_only": (net, name) in joint_only,
                                  "max_pred": float(np.nanmax(o["pred"][allq])),
                                  "max_pred_signed": float(np.nanmax(o["pred_signed"][allq])),
                                  "predicted_failures": int((o["pred_signed"][allq] > 1).sum()),
                                  "joint_failures": int(o["joint_fail"].sum())})
    scores, signed, labels = np.concatenate(das_scores), np.concatenate(das_signed), np.concatenate(das_labels)
    ranked = sorted(per_alignment, key=lambda x: -x["max_pred"])
    ranked_signed = sorted(per_alignment, key=lambda x: -x["max_pred_signed"])
    positions = [i + 1 for i, x in enumerate(ranked) if x["joint_only"]]
    positions_signed = [i + 1 for i, x in enumerate(ranked_signed) if x["joint_only"]]
    out = {"identity_max_abs": identity,
           "das_population": {"queries": int(len(labels)), "irreducible": int(labels.sum())},
           "auroc_predictor": auroc(scores, labels), "auroc_predictor_signed": auroc(signed, labels),
           "signed_rule_query_level": {"predicted_and_failed": int(((signed > 1) & labels).sum()),
                                       "predicted_but_passed": int(((signed > 1) & ~labels).sum()),
                                       "failed_not_predicted": int(((signed <= 1) & labels).sum())},
           "sign_accuracy": float(np.concatenate(sign_right).mean()),
           "sign_accuracy_irreducible": float(np.concatenate(sign_irr).mean()),
           "perfect_alignments": len(per_alignment), "joint_only_rank_positions": positions,
           "joint_only_rank_positions_signed": positions_signed,
           "alignment_level_auroc": auroc(np.array([x["max_pred"] for x in per_alignment]),
                                          np.array([x["joint_only"] for x in per_alignment])),
           "alignment_level_auroc_signed": auroc(np.array([x["max_pred_signed"] for x in per_alignment]),
                                                 np.array([x["joint_only"] for x in per_alignment])),
           "predicted_vs_actual": {"perfect_with_predicted_failure": sum(x["predicted_failures"] > 0 for x in per_alignment),
                                   "joint_only_with_predicted_failure": sum(x["predicted_failures"] > 0 for x in per_alignment if x["joint_only"]),
                                   "joint_only_missed": [x["network"] + "/" + x["candidate"] for x in per_alignment
                                                         if x["joint_only"] and x["predicted_failures"] == 0]},
           "mlp_lipschitz": {"joint_panel_queries": mlp_queries, "certified": certified, "certified_but_failed": cert_fail},
           "per_alignment": ranked}
    dest = STUDY / "posthoc/results"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "margin-bound.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_alignment"}, indent=1))


if __name__ == "__main__":
    main()
