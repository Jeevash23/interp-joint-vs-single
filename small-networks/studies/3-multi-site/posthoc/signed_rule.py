"""Out-of-sample check of the signed margin rule on Study 3 (code written after Study 3's results were known).

The rule was fixed on Study 2 (studies/2-trained-networks/posthoc/margin_bound.py, written 07:23 UTC and first run
07:24 UTC on 6 October 2026), before Study 3's protocol and predictions were recorded (07:36 UTC). It scores a joint query
by sigma, the smaller of the two first-order predicted relative drops of margin, -grad mu . u / mu, and flags it if
sigma > 1; an alignment is ranked by its largest sigma over the joint queries whose single queries pass, and flagged
if any query is flagged. Study 3 has one decomposition only (Remark 2: the B-query (r'', t) plus
u = (w - z_r'')(I - P_B)), so sigma is that one term; nothing else changes. Unsigned rho = |grad mu . u| / mu is
reported beside it.

Outputs (posthoc/results/signed-rule.json):
  * alignment level: the ranks of the joint-only alignments among all alignments that pass every single query,
    by largest sigma and by largest rho, the area under the ROC curve of each, and the alignments flagged by the rule;
  * query level, DAS alignments, joint-panel queries whose single queries (r'', t) and (r, s) both pass (the
    population of code/margins.py): area under the ROC curve of sigma, rho and the A-edit distance for unexplained
    failures, by kind and by architecture, and the rule's flagged/failed counts.
The by-kind values must equal those of the frozen code/margins.py.
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
from margins import auroc, margin_and_grad  # noqa: E402   (frozen code, imported read-only)
from sites import INPUTS, PAIRS, TRIPLES, a_to_b, class_of, natural_index  # noqa: E402
from common import build  # noqa: E402

DAS = ("das_r1", "das_r2", "das_r4")
TR, T1, T2 = TRIPLES.T
BOTH = (INPUTS[T1, 0] != INPUTS[TR, 0]) & (INPUTS[T2, 1] != INPUTS[TR, 1])
RB = natural_index(INPUTS[T1, 0], INPUTS[TR, 1], INPUTS[TR, 2])


def scores(folder: Path, name: str):
    rec = json.loads((folder / "record.json").read_text())
    task, kind = json.loads((folder / "task.json").read_text()), rec["kind"]
    model = build(rec["arch"], len(task["classes"]))
    model.load_state_dict({k: torch.tensor(v) for k, v in np.load(folder / "weights.npz").items()})
    model.eval()
    model.requires_grad_(False)
    c, cand = np.load(folder / "census.npz"), np.load(folder / "candidates.npz")
    i = c["names"].tolist().index(name)
    za, zb = c["states_a"], c["states_b"]
    pa, pb = cand[name + "_a"], cand[name + "_b"]
    r, d = PAIRS.T
    with torch.no_grad():
        wa = a_to_b(model, kind, torch.tensor(za[r] + (za[d] - za[r]) @ pa), r).numpy()
    wb = zb[r] + (zb[d] - zb[r]) @ pb
    mb, gb = margin_and_grad(model, kind, wb, class_of(task, INPUTS[r, 0], INPUTS[d, 1]))
    ib = RB * 27 + T2
    u = (wa[TR * 27 + T1] - zb[RB]) @ (np.eye(zb.shape[1]) - pb)
    dhat = (gb[ib] * u).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma, rho = -dhat / mb[ib], np.abs(dhat) / mb[ib]
    single, joint = c["single"][i], c["joint"][i]
    passing = (single[0][TR * 27 + T1] == 0) & (single[1][ib] == 0)
    return {"arch": rec["arch"], "kind": kind, "sigma": sigma, "rho": rho, "passing": passing,
            "fail": joint == 1, "dist": c["off_natural"][i][0][TR * 27 + T1]}


def rank_positions(items, key):
    ranked = sorted(items, key=lambda x: -x[key])
    return [k + 1 for k, x in enumerate(ranked) if x["joint_only"]]


def main():
    rows = list(csv.DictReader(open(STUDY / "results/final/candidates.csv")))
    perfect = [r for r in rows if int(r["single_fail"]) == 0 and int(r["single_tie"]) == 0]
    # Alignment level: every alignment that passes every single query.
    per = []
    for r in perfect:
        s = scores(STUDY / "runs/final" / r["network"], r["candidate"])
        q = s["passing"]
        per.append({"alignment": f"{r['network']}/{r['candidate']}", "arch": r["arch"],
                    "joint_only": int(r["joint_fail_all"]) > 0, "joint_failures": int(r["joint_fail_all"]),
                    "max_sigma": float(np.nanmax(s["sigma"][q])), "max_rho": float(np.nanmax(s["rho"][q])),
                    "flagged_queries": int((s["sigma"][q] > 1).sum()),
                    "flagged_and_failed": int(((s["sigma"] > 1) & q & s["fail"]).sum())})
    labels = np.array([x["joint_only"] for x in per])
    alignment = {"alignments": len(per), "joint_only": int(labels.sum()),
                 "ranks_sigma": rank_positions(per, "max_sigma"), "ranks_rho": rank_positions(per, "max_rho"),
                 "auroc_sigma": auroc(np.array([x["max_sigma"] for x in per]), labels),
                 "auroc_rho": auroc(np.array([x["max_rho"] for x in per]), labels),
                 "flagged": [x["alignment"] for x in per if x["flagged_queries"] > 0],
                 "flagged_joint_only": sum(x["flagged_queries"] > 0 and x["joint_only"] for x in per),
                 "per_alignment": sorted(per, key=lambda x: -x["max_sigma"])}
    # Query level: DAS alignments, joint panel, both single queries passing.
    pooled = {}
    for r in rows:
        if r["candidate"] not in DAS:
            continue
        s = scores(STUDY / "runs/final" / r["network"], r["candidate"])
        sel = BOTH & s["passing"]
        for group in (s["kind"], s["arch"]):
            p = pooled.setdefault(group, {"sigma": [], "rho": [], "dist": [], "fail": []})
            for k in ("sigma", "rho", "dist", "fail"):
                p[k].append(s[k][sel])
    query = {}
    for group, p in sorted(pooled.items()):
        sig, rho, dist, fail = (np.concatenate(p[k]) for k in ("sigma", "rho", "dist", "fail"))
        query[group] = {"queries": int(len(fail)), "unexplained_failures": int(fail.sum()),
                        "auroc_sigma": auroc(sig, fail), "auroc_rho": auroc(rho, fail),
                        "auroc_a_edit_distance": auroc(dist, fail),
                        "rule_flagged_and_failed": int(((sig > 1) & fail).sum()),
                        "rule_flagged_but_passed": int(((sig > 1) & ~fail).sum()),
                        "rule_missed_failures": int(((sig <= 1) & fail).sum())}
    frozen = json.loads((STUDY / "results/final/margins.json").read_text())
    for kind in ("transformer", "mlp"):
        assert abs(query[kind]["auroc_sigma"] - frozen[kind]["auroc_predictor_signed"]) < 1e-12, kind
        assert abs(query[kind]["auroc_rho"] - frozen[kind]["auroc_predictor"]) < 1e-12, kind
    out = {"alignment_level": alignment, "query_level": query}
    dest = STUDY / "posthoc/results"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "signed-rule.json").write_text(json.dumps(out, indent=1) + "\n")
    brief = {k: v for k, v in alignment.items() if k != "per_alignment"}
    print(json.dumps({"alignment_level": brief, "query_level": query}, indent=1))
    for x in alignment["per_alignment"][:12]:
        print(f"{x['max_sigma']:9.3f} {x['max_rho']:9.3f} {x['alignment']:32s} {'JOINT-ONLY' if x['joint_only'] else ''}")


if __name__ == "__main__":
    main()
