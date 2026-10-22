"""Where Study 3's positive detection contrast comes from (post hoc; written after the frozen results were known).

Two descriptions of the DAS alignments, from results/final/candidates.csv:
  * the contrast at 120 calls (3 calls per single, 4 per joint query) separately for the transformer alignments whose
    A-edits all land, for which every joint query is a single B-query (Remark 2), and for all other alignments;
  * joint queries against single tests that spend the whole budget on the variable that fails more often in that
    alignment (40 queries), which a tester could only do knowing that variable in advance.
Output: posthoc/results/mechanism.json.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
DAS = ("das_r1", "das_r2", "das_r4")


def alternating(r, n):
    pa, pb = float(r["p_full_single_A"]), float(r["p_full_single_B"])
    return 1 - (1 - pa) ** ((n + 1) // 2) * (1 - pb) ** (n // 2)


def worse_only(r, n):
    return 1 - (1 - max(float(r["p_full_single_A"]), float(r["p_full_single_B"]))) ** n


def joint(r, n):
    return 1 - (1 - float(r["p_joint"])) ** n


def describe(rs):
    if not rs:
        return None
    m = statistics.mean
    return {"alignments": len(rs),
            "mean_rate_A": m(float(r["p_full_single_A"]) for r in rs), "mean_rate_B": m(float(r["p_full_single_B"]) for r in rs),
            "mean_rate_joint": m(float(r["p_joint"]) for r in rs),
            "joint_minus_alternating_120": m(joint(r, 30) - alternating(r, 40) for r in rs),
            "joint_minus_worse_variable_only_120": m(joint(r, 30) - worse_only(r, 40) for r in rs)}


def main():
    rows = [r for r in csv.DictReader(open(STUDY / "results/final/candidates.csv")) if r["candidate"] in DAS]
    landing = [r for r in rows if r["kind"] == "transformer" and float(r["a_edits_landing"]) == 1.0]
    rest = [r for r in rows if not (r["kind"] == "transformer" and float(r["a_edits_landing"]) == 1.0)]
    out = {"all_das": describe(rows), "transformer_a_edits_land": describe(landing), "others": describe(rest),
           "mlp": describe([r for r in rows if r["kind"] == "mlp"]),
           "b_fails_more_often": sum(float(r["p_full_single_B"]) > float(r["p_full_single_A"]) for r in rows)}
    dest = STUDY / "posthoc/results"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "mechanism.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
