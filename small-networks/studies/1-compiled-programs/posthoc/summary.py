"""Summaries of the Study 1 census for the manuscript (posthoc/results/summary.json).

Reads posthoc/results/census.json, the frozen panel decisions (04-Results/v2/candidate-panel-results.csv)
and the saved CAE runs. Expected acceptance of a candidate by a panel after n queries is the
probability that n independent draws from the panel's query space contain no failure; singles
alternate between the two variables as in the frozen code. Ties are bounded by counting them
first as passes and then as failures.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
from scipy.stats import mannwhitneyu  # noqa: E402

STUDY = Path(__file__).resolve().parents[1]
RES = STUDY / "posthoc/results"
BUDGETS = [12, 24, 60, 120]
COST = {"ordinary": 3, "full_single": 3, "heldout_single": 3, "faithfulness": 3, "joint": 4, "raw_latent": 4,
        "transport": 6, "natural": 1}
SINGLE = ["ordinary", "full_single", "heldout_single"]


def p(c, ties_fail):
    return (c["fail"] + (c["tie"] if ties_fail else 0)) / c["queries"]


def accept_prob(rec, panel, n, ties_fail):
    if panel == "natural":
        return 1.0
    if panel in SINGLE:
        na, nb = (n + 1) // 2, n // 2
        return (1 - p(rec[f"panel_{panel}_A"], ties_fail)) ** na * (1 - p(rec[f"panel_{panel}_B"], ties_fail)) ** nb
    return (1 - p(rec[f"panel_{panel}"], ties_fail)) ** n


def rate(c):
    return c["fail"] / c["queries"]


def med_range(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def main():
    census = json.loads((RES / "census.json").read_text())
    out = {"candidates": len(census), "valid": sum(r["valid"] for r in census)}
    out["labels_equal_single_census"] = sum(r["valid"] == (not r["single_fail_any"]) for r in census)
    out["joint_only"] = sum(r["joint_only"] for r in census)
    out["irreducible_joint_failures"] = sum(r["joint_irreducible_fail"] for r in census)
    out["irreducible_in_candidates_failing_singles"] = sum(r["joint_irreducible_fail"] for r in census if r["single_fail_any"])
    out["irreducible_with_a_natural_edit"] = sum(r["irreducible_with_a_natural_edit"] for r in census)
    out["identity_max_abs"] = max(r["identity_max_abs"] for r in census)
    out["final_mlp_weight_on_varying_unedited_coordinates"] = max(r["final_mlp_weight_on_them"] for r in census)
    out["programs_with_varying_unedited_coordinates"] = len({r["program"] for r in census if r["varying_unedited_coordinates"]})
    valid = [r for r in census if r["valid"]]
    invalid = [r for r in census if not r["valid"]]
    out["valid_max_off_natural"] = max(max(r["max_off_natural_A"], r["max_off_natural_B"]) for r in valid)
    out["invalid_min_off_natural"] = min(max(r["max_off_natural_A"], r["max_off_natural_B"]) for r in invalid)
    out["valid_any_failure_or_tie"] = sum(
        any(r[k]["fail"] or r[k]["tie"] for k in r if k.startswith(("single_", "joint", "panel_")) and isinstance(r[k], dict)
            and k != "panel_raw_latent") for r in valid)
    out["invalid_with_ties"] = sum(any(r[k]["tie"] for k in r if isinstance(r[k], dict) and "tie" in r[k]) for r in invalid)

    # Per family x candidate class.
    cls = defaultdict(list)
    for r in census:
        cls[(r["family"], r["candidate"])].append(r)
    table = []
    for (fam, cand), rs in sorted(cls.items()):
        row = {"family": fam, "candidate": cand, "n": len(rs), "valid": sum(r["valid"] for r in rs),
               "fail_single": sum(r["single_fail_any"] for r in rs),
               "fail_minimal": sum(bool(r["minimal_A"]["fail"] or r["minimal_B"]["fail"]) for r in rs),
               "fail_restricted": sum(bool(r["panel_ordinary_A"]["fail"] or r["panel_ordinary_B"]["fail"]) for r in rs),
               "fail_joint": sum(bool(r["joint"]["fail"]) for r in rs),
               "joint_only": sum(r["joint_only"] for r in rs)}
        if row["valid"] < row["n"]:
            row["single_rate"] = med_range([(r["single_A"]["fail"] + r["single_B"]["fail"]) / 1458 for r in rs])
            row["joint_rate"] = med_range([r["joint"]["fail"] / r["joint"]["queries"] for r in rs])
            row["panel_full_single_rate"] = med_range(
                [(r["panel_full_single_A"]["fail"] + r["panel_full_single_B"]["fail"]) /
                 (r["panel_full_single_A"]["queries"] + r["panel_full_single_B"]["queries"]) for r in rs])
            row["panel_heldout_rate"] = med_range(
                [(r["panel_heldout_single_A"]["fail"] + r["panel_heldout_single_B"]["fail"]) /
                 (r["panel_heldout_single_A"]["queries"] + r["panel_heldout_single_B"]["queries"]) for r in rs])
            row["panel_joint_rate"] = med_range([rate(r["panel_joint"]) for r in rs])
        table.append(row)
    out["by_class"] = table

    # Expected versus observed acceptance of the 160 invalid candidates by every frozen panel and budget.
    observed = defaultdict(int)
    with open(STUDY / "04-Results/v2/candidate-panel-results.csv") as f:
        for row in csv.DictReader(f):
            if row["valid"] == "False" and row["accepted"] == "True":
                observed[(row["method"], int(row["budget"]))] += 1
    curves = []
    for panel in COST:
        for b in BUDGETS:
            n = b // COST[panel]
            lo = sum(accept_prob(r, panel, n, True) for r in invalid)
            hi = sum(accept_prob(r, panel, n, False) for r in invalid)
            curves.append({"panel": panel, "budget": b, "queries": n, "expected_low": lo, "expected_high": hi,
                           "observed": observed[(panel, b)]})
    out["invalid_acceptance_curves"] = curves
    # Chance that any invalid candidate survives 120 calls, ties counted as passes (the conservative side).
    out["prob_any_invalid_survives_120"] = {}
    for panel in ["full_single", "heldout_single", "joint", "faithfulness", "transport"]:
        n = 120 // COST[panel]
        none = 1.0
        for r in invalid:
            none *= 1 - accept_prob(r, panel, n, False)
        out["prob_any_invalid_survives_120"][panel] = 1 - none
    survivors = [r for r in invalid if r["family"] == "lookup" and r["candidate"] == "random"]
    out["random_lookup_expected_survivors_ordinary"] = {
        str(b): sum(accept_prob(r, "ordinary", b // 3, False) for r in survivors) for b in BUDGETS}

    # Restricted-support alignments in the lookup family: exact on the restricted panel, wrong beyond it.
    restricted = [r for r in census if r["family"] == "lookup" and r["candidate"] in ("mean_restricted", "das_restricted")]
    out["lookup_restricted"] = {
        "n": len(restricted),
        "restricted_panel_failures": sum(r["panel_ordinary_A"]["fail"] + r["panel_ordinary_B"]["fail"] for r in restricted),
        "restricted_panel_ties": sum(r["panel_ordinary_A"]["tie"] + r["panel_ordinary_B"]["tie"] for r in restricted),
        "all_fail_minimal_pairs": all(r["minimal_A"]["fail"] or r["minimal_B"]["fail"] for r in restricted)}

    # CAE runs saved by reproduce_official_v2.py (official setting): valid versus XOR-leak condition.
    cae = json.loads((STUDY / "04-Results/v2/official-cae-reproduction.json").read_text())
    rows = [r for r in cae["rows"] if r["mode"] == "official"]
    metrics = ["CAE_down_nf", "CAE_up_nf", "CAE_down", "CAE_up", "IIAMetric"]
    out["cae"] = {"runs_per_condition": sum(r["condition"] == "valid" for r in rows), "metrics": {}}
    for m in metrics:
        v = [r["scores"][m] for r in rows if r["condition"] == "valid"]
        x = [r["scores"][m] for r in rows if r["condition"] == "xor_leak"]
        out["cae"]["metrics"][m] = {"valid_mean": statistics.mean(v), "valid_min": min(v), "valid_max": max(v),
                                    "leak_mean": statistics.mean(x),
                                    "mann_whitney_p": float(mannwhitneyu(v, x, alternative="two-sided").pvalue)}
    out["cae"]["mediator_write_honored"] = cae["unmodified_source_semantics_check"]["mediator_write_honored"]
    # Distinct alignments: projector pairs that coincide within a program (allclose at 1e-8) count once.
    import numpy as np
    distinct, distinct_valid = 0, 0
    for prog in sorted({r["program"] for r in census}):
        c = np.load(STUDY / "03-Data/v2/final" / prog / "candidates.npz")
        valid = {r["candidate"]: r["valid"] for r in census if r["program"] == prog}
        reps = []
        for name in valid:
            pa, pb = c[name + "_a"], c[name + "_b"]
            if not any(np.allclose(pa, qa, atol=1e-8) and np.allclose(pb, qb, atol=1e-8) for qa, qb, _ in reps):
                reps.append((pa, pb, valid[name]))
        distinct += len(reps)
        distinct_valid += sum(v for _, _, v in reps)
    out["distinct_alignments"], out["distinct_valid"] = distinct, distinct_valid
    out["joint_panel_failures_invalid"] = sum(r["joint_panel_fail"] for r in invalid)
    (RES / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("by_class", "invalid_acceptance_curves")}, indent=1))


if __name__ == "__main__":
    main()
