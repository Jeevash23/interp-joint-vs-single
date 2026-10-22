"""Write paper/generated/numbers.tex: every study number quoted in the manuscript.

Each macro is computed from a saved, verified result file. `--check` regenerates the macros in memory
and fails if the committed file differs.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent
PROJECT = PAPER.parent
S1 = PROJECT / "studies/1-compiled-programs"
S1R = S1 / "04-Results/v2"
S1P = S1 / "posthoc/results"
S2 = PROJECT / "studies/2-trained-networks"
S2R = S2 / "results/final"
S3 = PROJECT / "studies/3-multi-site"
S3R = S3 / "results/final"
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def read(p):
    return json.loads(Path(p).read_text())


def f(x, d=2):
    text = f"{x:.{d}f}"
    if float(text) == 0:
        text = f"{0:.{d}f}"
    return "\\ensuremath{-}" + text[1:] if text.startswith("-") else text


def signed(x, d=3):
    """A difference with its sign: +0.012, \u22120.028 (zero stays unsigned)."""
    text = f(x, d)
    return text if text.startswith("\\ensuremath{-}") or float(f"{x:.{d}f}") == 0 else "\\ensuremath{+}" + text


def pct(x, d=1):
    return f(100 * x, d)


def sci(x, d=1):
    mant, exp = f"{x:.{d}e}".split("e")
    return f"\\ensuremath{{{mant}\\times 10^{{{int(exp)}}}}}"


def count(n):
    return f"{n:,}"


def ci(lo, hi, d=3):
    return f"[{f(lo, d)}, {f(hi, d)}]"


def positions(xs):
    """Ranks as text, with runs of consecutive ranks joined: [1, 2, 3, 7] -> '1--3 and 7'."""
    runs, start = [], xs[0]
    for a, b in zip(xs, xs[1:] + [None]):
        if b != a + 1:
            runs.append(str(start) if start == a else f"{start}--{a}")
            start = b
    return runs[0] if len(runs) == 1 else ", ".join(runs[:-1]) + " and " + runs[-1]


DAS_FULL = ("das_r1", "das_r2", "das_r4")
CANDIDATE_WORDS = {"das_r1": "rank-1 DAS alignment", "das_r2": "rank-2 DAS alignment", "das_r4": "rank-4 DAS alignment",
                   "das_restricted_r2": "restricted DAS alignment", "mean_full": "mean-contrast alignment",
                   "random_r2": "random alignment"}


def describe(network, candidate):
    """'additive-40015-M64', 'mean_full' -> 'the mean-contrast alignment of an additive-task MLP of width~64'."""
    family, _, arch = network.split("-")
    kind = {"T": "transformer", "M": "MLP"}[arch[0]]
    article = "an" if family[0] in "aeiou" else "a"
    return f"the {CANDIDATE_WORDS[candidate]} of {article} {family}-task {kind} of width~{arch[1:]}"


def study2_cluster_ci(rows, joint_queries, single_queries, seed=20261006, draws=10000):
    """Joint minus full-domain single detection probability of the full-domain DAS alignments, with the 95% interval
    of Study 2's program-cluster bootstrap (code/analyze.py: networks resampled within each architecture and family,
    10,000 draws, seed 20261006). A fresh generator is used for each comparison."""
    def detect(pa, pb, n, single):
        if single:
            return 1 - (1 - pa) ** ((n + 1) // 2) * (1 - pb) ** (n // 2)
        return 1 - (1 - pa) ** n
    das = [r for r in rows if r["candidate"] in DAS_FULL]
    keys = sorted({r["network"] for r in das})
    by_net = defaultdict(list)
    for r in das:
        by_net[r["network"]].append(r)
    groups = defaultdict(list)
    for k in keys:
        groups[(by_net[k][0]["family"], by_net[k][0]["arch"])].append(k)
    per_net = {k: np.array([detect(float(r["p_joint"]), None, joint_queries, False) -
                            detect(float(r["p_full_single_A"]), float(r["p_full_single_B"]), single_queries, True)
                            for r in by_net[k]]) for k in keys}
    point = float(np.mean(np.concatenate([per_net[k] for k in keys])))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        sample = []
        for g in groups.values():
            pick = rng.integers(len(g), size=len(g))
            sample.extend(per_net[g[i]] for i in pick)
        means.append(np.mean(np.concatenate(sample)))
    return point, np.quantile(means, [0.025, 0.975]).tolist()


def study1(n):
    a = read(S1R / "analysis.json")
    s = read(S1P / "summary.json")
    v = read(S1R / "independent-verification.json")
    n["CmpPrograms"] = count(a["programs"])
    n["CmpPerFamily"] = count(a["families"]["lookup"])
    n["CmpCandidates"] = count(a["candidate_instances"])
    n["CmpValid"] = count(a["reference_valid"])
    n["CmpInvalid"] = count(a["reference_invalid"])
    n["CmpQueries"] = count(a["query_records"])
    n["CmpNativeMax"] = sci(a["max_native_numpy_difference"])
    gates = [read(p)["gates"] | {"family": read(p)["family"]} for p in sorted((S1R / "final").glob("*-?????.json"))]
    for fam, key in [("lookup", "Lookup"), ("additive", "Additive")]:
        g = [x for x in gates if x["family"] == fam]
        lo, hi = min(x["residual_width"] for x in g), max(x["residual_width"] for x in g)
        n[f"CmpResid{key}"] = f"{lo}--{hi}" if lo != hi else str(lo)
        lo, hi = min(x["parameters"] for x in g), max(x["parameters"] for x in g)
        n[f"CmpParams{key}"] = f"{count(lo)}--{count(hi)}" if lo != hi else count(lo)
        n[f"CmpPool{key}"] = str(g[0]["pool_width"])
    freeze = read(S1 / "06-Study-History/V2-FROZEN-MANIFEST.json")["frozen_at_utc"]
    n["CmpFreezeTime"] = freeze[11:19] + " UTC on 29 September 2026"
    rows = list(csv.DictReader(open(S1R / "primary-table.csv")))
    t = {r["method"]: r for r in rows}
    n["CmpRestrictedAccepted"] = t["ordinary"]["invalid_accepted"]
    n["CmpRawValidAccepted"] = t["raw_latent"]["valid_accepted"]
    n["CmpRawFalseRejected"] = t["raw_latent"]["false_rejected"]
    # Census
    n["CmpLabelsMatch"] = count(s["labels_equal_single_census"])
    n["CmpIdentityMax"] = sci(s["identity_max_abs"])
    n["CmpValidOffMax"] = sci(s["valid_max_off_natural"])
    n["CmpInvalidOffMin"] = f(s["invalid_min_off_natural"], 2)
    n["CmpIrreducible"] = count(s["irreducible_joint_failures"])
    n["CmpDistinct"], n["CmpDistinctValid"] = count(s["distinct_alignments"]), count(s["distinct_valid"])
    sim = read(S1P / "panel-simulation.json")
    rows_sim = sim["rows"]
    n["CmpSimInside"] = count(sum(r["inside"] for r in rows_sim))
    n["CmpSimTotal"] = count(len(rows_sim))
    n["CmpSimReps"] = count(sim["replicates"])
    simr = {(r["panel"], r["budget"]): r for r in rows_sim}
    for key, panel in [("Single", "full_single"), ("Joint", "joint")]:
        r12 = simr[(panel, 12)]
        n[f"Cmp{key}TwelveRange"] = f"{r12['range95_low']:.0f}--{r12['range95_high']:.0f}"
    assert s["joint_only"] == 0 and s["irreducible_with_a_natural_edit"] == 0
    assert s["final_mlp_weight_on_varying_unedited_coordinates"] == 0
    assert s["irreducible_in_candidates_failing_singles"] == s["irreducible_joint_failures"]
    lr = s["lookup_restricted"]
    n["CmpRestrictedAlignments"] = count(lr["n"])
    n["CmpRestrictedQueries"] = "144"
    assert lr["restricted_panel_failures"] == 0 and lr["restricted_panel_ties"] == 0 and lr["all_fail_minimal_pairs"]
    bc = {(r["family"], r["candidate"]): r for r in s["by_class"]}
    for (fam, cand), key in [(("lookup", "das_restricted"), "Das"), (("lookup", "mean_restricted"), "Mean"),
                             (("lookup", "random"), "RandLook"), (("additive", "random"), "RandAdd")]:
        r = bc[(fam, cand)]
        for src, tag in [("panel_full_single_rate", "Single"), ("panel_joint_rate", "Joint")]:
            n[f"CmpRate{key}{tag}"] = pct(r[src]["median"], 0)
            n[f"CmpRate{key}{tag}Range"] = f"{pct(r[src]['min'], 0)}--{pct(r[src]['max'], 0)}"
    surv = s["prob_any_invalid_survives_120"]
    n["CmpSurviveSingle"] = sci(surv["full_single"])
    n["CmpSurviveJoint"] = sci(surv["joint"])
    n["CmpSurviveHeldout"] = sci(surv["heldout_single"])
    n["CmpSurviveFaith"] = f(surv["faithfulness"], 2)
    n["CmpSurviveTransport"] = f(surv["transport"], 2)
    n["CmpOtherResult"] = sci(surv["full_single"] + surv["joint"])
    curves = {(c["panel"], c["budget"]): c for c in s["invalid_acceptance_curves"]}
    c = curves[("ordinary", 120)]
    n["CmpRestrictedExpected"] = f"{f(c['expected_low'], 1)}--{f(c['expected_high'], 1)}"
    c = curves[("transport", 12)]
    n["CmpTransportTwelveExpected"] = f"{f(c['expected_low'], 0)}--{f(c['expected_high'], 0)}"
    n["CmpTransportTwelveObserved"] = str(c["observed"])
    c12s, c12j = curves[("full_single", 12)], curves[("joint", 12)]
    n["CmpSingleTwelveObserved"], n["CmpJointTwelveObserved"] = str(c12s["observed"]), str(c12j["observed"])
    n["CmpSingleTwelveExpected"] = f"{f(c12s['expected_low'], 1)}--{f(c12s['expected_high'], 1)}"
    n["CmpJointTwelveExpected"] = f"{f(c12j['expected_low'], 1)}--{f(c12j['expected_high'], 1)}"
    n["CmpRandomExpectedSurvivors"] = f(s["random_lookup_expected_survivors_ordinary"]["120"], 1)
    # Margin audit and replay
    n["CmpNearTies"] = count(v["categorical_argmax_disagreements"])
    n["CmpTieGap"] = sci(v["largest_tied_score_gap"])
    ma = {(m["method"], m["budget"]): m for m in v["margin_audit_summary"]}
    n["CmpIndeterminate"] = str(ma[("ordinary", 120)]["invalid_indeterminate"])
    clean = read(S1R / "clean-environment-replay.json")
    assert all(p["parameter_arrays_bitwise_equal"] and p["all_scientific_query_fields_exactly_equal"] for p in clean["programs"])
    n["CmpReplayRecords"] = count(sum(p["queries"] for p in clean["programs"]))
    # Worked example
    w = read(S1P / "worked-example.json")
    n["ExCarrierAfter"] = "(" + ", ".join(f(x, 1) for x in w["a_carrier_after_edit"]) + ")"
    n["ExScoreWinner"] = f(max(w["class_scores"]), 1)
    n["ExScoreExpected"] = f(w["class_scores"][w["class_values"].index(w["expected_answer"])], 1)
    n["ExRestrictedQueries"] = str(w["restricted_single_queries"])
    n["ExFullFailures"] = str(w["full_single_failures"])
    n["ExFullQueries"] = str(w["full_single_queries"])
    # CAE and Grant reproductions
    cae = s["cae"]["metrics"]
    n["CaeRuns"] = str(s["cae"]["runs_per_condition"])
    n["CaeValidDown"], n["CaeValidUp"] = f(cae["CAE_down"]["valid_mean"], 2), f(cae["CAE_up"]["valid_mean"], 2)
    n["CaeValidDownNf"], n["CaeValidUpNf"] = f(cae["CAE_down_nf"]["valid_mean"], 2), f(cae["CAE_up_nf"]["valid_mean"], 2)
    n["CaeValidIia"] = f(cae["IIAMetric"]["valid_mean"], 2)
    n["CaePDown"], n["CaePUp"] = sci(cae["CAE_down"]["mann_whitney_p"]), sci(cae["CAE_up"]["mann_whitney_p"])
    n["CaePOtherMin"] = f(min(cae[m]["mann_whitney_p"] for m in ("CAE_down_nf", "CAE_up_nf", "IIAMetric")), 2)
    fixed = read(S1P / "cae-mediator.json")
    assert fixed["released_scores_reproduced_runs_0_1"]
    assert all(v["valid_max"] == 0 for v in fixed["summary"].values())
    n["CaeFixedLeakDown"] = f(fixed["summary"]["CAE_down"]["leak_mean"], 3)
    n["CaeFixedLeakUp"] = f(fixed["summary"]["CAE_up"]["leak_mean"], 3)
    g = read(S1R / "official-grant-reproduction.json")
    n["GrantNatA"] = f"{f(g['sA1'], 2)} and {f(g['sA2'], 2)}"
    n["GrantPatched"] = f"{f(g['shat_B1'], 2)} and {f(g['shat_B2'], 2)}"


def design(n):
    """Counts that follow from the design: scoring every joint query from the single queries' gradients."""
    triples = 27 ** 3
    n["CostDotProducts"] = count(2 * triples)          # two dot products per joint query
    n["CostJointCalls"] = count(4 * triples)           # running every joint query at four calls each


def example(n):
    e = read(PROJECT / "analysis/results/constructed-example.json")
    b, o = e["both_tilted"], e["only_a_tilted"]
    assert b["single_failures"] == 0 and o["single_failures"] == 0 and o["joint_failures"] == 0
    n["ExTheta"] = f"{e['theta_degrees']:.0f}"
    n["ExSingleQueries"] = count(b["single_queries"])
    n["ExJointQueries"] = count(b["joint_queries"])
    n["ExJointFailures"] = count(b["joint_failures"])
    n["ExJointShare"] = pct(b["joint_failures"] / b["joint_queries"])
    n["ExPanelShare"] = pct(b["joint_panel_failures"] / b["joint_panel_queries"])


def study2(n):
    a = read(S2R / "analysis.json")
    ver = read(S2R / "verification.json")
    man = read(S2 / "FROZEN-MANIFEST.json")
    nets = a["networks"]
    n["TrnFreezeTime"] = man["frozen_at_utc"][11:19] + " UTC on 6 October 2026"
    n["TrnNetworks"] = count(nets["trained"])
    n["TrnCompetent"] = count(nets["competent"])
    n["TrnFirstAttempt"] = count(nets["attempts_needed"].get("1", 0))
    rows = list(csv.DictReader(open(S2R / "candidates.csv")))
    n["TrnCandidates"] = count(len(rows))
    das = [r for r in rows if r["candidate"] in ("das_r1", "das_r2", "das_r4")]
    n["TrnDas"] = count(len(das))
    sp = a["single_perfect"]
    n["TrnPerfect"] = count(sp["candidates"])
    n["TrnPerfectJointFail"] = count(sp["with_joint_failure"])
    n["TrnPerfectShare"] = pct(sp["with_joint_failure"] / sp["candidates"])
    n["TrnPerfectJointFailWord"] = WORDS[sp["with_joint_failure"]]
    n["TrnPerfectDas"] = count(sum(v for k, v in sp["by_candidate"].items() if k in DAS_FULL))
    perfect = [r for r in rows if int(r["single_fail"]) == 0 and int(r["single_tie"]) == 0]
    jo = [r for r in perfect if int(r["joint_fail_all"]) > 0]
    n["TrnJointOnlyMaxFailures"] = count(max([int(r["joint_fail_all"]) for r in jo], default=0))
    n["TrnPerfectMean"] = count(sp["by_candidate"].get("mean_full", 0))
    d = a["decomposition"]["das"]
    n["TrnIrrShare"] = pct(d["irreducible_share"])
    n["TrnIrrRate"] = pct(d["mean_irreducible_rate"])
    n["TrnSingleRate"] = pct(d["mean_single_rate"])
    n["TrnJointRate"] = pct(d["mean_joint_rate"])
    n["TrnIrrCandidates"] = count(d["candidates_with_irreducible"])
    n["TrnJointFailures"] = count(d["joint_failures"])
    n["TrnIrreducible"] = count(d["irreducible"])
    for b in ("12", "24", "60", "120"):
        c = a["primary_joint_minus_single"][b]
        key = {"12": "Twelve", "24": "TwentyFour", "60": "Sixty", "120": "OneTwenty"}[b]
        n[f"TrnContrast{key}"] = signed(c["difference"], 3)
        n[f"TrnContrast{key}CI"] = ci(*c["ci95"])
        for panel, tag in [("full_single", "Single"), ("minimal_single", "Minimal"), ("restricted_single", "Restricted"), ("joint", "Joint")]:
            n[f"TrnDetect{tag}{key}"] = f(a["detection"]["das"][panel][b], 2)
    cov = a["coverage"]
    n["TrnCovRestricted"] = pct(cov["restricted_rate"])
    n["TrnCovFull"] = pct(cov["full_rate"])
    n["TrnCovPassRestricted"] = count(cov["pass_all_restricted"])
    n["TrnCovFailFull"] = count(cov["fail_some_full"])
    n["TrnCovCandidates"] = count(cov["candidates"])
    ex = a["exploratory_distance_auc"]
    n["TrnAucMin"], n["TrnAucMax"] = f(ex["auc_min_distance"], 2), f(ex["auc_max_distance"], 2)
    assert ver["all_pass"]
    n["TrnVerifiedDecisions"] = count(ver["decisions_recomputed"])
    n["TrnVerifiedDiffer"] = count(ver["decisions_differing_within_margin"])
    for cand, key in [("mean_full", "Mean"), ("random_r2", "Random"), ("das_restricted_r2", "DasRestricted")]:
        dd = a["decomposition"][cand]
        n[f"TrnIrrShare{key}"] = pct(dd["irreducible_share"]) if dd["irreducible_share"] is not None else "--"
    # Post-freeze description of the joint-only failures.
    jo = read(S2 / "posthoc/results/joint-only.json")
    assert jo["joint_only"] == sp["with_joint_failure"] and jo["irreducible_with_a_natural_edit"] == 0
    cases = jo["cases"]
    n["TrnJointOnlyDasCount"] = WORDS[sum(c["candidate"] in DAS_FULL for c in cases)]
    outside = [c for c in cases if c["joint_failures_in_panel"] == 0]
    assert len(outside) == len(cases) - jo["joint_only_with_panel_failures"] == 1
    n["TrnJointOnlyOutsidePanel"] = describe(outside[0]["network"], outside[0]["candidate"])
    n["TrnJointOnlyMinFailures"] = count(min(c["joint_failures"] for c in cases))
    n["TrnJointOnlyMaxShare"] = pct(max(c["joint_failures"] / c["joint_queries"] for c in cases), 2)
    n["TrnJointOnlyDetect"] = pct(jo["mean_detect_joint_panel_120"])
    n["TrnJointOnlyDetectMax"] = pct(jo["max_detect_joint_panel_120"])
    meds = [c["median_off_natural"] for c in cases]
    n["TrnJointOnlyOffRange"] = f"{f(min(meds), 2)}--{f(max(meds), 2)}"
    o = jo["single_perfect_without_joint_failure_median_off_natural"]
    n["TrnPerfectOffMedian"] = f(o["median"], 2)
    n["TrnPerfectOffRange"] = f"{f(o['min'], 2)}--{f(o['max'], 2)}"
    n["TrnIrrChecked"] = count(jo["irreducible_das_failures_checked"])
    assert jo["all_alignments_landing_edits"] == 0 and jo["single_perfect_landing_edits"] == 0
    n["TrnLandMinAll"] = sci(jo["all_alignments_min_off_natural_changed_edits"])
    n["TrnLandMinPerfect"] = f(jo["single_perfect_min_off_natural_changed_edits"], 3)
    n["TrnJointOnlyPanelCount"] = WORDS[jo["joint_only_with_panel_failures"]]
    n["TrnJointOnlyMaxPanelShare"] = pct(jo["joint_only_max_panel_share"], 2)
    n["TrnIrrInPerfect"] = count(jo["irreducible_in_single_perfect_das"])
    ds = read(S2 / "posthoc/results/detection-sensitivity.json")
    o = ds["overall"]
    n["TrnEqualQueries"], n["TrnCached"] = signed(o["equal_queries_30"], 3), signed(o["cached_states_120"], 3)
    n["TrnAddonJoint"], n["TrnAddonSingle"] = pct(o["addon_joint_after_single_120"]), pct(o["addon_single_after_single_120"])
    n["TrnNeverFail"], n["TrnCeiling"] = count(o["fail_no_panel_query"]), f(o["detection_ceiling"], 2)
    cells = ds["by_cell"].values()
    words = WORDS
    n["TrnEqualPositiveCells"] = words[sum(c["equal_queries_30"] > 0 for c in cells)]
    n["TrnCachedPositiveCells"] = words[sum(c["cached_states_120"] > 0 for c in cells)]
    # Prespecified costing: "fewer alignments ... at each budget in Fig. 3b", interval below zero, "negative in every cell".
    assert all(a["primary_joint_minus_single"][b]["difference"] < 0 for b in ("12", "24", "60", "120"))
    assert a["primary_joint_minus_single"]["120"]["ci95"][1] < 0
    assert all(c["frozen_convention"] < 0 for c in cells)
    # The bootstrap below reproduces analysis.json's interval at 12 calls exactly (same method, seed and draw order).
    assert study2_cluster_ci(rows, 12 // 4, 12 // 3) == (a["primary_joint_minus_single"]["12"]["difference"],
                                                          a["primary_joint_minus_single"]["12"]["ci95"])
    # Post hoc costings with the same bootstrap: equal numbers of queries (30 each) and cached states (120 each).
    eq_point, eq_ci = study2_cluster_ci(rows, 30, 30)
    ca_point, ca_ci = study2_cluster_ci(rows, 120, 120)
    assert abs(eq_point - o["equal_queries_30"]) < 1e-12 and abs(ca_point - o["cached_states_120"]) < 1e-12
    assert eq_ci[1] < 0                                                   # "fewer" holds beyond sampling error
    assert ca_ci[0] < 0 < ca_ci[1]                                        # "within sampling error of zero"
    n["TrnEqualQueriesCI"], n["TrnCachedCI"] = ci(*eq_ci), ci(*ca_ci)
    n["TrnSmallestOff"] = f(jo["smallest_off_natural_of_both_edits"], 4)
    kinds = {"T": 0, "M": 0}
    for r in perfect:
        kinds[r["arch"][0]] += 1
    n["TrnPerfectMlp"], n["TrnPerfectTransformer"] = count(kinds["M"]), count(kinds["T"])
    n["TrnPerfectTransformerLookup"] = count(sum(r["arch"][0] == "T" and r["family"] == "lookup" for r in perfect))
    best = {}
    for r in rows:
        if r["arch"][0] == "T" and r["family"] == "lookup" and r["candidate"] in ("das_r1", "das_r2", "das_r4", "mean_full"):
            rate = (float(r["p_full_single_A"]) + float(r["p_full_single_B"])) / 2
            best[r["network"]] = min(best.get(r["network"], 1.0), rate)
    n["TrnLookupTransformerBest"] = pct(statistics.median(best.values()), 0)
    jo_arch = [r["arch"][0] for r in perfect if int(r["joint_fail_all"]) > 0]
    n["TrnJointOnlyMlp"], n["TrnJointOnlyTransformer"] = words[jo_arch.count("M")], words[jo_arch.count("T")]
    # Post hoc: the margin change of Proposition 2.
    mb = read(S2 / "posthoc/results/margin-bound.json")
    n["TrnMbIdentity"] = sci(mb["identity_max_abs"])
    n["TrnMbQueries"] = count(mb["das_population"]["queries"])
    n["TrnMbIrreducible"] = count(mb["das_population"]["irreducible"])
    assert mb["das_population"]["irreducible"] == d["irreducible"]
    n["TrnMbAuc"], n["TrnMbAucSigned"] = f(mb["auroc_predictor"], 2), f(mb["auroc_predictor_signed"], 2)
    n["TrnMbSign"], n["TrnMbSignIrr"] = pct(mb["sign_accuracy"], 0), pct(mb["sign_accuracy_irreducible"], 0)
    assert mb["perfect_alignments"] == sp["candidates"] and len(mb["joint_only_rank_positions"]) == sp["with_joint_failure"]
    n["TrnMbRanks"] = positions(mb["joint_only_rank_positions"])
    n["TrnMbRanksSigned"] = positions(mb["joint_only_rank_positions_signed"])
    n["TrnMbAlignAuc"], n["TrnMbAlignAucSigned"] = f(mb["alignment_level_auroc"], 2), f(mb["alignment_level_auroc_signed"], 2)
    pva = mb["predicted_vs_actual"]
    assert pva["perfect_with_predicted_failure"] == pva["joint_only_with_predicted_failure"]
    assert len(pva["joint_only_missed"]) == sp["with_joint_failure"] - pva["joint_only_with_predicted_failure"] == 1
    missed = [x for x in mb["per_alignment"] if f"{x['network']}/{x['candidate']}" in pva["joint_only_missed"]]
    assert missed[0]["arch"][0] == "T" and jo_arch.count("T") == 1        # "the only transformer among the seven"
    n["TrnMbMissed"] = describe(missed[0]["network"], missed[0]["candidate"])
    n["TrnMbRuleFlagged"] = words[pva["perfect_with_predicted_failure"]]
    q = mb["signed_rule_query_level"]
    n["TrnMbRuleTP"], n["TrnMbRuleFP"], n["TrnMbRuleFN"] = (count(q["predicted_and_failed"]), count(q["predicted_but_passed"]),
                                                          count(q["failed_not_predicted"]))
    lip = mb["mlp_lipschitz"]
    assert lip["certified_but_failed"] == 0
    n["TrnMbLipCertified"], n["TrnMbLipQueries"] = count(lip["certified"]), count(lip["joint_panel_queries"])
    n["TrnMbLipShare"] = pct(lip["certified"] / lip["joint_panel_queries"])


def heldout(n):
    """Post hoc held-out split of Study 2's single queries."""
    h = read(S2 / "posthoc/results/heldout-split.json")
    m = h["mean_failure_rate"]
    n["TrnHoAlignments"] = count(h["alignments"])
    n["TrnHoTrainRate"] = pct((m["train_A"] + m["train_B"]) / 2)
    n["TrnHoHeldRate"] = pct((m["held_A"] + m["held_B"]) / 2)
    n["TrnHoJointRate"] = pct(m["joint"])
    hv, tv = h["heldout_vs_joint"], h["training_vs_joint"]
    n["TrnHoContrast"], n["TrnHoContrastCI"] = signed(hv["calls_120"], 3), ci(*hv["calls_120_ci95"])
    n["TrnHoEqual"], n["TrnHoCached"] = signed(hv["equal_queries_30"], 3), signed(hv["cached_states_120"], 3)
    n["TrnHoAddonJoint"], n["TrnHoAddonSingle"] = pct(hv["addon_joint_after_single_120"]), pct(hv["addon_single_after_single_120"])
    n["TrnHoDetectSingle"], n["TrnHoDetectJoint"] = f(hv["single_detection_120"], 2), f(hv["joint_detection_120"], 2)
    n["TrnHoTrainContrast"] = signed(tv["calls_120"], 3)
    n["TrnHoPassTraining"] = count(h["pass_all_training"])
    n["TrnHoPassTrainingFailHeld"] = count(h["pass_all_training_fail_heldout"])
    n["TrnHoPassTrainingFailJoint"] = count(h["pass_all_training_fail_joint"])
    n["TrnHoPassBothFailJoint"] = count(h["pass_all_training_and_heldout_fail_joint"])
    n["TrnHoPassChanged"] = count(h["pass_all_training"] - h["pass_all_training_fail_heldout"])
    n["TrnHoCachedPositiveCells"] = WORDS[sum(c["cached_states_120"] > 0 for c in h["by_cell"].values())]


def study3(n):
    a = read(S3R / "analysis.json")
    ver = read(S3R / "verification.json")
    mar = read(S3R / "margins.json")
    man = read(S3 / "FROZEN-MANIFEST.json")
    pre = read(S3 / "PREDICTIONS-RECORD.json")
    assert ver["all_pass"]
    nets = a["networks"]
    n["MsPredictionsTime"] = pre["written_at_utc"][11:19] + " UTC"
    n["MsFreezeTime"] = man["frozen_at_utc"][11:19] + " UTC on 6 October 2026"
    n["MsNetworks"], n["MsCompetent"] = count(nets["trained"]), count(nets["competent"])
    n["MsFirstAttempt"] = count(nets["attempts_needed"].get("1", 0))
    rows = list(csv.DictReader(open(S3R / "candidates.csv")))
    n["MsAlignments"] = count(len(rows))
    das = [r for r in rows if r["candidate"] in ("das_r1", "das_r2", "das_r4")]
    n["MsDas"] = count(len(das))
    n["MsLookupWideDas"] = count(sum(r["family"] == "lookup" and r["arch"] == "T64" for r in das))
    assert all(sum(r["family"] == fam and r["arch"] == arch for r in das) == 75
               for fam in ("lookup", "additive") for arch in ("T16", "M16", "M64", "T64") if (fam, arch) != ("lookup", "T64"))
    jo = a["joint_only"]
    n["MsPerfect"], n["MsPerfectJointFail"] = count(jo["single_perfect"]), count(jo["with_joint_failure"])
    n["MsPerfectJointFailWord"] = WORDS[jo["with_joint_failure"]]
    n["MsPerfectShare"] = pct(jo["share"]) if jo["share"] is not None else "--"
    n["MsPerfectCI"] = f"{pct(jo['ci95'][0])}--{pct(jo['ci95'][1])}" if jo["ci95"] else "--"
    for kind, key in (("transformer", "Transformer"), ("mlp", "Mlp")):
        k = jo["by_kind"][kind]
        n[f"MsPerfect{key}"], n[f"MsPerfect{key}JointFail"] = count(k["single_perfect"]), count(k["with_joint_failure"])
    n["MsPerfectByCandidate"] = ", ".join(f"{v['single_perfect']} {c.replace('_', ' ')}"
                                          for c, v in jo["by_candidate"].items() if v["single_perfect"])
    p1, p2 = a["prediction_1"], a["prediction_2"]
    n["MsPredOne"] = "held" if p1["holds"] else "did not hold"
    n["MsPredOneStrict"] = "held" if p1["holds_beyond_sampling_error"] else "did not hold"
    n["MsPredTwo"] = "held" if p2["holds"] else "did not hold"
    n["MsPredTwoStrict"] = "held" if p2["holds_beyond_sampling_error"] else "did not hold"
    c = a["primary_joint_minus_single"]
    n["MsContrast"], n["MsContrastCI"] = signed(c["120"]["difference"], 3), ci(*c["120"]["ci95"])
    n["MsContrastTwelve"] = signed(c["12"]["difference"], 3)
    co = a["costings"]
    n["MsEqual"], n["MsCached"] = signed(co["das"]["equal_queries_30"], 3), signed(co["das"]["cached_states_120"], 3)
    n["MsAddonJoint"], n["MsAddonSingle"] = pct(co["das"]["addon_joint_after_single_120"]), pct(co["das"]["addon_single_after_single_120"])
    for kind, key in (("transformer", "Transformer"), ("mlp", "Mlp")):
        k = co["by_kind"][kind]
        n[f"MsContrast{key}"] = signed(k["calls_120"], 3)
        n[f"MsAddonJoint{key}"], n[f"MsAddonSingle{key}"] = pct(k["addon_joint_after_single_120"]), pct(k["addon_single_after_single_120"])
        dk = a["decomposition_by_kind"][kind]
        n[f"MsRateA{key}"], n[f"MsRateB{key}"] = pct(dk["mean_single_rate_A"]), pct(dk["mean_single_rate_B"])
        n[f"MsRateJoint{key}"] = pct(dk["mean_joint_rate"])
        n[f"MsUnexplained{key}"] = pct(dk["unexplained_share"])
        n[f"MsLand{key}"] = count(dk["all_a_edits_land"])
        n[f"MsDas{key}"] = count(dk["alignments"])
    d = a["decomposition"]["das"]
    n["MsRateA"], n["MsRateB"], n["MsRateJoint"] = pct(d["mean_single_rate_A"]), pct(d["mean_single_rate_B"]), pct(d["mean_joint_rate"])
    n["MsUnexplained"] = pct(d["unexplained_share"])
    n["MsJointFailures"], n["MsUnexplainedCount"] = count(d["joint_failures"]), count(d["unexplained"])
    mf = a["decomposition"]["mean_full"]
    n["MsMeanLand"], n["MsMeanAlignments"] = count(mf["all_a_edits_land"]), count(mf["alignments"])
    assert a["decomposition_by_kind"]["mlp"]["all_a_edits_land"] == 0      # "no MLP A-edit does"
    n["MsVerifiedDecisions"], n["MsVerifiedDiffer"] = count(ver["decisions_recomputed"]), count(ver["decisions_differing_within_margin"])
    n["MsMbIdentity"] = sci(mar["identity_max_abs"])
    for kind, key in (("transformer", "T"), ("mlp", "M")):
        m = mar[kind]
        n[f"MsMbAuc{key}"], n[f"MsMbAucSigned{key}"] = f(m["auroc_predictor"], 2), f(m["auroc_predictor_signed"], 2)
        n[f"MsMbAucDist{key}"] = f(m["auroc_a_edit_distance"], 2)
        n[f"MsMbQueries{key}"], n[f"MsMbFailures{key}"] = count(m["queries"]), count(m["unexplained_failures"])
    perfect = [r for r in rows if int(r["single_fail"]) == 0 and int(r["single_tie"]) == 0]
    jo_rows = [r for r in perfect if int(r["joint_fail_all"]) > 0]
    assert all(r["family"] == "additive" for r in perfect) and all(r["kind"] == "mlp" for r in jo_rows)
    assert all(int(r["joint_fail_all"]) == int(r["joint_fail"]) == int(r["joint_unexplained_fail"]) for r in jo_rows)
    n["MsJointOnlyMinFailures"] = count(min(int(r["joint_fail_all"]) for r in jo_rows))
    n["MsJointOnlyMaxFailures"] = count(max(int(r["joint_fail_all"]) for r in jo_rows))
    n["MsJointOnlyMaxPanelShare"] = pct(max(float(r["p_joint"]) for r in jo_rows), 2)
    n["MsJointOnlyDetect"] = pct(statistics.mean(1 - (1 - float(r["p_joint"])) ** 30 for r in jo_rows))
    n["MsPerfectTransformerLand"] = count(sum(r["kind"] == "transformer" and float(r["a_edits_landing"]) == 1.0 for r in perfect))
    mech = read(S3 / "posthoc/results/mechanism.json")
    n["MsMechLanding"] = count(mech["transformer_a_edits_land"]["alignments"])
    n["MsMechLandingContrast"] = signed(mech["transformer_a_edits_land"]["joint_minus_alternating_120"], 3)
    n["MsMechWorse"] = signed(mech["all_das"]["joint_minus_worse_variable_only_120"], 3)
    n["MsMechBMore"] = count(mech["b_fails_more_often"])
    assert mech["transformer_a_edits_land"]["mean_rate_A"] == 0
    assert abs(mech["transformer_a_edits_land"]["mean_rate_B"] - mech["transformer_a_edits_land"]["mean_rate_joint"]) < 1e-12
    cells = a["costings"]["by_cell"].values()
    assert all(c["calls_120"] > 0 for c in cells)                    # "positive in every cell"
    assert co["das"]["equal_queries_30"] > 0 and co["das"]["cached_states_120"] > 0
    mt = sum(r["kind"] == "transformer" for r in rows if r["candidate"] == "mean_full")
    assert mf["all_a_edits_land"] == mt                                   # "and in every mean-contrast alignment"
    n["MsFailedNetwork"] = a["networks"]["failed"][0] if len(a["networks"]["failed"]) == 1 else "--"
    sr = read(S3 / "posthoc/results/signed-rule.json")
    al, ql = sr["alignment_level"], sr["query_level"]
    assert al["alignments"] == jo["single_perfect"] and al["joint_only"] == jo["with_joint_failure"]
    assert al["flagged_joint_only"] == len(al["flagged"])                 # "flags exactly these"
    n["MsSrRanksSigma"], n["MsSrRanksRho"] = positions(al["ranks_sigma"]), positions(al["ranks_rho"])
    n["MsSrAlignAucSigma"], n["MsSrAlignAucRho"] = f(al["auroc_sigma"], 2), f(al["auroc_rho"], 2)
    n["MsSrFlagged"] = WORDS[len(al["flagged"])]
    for arch, tag in (("T16", "Ts"), ("T64", "Tl"), ("M16", "Ms"), ("M64", "Ml")):
        q = ql[arch]
        n[f"MsSrAucSigma{tag}"], n[f"MsSrAucRho{tag}"] = f(q["auroc_sigma"], 2), f(q["auroc_rho"], 2)
        n[f"MsSrAucDist{tag}"] = f(q["auroc_a_edit_distance"], 2)
    assert ql["transformer"]["auroc_rho"] > ql["transformer"]["auroc_sigma"]   # "rho does better than sigma there"
    dev = read(S3 / "results/development/analysis.json")
    n["MsDevPerfect"] = count(dev["joint_only"]["single_perfect"])
    n["MsDevAlignments"] = count(sum(1 for _ in csv.DictReader(open(S3 / "results/development/candidates.csv"))))
    n["MsDevContrast"] = signed(dev["primary_joint_minus_single"]["120"]["difference"], 3)
    assert dev["networks"]["trained"] == dev["networks"]["competent"]
    n["MsDevNetworks"] = count(dev["networks"]["competent"])


BUDGET_WORDS = {"12": "Twelve", "24": "TwentyFour", "60": "Sixty", "120": "OneTwenty"}
STRATEGIES = [("alternating", "Alt", "Alternating single"), ("adaptive_margin", "AdaptMargin", "Adaptive single (margins)"),
              ("oracle", "Oracle", "Oracle-targeted single"), ("joint", "Joint", "Joint"),
              ("guided", "Guided", "Margin-guided joint"), ("guided_random", "GuidedRandom", "Margin-guided, random order")]


def strategies(n):
    """Post hoc test strategies at small budgets (analysis/test_strategies.py)."""
    st = read(PROJECT / "analysis/results/test-strategies.json")
    cfg = module_constants(PROJECT / "analysis/test_strategies.py")
    n["StReplicates"] = count(cfg["R"])
    n["StAdaptiveDiff"] = sci(max(st[k]["adaptive_equals_alternating_max_abs"] for k in ("study2", "study3")))
    n["StMaxSe"] = f(max(st[k]["overall"][b][key] for k in ("study2", "study3") for b in BUDGET_WORDS
                         for key in ("guided_se", "guided_random_se")), 4)
    assert max(st[k]["adaptive_equals_alternating_max_abs"] for k in ("study2", "study3")) < 1e-12
    for study, word, gradients in (("study2", "Two", 2), ("study3", "Three", 1)):
        cost = 6 + gradients                                    # two single queries and the gradients of one candidate
        cand = 120 // (cost + 2)
        n[f"St{word}GuidedCandidates"] = count(cand)
        n[f"St{word}GuidedRuns"] = count((120 - cand * cost) // 4)
        for b, bw in BUDGET_WORDS.items():
            for key, tag, _ in STRATEGIES:
                n[f"St{word}{tag}{bw}"] = f(st[study]["overall"][b][key], 3)
            for key, tag in (("joint", "Joint"), ("guided", "Guided"), ("guided_random", "GuidedRandom")):
                n[f"St{word}Jo{tag}{bw}"] = pct(st[study]["joint_only"][b][key])
            assert st[study]["overall"][b]["guided"] < st[study]["overall"][b]["alternating"]   # "at every budget"
        n[f"St{word}JoCount"] = WORDS[len(st[study]["joint_only"]["alignments"])]
    s3 = st["study3"]["overall"]
    assert s3["24"]["adaptive_margin"] > s3["24"]["alternating"] and s3["120"]["adaptive_margin"] < s3["120"]["alternating"]
    for study in ("study2", "study3"):
        o = st[study]["overall"]
        assert all(o[b]["oracle"] >= max(o[b]["alternating"], o[b]["adaptive_margin"]) for b in BUDGET_WORDS)


def strategies_table():
    st = read(PROJECT / "analysis/results/test-strategies.json")
    lines = []
    for key, _, label in STRATEGIES:
        cells = [f(st[s]["overall"][b][key], 3) for s in ("study2", "study3") for b in BUDGET_WORDS]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n\\bottomrule\n"


def module_constants(path):
    import ast
    out = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def numbers():
    n = {}
    study1(n)
    design(n)
    example(n)
    if (S2R / "analysis.json").exists():
        study2(n)
    if (S2 / "posthoc/results/heldout-split.json").exists():
        heldout(n)
    if (S3R / "analysis.json").exists():
        study3(n)
    if (PROJECT / "analysis/results/test-strategies.json").exists():
        strategies(n)
    return n


def tables():
    """Table fragments: Study 1 candidates with their census outcomes, and the frozen panels at 120 calls."""
    s = read(S1P / "summary.json")
    bc = {(r["family"], r["candidate"]): r for r in s["by_class"]}
    names = [("compiler", "Compiler", "the carrier coordinates of $A$ and $B$"),
             ("compiler_null", "Compiler + nuisance", "adds the nuisance contrasts to $P_A$"),
             ("mean_full", "Full mean", "conditional-mean contrasts, all 27 inputs"),
             ("mean_restricted", "Restricted mean", "the same from the 12 inputs with $a,b\\in\\{0,1\\}$"),
             ("das_restricted", "Restricted DAS", "rotation trained on those inputs' single interchanges"),
             ("random", "Random", "random subspaces of the same rank")]
    lines = []
    for key, label, how in names:
        lk, ad = bc[("lookup", key)], bc[("additive", key)]
        cells = [f"{lk['valid']} / {ad['valid']}", f"{lk['fail_single']} / {ad['fail_single']}",
                 f"{lk['fail_minimal']} / {ad['fail_minimal']}", f"{lk['fail_restricted']} / {ad['fail_restricted']}",
                 f"{lk['fail_joint']} / {ad['fail_joint']}", f"{lk['joint_only'] + ad['joint_only']}"]
        lines.append(f"{label} & {how} & " + " & ".join(cells) + " \\\\")
    cand = "\n".join(lines) + "\n\\bottomrule\n"
    rows = {r["method"]: r for r in csv.DictReader(open(S1R / "primary-table.csv"))}
    a = read(S1R / "analysis.json")
    queries = {m["method"]: None for m in a["metrics"]}
    cost = {"ordinary": 3, "full_single": 3, "heldout_single": 3, "joint": 4, "faithfulness": 3, "raw_latent": 4,
            "transport": 6, "natural": 1}
    labels = [("ordinary", "Restricted singles"), ("full_single", "Full-domain singles"), ("heldout_single", "Held-out singles"),
              ("joint", "Joint"), ("faithfulness", "Complement resampling"), ("raw_latent", "Raw state distance"),
              ("transport", "Transported divergence"), ("natural", "Natural execution")]
    lines = []
    for key, label in labels:
        r = rows[key]
        lines.append(f"{label} & {120 // cost[key]} & {r['invalid_accepted']} & {r['valid_accepted']} \\\\")
    panels = "\n".join(lines) + "\n\\bottomrule\n"
    out = {"table-compiled-candidates.tex": cand, "table-compiled-panels.tex": panels}
    if (S2R / "analysis.json").exists():
        rows = list(csv.DictReader(open(S2R / "candidates.csv")))
        a2 = read(S2R / "analysis.json")
        lines = []
        for arch, label in [("T16", "Transformer, width 16"), ("T64", "Transformer, width 64"),
                            ("M16", "MLP, width 16"), ("M64", "MLP, width 64")]:
            for fam in ("lookup", "additive"):
                das = [r for r in rows if r["arch"] == arch and r["family"] == fam and r["candidate"] in ("das_r1", "das_r2", "das_r4")]
                single = statistics.mean((float(r["p_full_single_A"]) + float(r["p_full_single_B"])) / 2 for r in das)
                joint = statistics.mean(float(r["p_joint"]) for r in das)
                perfect = sum(int(r["single_fail"]) == 0 and int(r["single_tie"]) == 0 for r in das)
                jo = sum(int(r["single_fail"]) == 0 and int(r["single_tie"]) == 0 and int(r["joint_fail_all"]) > 0 for r in das)
                cell = a2["decomposition_by_cell"][f"{fam}/{arch}"]
                diff = statistics.mean((1 - (1 - float(r["p_joint"])) ** 30) -
                                       (1 - (1 - float(r["p_full_single_A"])) ** 20 * (1 - float(r["p_full_single_B"])) ** 20) for r in das)
                lines.append(f"{label} & {fam} & {pct(single, 0)} & {pct(joint, 0)} & {pct(cell['irreducible_share'])} & "
                             f"{perfect} & {jo} & {signed(diff, 3)} \\\\")
        out["table-trained-cells.tex"] = "\n".join(lines) + "\n\\bottomrule\n"
    if (S3R / "analysis.json").exists():
        rows = list(csv.DictReader(open(S3R / "candidates.csv")))
        a3 = read(S3R / "analysis.json")
        lines = []
        for arch, label in [("T16", "Transformer, width 16"), ("T64", "Transformer, width 64"),
                            ("M16", "MLP, width 16"), ("M64", "MLP, width 64")]:
            for fam in ("lookup", "additive"):
                key = f"{fam}/{arch}"
                cell = a3["decomposition_by_cell"][key]
                perfect = a3["joint_only"]["by_cell"][key]
                share = pct(cell["unexplained_share"]) if cell["unexplained_share"] is not None else "--"
                lines.append(f"{label} & {fam} & {pct(cell['mean_single_rate_A'], 0)} & {pct(cell['mean_single_rate_B'], 0)} & "
                             f"{pct(cell['mean_joint_rate'], 0)} & {share} & {perfect['single_perfect']} & "
                             f"{perfect['with_joint_failure']} & {signed(a3['costings']['by_cell'][key]['calls_120'], 3)} \\\\")
        out["table-multisite-cells.tex"] = "\n".join(lines) + "\n\\bottomrule\n"
    if (PROJECT / "analysis/results/test-strategies.json").exists():
        out["table-strategies.tex"] = strategies_table()
    return out


def render(n):
    lines = ["% Generated by paper/make_numbers.py from saved study outputs. Do not edit."]
    for k in sorted(n):
        lines.append(f"\\newcommand{{\\{k}}}{{{n[k]}}}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    files = {"numbers.tex": render(numbers()), **tables()}
    if args.check:
        stale = [k for k, v in files.items() if not (PAPER / "generated" / k).exists() or (PAPER / "generated" / k).read_text() != v]
        if stale:
            print("out of date: " + ", ".join(stale), file=sys.stderr)
            sys.exit(1)
        return
    (PAPER / "generated").mkdir(exist_ok=True)
    for k, v in files.items():
        (PAPER / "generated" / k).write_text(v)
    print(f"wrote {len(files['numbers.tex'].splitlines()) - 1} macros and {len(files) - 1} tables")


if __name__ == "__main__":
    main()
