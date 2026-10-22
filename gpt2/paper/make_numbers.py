"""Write paper/generated/numbers.tex and the generated tables from the saved study outputs.

Every study number in the manuscript is a macro defined here, computed from: the configuration and prompt files; the
development settings, analyses and records; the confirmation analyses of GPT-2 Small (local) and GPT-2 Medium (cloud);
arm d's analyses; the pilot, cloud check and cost records; the post hoc verification; and AI-05's generated numbers
(for the companion study's results). `--check` recomputes everything in memory and fails if a written file differs.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
STUDY = ROOT / "study"
RES = STUDY / "results"
AI05 = ROOT / "reference/joint-vs-single-interchange"  # release: the companion study's numbers
sys.path.insert(0, str(STUDY / "code"))
import interchange as ic  # noqa: E402
import prompts  # noqa: E402


def load(path: Path):
    return json.loads(path.read_text())


CONFIG = load(STUDY / "config.json")
CLOUD = load(STUDY / "cloud/cloud-config.json")
SET_S = load(RES / "development-settings.json")
SET_M = load(RES / "cloud/gpt2-medium/development-settings.json")
DEV_S = load(RES / "analysis/development/summary.json")
CONF_S = load(RES / "analysis/confirmation/summary.json")
CONF_M = load(RES / "cloud/gpt2-medium/analysis/confirmation/summary.json")
D_FINAL = load(RES / "analysis/overlap-final/summary.json")
D_DEV = load(RES / "analysis/overlap-development/summary.json")
PILOT = load(RES / "pilot.json")
PARITY = load(RES / "pilot-parity-float64.json")
CHECK = load(RES / "cloud/check.json")
COST = load(RES / "cloud/cost-decision.json")
SPEND = load(RES / "cloud/final-spend.json")
GATE_S = load(RES / "confirmation-gate.json")
GATE_M = load(RES / "cloud/gpt2-medium/confirmation-gate.json")
VERIFY = load(ROOT / "posthoc/results/verify-confirmation.json")


# ------------------------------------------------------------------------------------------------ formatting
def f(x, d=3):
    text = f"{x:.{d}f}"
    if float(text) == 0:
        text = f"{0:.{d}f}"
    return "\\ensuremath{-}" + text[1:] if text.startswith("-") else text


def signed(x, d=3):
    text = f(x, d)
    return text if text.startswith("\\ensuremath") else "\\ensuremath{+}" + text


def pct(x, d=0):
    return f(100 * x, d)


def num(n):
    return f"{int(n):,}"


def sci(x):
    mant, exp = f"{x:.1e}".split("e")
    return f"\\ensuremath{{{mant}\\times 10^{{{int(exp)}}}}}"


def interval(lo, hi, d=3, sign=False):
    g = signed if sign else f
    return f"[{g(lo, d)}, {g(hi, d)}]"


def words(n):
    return ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][n] if n <= 10 else num(n)


def ai05_macro(name: str) -> str:
    text = (AI05 / "paper/generated/numbers.tex").read_text()
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{(.*)\}\s*$", text, re.M)
    if not m:
        raise SystemExit(f"AI-05 macro {name} not found")
    return m.group(1)


# ---------------------------------------------------------------------------------------------------- macros
def macros() -> dict[str, str]:
    m = {}
    M = CONFIG["templates_per_task"]
    x = prompts.inputs(M)
    dom = ic.Domain(x, prompts.answer_class(x[:, 0], x[:, 1]))
    panels = dom.panels()
    m.update(NTemplates=str(M), NPrompts=str(len(x)), NTokens=str(prompts.LENGTH), NTasksDev=str(CONFIG["development_tasks"]),
             NTasksConf=str(CONFIG["confirmation_tasks"]), CompetenceMargin=f(CONFIG["competence_margin"], 0),
             CostSingle=str(CONFIG["costs"]["single"]), CostJoint=str(CONFIG["costs"]["joint"]), Budget=str(CONFIG["budgets"][-1]),
             NBoot=num(CONFIG["bootstrap_replicates"]), NSingle=num(2 * dom.n ** 2), NSinglePerVar=num(dom.n ** 2),
             NJoint=num(dom.n ** 3), NFullA=num(panels[("full_single", 0)].sum()), NFullB=num(panels[("full_single", 1)].sum()),
             NJointPanel=num(panels["joint"].sum()), SiteDim=num(len(prompts.SITE_POSITIONS) * 768))
    lr_loss = sorted(SET_S["lr_losses"].items(), key=lambda kv: kv[1])
    m.update(LrLossBest=sci(lr_loss[0][1]), LrLossSecond=sci(lr_loss[1][1]), LrLossThird=sci(lr_loss[2][1]))
    m.update(LrSmall=f(SET_S["lr"], 3), StepsSmall=num(SET_S["steps"]), LayerSmall=str(SET_S["layer"]),
             TieBand=f(SET_S["eps"], 3), LayerMedium=str(SET_M["layer"]),
             CandSmall=", ".join(map(str, CONFIG["candidate_layers"][:-1])) + f" and {CONFIG['candidate_layers'][-1]}",
             CandMedium=", ".join(map(str, CLOUD["candidate_layers"]["gpt2-medium"][:-1]))
             + f" and {CLOUD['candidate_layers']['gpt2-medium'][-1]}")
    # The companion study (AI-05), from its generated numbers.
    m.update(AiFiveTrainedNetworks=ai05_macro("TrnNetworks"), AiFiveJointOnly=ai05_macro("TrnPerfectJointFail"),
             AiFivePerfect=ai05_macro("TrnPerfect"), AiFiveFlagged=ai05_macro("TrnMbRuleFlagged"),
             AiFiveContrast=ai05_macro("TrnContrastOneTwenty"))
    # Pilot, Amendment 1, the cloud check and cost.
    m.update(PilotMarginGap=sci(max(c["max_margin_gap"] for c in PILOT["checks"].values())),
             ParityFloatSixtyFour=sci(max(v["max_logit_gap"] for v in PARITY["float64"].values())),
             NReproQueries=num(CHECK["reproduction"]["queries"]), CheckReproGap=sci(CHECK["reproduction"]["max_abs_margin_difference"]),
             TfLoss=f(CHECK["tf32_check"]["best_loss_tf32"], 7), LocalLoss=f(CHECK["tf32_check"]["best_loss_local_float32"], 7),
             TfAgreePct=pct(CHECK["tf32_check"]["single_decisions_agreeing"]),
             NTasksMedium=str(COST["confirmation_tasks"]["gpt2-medium"]), ArmBudget=f"\\${CLOUD['budget']['arm_stop_usd']:.0f}",
             ArmSpent=f"\\${SPEND['arm_usd']:.2f}")
    # Gates.
    for key, gate in (("Small", GATE_S), ("Medium", GATE_M)):
        drawn = len(gate["passed"]) + len(gate["failed"])
        m[f"GateDrawn{key}"] = str(drawn)
        m[f"GatePassed{key}"] = str(len(gate["passed"]))
    # Development numbers that led to the addendum (GPT-2 Small).
    unchanged = {2: [], 4: []}
    for path in sorted((RES / "runs/development").glob("*/arm-a.npz")):
        arr = np.load(path)
        for r in unchanged:
            single = arr[f"das_r{r}__single"][0]
            unchanged[r].append((single[~panels[("full_single", 0)]] == 1).mean())
    m.update(DevUnchangedRTwoPct=pct(np.mean(unchanged[2])), DevUnchangedRFourPct=pct(np.mean(unchanged[4])),
             DevContrastSmall=signed(DEV_S["a"]["detection"]["120"]["contrast"]))
    # Confirmation, one site and the other arms.
    for key, summ in (("S", CONF_S), ("M", CONF_M)):
        a = summ["a"]
        jo, mr, det = a["joint_only"], a["margin_rule"], a["detection"]["120"]
        m[f"Align{key}"] = num(jo["alignments"])
        m[f"Perfect{key}"] = str(jo["pass_all_singles"])
        m[f"JointOnly{key}"] = str(jo["joint_only"])
        m[f"PerfectWord{key}"] = words(jo["pass_all_singles"])
        m[f"JointOnlyWord{key}"] = words(jo["joint_only"])
        if jo["pass_all_singles"]:
            m[f"JointOnlyShare{key}"] = f(jo["share"], 2)
            m[f"JointOnlyCI{key}"] = interval(*jo["ci95"], d=2)
        if mr.get("testable"):
            m[f"Flagged{key}"] = str(mr["flagged_joint_only"])
            m[f"FlaggedOther{key}"] = str(mr["flagged_other"])
        m[f"Contrast{key}"] = signed(det["contrast"])
        m[f"ContrastCILo{key}"], m[f"ContrastCIHi{key}"] = signed(det["ci95"][0]), signed(det["ci95"][1])
        m[f"ContrastAll{key}"] = signed(det["contrast_all_single"])
        m[f"ContrastAllCILo{key}"] = signed(det["ci95_all_single"][0])
        m[f"ContrastAllCIHi{key}"] = signed(det["ci95_all_single"][1])
        m[f"ContrastAll{key}AtTwelve"] = signed(a["detection"]["12"]["contrast_all_single"])
        m[f"DetSingle{key}"] = f(det["single"])
        m[f"DetJoint{key}"] = f(det["joint"])
        m[f"DetAllSingle{key}"] = f(det["all_single"])
        for b, word in (("12", "Twelve"), ("24", "TwentyFour"), ("60", "Sixty")):
            m[f"Contrast{key}At{word}"] = signed(a["detection"][b]["contrast"])
        rates = a["mean_rates_das"]
        m[f"RateFullA{key}"] = pct(rates["p_full_single_A"], 1)
        m[f"RateFullB{key}"] = pct(rates["p_full_single_B"], 1)
        m[f"RateJoint{key}"] = pct(rates["p_joint"], 1)
        m[f"RateUnchangedA{key}"] = pct(rates["p_unchanged_single_A"], 1)
        m[f"RateUnchangedB{key}"] = pct(rates["p_unchanged_single_B"], 1)
        m[f"IrrShare{key}"] = pct(a["irreducible_share"], 0)
    for arm, key in (("a_end", "End"), ("b", "Heads"), ("c", "Sep")):
        a = CONF_S[arm]
        rates = a["mean_rates_das"]
        m[f"{key}RateA"] = pct(rates["p_full_single_A"], 1)
        m[f"{key}RateB"] = pct(rates["p_full_single_B"], 1)
        m[f"{key}RateJoint"] = pct(rates["p_joint"], 1)
        m[f"{key}Perfect"] = str(a["joint_only"]["pass_all_singles"])
        m[f"{key}Contrast"] = signed(a["detection"]["120"]["contrast"])
        m[f"{key}ContrastCI"] = interval(*a["detection"]["120"]["ci95"], sign=True)
    bma = CONF_S["b_minus_a"]
    m["HeadsMinusOneAbs"] = f(abs(bma["difference"]))
    m["HeadsMinusOneCILoAbs"], m["HeadsMinusOneCIHiAbs"] = f(abs(bma["ci95"][1])), f(abs(bma["ci95"][0]))
    # Per-kind rates, the held-out split and the GPT-2 Medium alignments that pass every single query.
    rows_s = [r for r in csv.DictReader(open(RES / "analysis/confirmation/alignments.csv")) if r["arm"] == "a"]
    rows_m = list(csv.DictReader(open(RES / "cloud/gpt2-medium/analysis/confirmation/alignments.csv")))
    kind = lambda rows, k, col: np.mean([float(r[col]) for r in rows if r["alignment"] == k])  # noqa: E731
    m.update(UnchRTwoS=pct(kind(rows_s, "das_r2", "p_unchanged_single_A"), 1),
             UnchRFourS=pct(kind(rows_s, "das_r4", "p_unchanged_single_A"), 1),
             JointRTwoS=pct(kind(rows_s, "das_r2", "p_joint"), 1), JointRFourS=pct(kind(rows_s, "das_r4", "p_joint"), 1),
             MinSingleFailRTwoS=str(min(int(r["single_fail"]) for r in rows_s if r["alignment"] == "das_r2")))
    perfect_m = [int(r["joint_fail_all"]) for r in rows_m if int(r["single_fail"]) == 0 and int(r["single_tie"]) == 0]
    m.update(MedJointFailMin=num(min(perfect_m)), MedJointFailMax=num(max(perfect_m)))
    held = []
    for path in sorted((RES / "runs/confirmation").glob("*/arm-a.npz")):
        arr = np.load(path)
        mask, single = arr["das_r2_split__train_mask"], arr["das_r2_split__single"]
        for v in (0, 1):
            fs = panels[("full_single", v)]
            held.append((single[v][fs & ~mask] == 1).mean())
    m["SplitHeldOutS"] = pct(np.mean(held), 1)
    m["GateWrongNamePctSmall"] = pct(np.mean([g["min_margin"] < 0 for g in GATE_S["failed"]]))
    ov = CONF_S["c"]["overlap"]
    m.update(SepOverlapMedian=f(ov["overlap_median"], 2), SepOverlapMin=f(ov["overlap_range"][0], 3),
             SepOverlapMax=f(ov["overlap_range"][1], 2), SepSpearman=signed(ov["spearman"], 2),
             SepSpearmanCI=interval(*ov["ci95"], d=2, sign=True))
    # Arm d (AI-05's networks).
    sep, ref = D_FINAL["separate"], D_FINAL["study2_orthogonal"]
    m.update(DAlign=num(sep["alignments"]), DPerfect=str(sep["pass_all_singles"]), DJointOnly=str(sep["joint_only"]),
             DShare=pct(sep["share"], 1) if sep["share"] is not None else "--",
             DShareCI=f"{pct(sep['ci95'][0], 1)}\\% to {pct(sep['ci95'][1], 1)}\\%",
             DRefPerfect=str(ref["pass_all_singles"]), DRefJointOnly=str(ref["joint_only"]), DRefShare=pct(ref["share"], 1),
             DAuroc=f(D_FINAL["overlap_auroc_for_joint_only"], 2) if D_FINAL["overlap_auroc_for_joint_only"] is not None else "--",
             DAurocCI=interval(*D_FINAL["overlap_auroc_ci95"], d=2) if D_FINAL["overlap_auroc_ci95"] else "--",
             DSpearman=signed(D_FINAL["spearman_overlap_vs_joint_rate_das"], 2))
    if D_FINAL["margin_rule"].get("testable"):
        m.update(DFlagged=str(D_FINAL["margin_rule"]["flagged_joint_only"]),
                 DFlaggedOther=words(D_FINAL["margin_rule"]["flagged_other"]))
    # Verification.
    m.update(VerifyQueriesConf=num(VERIFY["queries"]), VerifyDisagreeConf=words(len(VERIFY["disagreements"])),
             VerifyMaxGapConf=sci(VERIFY["max_abs_margin_difference"]))
    return m


def verdict(estimate, lo, hi, predicted_above: bool, threshold: float, text: str) -> str:
    holds = estimate > threshold if predicted_above else estimate < threshold
    beyond = (lo > threshold if holds else hi < threshold) if predicted_above else (hi < threshold if holds else lo > threshold)
    return f"{'Holds' if holds else 'Does not hold'}{', beyond sampling error' if beyond else ''}: {text}"


def predictions_table() -> str:
    rows = []
    a_s, d = CONF_S["a"], D_FINAL
    for model, summ in (("GPT-2 Small", CONF_S), ("GPT-2 Medium", CONF_M)):
        n = summ["a"]["joint_only"]["pass_all_singles"]
        why = ("not testable: no alignment passes every single query" if n == 0 else
               f"not testable: {words(n)} alignment{'s pass' if n != 1 else ' passes'} every single query (10 needed)")
        rows.append(("1", f"{model}: at most 10\\% of one-site alignments that pass every single query fail a joint query",
                     why[0].upper() + why[1:]))
        rows.append(("2", f"{model}: the margin rule flags at least half of the joint-only alignments and at most 10\\% "
                          "of the others", why[0].upper() + why[1:]))
        det = summ["a"]["detection"]["120"]
        rows.append(("3", f"{model}: joint minus full-domain single detection at {CONFIG['budgets'][-1]} calls below 0",
                     verdict(det["contrast"], *det["ci95"], False, 0.0,
                             f"{signed(det['contrast'])} {interval(*det['ci95'], sign=True)}")))
    bma = CONF_S["b_minus_a"]
    rows.append(("4", "Arm b's detection contrast above arm a's at one site",
                 verdict(bma["difference"], *bma["ci95"], True, 0.0,
                         f"{signed(bma['difference'])} {interval(*bma['ci95'], sign=True)}")))
    b120 = CONF_S["b"]["detection"]["120"]
    rows.append(("4", "Arm b's own detection contrast at least 0",
                 f"Holds: zero to rounding ({sci(b120['contrast'])}; both tests detect every arm-b alignment)"))
    ov = CONF_S["c"]["overlap"]
    rows.append(("5a", "Arm c: median $\\|P_AP_B\\|$ below 0.2",
                 f"{'Holds' if ov['overlap_median'] < 0.2 else 'Does not hold'}: {f(ov['overlap_median'], 2)}"))
    rows.append(("5b", "Arm c: joint failure rate rises with $\\|P_AP_B\\|$ (Spearman above 0)",
                 verdict(ov["spearman"], *ov["ci95"], True, 0.0,
                         f"{signed(ov['spearman'], 2)} {interval(*ov['ci95'], d=2, sign=True)}")))
    rows.append(("5c", "Arm c's joint-only share at least arm a's",
                 "Not testable: no alignment passes every single query in either arm"))
    sep, ref = d["separate"], d["study2_orthogonal"]
    rows.append(("6a", f"Arm d: joint-only share above the companion study's {pct(ref['share'], 1)}\\%",
                 verdict(sep["share"], *sep["ci95"], True, ref["share"],
                         f"{pct(sep['share'], 1)}\\% [{pct(sep['ci95'][0], 1)}, {pct(sep['ci95'][1], 1)}]")))
    rows.append(("6b", "Arm d: $\\|P_AP_B\\|$ separates joint-only alignments (area under the ROC curve above 0.5)",
                 verdict(d["overlap_auroc_for_joint_only"], *d["overlap_auroc_ci95"], True, 0.5,
                         f"{f(d['overlap_auroc_for_joint_only'], 2)} {interval(*d['overlap_auroc_ci95'], d=2)}")))
    return "".join(f"{a} & {b} & {c} \\\\\n" for a, b, c in rows)


def numbers_tex(m: dict[str, str]) -> str:
    lines = ["% Generated by make_numbers.py from the saved study outputs. Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    out = {PAPER / "generated/numbers.tex": numbers_tex(macros()),
           PAPER / "generated/table-predictions.tex": predictions_table()}
    if args.check:
        bad = [str(k) for k, v in out.items() if not k.exists() or k.read_text() != v]
        if bad:
            raise SystemExit(f"generated files differ: {bad}")
        print("generated files match")
        return
    for path, text in out.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(f"wrote {len(out)} files")


if __name__ == "__main__":
    main()
