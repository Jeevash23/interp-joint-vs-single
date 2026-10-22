"""AI-10 analysis from the saved censuses (no model): PROTOCOL.md Sections 8 and 9.

    python -B analyse.py development     means and counts, no decisions (development tasks)
    python -B analyse.py confirmation    intervals, decisions and the scored predictions (refused before the freeze)

Writes study/results/analysis/<phase>/summary.json and alignments.csv.

Status (17 October 2026): the primary quantities, the AI-05 detection comparisons (Section 8.3) and the bootstrap are
implemented and unit-tested on synthetic censuses. The margin-adaptive and margin-guided tests of Section 8.5 are to be
implemented in the development phase, before the freeze, following AI-05 analysis/test_strategies.py.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import beta, spearmanr

import ai10_common as common
from interchange import detect
from margin_rule import auroc

C = common.CONFIG
DAS = {"a": ("das_r1", "das_r2", "das_r4"), "a_end": ("das_r1", "das_r2", "das_r4"),
       "b": ("das_r1", "das_r2", "das_r4"), "c": ("sep_r1", "sep_r2", "sep_r4")}


def load_rows(phase: str) -> list[dict]:
    rows = []
    for rec_path in sorted((common.RESULTS / "runs" / phase).glob("*/record.json")):
        rec = json.loads(rec_path.read_text())
        for arm, body in rec["arms"].items():
            for a in body["alignments"]:
                rows.append({"task": rec["task"]["draw"], "arm": arm, **{k: v for k, v in a.items() if k != "fit"}})
    return rows


def clopper_pearson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return [float(lo), float(hi)]


def joint_only(rows: list[dict]) -> dict:
    """Primary 1: among alignments with no failure and no tie in every single query, how many fail a joint query."""
    perfect = [r for r in rows if r["single_fail"] == 0 and r["single_tie"] == 0]
    jo = [r for r in perfect if r["joint_fail_all"] > 0]
    return {"alignments": len(rows), "pass_all_singles": len(perfect), "joint_only": len(jo),
            "share": len(jo) / len(perfect) if perfect else None, "ci95": clopper_pearson(len(jo), len(perfect)),
            "joint_only_alignments": [f"{r['task']}/{r['alignment']}" for r in jo]}


def margin_rule(rows: list[dict]) -> dict:
    """Primary 2: rank the alignments that pass every single query by their largest sigma; the rule sigma > 1."""
    perfect = [r for r in rows if r["single_fail"] == 0 and r["single_tie"] == 0]
    if not perfect:
        return {"testable": False}
    label = np.array([r["joint_fail_all"] > 0 for r in perfect])
    score = np.array([r["max_sigma"] for r in perfect])
    order = np.argsort(-score, kind="stable")
    flagged = np.array([r["flagged_queries"] > 0 for r in perfect])
    return {"testable": bool(label.any()), "pass_all_singles": len(perfect), "joint_only": int(label.sum()),
            "ranks_of_joint_only": [int(np.flatnonzero(order == i)[0]) + 1 for i in np.flatnonzero(label)],
            "auroc": auroc(score, label), "flagged": int(flagged.sum()),
            "flagged_joint_only": int((flagged & label).sum()), "flagged_other": int((flagged & ~label).sum())}


def per_alignment_detection(r: dict, budget: int) -> dict:
    ns, nj = budget // C["costs"]["single"], budget // C["costs"]["joint"]
    pa, pb, pj = r["p_full_single_A"], r["p_full_single_B"], r["p_joint"]
    base_miss = (1 - pa) ** ((ns + 1) // 2) * (1 - pb) ** (ns // 2)
    return {"single": detect(pa, pb, ns, True), "joint": detect(pj, None, nj, False),
            "equal_queries_joint": detect(pj, None, budget // 4, False),       # 30 queries of each at 120 calls
            "equal_queries_single": detect(pa, pb, budget // 4, True),
            "cached_single": detect(pa, pb, budget, True), "cached_joint": detect(pj, None, budget, False),
            "oracle_single": max(1 - (1 - pa) ** ns, 1 - (1 - pb) ** ns),
            "addon_joint": base_miss * detect(pj, None, nj, False),
            "addon_single": base_miss * detect(pa, pb, ns, True),
            # Section 17: an alternating single test drawing from every single query of each variable.
            **({"all_single": detect(r["p_all_single_A"], r["p_all_single_B"], ns, True)}
               if "p_all_single_A" in r else {})}


def bootstrap(per_task: dict[int, np.ndarray], seed: int, reps: int) -> list[float]:
    """95% percentile interval of the mean over alignments, resampling tasks with replacement."""
    keys = sorted(per_task)
    rng = np.random.default_rng(seed)
    draws = [np.mean(np.concatenate([per_task[keys[i]] for i in rng.integers(len(keys), size=len(keys))]))
             for _ in range(reps)]
    return np.quantile(draws, [0.025, 0.975]).tolist()


def detection(rows: list[dict], reps: int) -> dict:
    """Primary 3 and Section 8.3: detection of the DAS alignments at each budget, joint minus alternating single."""
    out = {}
    for b in C["budgets"]:
        per_task, per_task_all = defaultdict(list), defaultdict(list)
        cells = defaultdict(list)
        for r in rows:
            d = per_alignment_detection(r, b)
            per_task[r["task"]].append(d["joint"] - d["single"])
            if "all_single" in d:
                per_task_all[r["task"]].append(d["joint"] - d["all_single"])
            for k, v in d.items():
                cells[k].append(v)
        per_task = {k: np.array(v) for k, v in per_task.items()}
        out[str(b)] = {"contrast": float(np.mean(np.concatenate(list(per_task.values())))),
                       "ci95": bootstrap(per_task, C["seeds"]["bootstrap"], reps),
                       **{k: float(np.mean(v)) for k, v in cells.items()}}
        if per_task_all and len(per_task_all) == len(per_task):          # Section 17 (development-informed)
            per_task_all = {k: np.array(v) for k, v in per_task_all.items()}
            out[str(b)]["contrast_all_single"] = float(np.mean(np.concatenate(list(per_task_all.values()))))
            out[str(b)]["ci95_all_single"] = bootstrap(per_task_all, C["seeds"]["bootstrap"] + 4, reps)
    return out


def paired_contrast(rows_x: list[dict], rows_y: list[dict], reps: int, budget: int = 120) -> dict:
    """Per-task mean contrast of arm x minus arm y (same tasks), with a task bootstrap."""
    def per_task(rows):
        acc = defaultdict(list)
        for r in rows:
            d = per_alignment_detection(r, budget)
            acc[r["task"]].append(d["joint"] - d["single"])
        return {k: np.mean(v) for k, v in acc.items()}
    x, y = per_task(rows_x), per_task(rows_y)
    keys = sorted(set(x) & set(y))
    diff = {k: np.array([x[k] - y[k]]) for k in keys}
    return {"tasks": len(keys), "difference": float(np.mean([x[k] - y[k] for k in keys])) if keys else None,
            "ci95": bootstrap(diff, C["seeds"]["bootstrap"] + 1, reps) if keys else None}


def overlap_relation(rows: list[dict], reps: int) -> dict:
    """Arm c: does the joint-panel failure rate rise with ||P_A P_B||? Spearman over the separately fitted DAS
    alignments, with a task bootstrap of the coefficient."""
    ov = np.array([r["overlap"] for r in rows])
    pj = np.array([r["p_joint"] for r in rows])
    tasks = np.array([r["task"] for r in rows])
    keys = sorted(set(tasks.tolist()))
    rng = np.random.default_rng(C["seeds"]["bootstrap"] + 2)
    draws = []
    for _ in range(reps):
        pick = np.concatenate([np.flatnonzero(tasks == keys[i]) for i in rng.integers(len(keys), size=len(keys))])
        draws.append(spearmanr(ov[pick], pj[pick]).statistic)
    return {"alignments": len(rows), "overlap_median": float(np.median(ov)), "overlap_range": [float(ov.min()),
            float(ov.max())], "spearman": float(spearmanr(ov, pj).statistic),
            "ci95": np.nanquantile(draws, [0.025, 0.975]).tolist()}


def summarise(rows: list[dict], reps: int) -> dict:
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    out = {}
    for arm, rs in by_arm.items():
        das_rows = [r for r in rs if r["alignment"] in DAS[arm]]
        out[arm] = {"joint_only": joint_only(rs), "margin_rule": margin_rule(rs), "detection": detection(das_rows, reps),
                    "mean_rates_das": {k: float(np.mean([r[k] for r in das_rows])) for k in
                                       ("p_full_single_A", "p_full_single_B", "p_joint", "p_all_single_A",
                                        "p_all_single_B", "p_unchanged_single_A", "p_unchanged_single_B")
                                       if all(k in r for r in das_rows)}}
        split_key = "irreducible" if arm != "b" else "unexplained"
        jf = sum(r["joint_panel_failures"] for r in das_rows)
        out[arm][f"{split_key}_share"] = sum(r[split_key] for r in das_rows) / jf if jf else None
    if "a" in by_arm and "b" in by_arm:
        out["b_minus_a"] = paired_contrast([r for r in by_arm["b"] if r["alignment"] in DAS["b"]],
                                           [r for r in by_arm["a"] if r["alignment"] in DAS["a"]], reps)
    if "c" in by_arm:
        out["c"]["overlap"] = overlap_relation([r for r in by_arm["c"] if r["alignment"] in DAS["c"]], reps)
    return out


def overlap_ai05(split: str, reps: int) -> dict:
    """Arm d: separately fitted alignments on AI-05 Study 2's networks against Study 2's own orthogonal ones."""
    rows = list(csv.DictReader(open(common.RESULTS / "overlap-ai05" / split / "alignments.csv")))
    for r in rows:
        for k in ("overlap", "p_joint", "max_sigma"):
            r[k] = float(r[k])
        for k in ("single_fail", "single_tie", "joint_fail_all", "flagged_queries"):
            r[k] = int(r[k])
    ref = list(csv.DictReader(open(common.AI05 / "studies/2-trained-networks/results" / split / "candidates.csv")))
    ref_perfect = [r for r in ref if int(r["single_fail"]) == 0 and int(r["single_tie"]) == 0]
    ref_jo = sum(int(r["joint_fail_all"]) > 0 for r in ref_perfect)
    perfect = [r for r in rows if r["single_fail"] == 0 and r["single_tie"] == 0]
    label = np.array([r["joint_fail_all"] > 0 for r in perfect])
    ov = np.array([r["overlap"] for r in perfect])
    nets = sorted({r["network"] for r in perfect})
    rng = np.random.default_rng(C["seeds"]["bootstrap"] + 3)
    draws = []
    for _ in range(reps):
        pick = [nets[i] for i in rng.integers(len(nets), size=len(nets))]
        idx = np.concatenate([[j for j, r in enumerate(perfect) if r["network"] == n] for n in pick]).astype(int)
        draws.append(auroc(ov[idx], label[idx]) if label[idx].any() and (~label[idx]).any() else np.nan)
    das_rows = [r for r in rows if r["candidate"].startswith("sep_")]
    return {"separate": {**joint_only([{**r, "task": r["network"], "alignment": r["candidate"]} for r in rows])},
            "study2_orthogonal": {"pass_all_singles": len(ref_perfect), "joint_only": ref_jo,
                                  "share": ref_jo / len(ref_perfect) if ref_perfect else None},
            "overlap_auroc_for_joint_only": auroc(ov, label) if len(perfect) else None,
            "overlap_auroc_ci95": np.nanquantile(draws, [0.025, 0.975]).tolist() if draws else None,
            "spearman_overlap_vs_joint_rate_das": float(spearmanr([r["overlap"] for r in das_rows],
                                                                  [r["p_joint"] for r in das_rows]).statistic),
            "margin_rule": margin_rule([{**r, "task": r["network"], "alignment": r["candidate"]} for r in rows])}


def main() -> None:
    phase = sys.argv[1]
    if phase in ("confirmation", "overlap-final"):
        common.require_freeze()
    if phase.startswith("overlap-"):
        split = "final" if phase == "overlap-final" else "development"
        out = overlap_ai05(split, C["bootstrap_replicates"])
        common.save_json(common.RESULTS / "analysis" / phase / "summary.json", out, overwrite=split == "development")
        print(json.dumps(out, indent=1))
        return
    rows = load_rows(phase)
    out = summarise(rows, C["bootstrap_replicates"])
    dest = common.RESULTS / "analysis" / phase
    common.save_json(dest / "summary.json", out, overwrite=phase == "development")
    with open(dest / "alignments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
