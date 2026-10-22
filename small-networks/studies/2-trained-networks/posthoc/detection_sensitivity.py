"""How Study 2's detection comparison depends on the cost convention, and what a joint panel adds (post hoc).

The frozen contrast charges 3 model calls per single query and 4 per joint query (two or three natural states
read plus one edited suffix). Here, for the full-domain DAS alignments:
  * equal queries: 30 single and 30 joint queries (the joint panel's count at 120 calls);
  * cached natural states: one call per query of either kind, 120 queries each;
  * add-on: after a 120-call full-domain single panel, the expected extra detection from a further 120 calls
    spent on joint queries versus on more single queries;
  * the share of alignments that fail no query of either panel, which caps any detection curve.
Overall values and values per family x architecture cell. Output: posthoc/results/detection-sensitivity.json.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
DAS = ("das_r1", "das_r2", "das_r4")


def single(r, n):
    pa, pb = float(r["p_full_single_A"]), float(r["p_full_single_B"])
    return 1 - (1 - pa) ** ((n + 1) // 2) * (1 - pb) ** (n // 2)


def joint(r, n):
    return 1 - (1 - float(r["p_joint"])) ** n


def summary(rs):
    eq = [joint(r, 30) - single(r, 30) for r in rs]
    cached = [joint(r, 120) - single(r, 120) for r in rs]
    frozen = [joint(r, 30) - single(r, 40) for r in rs]
    add_joint = [(1 - single(r, 40)) * joint(r, 30) for r in rs]
    add_single = [single(r, 80) - single(r, 40) for r in rs]
    never = sum(float(r["p_full_single_A"]) == 0 and float(r["p_full_single_B"]) == 0 and float(r["p_joint"]) == 0 for r in rs)
    return {"alignments": len(rs), "frozen_convention": statistics.mean(frozen), "equal_queries_30": statistics.mean(eq),
            "cached_states_120": statistics.mean(cached), "addon_joint_after_single_120": statistics.mean(add_joint),
            "addon_single_after_single_120": statistics.mean(add_single), "fail_no_panel_query": never,
            "detection_ceiling": 1 - never / len(rs)}


def main():
    rows = [r for r in csv.DictReader(open(STUDY / "results/final/candidates.csv")) if r["candidate"] in DAS]
    cells = defaultdict(list)
    for r in rows:
        cells[f"{r['family']}/{r['arch']}"].append(r)
    out = {"overall": summary(rows), "by_cell": {k: summary(v) for k, v in sorted(cells.items())}}
    dest = STUDY / "posthoc/results"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "detection-sensitivity.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["overall"], indent=1))
    for k, v in out["by_cell"].items():
        print(f"{k:14s} frozen {v['frozen_convention']:+.4f}  equal-queries {v['equal_queries_30']:+.4f}  "
              f"cached {v['cached_states_120']:+.4f}  add-on joint {v['addon_joint_after_single_120']:.4f} vs single {v['addon_single_after_single_120']:.4f}")


if __name__ == "__main__":
    main()
