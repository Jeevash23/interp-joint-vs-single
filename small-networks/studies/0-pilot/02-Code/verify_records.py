#!/usr/bin/env python3
"""Independent aggregation and state-membership audit of saved final records."""
import json
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root / "04-Results/final.json").read_text())
groups = defaultdict(list)
for record in data["records"]:
    groups[(record["family"], record["explanation"], record["panel"])].append(record)
for row in data["rows"]:
    for panel in ("ordinary_single", "extra_single", "heldout_single", "joint", "natural", "faithfulness"):
        records = groups[row["family"], row["explanation"], panel]
        assert len(records) == 4
        assert row[panel]["matches"] == sum(x["observed"] == x["predicted"] for x in records)
        assert row[panel]["accept"] == all(x["observed"] == x["predicted"] for x in records)
for panel in ("ordinary_single", "extra_single", "heldout_single", "joint", "natural", "faithfulness"):
    invalid = [r for r in data["rows"] if not r["valid"]]
    valid = [r for r in data["rows"] if r["valid"]]
    assert data["summary"][panel]["false_acceptance"] == sum(r[panel]["accept"] for r in invalid)
    assert data["summary"][panel]["valid_acceptance"] == sum(r[panel]["accept"] for r in valid)

# A state is natural iff the q bit agrees with its constructive feature equation.
membership = defaultdict(lambda: {"natural": 0, "off_manifold": 0})
for r in data["records"]:
    if "state" not in r or r["explanation"] not in ("hidden", "conditional_hidden"):
        continue
    s = r["state"]
    q_expected = s["x"] & s["y"]
    if r["explanation"] == "conditional_hidden":
        q_expected &= s["n"]
    label = "natural" if s["q"] == q_expected else "off_manifold"
    membership[(r["panel"], r["explanation"])][label] += 1

result = {"independent_aggregation": "passed", "groups": len(groups),
          "records": len(data["records"]),
          "state_membership": {"/".join(k): v for k, v in sorted(membership.items())}}
# Recompute all saved joint states from base/donor tuples without importing study.py.
# This is separate from both its patch() and edit() implementations.
joint_checked = 0
for r in data["records"]:
    if r["panel"] != "joint":
        continue
    a, b, n = r["base"]
    for variable, donor in r["edits"]:
        if variable == "a":
            a = donor[0]
        else:
            b = donor[1]
    cls = r["explanation"]
    bx, by, bn = r["base"]
    q = None if cls in ("direct", "rotated") else (bx & by & (bn if cls == "conditional_hidden" else 1))
    expected_state = {"x": a+b if cls == "rotated" else a,
                      "y": a-b if cls == "rotated" else b, "n": n, "q": q}
    assert r["state"] == expected_state
    expected_output = ((a ^ b) if r["family"] == "parity" else (a & b)) ^ n
    if cls == "hidden": expected_output ^= a & b & (1-q)
    if cls == "conditional_hidden": expected_output ^= a & b & n & (1-q)
    assert r["observed"] == expected_output
    joint_checked += 1
assert joint_checked == 32
result["independent_joint_composition_checks"] = joint_checked
(root / "04-Results/independent-verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
