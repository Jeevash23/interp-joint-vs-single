#!/usr/bin/env python3
"""Frozen finite-domain AI-05 study; Python standard library only."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("parity", "conjunction")
CLASSES = ("direct", "rotated", "hidden", "conditional_hidden")
VALID = {"direct": True, "rotated": True, "hidden": False, "conditional_hidden": False}


def oracle(family: str, a: int, b: int, n: int) -> int:
    return ((a ^ b) if family == "parity" else (a & b)) ^ n


@dataclass(frozen=True)
class State:
    x: int
    y: int
    n: int
    q: int | None


def encode(cls: str, a: int, b: int, n: int) -> State:
    if cls == "rotated":
        return State(a + b, a - b, n, None)
    if cls == "hidden":
        return State(a, b, n, a & b)
    if cls == "conditional_hidden":
        return State(a, b, n, a & b & n)
    return State(a, b, n, None)


def decode(cls: str, s: State) -> tuple[int, int]:
    if cls == "rotated":
        assert (s.x + s.y) % 2 == 0 and (s.x - s.y) % 2 == 0
        return (s.x + s.y) // 2, (s.x - s.y) // 2
    return s.x, s.y


def native(family: str, cls: str, a: int, b: int, n: int) -> int:
    return instrumented(family, cls, encode(cls, a, b, n))


def instrumented(family: str, cls: str, s: State) -> int:
    a, b = decode(cls, s)
    out = oracle(family, a, b, s.n)
    if cls == "hidden":
        out ^= a & b & (1 - int(s.q))
    elif cls == "conditional_hidden":
        out ^= a & b & s.n & (1 - int(s.q))
    return out


def edit(cls: str, base: State, variable: str, value: int) -> State:
    a, b = decode(cls, base)
    if variable == "a":
        a = value
    else:
        b = value
    if cls == "rotated":
        return State(a + b, a - b, base.n, base.q)
    return State(a, b, base.n, base.q)


def patch(family: str, cls: str, base: tuple[int, int, int],
          edits: tuple[tuple[str, tuple[int, int, int]], ...]) -> tuple[int, int, State]:
    s = encode(cls, *base)
    a, b, n = base
    for variable, donor in edits:
        d = encode(cls, *donor)
        da, db = decode(cls, d)
        value = da if variable == "a" else db
        s = edit(cls, s, variable, value)
        if variable == "a":
            a = value
        else:
            b = value
    return instrumented(family, cls, s), oracle(family, a, b, n), s


def query_specs():
    # Each panel has exactly four target evaluations. Donor nuisance is never copied.
    ordinary = []
    extra = []
    for v in ("a", "b"):
        for dn in (0, 1):
            donor = (1, 0, dn) if v == "a" else (0, 1, dn)
            ordinary.append(((0, 0, 0), ((v, donor),)))
            donor2 = (1, 1, dn)
            extra.append(((0, 0, 0), ((v, donor2),)))
    heldout = []
    for n in (0, 1):
        heldout.extend([
            ((0, 1, n), (("a", (1, 0, 1 - n)),)),
            ((1, 0, n), (("b", (0, 1, 1 - n)),)),
        ])
    joint = []
    for n in (0, 1):
        for dn in (0, 1):
            joint.append(((0, 0, n), (("a", (1, 0, dn)), ("b", (0, 1, 1 - dn)))))
    return {"ordinary_single": ordinary, "extra_single": extra,
            "heldout_single": heldout, "joint": joint}


def reproduce_grant() -> dict:
    # Grant et al., arXiv:2511.04638v5, section 4.2.1, printed matrix and states.
    W = ((.75, .25, 0, .5), (0, 1, 0, 0), (1, 1, -1, -1))
    bias = (-.5, -.5, 0)
    A = ((1, 0, 1, 0), (0, 1, 1, 0))
    B = ((0, 0, 1, 0), (0, 0, 1, 1))
    delta = tuple(sum(A[i][j] - B[i][j] for i in range(2)) / 2 for j in range(4))
    def forward(h):
        z = tuple(max(0., sum(wj * hj for wj, hj in zip(row, h)) + bj)
                  for row, bj in zip(W, bias))
        return {"units": z, "score": sum(z)}
    patched = tuple(tuple(B[i][j] + delta[j] for j in range(4)) for i in range(2))
    natural = [forward(x) for x in (*A, *B)]
    changed = [forward(x) for x in patched]
    assert delta == (.5, .5, 0., -.5)
    assert [x["score"] for x in natural] == [.25, .5, 0., 0.]
    assert [x["score"] for x in changed] == [.5, .25]
    assert all(x["units"][2] == 0 for x in natural)
    assert changed[0]["units"][2] == .5
    return {"citation": "Grant et al. 2511.04638v5 section 4.2.1",
            "status": "equation-level methodological reconstruction; no official code",
            "delta": delta, "natural": natural, "patched_states": patched,
            "patched": changed, "dormant_unit_activated": True}


def verify() -> dict:
    checks = 0
    for family in FAMILIES:
        for cls in CLASSES:
            for a in (0, 1):
                for b in (0, 1):
                    for n in (0, 1):
                        s = encode(cls, a, b, n)
                        assert decode(cls, s) == (a, b)
                        assert native(family, cls, a, b, n) == oracle(family, a, b, n)
                        assert instrumented(family, cls, s) == native(family, cls, a, b, n)
                        for v in ("a", "b"):
                            assert edit(cls, s, v, a if v == "a" else b) == s
                            d = edit(cls, s, v, 1 - (a if v == "a" else b))
                            assert d.n == s.n and d.q == s.q
                            assert all(x in (0, 1) for x in decode(cls, d))
                        checks += 1
            # Independent exhaustive reference label check, all 8 bases x 4 target combinations.
            mismatches = 0
            for a in (0, 1):
                for b in (0, 1):
                    for n in (0, 1):
                        for ta in (0, 1):
                            for tb in (0, 1):
                                s = edit(cls, edit(cls, encode(cls, a, b, n), "a", ta), "b", tb)
                                mismatches += instrumented(family, cls, s) != oracle(family, ta, tb, n)
                                # Explicitly check edit composition order on states.
                                assert s == edit(cls, edit(cls, encode(cls, a, b, n), "b", tb), "a", ta)
            assert (mismatches == 0) == VALID[cls]
            checks += 32
    return {"passed": True, "checks": checks,
            "independent_label_rule": "exhaustive 8 natural states and 32 joint targets per pair"}


def run() -> dict:
    start = time.perf_counter()
    verification = verify()
    reproduction = reproduce_grant()
    records = []
    specs = query_specs()
    for family in FAMILIES:
        for cls in CLASSES:
            for panel, items in specs.items():
                for ix, (base, edits) in enumerate(items):
                    observed, predicted, state = patch(family, cls, base, edits)
                    records.append({"family": family, "explanation": cls, "reference_valid": VALID[cls],
                                    "panel": panel, "query": ix, "base": base, "edits": edits,
                                    "state": state.__dict__, "observed": observed,
                                    "predicted": predicted, "match": observed == predicted,
                                    "target_queries": 1, "donor_reads": len(edits), "recipient_reads": 1})
            for ix, (a, b, n) in enumerate(((1, 1, 0), (0, 0, 0), (1, 1, 1), (0, 0, 1))):
                observed = native(family, cls, a, b, n)
                records.append({"family": family, "explanation": cls, "reference_valid": VALID[cls],
                                "panel": "natural", "query": ix, "input": (a,b,n),
                                "observed": observed, "predicted": oracle(family,a,b,n),
                                "match": observed == oracle(family,a,b,n),
                                "target_queries": 1, "donor_reads": 0, "recipient_reads": 0})
            for ix, n in enumerate((0, 0, 1, 1)):
                s = encode(cls, 1, 1, n)
                changed = State(s.x, s.y, s.n, 0) if s.q is not None else s
                observed = instrumented(family, cls, changed)
                predicted = native(family, cls, 1, 1, n)  # unmapped q claimed inert
                records.append({"family": family, "explanation": cls, "reference_valid": VALID[cls],
                                "panel": "faithfulness", "query": ix, "input": (1,1,n),
                                "state": changed.__dict__, "observed": observed,
                                "predicted": predicted, "match": observed == predicted,
                                "target_queries": 1, "donor_reads": 0, "recipient_reads": 1})
    panels = (*specs.keys(), "natural", "faithfulness")
    rows = []
    for family in FAMILIES:
        for cls in CLASSES:
            row = {"family": family, "explanation": cls, "valid": VALID[cls]}
            for panel in panels:
                rr = [r for r in records if r["family"] == family and r["explanation"] == cls and r["panel"] == panel]
                assert len(rr) == 4
                row[panel] = {"matches": sum(r["match"] for r in rr), "accept": all(r["match"] for r in rr),
                              "target_queries": 4, "donor_reads": sum(r["donor_reads"] for r in rr),
                              "recipient_reads": sum(r["recipient_reads"] for r in rr)}
            rows.append(row)
    summary = {}
    for panel in panels:
        summary[panel] = {"false_acceptance": sum(r[panel]["accept"] for r in rows if not r["valid"]),
                          "invalid_total": sum(not r["valid"] for r in rows),
                          "valid_acceptance": sum(r[panel]["accept"] for r in rows if r["valid"]),
                          "valid_total": sum(r["valid"] for r in rows),
                          "target_queries_per_candidate": 4}
    ordinary_pass = [r for r in rows if r["ordinary_single"]["accept"]]
    summary["ordinary_pass_subset"] = {"n": len(ordinary_pass), "invalid": sum(not r["valid"] for r in ordinary_pass)}
    return {"metadata": {"python": sys.version, "platform": platform.platform(),
                         "protocol_sha256": hashlib.sha256((ROOT/'06-Study-History'/'PROTOCOL-FROZEN-2026-09-29.md').read_bytes()).hexdigest(),
                         "elapsed_seconds": time.perf_counter()-start},
            "verification": verification, "reproduction": reproduction,
            "records": records, "rows": rows, "summary": summary}


if __name__ == "__main__":
    out = run()
    dest = ROOT / '04-Results' / 'final.json'
    dest.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out['summary'], indent=2))
