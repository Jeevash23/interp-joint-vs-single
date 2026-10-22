"""A small explicit network in which joint interchange tests catch what every single test misses.

Inputs (a, b, n) in {0,1,2}^3; the target is y = a + b. The site holds z = (u_a, u_b, e_A, e_B) with
natural value (a, b, 0, 0). The readout is a two-layer ReLU network:

    y_hat = u_a + u_b + t (e_A + e_B) - k * relu(|e_A| + |e_B| - tau),   |e| = relu(e) + relu(-e),

with t = tan(theta), tau = 2.5 sin(theta) cos(theta) and k = 1 / (sin(theta) cos(theta)). On natural states
e_A = e_B = 0, so y_hat = a + b. A candidate alignment claims A lives on the unit vector
v_A = (cos theta, 0, sin theta, 0) and B on v_B = (0, cos theta, 0, sin theta); the two are orthogonal.

A single A-edit moves u_a by cos^2(theta) * (a_d - a_r) and e_A by sin(theta)cos(theta) * (a_d - a_r); the readout
recombines them into exactly a_d, and |e_A| <= 2 sin(theta) cos(theta) < tau keeps the dormant unit off with a
margin. So every single edit is output-correct and misses the natural state it simulates. A joint edit with
|a_d1 - a_r| + |b_d2 - b_r| >= 3 switches the dormant unit on and is wrong by at least 0.5. The counts do not
depend on theta (0 < theta < 90 degrees). The second variant tilts only v_A (v_B = (0, 1, 0, 0)), so B-edits land on natural
states and, by the composition identity, no joint failure can occur.

Output: analysis/results/constructed-example.json
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

X = np.array(list(itertools.product(range(3), repeat=3)))
TOL = 1e-4
THETA = np.deg2rad(36.0)


def readout(z, theta):
    s, c = np.sin(theta), np.cos(theta)
    relu = lambda v: np.maximum(v, 0.0)  # noqa: E731
    mag = relu(z[:, 2]) + relu(-z[:, 2]) + relu(z[:, 3]) + relu(-z[:, 3])
    return z[:, 0] + z[:, 1] + np.tan(theta) * (z[:, 2] + z[:, 3]) - relu(mag - 2.5 * s * c) / (s * c)


def census(v_a, v_b, theta):
    Z = np.column_stack([X[:, 0], X[:, 1], np.zeros(27), np.zeros(27)]).astype(float)
    pa, pb = np.outer(v_a, v_a), np.outer(v_b, v_b)
    assert abs(v_a @ v_b) < 1e-12
    assert np.allclose(readout(Z, theta), X[:, 0] + X[:, 1])
    r, d = np.array(list(itertools.product(range(27), repeat=2))).T
    single_fail, off = 0, 0.0
    for v, p in ((0, pa), (1, pb)):
        tgt = X[r].copy()
        tgt[:, v] = X[d, v]
        edit = Z[r] + (Z[d] - Z[r]) @ p
        single_fail += int((np.abs(readout(edit, theta) - tgt[:, 0] - tgt[:, 1]) > TOL).sum())
        off = max(off, float(np.abs(edit[:, 2:]).max()))
    r, d1, d2 = np.array(list(itertools.product(range(27), repeat=3))).T
    joint = Z[r] + (Z[d1] - Z[r]) @ pa + (Z[d2] - Z[r]) @ pb
    err = np.abs(readout(joint, theta) - X[d1, 0] - X[d2, 1])
    both_changed = (X[d1, 0] != X[r, 0]) & (X[d2, 1] != X[r, 1])
    return {"single_queries": 2 * 729, "single_failures": single_fail,
            "joint_queries": len(r), "joint_failures": int((err > TOL).sum()),
            "joint_panel_queries": int(both_changed.sum()), "joint_panel_failures": int(((err > TOL) & both_changed).sum()),
            "max_joint_error": float(err.max()), "max_single_edit_off_natural": off}


def main():
    s, c = np.sin(THETA), np.cos(THETA)
    out = {"theta_degrees": 36.0, "tau": 2.5 * s * c, "k": 1 / (s * c),
           "both_tilted": census(np.array([c, 0, s, 0]), np.array([0, c, 0, s]), THETA),
           "only_a_tilted": census(np.array([c, 0, s, 0]), np.array([0, 1.0, 0, 0]), THETA)}
    dest = Path(__file__).resolve().parent / "results"
    dest.mkdir(exist_ok=True)
    (dest / "constructed-example.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
