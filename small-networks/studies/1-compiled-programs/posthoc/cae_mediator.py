"""CAE controlled XOR experiment (Méloux et al., Experiment 2) with the mediator write honoured.

The released valid low-level model sets a = x and ignores an intervention on a. This script reruns
the same 20 official runs per condition as 02-Code/reproduce_official_v2.py, first unchanged (to
confirm the saved scores are reproduced) and then with a single change: an intervention on a
replaces a = x. Output: posthoc/results/cae-mediator.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402

STUDY = Path(__file__).resolve().parents[1]
VENDOR = STUDY / "02-Code/vendor/cae"
sys.path.insert(0, str(VENDOR / "test"))
from utils import collect_scores, load_system  # noqa: E402
from causal_abstraction import SystemState  # noqa: E402

suite = load_system(VENDOR / "test/11_controlled_experiments.py", "cae_controlled_suite")
ctrl = suite._get_ctrl()
METRICS = {"CAE_down_nf", "CAE_up_nf", "CAE_down", "CAE_up"}
released = ctrl.XORBackupLowLevelModel.forward_with_interventions


def honoured(self, input_state, interventions):
    """The released method with one change: an intervention on a replaces a = x."""
    x = np.atleast_2d(np.asarray(interventions.get("x", input_state.get("x", np.array([[0.0]]))), dtype=float))
    a = np.atleast_2d(np.asarray(interventions.get("a", x.copy()), dtype=float))
    b1 = np.atleast_2d(np.asarray(interventions.get("b1", x.copy()), dtype=float))
    b2 = np.atleast_2d(np.asarray(interventions.get("b2", x.copy()), dtype=float))
    a_b, b1_b, b2_b = [(v > 0.5).astype(float) for v in (a, b1, b2)]
    y = np.maximum(a_b, np.abs(b1_b - b2_b)) if self.xor_active else a_b
    return SystemState(values={"x": x, "a": a, "b1": b1, "b2": b2, "y": y})


def run(runs):
    rows = []
    for index in runs:
        for cond in ("valid", "xor_leak"):
            c = suite.build_exp2(cond, index)
            tasks = [t for t in c.tasks if t.name in METRICS]
            scores = collect_scores(c.engine.run_tasks(tasks, **c.task_kwargs), {})
            iia = [m for m in c.analytical if type(m).__name__ == "IIAMetric"]
            scores.update(collect_scores({}, c.engine.run_analytical_metrics(iia, sampler=c.sampler, n_samples=500)))
            rows.append({"run_index": index, "condition": cond, "scores": scores})
    return rows


def main():
    saved = json.loads((STUDY / "04-Results/v2/official-cae-reproduction.json").read_text())
    saved = {(r["run_index"], r["condition"]): r["scores"] for r in saved["rows"] if r["mode"] == "official"}
    check = run(range(2))
    reproduced = all(abs(r["scores"][m] - saved[(r["run_index"], r["condition"])][m]) < 1e-12
                     for r in check for m in r["scores"])
    probe = ctrl.XORBackupLowLevelModel(xor_active=False)
    ctrl.XORBackupLowLevelModel.forward_with_interventions = honoured
    state = probe.forward_with_interventions(SystemState(values={"x": np.array([[0.0]])}), {"a": np.array([[1.0]])})
    rows = run(range(20))
    ctrl.XORBackupLowLevelModel.forward_with_interventions = released
    out = {"released_scores_reproduced_runs_0_1": reproduced,
           "honoured_write_check": {"input_x": 0, "requested_a": 1, "returned_a": float(state.values["a"][0, 0]),
                                    "returned_y": float(state.values["y"][0, 0])},
           "change": "a = interventions.get('a', x) instead of a = x; everything else as released",
           "rows": rows}
    summary = {}
    for m in sorted(rows[0]["scores"]):
        v = [r["scores"][m] for r in rows if r["condition"] == "valid"]
        x = [r["scores"][m] for r in rows if r["condition"] == "xor_leak"]
        summary[m] = {"valid_mean": float(np.mean(v)), "valid_max": float(np.max(v)), "leak_mean": float(np.mean(x))}
    out["summary"] = summary
    dest = STUDY / "posthoc/results"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "cae-mediator.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
