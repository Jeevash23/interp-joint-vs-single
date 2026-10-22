"""Train, align and take the exhaustive census for every network of one Study 3 split (resumable).

    python code/run.py development [workers]   # seeds 70-73 per family
    python code/run.py final [workers]         # frozen seeds; refuses to run unless the freeze manifest matches

Networks are trained by Study 2's frozen code (same settings, competence gate and retry rule). Each network gets
runs/<split>/<family>-<seed>-<arch>/ with task.json, weights.npz, candidates.npz, census.npz and record.json.
A network whose record.json exists is skipped.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.dont_write_bytecode = True
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "code"))

SPLITS = {"development": {"lookup": list(range(70, 74)), "additive": list(range(70, 74))}}


def frozen_split():
    manifest = json.loads((STUDY / "FROZEN-MANIFEST.json").read_text())
    for path, digest in manifest["sha256"].items():
        if hashlib.sha256((STUDY / path).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"frozen file changed: {path}")
    return manifest["final_seeds"]


def one(job):
    import numpy as np
    import torch
    from sites import ARCHS, CANDIDATES, a_to_b, census, make_candidates, natural_scale, natural_states
    from common import RETRIES, make_task, train

    split, family, seed, arch = job
    folder = STUDY / "runs" / split / f"{family}-{seed}-{arch}"
    if (folder / "record.json").exists():
        return f"{folder.name}: done earlier"
    started = time.perf_counter()
    task = make_task(family, seed)
    attempts = []
    for attempt in range(RETRIES):
        model, info = train(task, arch, init_seed=seed * 10 + attempt)
        attempts.append(info)
        if info["passed"]:
            break
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "task.json").write_text(json.dumps(task, indent=1) + "\n")
    kind = ARCHS[arch][0]
    record = {"family": family, "seed": seed, "arch": arch, "kind": kind, "width": ARCHS[arch][1],
              "training_attempts": attempts, "competent": attempts[-1]["passed"]}
    if not record["competent"]:
        record["seconds"] = time.perf_counter() - started
        (folder / "record.json").write_text(json.dumps(record, indent=1) + "\n")
        return f"{folder.name}: failed the competence gate"
    np.savez_compressed(folder / "weights.npz", **{k: v.detach().numpy() for k, v in model.state_dict().items()})
    za, zb = natural_states(model, kind)
    with torch.no_grad():
        check = a_to_b(model, kind, torch.tensor(za), np.arange(27)).numpy()
    record["propagation_check_max_abs"] = float(np.abs(check - zb).max())
    assert record["propagation_check_max_abs"] < 1e-10, record["propagation_check_max_abs"]
    t0 = time.perf_counter()
    cands, info = make_candidates(model, kind, za, zb, task, seed=seed * 1000 + list(ARCHS).index(arch))
    record["alignment"] = info
    record["alignment_seconds"] = time.perf_counter() - t0
    scale_a, scale_b = natural_scale(za), natural_scale(zb)
    results = [census(model, kind, za, zb, task, *cands[n], scale_a, scale_b) for n in CANDIDATES]
    np.savez_compressed(folder / "candidates.npz", **{f"{n}_{v}": cands[n][i] for n in CANDIDATES for i, v in enumerate("ab")})
    np.savez_compressed(folder / "census.npz", names=np.array(CANDIDATES), states_a=za, states_b=zb,
                        scale_a=scale_a, scale_b=scale_b,
                        single=np.stack([r["single"] for r in results]),
                        joint=np.stack([r["joint"] for r in results]),
                        off_natural=np.stack([r["off_natural"] for r in results]))
    record["seconds"] = time.perf_counter() - started
    (folder / "record.json").write_text(json.dumps(record, indent=1) + "\n")
    return f"{folder.name}: {record['seconds']:.1f} s"


def main():
    split = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    seeds = frozen_split() if split == "final" else SPLITS[split]
    archs = ["T16", "T64", "M16", "M64"]
    jobs = [(split, family, seed, arch) for family, ss in seeds.items() for seed in ss for arch in archs]
    with Pool(workers) as pool:
        for message in pool.imap_unordered(one, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
