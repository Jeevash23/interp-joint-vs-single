"""Train, align and take the exhaustive census for every network of one split (resumable).

    python code/run.py development      # seeds 50-53 per family, used to fix the settings
    python code/run.py final            # frozen seeds; refuses to run unless the freeze manifest matches

Each network gets runs/<split>/<family>-<seed>-<arch>/ with task.json, weights.npz, candidates.npz,
census.npz and record.json. A network whose record.json exists is skipped.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.dont_write_bytecode = True
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "code"))

SPLITS = {"development": {"lookup": list(range(50, 54)), "additive": list(range(50, 54))}}


def frozen_split():
    manifest = json.loads((STUDY / "FROZEN-MANIFEST.json").read_text())
    for path, digest in manifest["sha256"].items():
        if hashlib.sha256((STUDY / path).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"frozen file changed: {path}")
    return manifest["final_seeds"]


def one(job):
    import numpy as np
    import torch
    from common import ARCHS, RETRIES, census, encode, make_candidates, make_task, natural_scale, train

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
    record = {"family": family, "seed": seed, "arch": arch, "kind": ARCHS[arch][0], "width": ARCHS[arch][1],
              "training_attempts": attempts, "competent": attempts[-1]["passed"]}
    if not record["competent"]:
        record["seconds"] = time.perf_counter() - started
        (folder / "record.json").write_text(json.dumps(record, indent=1) + "\n")
        return f"{folder.name}: failed the competence gate"
    np.savez_compressed(folder / "weights.npz", **{k: v.detach().numpy() for k, v in model.state_dict().items()})
    with torch.no_grad():
        z = model.site(encode(ARCHS[arch][0])).numpy()
    t0 = time.perf_counter()
    cands, info = make_candidates(model, z, task, seed=seed * 1000 + list(ARCHS).index(arch))
    record["alignment"] = info
    record["alignment_seconds"] = time.perf_counter() - t0
    scale = natural_scale(z)
    names = list(cands)
    results = [census(model, z, task, *cands[n], scale) for n in names]
    np.savez_compressed(folder / "candidates.npz", **{f"{n}_{v}": cands[n][i] for n in names for i, v in enumerate("ab")})
    np.savez_compressed(folder / "census.npz", names=np.array(names), natural_states=z, scale=scale,
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
    from common import ARCHS
    jobs = [(split, family, seed, arch) for family, ss in seeds.items() for seed in ss for arch in ARCHS]
    with Pool(workers) as pool:
        for message in pool.imap_unordered(one, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
