"""CPU code test of the bigger-model arm's jobs on tiny random GPT-2 models (no pretrained weights; outputs go to a
scratch folder given as the first argument). Settings are shrunk so that it runs in a few minutes:

    CUDA_VISIBLE_DEVICES= python -B study/cloud/test_cloud.py <scratch folder>

It builds three tiny random models in the Hugging Face layout under the names gpt2, gpt2-medium and gpt2-large, runs
GPT-2 Small's arm a on development draw 0 to make the "saved" reproduction inputs, then runs every job of
cloud_jobs.py: check (reproduction within 1e-6 here, parity must pass), development, gate, tasks and analyse.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE))
torch.set_num_threads(2)


def main() -> None:
    scratch = Path(sys.argv[1])
    if scratch.exists():
        shutil.rmtree(scratch)
    models = scratch / "models"
    from transformers import GPT2Config, GPT2LMHeadModel
    import ai10_common as common
    tokenizer = common.model_dir() / "tokenizer.json"
    for k, name in enumerate(("gpt2", "gpt2-medium", "gpt2-large")):
        torch.manual_seed(k)
        cfg = GPT2Config(vocab_size=50257, n_positions=16, n_embd=32, n_layer=3 + k, n_head=4, resid_pdrop=0.0,
                         embd_pdrop=0.0, attn_pdrop=0.0)
        GPT2LMHeadModel(cfg).eval().save_pretrained(models / name)
        shutil.copy(tokenizer, models / name / "tokenizer.json")
    cfg = json.loads((HERE / "cloud-config.json").read_text())
    for name in cfg["models"]:
        cfg["models"][name]["safetensors_sha256"] = ""
    cfg["candidate_layers"] = {"gpt2-medium": [1, 2], "gpt2-large": [2, 3]}
    cfg["timing_steps"] = 2
    paths = {"code_dir": CODE, "models_dir": models, "volume_dir": scratch / "vol", "repro_dir": scratch / "repro"}
    import cloud_jobs as cj
    common, rs = cj.setup(paths, "gpt2")
    C = common.CONFIG
    C["das"].update(development_steps=4, step_candidates=[2, 4], learning_rate_candidates=[0.003])
    C.update(candidate_layers=[1, 2], development_tasks=2, competence_margin=-1e9, bootstrap_replicates=50)
    # "Saved" GPT-2 Small inputs for the reproduction, made with the local pipeline on the tiny model.
    import prompts
    dev = cj.device()
    small = cj.load(paths, cfg, "gpt2", dev)
    tok = prompts.load_tokenizer()
    draws = json.loads((common.PROMPTS / "development-draws.json").read_text())
    settings = {"eps": 1e-3, "lr": 0.003, "layer": 1, "steps": 4}
    task = rs.Task(small, tok, draws[0], dev)
    rs.run_task(task, "development", settings, arms_to_run=("a",))
    src = common.RESULTS / "runs/development" / task.name
    paths["repro_dir"].mkdir(parents=True)
    for f in ("record.json", "arm-a.npz"):
        shutil.copy(src / f, paths["repro_dir"] / f)
    (paths["repro_dir"] / "development-settings.json").write_text(json.dumps(settings))
    out = cj.check(paths, cfg)
    print(json.dumps({k: out[k] for k in ("reproduction", "tf32_check", "passed")}, indent=1))
    assert out["reproduction"]["max_abs_margin_difference"] <= 1e-6 and out["passed"], out
    assert set(out["models"]) == {"gpt2-medium", "gpt2-large"} and all(len(m["timings"]) == 2 for m in out["models"].values())
    d = cj.development(paths, cfg, "gpt2-medium")
    assert d["settings"]["layer"] in (1, 2) and len(d["gate"]["passed"]) == 2, d
    recs = sorted((paths["volume_dir"] / "gpt2-medium/runs/development").glob("*/record.json"))
    assert len(recs) == 2 and all(set(json.loads(r.read_text())["arms"]) == {"a"} for r in recs)
    a = json.loads(recs[0].read_text())["arms"]["a"]["alignments"]
    assert [x["alignment"] for x in a] == ["das_r1", "das_r2", "das_r4", "mean", "random_r2"], [x["alignment"] for x in a]
    g = cj.gate(paths, cfg, "gpt2-medium", "confirmation", 3)
    assert g["passed"] == [0, 1, 2]
    t = cj.tasks(paths, cfg, "gpt2-medium", "confirmation", g["passed"][:2])
    assert len(t["tasks"]) == 2
    s = cj.analyse(paths, cfg, "gpt2-medium", "development")
    summary = json.loads((paths["volume_dir"] / "gpt2-medium/analysis/development/summary.json").read_text())
    assert set(summary) == {"a"} and "joint_only" in summary["a"], summary.keys()
    print("cloud jobs test passed:", s, {m: o["parity"]["passed"] for m, o in out["models"].items()}, flush=True)


if __name__ == "__main__":
    main()
