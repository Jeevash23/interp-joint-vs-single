"""GPU-side work of AI-10's bigger-model arm (PROTOCOL.md Section 16, Amendment 2).

Runs inside Modal (ai10_modal.py) with the study's own code from study/code imported unchanged; the GPT-2 Small
pipeline is reused, with arm a only, each model's candidate layers, and TF32 for DAS training (run_study.TRAIN_TF32).
test_cloud.py runs the same functions on the CPU on a tiny random GPT-2.

    check(paths, cfg)                       stage 1: reproduction, parity, TF32 check, timings
    development(paths, cfg, model)          settings rules, then arm a on the model's 8 development tasks
    gate(paths, cfg, model, phase, wanted)  the first `wanted` draws of a phase that pass the model's competence gate
    tasks(paths, cfg, model, phase, draws)  arm a on given draws (a slice of the confirmation tasks)
    analyse(paths, cfg, model, phase)       the GPT-2 Small analysis (analyse.summarise) on the model's records

`paths` holds code_dir (study/code), models_dir (one folder per model, Hugging Face layout), volume_dir (records) and
repro_dir (the local machine's saved GPT-2 Small inputs for the reproduction).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def setup(paths: dict, model: str):
    """Import the study code with results going to the model's folder on the Volume."""
    if str(paths["code_dir"]) not in sys.path:
        sys.path.insert(0, str(paths["code_dir"]))
    import ai10_common as common
    import run_study
    common.RESULTS = Path(paths["volume_dir"]) / model
    common.CONFIG["model"]["snapshot"] = str(Path(paths["models_dir"]) / "gpt2")   # tokeniser (same for all sizes)
    return common, run_study


def device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        return torch.device("cuda")
    return torch.device("cpu")


def sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(paths, cfg, name, dev, dtype=torch.float32):
    from gpt2 import GPT2
    from safetensors.torch import load_file
    folder = Path(paths["models_dir"]) / name
    want = cfg["models"][name]["safetensors_sha256"]
    if want and sha256(folder / "model.safetensors") != want:
        raise RuntimeError(f"{name}: weights do not match the recorded SHA-256")
    heads = json.loads((folder / "config.json").read_text())["n_head"]
    return GPT2(load_file(str(folder / "model.safetensors")), dev, heads=heads, dtype=dtype)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _single_margins(run_study, site, z, fa, fb, dom, dev):
    import interchange as ic
    f = run_study.margin_fn(site.suffix, dev)
    return np.stack([np.concatenate([f(ic.single_states(z, frame, dom.pairs[i:i + 256]), dom.pairs[i:i + 256, 0],
                                       dom.cls[dom.single_target(v)[i:i + 256]]) for i in range(0, len(dom.pairs), 256)])
                     for v, frame in ((0, fa), (1, fb))])


def check(paths: dict, cfg: dict) -> dict:
    """Stage 1 (Section 16): no bigger-model outcome is computed here."""
    common, rs = setup(paths, "check")
    import das
    import interchange as ic
    import prompts
    from gpt2 import END, ResidualSite, margins_from_logits
    dev = device()
    tok = prompts.load_tokenizer()
    draws = json.loads((common.PROMPTS / "development-draws.json").read_text())
    repro = cfg["reproduction"]
    repro_dir = Path(paths["repro_dir"])
    settings = json.loads((repro_dir / "development-settings.json").read_text())
    saved = np.load(repro_dir / "arm-a.npz")
    record = json.loads((repro_dir / "record.json").read_text())
    out = {"started_utc": common.utc_now(), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    # (i) reproduction of a saved GPT-2 Small number, float32, TF32 off.
    small = load(paths, cfg, "gpt2", dev)
    task = rs.Task(small, tok, draws[repro["draw"]], dev)
    if task.name != repro["task"]:
        raise RuntimeError(f"reproduction task is {task.name}, expected {repro['task']}")
    site = ResidualSite(small, task.nat, settings["layer"], prompts.SITE_POSITIONS, task.ids)
    z = site.natural_states()
    name = repro["alignment"]
    fa, fb = saved[f"{name}__fa"].astype(np.float64), saved[f"{name}__fb"].astype(np.float64)
    got = _single_margins(rs, site, z, fa, fb, task.dom, dev)
    gap = float(np.abs(got - saved[f"{name}__single_margin"]).max())
    out["reproduction"] = {"task": task.name, "alignment": name, "layer": settings["layer"], "queries": int(got.size),
                           "max_abs_margin_difference": gap, "tolerance": repro["tolerance"],
                           "passed": gap <= repro["tolerance"]}
    # (iii) TF32 check on the same alignment: refit with the local machine's seed and settings, TF32 on for training only.
    fit_info = next(a["fit"] for a in record["arms"]["a"]["alignments"] if a["alignment"] == name)
    rs.TRAIN_TF32 = True
    seed = common.CONFIG["seeds"]["alignments"] + task.spec["draw"] * 100 + 11 * 2
    _sync()
    t0 = time.perf_counter()
    (ta, tb), info = rs.fitting(das.fit_joint, site.suffix, z, task.dom, 2, seed, lr=settings["lr"],
                                steps=settings["steps"], batch=common.CONFIG["batch_queries"], device=dev)
    _sync()
    fit_seconds = time.perf_counter() - t0
    tf32_margins = _single_margins(rs, site, z, ta, tb, task.dom, dev)
    eps = settings["eps"]
    agree = float((ic.decide(tf32_margins, eps) == saved[f"{name}__single"]).mean())
    out["tf32_check"] = {"best_loss_tf32": info["best_loss"], "best_loss_local_float32": fit_info["best_loss"],
                         "single_decisions_agreeing": agree, "fit_seconds": fit_seconds, "steps": settings["steps"]}
    rs.TRAIN_TF32 = False
    del small
    # (ii) parity and (iv) timings for each bigger model.
    from transformers import GPT2LMHeadModel
    spec = draws[0]
    tokens = torch.tensor(prompts.task_token_ids(tok, spec), device=dev)
    ids = list(prompts.class_token_ids(tok, spec))
    x = prompts.inputs(len(spec["templates"]))
    cls = prompts.answer_class(x[:, 0], x[:, 1])
    out["models"] = {}
    for model_name in cfg["bigger_models"]:
        m_out = {"parity": {}, "timings": {}}
        for dtype, key in ((torch.float64, "float64"), (torch.float32, "float32")):
            ref = GPT2LMHeadModel.from_pretrained(str(Path(paths["models_dir"]) / model_name)).to(dev, dtype).eval()
            ours = load(paths, cfg, model_name, dev, dtype)
            with torch.no_grad():
                hf = ref(tokens).logits[:, END]
                mine = ours.ln(ours.natural(tokens)["resid"][:, -1, END], *ours.lnf) @ ours.wte.T
            m_out["parity"][key] = {"max_logit_gap": float((hf - mine).abs().max()),
                                    "max_margin_gap": float((margins_from_logits(hf[:, ids], cls)
                                                             - margins_from_logits(mine[:, ids], cls)).abs().max())}
            del ref, ours
        tol = cfg["parity"]
        m_out["parity"]["passed"] = (max(m_out["parity"]["float64"].values()) <= tol["float64_tolerance"]
                                     and max(m_out["parity"]["float32"].values()) <= tol["float32_tolerance"])
        model = load(paths, cfg, model_name, dev)
        t = rs.Task(model, tok, spec, dev)
        rs.TRAIN_TF32 = True
        for layer in cfg["candidate_layers"][model_name]:
            site = ResidualSite(model, t.nat, layer, prompts.SITE_POSITIONS, t.ids)
            z = site.natural_states()
            _sync()
            t0 = time.perf_counter()
            (fa, fb), _ = rs.fitting(das.fit_joint, site.suffix, z, t.dom, 2, 1, lr=settings["lr"],
                                     steps=cfg["timing_steps"], batch=common.CONFIG["batch_queries"], device=dev)
            _sync()
            t1 = time.perf_counter()
            ic.census(z, fa, fb, t.dom, rs.margin_fn(site.suffix, dev), eps, chunk=common.CONFIG["batch_queries"])
            _sync()
            m_out["timings"][str(layer)] = {"das_seconds_per_step": (t1 - t0) / cfg["timing_steps"],
                                            "census_seconds": time.perf_counter() - t1}
        rs.TRAIN_TF32 = False
        out["models"][model_name] = m_out
        del model, t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out["passed"] = out["reproduction"]["passed"] and all(m["parity"]["passed"] for m in out["models"].values())
    out["finished_utc"] = common.utc_now()
    return out


def development(paths: dict, cfg: dict, model_name: str, commit=lambda: None) -> dict:
    """The settings rules of Section 11 on the model's own development tasks, then arm a on its first 8 development
    draws that pass its gate. Resumes: saved settings are reused and finished tasks are skipped."""
    common, rs = setup(paths, model_name)
    import prompts
    dev = device()
    rs.TRAIN_TF32 = cfg["train_tf32"]
    model = load(paths, cfg, model_name, dev)
    tok = prompts.load_tokenizer()
    draws = json.loads((common.PROMPTS / "development-draws.json").read_text())
    path = common.RESULTS / "development-settings.json"
    t0 = time.perf_counter()
    if not path.exists():
        settings = rs.choose_settings(model, tok, dev, draws, layers=cfg["candidate_layers"][model_name])
        common.save_json(path, {"written_at_utc": common.utc_now(), "model": model_name,
                                "seconds": time.perf_counter() - t0, **settings})
        commit()
    settings = json.loads(path.read_text())
    out = {"passed": [], "failed": []}
    for spec in draws:
        if len(out["passed"]) == common.CONFIG["development_tasks"]:
            break
        t = rs.Task(model, tok, spec, dev)
        if not t.passes_gate():
            out["failed"].append({"draw": spec["draw"], "min_margin": float(t.natural_margin.min())})
            continue
        out["passed"].append(spec["draw"])
        rs.run_task(t, "development", settings, arms_to_run=tuple(cfg["arms"]), split=cfg["heldout_split"])
        commit()
        print(f"{model_name} development {t.name} done", flush=True)
    common.save_json(common.RESULTS / "development-gate.json", out, overwrite=True)
    commit()
    return {"model": model_name, "settings": settings, "gate": out, "seconds": time.perf_counter() - t0}


def gate(paths: dict, cfg: dict, model_name: str, phase: str, wanted: int) -> dict:
    """Natural margins only: which draws, in order, pass the model's competence gate."""
    common, rs = setup(paths, model_name)
    import prompts
    dev = device()
    model = load(paths, cfg, model_name, dev)
    tok = prompts.load_tokenizer()
    name = "development-draws.json" if phase == "development" else "confirmation-draws.json"
    out = {"passed": [], "failed": []}
    for spec in json.loads((common.PROMPTS / name).read_text()):
        if len(out["passed"]) == wanted:
            break
        t = rs.Task(model, tok, spec, dev)
        if t.passes_gate():
            out["passed"].append(spec["draw"])
        else:
            out["failed"].append({"draw": spec["draw"], "min_margin": float(t.natural_margin.min())})
    return out


def tasks(paths: dict, cfg: dict, model_name: str, phase: str, draws: list[int], commit=lambda: None) -> dict:
    common, rs = setup(paths, model_name)
    import prompts
    dev = device()
    rs.TRAIN_TF32 = cfg["train_tf32"]
    model = load(paths, cfg, model_name, dev)
    tok = prompts.load_tokenizer()
    settings = json.loads((common.RESULTS / "development-settings.json").read_text())
    name = "development-draws.json" if phase == "development" else "confirmation-draws.json"
    specs = {s["draw"]: s for s in json.loads((common.PROMPTS / name).read_text())}
    done = []
    for d in draws:
        t = rs.Task(model, tok, specs[d], dev)
        if not t.passes_gate():
            raise RuntimeError(f"draw {d} no longer passes the gate")
        t0 = time.perf_counter()
        rs.run_task(t, phase, settings, arms_to_run=tuple(cfg["arms"]), split=cfg["heldout_split"])
        commit()
        done.append({"draw": d, "task": t.name, "seconds": time.perf_counter() - t0})
        print(f"{model_name} {phase} {t.name} done", flush=True)
    return {"model": model_name, "phase": phase, "tasks": done}


def analyse(paths: dict, cfg: dict, model_name: str, phase: str) -> dict:
    common, _ = setup(paths, model_name)
    import analyse as an
    rows = an.load_rows(phase)
    out = an.summarise(rows, common.CONFIG["bootstrap_replicates"])
    dest = common.RESULTS / "analysis" / phase
    common.save_json(dest / "summary.json", out, overwrite=phase == "development")
    import csv
    with open(dest / "alignments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    return {"model": model_name, "phase": phase, "alignments": len(rows)}
