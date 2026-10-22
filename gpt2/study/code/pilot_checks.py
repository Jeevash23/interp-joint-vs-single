"""Pilot (PROTOCOL Section 11, step 2): numerical checks and timings on two development draws; no outcome is recorded.

Called by `run_study.py pilot` after the predictions are hashed. Checks:
  1. AI-10's GPT-2 code against transformers' GPT2LMHeadModel on the real weights, float32 on the GPU: every logit at
     the final position and every margin within 1e-3 (Amendment 1, PROTOCOL Section 15; the float64 part of the gate is
     pilot_parity_float64.py);
  2. null edits reproduce the natural margins (within 1e-4) at every candidate layer for the one-site and final-position
     sites, and at the head sites;
  3. timings: one rank-2 joint DAS fit of 100 steps per candidate layer, and one census, to replace the estimates of
     PROTOCOL Section 10. Failure counts of the timed census are not recorded.
"""
from __future__ import annotations

import time

import numpy as np
import torch

import ai10_common as common
import das
import interchange as ic
import prompts
from gpt2 import END, HeadSites, ResidualSite, margins_from_logits

C = common.CONFIG


def run(model, tok, dev, draws) -> None:
    from transformers import GPT2LMHeadModel
    import run_study
    ref = GPT2LMHeadModel.from_pretrained(str(common.model_dir()), local_files_only=True).to(dev).eval()
    out = {"written_at_utc": common.utc_now(), "draws": [d["draw"] for d in draws], "checks": {}, "timings": {}}
    for spec in draws:
        task = run_study.Task(model, tok, spec, dev)
        with torch.no_grad():
            hf = ref(task.tokens).logits[:, END]
            ours = model.ln(task.nat["resid"][:, -1, END], *model.lnf) @ model.wte.T
        hf_margin = margins_from_logits(hf[:, list(task.ids)], task.dom.cls).cpu().numpy()
        c = {"max_logit_gap": float((hf - ours).abs().max()),
             "max_margin_gap": float(np.abs(hf_margin - task.natural_margin).max())}
        idx = np.arange(task.dom.n)
        with torch.no_grad():
            for layer in C["candidate_layers"]:
                for label, pos in (("site_a", prompts.SITE_POSITIONS), ("site_end", [END])):
                    site = ResidualSite(model, task.nat, layer, pos, task.ids)
                    z = torch.tensor(site.natural_states(), dtype=torch.float32, device=dev)
                    m = margins_from_logits(site.suffix(z, idx), task.dom.cls).cpu().numpy()
                    c[f"null_{label}_L{layer}"] = float(np.abs(m - task.natural_margin).max())
            hs = HeadSites(model, task.nat, C["heads"]["position"], C["heads"]["token"][0], task.ids)
            za, zb = (torch.tensor(a, dtype=torch.float32, device=dev) for a in hs.natural_states())
            m = margins_from_logits(hs.forward(idx, a_states=za, b_frame=torch.eye(64, device=dev), b_donor=zb)[0],
                                    task.dom.cls).cpu().numpy()
            c["null_heads"] = float(np.abs(m - task.natural_margin).max())
        # Parity limits as amended (PROTOCOL Section 15): float32 on the GPU within 1e-3; the float64 part of the gate
        # is pilot_parity_float64.py. Null edits within 1e-4.
        c["passed"] = (c["max_logit_gap"] <= 1e-3 and c["max_margin_gap"] <= 1e-3
                       and all(v <= 1e-4 for k, v in c.items() if k.startswith("null")))
        out["checks"][task.name] = c
    for layer in C["candidate_layers"]:
        site = ResidualSite(model, task.nat, layer, prompts.SITE_POSITIONS, task.ids)
        z = site.natural_states()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        (fa, fb), _ = das.fit_joint(site.suffix, z, task.dom, 2, 1, lr=0.003, steps=100, batch=C["batch_queries"],
                                    device=dev)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ic.census(z, fa, fb, task.dom, run_study.margin_fn(site.suffix, dev), 1e-3, chunk=C["batch_queries"])
        torch.cuda.synchronize()
        out["timings"][f"L{layer}"] = {"das_seconds_per_step": (t1 - t0) / 100,
                                       "census_seconds": time.perf_counter() - t1}
    out["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    out["passed"] = all(c["passed"] for c in out["checks"].values())
    common.save_json(common.RESULTS / "pilot.json", out)
    print(out)
