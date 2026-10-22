"""AI-10 runner (GPU, under the portfolio lock): pilot, development and confirmation. PROTOCOL.md Sections 4-7, 11.

    python -B run_study.py pilot          numerical checks and timings on two development draws (Section 11, step 2)
    python -B run_study.py development    gate, tie band, learning rate, layer and steps, then every arm (step 3)
    python -B run_study.py confirmation   every arm on the first 40 held-out draws that pass the gate (step 5)

Nothing runs before study/PREDICTIONS-RECORD.json exists, which is written only after the author approves the protocol.
Confirmation draws are refused until study/freeze.json exists and every frozen file matches it. Records go to
study/results/runs/<phase>/<draw>-<names>/ and are never overwritten; a rerun skips finished tasks.

Status (17 October 2026): written and unit-tested on toy tensors only. It has never loaded GPT-2. The pilot is its
first real test, and development may change it (PROTOCOL Section 11); every change is recorded before the freeze.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

import ai10_common as common
import das
import interchange as ic
import margin_rule as mr
import prompts
from gpt2 import END, GPT2, HeadSites, ResidualSite, margins_from_logits

C = common.CONFIG
SITE_A = prompts.SITE_POSITIONS
TRAIN_TF32 = False      # Amendment 2: the bigger-model arm trains DAS with TF32; censuses, margins and gradients never do


def fitting(fn, *args, **kwargs):
    """Run a DAS fit with TF32 matrix products if TRAIN_TF32 is set, then switch TF32 off again."""
    torch.backends.cuda.matmul.allow_tf32 = TRAIN_TF32
    try:
        return fn(*args, **kwargs)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False


# ---------------------------------------------------------------------------------------------------- setup
def device():
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible; run through gpu_lock.py (make development / confirmation)")
    torch.cuda.set_per_process_memory_fraction(C["resources"]["max_gpu_memory_fraction"])
    torch.backends.cuda.matmul.allow_tf32 = False      # census, margins and gradients in full float32
    return torch.device("cuda")


def load_model(dev):
    path = common.model_dir() / "model.safetensors"
    if common.sha256(path) != C["model"]["safetensors_sha256"]:
        raise SystemExit("GPT-2 Small weights do not match the recorded SHA-256")
    return GPT2.from_snapshot(common.model_dir(), dev)


class Task:
    def __init__(self, model: GPT2, tok, spec: dict, dev):
        self.spec, self.m, self.dev = spec, model, dev
        self.tokens = torch.tensor(prompts.task_token_ids(tok, spec), device=dev)
        self.ids = prompts.class_token_ids(tok, spec)
        x = prompts.inputs(len(spec["templates"]))
        self.dom = ic.Domain(x, prompts.answer_class(x[:, 0], x[:, 1]))
        with torch.no_grad():
            self.nat = model.natural(self.tokens)
            logits = model.class_logits(self.nat["resid"][:, -1, END], self.ids)
        self.natural_margin = margins_from_logits(logits, self.dom.cls).cpu().double().numpy()

    @property
    def name(self):
        return f"{self.spec['draw']:03d}-" + "-".join(self.spec["names"])

    def passes_gate(self) -> bool:
        return bool(self.natural_margin.min() >= C["competence_margin"])


def margin_fn(suffix, dev):
    def f(states, recipients, cls):
        with torch.no_grad():
            logits = suffix(torch.tensor(states, dtype=torch.float32, device=dev), recipients)
        return margins_from_logits(logits, cls).cpu().double().numpy()
    return f


def margin_grads(suffix, states: np.ndarray, recipients, cls, dev, chunk: int):
    out = []
    for i in range(0, len(states), chunk):
        s = torch.tensor(states[i:i + chunk], dtype=torch.float32, device=dev, requires_grad=True)
        margins_from_logits(suffix(s, recipients[i:i + chunk]), cls[i:i + chunk]).sum().backward()
        out.append(s.grad.cpu().double().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------------------------- one-site arms
def one_site_record(name, suffix, z, fa, fb, dom, eps, dev, extra=None):
    """Census, margin rule and classification of one alignment at a single site (arms a, a' and c)."""
    cen = ic.census(z, fa, fb, dom, margin_fn(suffix, dev), eps, chunk=C["batch_queries"])
    r = dom.pairs[:, 0]
    grads = [margin_grads(suffix, ic.single_states(z, f, dom.pairs), r, dom.cls[dom.single_target(v)], dev,
                          C["batch_queries"]) for v, f in ((0, fa), (1, fb))]
    scores = mr.one_site_scores(z, fa, fb, dom, cen["single_margin"], grads[0], grads[1], cen["joint_margin"],
                                chunk=256)
    if scores["identity_max_abs"] > 1e-4 or (scores["ai05_form_max_abs"] or 0.0) > 1e-4:
        raise SystemExit(f"{name}: numerical gate failed (Lemma 1 split {scores['identity_max_abs']:.1e}, "
                         f"AI-05 form {scores['ai05_form_max_abs']})")
    split = ic.classify_one_site(cen["single"], cen["joint"], dom)
    scale = np.sqrt(np.mean([np.sum((z[i] - z[j]) ** 2) for i in range(dom.n) for j in range(i + 1, dom.n)]))
    masks = dom.panels()
    off = {}
    for v, (label, f) in enumerate((("a", fa), ("b", fb))):
        miss = np.linalg.norm(ic.single_states(z, f, dom.pairs) - z[dom.single_target(v)], axis=1) / scale
        off[f"off_target_median_{label}"] = float(np.median(miss[masks[("full_single", v)]]))
        off[f"off_target_min_{label}"] = float(miss[masks[("full_single", v)]].min())
    rec = {"alignment": name, "rank_a": int(fa.shape[1]), "rank_b": int(fb.shape[1]), "overlap": das.overlap(fa, fb),
           "single_fail": int((cen["single"] == 1).sum()), "single_tie": int((cen["single"] == 2).sum()),
           "joint_fail_all": int((cen["joint"] == 1).sum()), "joint_tie_all": int((cen["joint"] == 2).sum()),
           **ic.panel_rates(cen["single"], cen["joint"], dom),
           **{k: v for k, v in split.items() if k != "both_reduced_pass"},
           **mr.alignment_flag(scores["sigma"], split["both_reduced_pass"], C["rule_threshold"]),
           "identity_max_abs": scores["identity_max_abs"], "ai05_form_max_abs": scores["ai05_form_max_abs"],
           "sign_agreement": float(np.mean(np.concatenate([scores["sign_right_a"], scores["sign_right_b"]]))),
           **off, **(extra or {})}
    arrays = {"fa": fa.astype(np.float32), "fb": fb.astype(np.float32), **cen, "sigma": scores["sigma"],
              "rho": scores["rho"], "both_reduced_pass": split["both_reduced_pass"]}
    return rec, arrays


def arm_one_site(task: Task, layer: int, positions, settings, separate: bool, split: bool = True):
    site = ResidualSite(task.m, task.nat, layer, positions, task.ids)
    z = site.natural_states()
    seed = C["seeds"]["alignments"] + task.spec["draw"] * 100
    fit = dict(lr=settings["lr"], steps=settings["steps"], batch=C["batch_queries"], device=task.dev)
    frames, info = {}, {}
    for r in C["das"]["ranks"]:
        if separate:
            frames[f"sep_r{r}"], info[f"sep_r{r}"] = fitting(das.fit_separate, (site.suffix, site.suffix), (z, z), task.dom,
                                                                      r, seed + 11 * r, **fit)
        else:
            frames[f"das_r{r}"], info[f"das_r{r}"] = fitting(das.fit_joint, site.suffix, z, task.dom, r, seed + 11 * r, **fit)
    if separate:
        frames["mean_raw"] = das.mean_frames(z, task.dom.x, orthogonalise=False)
    else:
        frames["mean"] = das.mean_frames(z, task.dom.x, orthogonalise=True)
        frames["random_r2"] = das.random_frames(z.shape[1], 2, 2, seed + 131)
        if positions == SITE_A and split:               # held-out split: rank 2 trained on half the single panel
            rng = np.random.default_rng([C["seeds"]["heldout_split"], task.spec["draw"]])
            train = rng.random(len(task.dom.pairs)) < 0.5
            frames["das_r2_split"], info["das_r2_split"] = fitting(das.fit_joint, site.suffix, z, task.dom, 2, seed + 97,
                                                                         mask=train, **fit)
            info["das_r2_split"]["train_mask"] = train.tolist()
    out = []
    for name, (fa, fb) in frames.items():
        rec, arrays = one_site_record(name, site.suffix, z, fa, fb, task.dom, settings["eps"], task.dev,
                                      {"fit": {k: v for k, v in info.get(name, {}).items() if k != "train_mask"}})
        if "train_mask" in info.get(name, {}):
            arrays["train_mask"] = np.array(info[name]["train_mask"])
        out.append((rec, arrays))
    return out, info


# --------------------------------------------------------------------------------------------- two-site arm
def arm_heads(task: Task, settings):
    hs = HeadSites(task.m, task.nat, C["heads"]["position"], C["heads"]["token"][0], task.ids)
    za, zb = hs.natural_states()
    dom, dev = task.dom, task.dev
    seed = C["seeds"]["alignments"] + task.spec["draw"] * 100 + 50
    fit = dict(lr=settings["lr"], steps=settings["steps"], batch=C["batch_queries"], device=dev)
    frames = {f"das_r{r}": fitting(das.fit_separate, (hs.suffix_a, hs.suffix_b), (za, zb), dom, r, seed + 11 * r, **fit)[0]
              for r in C["das"]["ranks"]}
    frames["mean"] = (das.contrast_basis(za, dom.x, 0), das.contrast_basis(zb, dom.x, 1))
    frames["random_r2"] = (das.random_frames(za.shape[1], 2, 0, seed + 131)[0],
                           das.random_frames(zb.shape[1], 2, 0, seed + 137)[0])
    out = []
    for name, (fa, fb) in frames.items():
        out.append(two_site_record(name, hs, za, zb, fa, fb, dom, settings["eps"], dev))
    return out


def two_site_record(name, hs: HeadSites, za, zb, fa, fb, dom, eps, dev):
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    r, d = dom.pairs.T
    ea = ic.single_states(za, fa, dom.pairs)                       # A-edits (r, s)
    eb = ic.single_states(zb, fb, dom.pairs)                       # B-edits (r, t)
    cls_a, cls_b = dom.cls[dom.single_target(0)], dom.cls[dom.single_target(1)]
    b = C["batch_queries"]
    ma, mb, w = [], [], []
    with torch.no_grad():
        for i in range(0, len(r), b):
            logits, wi = hs.forward(r[i:i + b], a_states=t(ea[i:i + b]), keep_w=True)
            ma.append(margins_from_logits(logits, cls_a[i:i + b]).cpu().double().numpy())
            w.append(wi.cpu().double().numpy())
            mb.append(margins_from_logits(hs.suffix_b(t(eb[i:i + b]), r[i:i + b]), cls_b[i:i + b]).cpu().double().numpy())
    single_m = np.stack([np.concatenate(ma), np.concatenate(mb)])
    w = np.concatenate(w)
    tr, ts, tt = dom.triples.T
    cls_j = dom.cls[dom.joint_target()]
    fbt = t(fb)
    jm = []
    with torch.no_grad():
        for i in range(0, len(tr), b):
            sl = slice(i, i + b)
            logits, _ = hs.forward(tr[sl], a_states=t(ea[tr[sl] * dom.n + ts[sl]]), b_frame=fbt, b_donor=t(zb[tt[sl]]))
            jm.append(margins_from_logits(logits, cls_j[sl]).cpu().double().numpy())
    jm = np.concatenate(jm)
    grad_b = margin_grads(hs.suffix_b, eb, r, cls_b, dev, b)
    scores = mr.two_site_scores(zb, fb, w, dom, single_m[1], grad_b)
    single, joint = ic.decide(single_m, eps), ic.decide(jm, eps)
    split = ic.classify_two_site(single, joint, dom)
    landed = float(np.linalg.norm(w - zb[dom.single_target(0)], axis=1).max())
    _, ib = dom.reduced()
    tracking = float(np.mean(np.sign(scores["d_hat"]) == np.sign(jm - single_m[1][ib])))
    rec = {"alignment": name, "rank_a": int(fa.shape[1]), "rank_b": int(fb.shape[1]),
           "single_fail": int((single == 1).sum()), "single_tie": int((single == 2).sum()),
           "joint_fail_all": int((joint == 1).sum()), "joint_tie_all": int((joint == 2).sum()),
           **ic.panel_rates(single, joint, dom), **{k: v for k, v in split.items() if k != "both_reduced_pass"},
           **mr.alignment_flag(scores["sigma"], split["both_reduced_pass"], C["rule_threshold"]),
           "max_propagated_distance": landed, "first_order_sign_agreement": tracking}
    arrays = {"fa": fa.astype(np.float32), "fb": fb.astype(np.float32), "single_margin": single_m.astype(np.float32),
              "joint_margin": jm.astype(np.float32), "single": single, "joint": joint, "sigma": scores["sigma"],
              "rho": scores["rho"], "both_reduced_pass": split["both_reduced_pass"]}
    return rec, arrays


# ---------------------------------------------------------------------------------------------- one task
def run_task(task: Task, phase: str, settings: dict, arms_to_run=("a", "c", "a_end", "b"), split: bool = True) -> None:
    folder = common.RESULTS / "runs" / phase / task.name
    if (folder / "record.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record = {"task": task.spec, "settings": settings, "natural_margin": task.natural_margin.tolist(), "arms": {}}
    arms = {"a": lambda: arm_one_site(task, settings["layer"], SITE_A, settings, separate=False, split=split)[0],
            "c": lambda: arm_one_site(task, settings["layer"], SITE_A, settings, separate=True)[0],
            "a_end": lambda: arm_one_site(task, settings["layer"], [END], settings, separate=False)[0],
            "b": lambda: arm_heads(task, settings)}
    for arm, run in arms.items():
        if arm not in arms_to_run:
            continue
        t0 = time.perf_counter()
        results = run()
        record["arms"][arm] = {"seconds": time.perf_counter() - t0, "alignments": [rec for rec, _ in results]}
        np.savez_compressed(folder / f"arm-{arm}.npz",
                            **{f"{rec['alignment']}__{k}": v for rec, arrays in results for k, v in arrays.items()})
    record["seconds"] = time.perf_counter() - started
    record["arrays_sha256"] = {p.name: common.sha256(p) for p in sorted(folder.glob("arm-*.npz"))}
    common.save_json(folder / "record.json", record)


def run_phase(phase: str, draws: list[dict], wanted: int, settings: dict, model, tok, dev, **task_options) -> dict:
    gate = {"passed": [], "failed": []}
    for spec in draws:
        if len(gate["passed"]) == wanted:
            break
        common.check_resources()
        task = Task(model, tok, spec, dev)
        if not task.passes_gate():
            gate["failed"].append({"draw": spec["draw"], "min_margin": float(task.natural_margin.min())})
            continue
        gate["passed"].append(spec["draw"])
        run_task(task, phase, settings, **task_options)
        print(f"{phase} {task.name} done", flush=True)
    return gate


# ----------------------------------------------------------------------------------------- development rules
def choose_settings(model, tok, dev, draws, layers=None) -> dict:
    """PROTOCOL Section 11, step 3: the tie band, learning rate, layer and number of steps, from single queries only.
    `layers` are the candidate layers (Amendment 2 gives each bigger model its own); default: the configuration's."""
    layers = list(layers or C["candidate_layers"])
    tasks = []
    for spec in draws:
        if len(tasks) == C["development_tasks"]:
            break
        t = Task(model, tok, spec, dev)
        if t.passes_gate():
            tasks.append(t)
    # Tie band: the same natural margins through the site code and the full forward pass, at two batch sizes.
    gaps = []
    for t in tasks:
        for layer in layers:
            site = ResidualSite(model, t.nat, layer, SITE_A, t.ids)
            f = margin_fn(site.suffix, dev)
            z = site.natural_states()
            idx = np.arange(t.dom.n)
            gaps.append(np.abs(f(z, idx, t.dom.cls) - t.natural_margin).max())
            gaps.append(np.abs(np.concatenate([f(z[i:i + 1], idx[i:i + 1], t.dom.cls[i:i + 1]) for i in idx])
                               - t.natural_margin).max())
    noise = float(max(gaps))
    eps = next((e for e in C["tie_band_candidates"] if e >= 10 * noise), None)
    if eps is None:
        raise SystemExit(f"numerical noise {noise:.2e} exceeds every tie-band candidate; stop and report")
    steps_max = C["das"]["development_steps"]
    batch = C["batch_queries"]
    # Learning rate: rank-2 joint DAS at the middle candidate layer on the first four tasks, run for the full number
    # of development steps; the lowest mean best loss wins.
    mid = layers[len(layers) // 2]
    losses, curves = {}, {}
    for lr in C["das"]["learning_rate_candidates"]:
        vals = []
        for t in tasks[:4]:
            site = ResidualSite(model, t.nat, mid, SITE_A, t.ids)
            info = fitting(das.fit_joint, site.suffix, site.natural_states(), t.dom, 2, 1, lr=lr, steps=steps_max,
                                 batch=batch, device=dev, checkpoints=tuple(C["das"]["step_candidates"]))[1]
            vals.append(info["best_loss"])
            curves.setdefault(str(lr), []).append(info["best_loss_at_step"])
        losses[str(lr)] = float(np.mean(vals))
    lr = float(min(losses, key=losses.get))
    # Steps: the smallest candidate whose best loss is within the tolerance of the full run's in 90% of those fits.
    tol = C["das"]["convergence_tolerance"]
    steps = next(s for s in C["das"]["step_candidates"]
                 if np.mean([c[str(s)] <= (1 + tol) * c[str(steps_max)] + 1e-6 for c in curves[str(lr)]]) >= 0.9)
    # Layer: rank-2 joint DAS at every candidate layer on every task; lowest mean full-domain single failure rate
    # (single queries only; ties go to the later layer).
    layer_rate = {}
    for layer in layers:
        rates = []
        for t in tasks:
            site = ResidualSite(model, t.nat, layer, SITE_A, t.ids)
            z = site.natural_states()
            (fa, fb), _ = fitting(das.fit_joint, site.suffix, z, t.dom, 2, 1, lr=lr, steps=steps, batch=batch, device=dev)
            f = margin_fn(site.suffix, dev)
            masks = t.dom.panels()
            for v, frame in ((0, fa), (1, fb)):
                m = f(ic.single_states(z, frame, t.dom.pairs), t.dom.pairs[:, 0], t.dom.cls[t.dom.single_target(v)])
                rates.append(float((ic.decide(m, eps)[masks[("full_single", v)]] == 1).mean()))
        layer_rate[str(layer)] = float(np.mean(rates))
    best = min(layer_rate.values())
    layer = max(int(k) for k, v in layer_rate.items() if v == best)
    return {"eps": eps, "numerical_noise": noise, "lr": lr, "lr_losses": losses, "layer": layer,
            "layer_single_failure_rate": layer_rate, "steps": int(steps), "development_tasks": [t.spec["draw"] for t in tasks]}


def main() -> None:
    phase = sys.argv[1]
    common.require_predictions()
    common.check_resources()
    dev = device()
    model = load_model(dev)
    tok = prompts.load_tokenizer()
    dev_draws = json.loads((common.PROMPTS / "development-draws.json").read_text())
    if phase == "pilot":
        import pilot_checks
        pilot_checks.run(model, tok, dev, dev_draws[:2])
    elif phase == "development":
        path = common.RESULTS / "development-settings.json"
        if not path.exists():
            settings = choose_settings(model, tok, dev, dev_draws)
            common.save_json(path, {"written_at_utc": common.utc_now(), **settings})
        settings = json.loads(path.read_text())
        gate = run_phase("development", dev_draws, C["development_tasks"], settings, model, tok, dev)
        common.save_json(common.RESULTS / "development-gate.json", gate, overwrite=True)
    elif phase == "confirmation":
        common.require_freeze()
        settings = json.loads((common.RESULTS / "development-settings.json").read_text())
        draws = json.loads((common.PROMPTS / "confirmation-draws.json").read_text())
        gate = run_phase("confirmation", draws, C["confirmation_tasks"], settings, model, tok, dev)
        common.save_json(common.RESULTS / "confirmation-gate.json", gate, overwrite=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
