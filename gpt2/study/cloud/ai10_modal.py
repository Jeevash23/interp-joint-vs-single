"""Modal app for AI-10's bigger-model arm (PROTOCOL.md Section 16, Amendment 2). Run from study/cloud with the CLI,
through run_stage.sh (which retries a start that could not connect):

    modal run --detach ai10_modal.py::check                                 stage 1: reproduction, parity, TF32, timings
    modal run --detach ai10_modal.py::development --model gpt2-medium       stage 2 (then pull: cost decision)
    modal run --detach ai10_modal.py::development --model gpt2-large        stage 4, only if the cost decision allows
    modal run --detach ai10_modal.py::confirmation --model gpt2-medium      stage 6, after the freeze
    modal run ai10_modal.py::pull --stage <stage>                           records back to results/cloud/

Each stage runs inside Modal in orchestrate(), so a dropped local connection cannot stop it (AI-03 Study 5's pattern).
Weights download inside the image build at pinned revisions with every SHA-256 asserted; the environment is AI-01's
locked one, as on the local machine. Records and arrays stay on the Volume `ai10-bigger`; pull copies back only the small
records. Before each stage the billing report is read; no stage starts once the arm has spent $6, or if the month's
total would pass $27 (Section 16, "Budget and stopping").
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

# Duplicated from cloud-config.json because the image build runs this file before the study files are mounted; the
# local side asserts that the two agree.
MODELS = {"gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e",
                   "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"),
          "gpt2-medium": ("openai-community/gpt2-medium", "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
                          "fc5a354a19255ad494f3d71549390baca1ccf61d1d822b9408971705c687c9cd"),
          "gpt2-large": ("openai-community/gpt2-large", "32b71b12589c2f8d625668d2335a01cac3249519",
                         "5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce")}
TOKENIZER_SHA = "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
REMOTE_MODELS, ROOT, VOL = "/root/models", "/root/ai10", "/vol"
APP = "ai10-bigger"


def _sha(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download_weights():
    from huggingface_hub import hf_hub_download
    record = {}
    for name, (repo, revision, want) in MODELS.items():
        folder = Path(REMOTE_MODELS) / name
        folder.mkdir(parents=True, exist_ok=True)
        for f in ("config.json", "tokenizer.json", "model.safetensors"):
            hf_hub_download(repo, f, revision=revision, local_dir=folder)
        assert _sha(folder / "model.safetensors") == want, name
        assert _sha(folder / "tokenizer.json") == TOKENIZER_SHA, name
        record[name] = dict(repo=repo, revision=revision, safetensors_sha256=want,
                            bytes=(folder / "model.safetensors").stat().st_size)
    (Path(REMOTE_MODELS) / "build-record.json").write_text(json.dumps(record, indent=2))


if modal.is_local():
    HERE = Path(__file__).resolve().parent
    STUDY = HERE.parent
    AI01 = STUDY.parent  # release: pyproject.toml and uv.lock are at the release root
    CFG = json.loads((HERE / "cloud-config.json").read_text())
    assert {k: (v["repo"], v["revision"], v["safetensors_sha256"]) for k, v in CFG["models"].items()} == MODELS
    assert CFG["tokenizer_sha256"] == TOKENIZER_SHA and CFG["app"] == APP
    REPRO = STUDY / "results/runs/development" / CFG["reproduction"]["task"]
else:
    HERE = STUDY = AI01 = REPRO = Path("/nonexistent")
    CFG = None

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(uv_project_dir=str(AI01), extras=["experiment"], frozen=True)
         .run_function(download_weights)
         .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
               "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
               "OPENBLAS_NUM_THREADS": "4"})
         .add_local_dir(STUDY / "code", f"{ROOT}/study/code", ignore=["__pycache__"])
         .add_local_file(STUDY / "config.json", f"{ROOT}/study/config.json")
         .add_local_dir(STUDY / "prompts", f"{ROOT}/study/prompts")
         .add_local_file(HERE / "cloud_jobs.py", f"{ROOT}/study/cloud/cloud_jobs.py")
         .add_local_file(HERE / "cloud-config.json", f"{ROOT}/study/cloud/cloud-config.json")
         .add_local_file(STUDY / "results/development-settings.json", f"{ROOT}/repro/development-settings.json")
         .add_local_file(REPRO / "record.json", f"{ROOT}/repro/record.json")
         .add_local_file(REPRO / "arm-a.npz", f"{ROOT}/repro/arm-a.npz"))
app = modal.App(APP, image=image, tags={"project": "AI-10", "arm": "bigger-models"})
volume = modal.Volume.from_name(APP, create_if_missing=True)
PATHS = {"code_dir": f"{ROOT}/study/code", "models_dir": REMOTE_MODELS, "volume_dir": VOL, "repro_dir": f"{ROOT}/repro"}
GPU_OPTS = dict(gpu="H100", timeout=6 * 3600, cpu=4.0, memory=32768, volumes={VOL: volume}, scaledown_window=10)


# ============================================================================================== remote side
def _remote_cfg():
    return json.loads(Path(f"{ROOT}/study/cloud/cloud-config.json").read_text())


def _check_frozen(frozen: dict | None):
    """Confirmation only: the mounted code, configuration and prompts must match the freeze's hashes."""
    if not frozen:
        return
    changed = [rel for rel, want in frozen.items() if _sha(Path(ROOT) / rel) != want]
    if changed:
        raise RuntimeError(f"mounted files differ from the freeze: {changed}")


@app.function(**GPU_OPTS, max_containers=4)
def gpu_job(task: dict) -> str:
    sys.path.insert(0, f"{ROOT}/study/cloud")
    import cloud_jobs as cj
    cfg = _remote_cfg()
    _check_frozen(task.get("frozen"))
    volume.reload()
    kind = task["kind"]
    if kind == "check":
        target = Path(VOL) / "check" / "check.json"
        if target.exists():
            return target.read_text()
        out = cj.check(PATHS, cfg)
        out["build_record"] = json.loads((Path(REMOTE_MODELS) / "build-record.json").read_text())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(out, indent=1))
    elif kind == "development":
        out = cj.development(PATHS, cfg, task["model"], commit=volume.commit)
    elif kind == "gate":
        out = cj.gate(PATHS, cfg, task["model"], task["phase"], task["wanted"])
        target = Path(VOL) / task["model"] / f"{task['phase']}-gate.json"
        target.write_text(json.dumps(out, indent=1))
    elif kind == "tasks":
        out = cj.tasks(PATHS, cfg, task["model"], task["phase"], task["draws"], commit=volume.commit)
    else:
        raise ValueError(kind)
    volume.commit()
    return json.dumps(out)


@app.function(cpu=4.0, memory=16384, timeout=3600, volumes={VOL: volume}, scaledown_window=10, max_containers=1)
def cpu_job(task: dict) -> str:
    sys.path.insert(0, f"{ROOT}/study/cloud")
    import cloud_jobs as cj
    volume.reload()
    out = cj.analyse(PATHS, _remote_cfg(), task["model"], task["phase"])
    volume.commit()
    return json.dumps(out)


@app.function(cpu=0.25, memory=1024, timeout=12 * 3600, volumes={VOL: volume}, max_containers=1)
def orchestrate(task: dict) -> str:
    """Runs one stage inside Modal; writes orchestration/<stage>.json on the Volume with its status when it ends."""
    stage = task["stage"]
    summary = dict(stage=stage, started_utc=datetime.now(timezone.utc).isoformat(), status="running")
    target = Path(VOL) / "orchestration" / f"{stage}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=1))
    volume.commit()
    try:
        if stage == "check":
            out = json.loads(gpu_job.remote(dict(kind="check")))
            summary.update(passed=out["passed"])
        elif stage.startswith("development-"):
            model = task["model"]
            summary.update(result=json.loads(gpu_job.remote(dict(kind="development", model=model))))
            summary.update(analysis=json.loads(cpu_job.remote(dict(model=model, phase="development"))))
        elif stage.startswith("confirmation-"):
            model, frozen = task["model"], task["frozen"]
            g = json.loads(gpu_job.remote(dict(kind="gate", model=model, phase="confirmation", wanted=task["tasks"],
                                               frozen=frozen)))
            draws = g["passed"]
            k = max(1, min(task["containers"], len(draws)))
            chunks = [draws[i::k] for i in range(k)]
            results, failures = [], []
            for r in gpu_job.map([dict(kind="tasks", model=model, phase="confirmation", draws=c, frozen=frozen)
                                  for c in chunks], return_exceptions=True):
                (failures.append(repr(r)) if isinstance(r, Exception) else results.append(json.loads(r)))
            summary.update(gate=g, tasks=[t for r in results for t in r["tasks"]], failures=failures)
            if failures:
                raise RuntimeError(f"confirmation stopped: {failures}")
            summary.update(analysis=json.loads(cpu_job.remote(dict(model=model, phase="confirmation"))))
        else:
            raise ValueError(stage)
        summary["status"] = "finished"
    except Exception as error:                                                   # noqa: BLE001
        summary.update(status="failed", error=repr(error))
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    target.write_text(json.dumps(summary, indent=1))
    volume.commit()
    return summary["status"]


# =============================================================================================== local side
def _results() -> Path:
    path = STUDY / "results" / "cloud"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _billing() -> dict:
    """The arm's and the month's spend by Modal's billing report (retries while the local connection drops)."""
    for _ in range(6):
        proc = subprocess.run([str(Path.home() / ".local/bin/modal"), "billing", "report", "--for", "this month",
                               "--json"], capture_output=True, text=True)
        if proc.returncode == 0:
            break
        time.sleep(20)
    else:
        raise SystemExit(f"could not read the billing report: {proc.stderr[-300:]}")
    rows = json.loads(proc.stdout)
    arm = sum(float(r["cost"]) for r in rows if r["description"].startswith(APP))
    month = sum(float(r["cost"]) for r in rows)
    entry = dict(utc=datetime.now(timezone.utc).isoformat(), arm_usd=round(arm, 4), month_usd=round(month, 4))
    with (_results() / "billing-log.jsonl").open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def _guard(stage: str, projected_usd: float) -> dict:
    b = _billing()
    budget = CFG["budget"]
    print(f"[{stage}] arm spend ${b['arm_usd']:.2f}; month ${b['month_usd']:.2f}; projected ${projected_usd:.2f}",
          flush=True)
    if b["arm_usd"] >= budget["arm_stop_usd"]:
        raise SystemExit(f"stop: the arm has spent ${b['arm_usd']:.2f} (limit ${budget['arm_stop_usd']})")
    if b["month_usd"] + projected_usd > budget["month_cap_usd"]:
        raise SystemExit(f"stop: the month would reach ${b['month_usd'] + projected_usd:.2f} (cap ${budget['month_cap_usd']})")
    with (_results() / "stage-log.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(stage=stage, started_utc=b["utc"], projected_usd=projected_usd, **b)) + "\n")
    return b


def _start(stage: str, projected_usd: float, **task):
    _guard(stage, projected_usd)
    call = orchestrate.spawn(dict(task, stage=stage))
    with (_results() / "stage-log.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(stage=stage, spawned_utc=datetime.now(timezone.utc).isoformat(),
                                 call=call.object_id)) + "\n")
    print(f"[{stage}] running in Modal as {call.object_id}; pull with: modal run ai10_modal.py::pull --stage {stage}",
          flush=True)


def _hourly() -> float:
    r, c = CFG["rates_usd_per_hour"], CFG["container"]
    return r[CFG["gpu"]] + r["cpu_core"] * c["cpu"] + r["memory_gib"] * c["memory_mib"] / 1024


def _read(path: str) -> str:
    return b"".join(volume.read_file(path)).decode()


def _listing(path: str) -> list[str]:
    try:
        return [e.path for e in volume.listdir(path, recursive=True)]
    except Exception:                                                          # noqa: BLE001
        return []


@app.local_entrypoint()
def check():
    if not (STUDY / "AMENDMENT-2-RECORD.json").exists():
        raise SystemExit("Amendment 2 and the bigger-model predictions must be recorded first (freeze.py amendment2)")
    _start("check", projected_usd=0.6)


@app.local_entrypoint()
def development(model: str):
    if model not in CFG["bigger_models"]:
        raise SystemExit(model)
    check_record = _results() / "check.json"
    if not check_record.exists() or not json.loads(check_record.read_text())["passed"]:
        raise SystemExit("the check stage must have passed (pull --stage check)")
    if model == "gpt2-large":
        decision = json.loads((_results() / "cost-decision.json").read_text())
        if not decision["run_large"]:
            raise SystemExit("the cost decision does not allow GPT-2 Large")
        projected = decision["large_development_usd"]
    else:
        projected = 2.5
    _start(f"development-{model}", projected_usd=projected, model=model)


@app.local_entrypoint()
def confirmation(model: str):
    sys.path.insert(0, str(STUDY / "code"))
    import ai10_common as common
    freeze = common.require_freeze()
    decision = json.loads((_results() / "cost-decision.json").read_text())
    n = decision["confirmation_tasks"].get(model, 0)
    if n < CFG["budget"]["minimum_confirmation_tasks"]:
        raise SystemExit(f"{model}: the cost decision allows {n} confirmation tasks")
    frozen = {rel: freeze["files"][rel] for rel in freeze["files"]
              if rel.startswith(("study/code/", "study/prompts/")) or rel in ("study/config.json",
                                                                               "study/cloud/cloud_jobs.py",
                                                                               "study/cloud/cloud-config.json")}
    _start(f"confirmation-{model}", projected_usd=decision["confirmation_usd"][model], model=model, tasks=n,
           containers=CFG["confirmation_containers"], frozen=frozen)


@app.local_entrypoint()
def pull(stage: str):
    """Copy a stage's small records from the Volume to results/cloud/ (arrays stay on the Volume)."""
    status = json.loads(_read(f"orchestration/{stage}.json"))
    print(json.dumps({k: status.get(k) for k in ("stage", "status", "started_utc", "finished_utc", "error")}),
          flush=True)
    if status["status"] == "running":
        return
    out = _results()
    (out / "orchestration").mkdir(exist_ok=True)
    (out / "orchestration" / f"{stage}.json").write_text(json.dumps(status, indent=1))
    if stage == "check":
        (out / "check.json").write_text(_read("check/check.json"))
    else:
        model = status.get("result", {}).get("model") or stage.split("-", 1)[1]
        for path in _listing(model):
            if path.endswith((".json", ".csv")):
                target = out / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(_read(path))
        if stage == "development-gpt2-medium" and status["status"] == "finished":
            _cost_decision(out)
    _billing()


def _per_task(root: Path, phase: str) -> list[float]:
    return [json.loads(p.read_text())["seconds"] for p in sorted((root / "runs" / phase).glob("*/record.json"))]


def _cost_decision(out: Path):
    """Section 16, cost rule: project Medium's confirmation and Large's development and confirmation from measured
    times (Large scaled from the check stage's timings by Medium's ratio of measured to projected time)."""
    check = json.loads((out / "check.json").read_text())
    hourly = _hourly()
    spent = _billing()["arm_usd"]
    budget = CFG["budget"]
    med = out / "gpt2-medium"
    s_med = json.loads((med / "development-settings.json").read_text())
    task_med = sum(_per_task(med, "development")) / len(_per_task(med, "development"))
    t = check["models"]
    steps, layers_m = s_med["steps"], CFG["candidate_layers"]["gpt2-medium"]
    index = layers_m.index(s_med["layer"])

    def projected_task(model, layer):
        tm = t[model]["timings"][str(layer)]
        return 3 * steps * tm["das_seconds_per_step"] + 5 * tm["census_seconds"]

    dev_steps = json.loads((STUDY / "config.json").read_text())["das"]["development_steps"]

    def projected_settings(model):
        lays = CFG["candidate_layers"][model]
        tm = t[model]["timings"]
        lr = 3 * 4 * dev_steps * tm[str(lays[len(lays) // 2])]["das_seconds_per_step"]
        sweep = 8 * sum(steps * tm[str(x)]["das_seconds_per_step"] + tm[str(x)]["census_seconds"] / 10 for x in lays)
        return lr + sweep

    ratio_task = task_med / projected_task("gpt2-medium", s_med["layer"])
    ratio_settings = s_med["seconds"] / projected_settings("gpt2-medium")
    overhead = 4 * 180                                   # four containers: image start, model load and the gate
    usd = lambda seconds: seconds * hourly / 3600        # noqa: E731
    medium_conf = {n: usd(n * task_med + overhead) for n in range(1, 41)}
    large_layer = CFG["candidate_layers"]["gpt2-large"][index]
    large_task = ratio_task * projected_task("gpt2-large", large_layer)
    large_dev = usd(ratio_settings * projected_settings("gpt2-large") + 8 * large_task + 300)
    large_conf = usd(40 * large_task + overhead)
    limit = budget["large_rule_usd"]
    n_med = max([n for n in range(budget["minimum_confirmation_tasks"], 41) if spent + medium_conf[n] <= limit],
                default=0)
    run_large = n_med == 40 and spent + medium_conf[40] + large_dev + large_conf <= limit
    decision = dict(created_utc=datetime.now(timezone.utc).isoformat(), arm_spent_usd=spent, hourly_usd=hourly,
                    medium_seconds_per_task=task_med, medium_ratio_task=ratio_task, medium_ratio_settings=ratio_settings,
                    large_layer_if_same_depth=large_layer, large_seconds_per_task=large_task,
                    large_development_usd=large_dev, confirmation_usd={"gpt2-medium": medium_conf.get(n_med, 0.0),
                                                                       "gpt2-large": large_conf},
                    confirmation_tasks={"gpt2-medium": n_med, "gpt2-large": 40 if run_large else 0},
                    run_large=run_large, rule=f"Large only if spent + Medium confirmation + Large in all <= ${limit}")
    (out / "cost-decision.json").write_text(json.dumps(decision, indent=1))
    print(json.dumps(decision, indent=1), flush=True)
