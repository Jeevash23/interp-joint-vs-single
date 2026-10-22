"""Record the predictions (PROTOCOL Section 11, step 1) and freeze the study (step 4). Neither overwrites.

  python -B study/code/freeze.py predictions   writes study/PREDICTIONS-RECORD.json; only after the author approves the
                                               protocol and before any model has run
  python -B study/code/freeze.py amendment2    writes study/AMENDMENT-2-RECORD.json (bigger-model arm, Section 16)
  python -B study/code/freeze.py freeze        writes study/freeze.json after the development phase
"""
from __future__ import annotations

import sys

from ai10_common import AI05, CONFIG, PROMPTS, RESULTS, ROOT, STUDY, model_dir, save_json, sha256, utc_now

AI05_INPUTS = ["studies/2-trained-networks/code/common.py", "studies/2-trained-networks/FROZEN-MANIFEST.json",
               "studies/2-trained-networks/results/final/candidates.csv",
               "studies/2-trained-networks/results/development/candidates.csv",
               "analysis/results/constructed-example.json"]


def record_predictions() -> None:
    path = STUDY / "PREDICTIONS-RECORD.json"
    if path.exists():
        raise SystemExit(f"{path} exists; predictions are recorded once")
    if (RESULTS / "runs").exists() or (RESULTS / "overlap-ai05").exists() or (RESULTS / "pilot.json").exists():
        raise SystemExit("AI-10 model outputs exist already; predictions must be recorded before any run")
    save_json(path, {"written_at_utc": utc_now(), "predictions_sha256": sha256(STUDY / "PREDICTIONS.md"),
                     "protocol_sha256": sha256(STUDY / "PROTOCOL.md"), "config_sha256": sha256(STUDY / "config.json"),
                     "prompts_sha256": {p.name: sha256(p) for p in sorted(PROMPTS.glob("*.json"))},
                     "predictions_text": (STUDY / "PREDICTIONS.md").read_text(),
                     "note": "Recorded after the author approved the protocol and before any model was run."})


def record_amendment2() -> None:
    """Amendment 2 (PROTOCOL Section 16): hash the amended protocol, the bigger-model predictions and the cloud code
    before any cloud computation."""
    path = STUDY / "AMENDMENT-2-RECORD.json"
    if path.exists():
        raise SystemExit(f"{path} exists; Amendment 2 is recorded once")
    if (RESULTS / "cloud").exists():
        raise SystemExit("cloud outputs exist; Amendment 2 must be recorded before any cloud computation")
    if (STUDY / "freeze.json").exists():
        raise SystemExit("the study is frozen; Amendment 2 belongs before the freeze")
    files = ["study/PROTOCOL.md", "study/PREDICTIONS-BIGGER.md", "study/DEVELOPMENT-NOTES.md", "study/config.json"]
    files += sorted(str(p.relative_to(ROOT)) for p in (STUDY / "cloud").glob("*") if p.is_file())
    files += sorted(str(p.relative_to(ROOT)) for p in (STUDY / "code").glob("*.py"))
    save_json(path, {"written_at_utc": utc_now(), "files": {rel: sha256(ROOT / rel) for rel in files},
                     "predictions_text": (STUDY / "PREDICTIONS-BIGGER.md").read_text(),
                     "note": "Recorded after the author confirmed the arm and its $6 budget, before any bigger-model "
                             "computation and before any GPT-2 Small joint-query outcome was read."})


def freeze() -> None:
    path = STUDY / "freeze.json"
    if path.exists():
        raise SystemExit(f"{path} exists; the study is frozen once")
    for need in ("PREDICTIONS-RECORD.json", "DEVELOPMENT-NOTES.md"):
        if not (STUDY / need).exists():
            raise SystemExit(f"study/{need} is missing")
    if (RESULTS / "runs/confirmation").exists() or (RESULTS / "overlap-ai05/final").exists():
        raise SystemExit("Confirmation outputs exist before the freeze")
    files = ["study/PROTOCOL.md", "study/PROTOCOL-approved-2026-10-17.md", "study/PREDICTIONS.md",
             "study/PREDICTIONS-RECORD.json", "study/config.json", "study/DEVELOPMENT-NOTES.md",
             "study/results/development-settings.json", "Makefile"]
    files += sorted(str(p.relative_to(ROOT)) for p in PROMPTS.glob("*.json"))
    files += sorted(str(p.relative_to(ROOT)) for p in (STUDY / "code").glob("*.py"))
    # What was seen before confirmation: the development gate, analyses and arm d's development records.
    files += [str(p.relative_to(ROOT)) for p in [RESULTS / "development-gate.json",
                                                RESULTS / "analysis/development/summary.json",
                                                RESULTS / "analysis/overlap-development/summary.json",
                                                RESULTS / "overlap-ai05/development/alignments.csv",
                                                RESULTS / "overlap-ai05/development/record.json"] if p.exists()]
    # Amendment 2: the bigger-model arm's predictions, record, cloud code and each model's development records.
    if (STUDY / "AMENDMENT-2-RECORD.json").exists():
        files += ["study/PREDICTIONS-BIGGER.md", "study/AMENDMENT-2-RECORD.json"]
        files += sorted(str(p.relative_to(ROOT)) for p in (STUDY / "cloud").glob("*") if p.is_file())
        files += sorted(str(p.relative_to(ROOT)) for p in (RESULTS / "cloud").rglob("*")
                        if p.is_file() and "/confirmation" not in str(p) and p.suffix in (".json", ".csv"))
    save_json(path, {"utc": utc_now(), "kind": "local analysis freeze; not external preregistration",
                     "confirmation_outputs_exist": False, "files": {rel: sha256(ROOT / rel) for rel in files},
                     "model_safetensors_sha256": sha256(model_dir() / "model.safetensors"),
                     "ai05_inputs": {rel: sha256(AI05 / rel) for rel in AI05_INPUTS},
                     "config_model": CONFIG["model"]})


if __name__ == "__main__":
    {"predictions": record_predictions, "amendment2": record_amendment2, "freeze": freeze}[sys.argv[1]]()
