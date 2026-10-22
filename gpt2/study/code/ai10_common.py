"""Paths, configuration, hashing and resource guards shared by the AI-10 scripts.

AI-05 and AI-01 are read, never written: AI-10 imports AI-05's frozen Study 2 code read-only (overlap arm only) and runs
in AI-01's environment with `python -B`, so no bytecode is written into either project.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
import sys as _release_sys  # release: frozen-file checks accept documented changes
_release_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import release_paths as _release_paths  # noqa: E402

sys.dont_write_bytecode = True
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

CODE = Path(__file__).resolve().parent
STUDY = CODE.parent
ROOT = STUDY.parent
AI_FOLDER = ROOT.parent
AI05 = ROOT / "reference/joint-vs-single-interchange"  # release: the companion study's files are shipped here
RESULTS = STUDY / "results"
PROMPTS = STUDY / "prompts"
CONFIG = json.loads((STUDY / "config.json").read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save_json(path: Path, obj, overwrite: bool = False) -> None:
    """Write JSON; records are written once unless the caller says otherwise."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} exists; not overwriting")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def model_dir() -> Path:
    return Path(os.environ.get("MODELS_DIR", ROOT / "models")) / "gpt2-small"  # release: see models/README.md


def require_predictions() -> None:
    """No model is run before the author approves the protocol and the predictions are hashed."""
    if not (STUDY / "PREDICTIONS-RECORD.json").exists():
        raise SystemExit("study/PREDICTIONS-RECORD.json is missing: the protocol is not approved and the predictions "
                         "are not recorded, so no model may run (PROTOCOL.md Section 11)")


def require_freeze() -> dict:
    """Confirmation work needs the freeze, and every frozen file must still match its hash."""
    path = STUDY / "freeze.json"
    if not path.exists():
        raise SystemExit("study/freeze.json is missing: confirmation tasks are refused before the freeze")
    freeze = json.loads(path.read_text())
    changed = [rel for rel, digest in freeze["files"].items() if not _release_paths.matches(ROOT / rel, digest, rel)]
    if changed:
        raise SystemExit(f"Frozen files changed since the freeze: {changed}")
    return freeze


def available_ram_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    return float("nan")


def check_resources() -> dict:
    """Refuse to start when RAM or disk is below the configured floors."""
    limits = CONFIG["resources"]
    ram = available_ram_gib()
    disk = shutil.disk_usage(ROOT).free / 2 ** 30
    if ram < limits["minimum_ram_gib"]:
        raise SystemExit(f"Only {ram:.1f} GiB of RAM available (floor {limits['minimum_ram_gib']} GiB)")
    if disk < limits["minimum_disk_gib"]:
        raise SystemExit(f"Only {disk:.1f} GiB of disk free (floor {limits['minimum_disk_gib']} GiB)")
    return {"ram_available_gib": round(ram, 2), "disk_free_gib": round(disk, 1)}
