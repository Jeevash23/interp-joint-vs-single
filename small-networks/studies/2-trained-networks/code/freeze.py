"""Freeze Study 2: hash the protocol and code and fix the final seeds before any final network exists."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
FILES = ["PROTOCOL.md", "code/common.py", "code/run.py", "code/analyze.py", "code/verify.py", "code/freeze.py"]


def main():
    if (STUDY / "runs/final").exists() or (STUDY / "FROZEN-MANIFEST.json").exists():
        raise SystemExit("already frozen or final runs exist")
    manifest = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "final_seeds": {"lookup": list(range(30000, 30025)), "additive": list(range(40000, 40025))},
                "sha256": {f: hashlib.sha256((STUDY / f).read_bytes()).hexdigest() for f in FILES},
                "development_networks": len(list((STUDY / "runs/development").iterdir()))}
    (STUDY / "FROZEN-MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
