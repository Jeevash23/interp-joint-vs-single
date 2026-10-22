"""Freeze Study 3: hash the protocol, the prediction record and the code, and fix the final seeds before any final
network exists. Study 2's common.py, which Study 3 imports, is hashed too."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
FILES = ["PROTOCOL.md", "PREDICTIONS-RECORD.json", "code/sites.py", "code/run.py", "code/analyze.py", "code/verify.py",
         "code/margins.py", "code/freeze.py", "../2-trained-networks/code/common.py"]


def main():
    if (STUDY / "runs/final").exists() or (STUDY / "FROZEN-MANIFEST.json").exists():
        raise SystemExit("already frozen or final runs exist")
    manifest = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "final_seeds": {"lookup": list(range(50000, 50025)), "additive": list(range(60000, 60025))},
                "sha256": {f: hashlib.sha256((STUDY / f).read_bytes()).hexdigest() for f in FILES},
                "development_networks": len(list((STUDY / "runs/development").iterdir()))}
    (STUDY / "FROZEN-MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
