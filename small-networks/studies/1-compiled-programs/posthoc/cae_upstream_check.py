"""Check the CAE mediator finding against the authors' repository (post hoc; needs network access).

Fetches systems/11_controlled_experiments.py from github.com/MelouxM/CAE at the pinned commit and on the default
branch, compares them with the vendored copy, and records the line that sets the mediator of the XOR backup model.
Output: posthoc/results/cae-upstream-check.json.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
VENDOR = STUDY / "02-Code/vendor/cae"
PATH = "systems/11_controlled_experiments.py"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "AI-05-check"}), timeout=60) as r:
        return r.read()


def describe(data: bytes):
    lines = data.decode().splitlines()
    hits = [i + 1 for i, line in enumerate(lines) if line.strip() == "a = x.copy()"]
    return {"sha256": hashlib.sha256(data).hexdigest(), "lines_with_a_equals_x": hits,
            "honours_intervention_on_a": any('interventions.get("a"' in line or "interventions.get('a'" in line
                                             for line in lines)}


def main():
    commit = subprocess.run(["git", "--no-optional-locks", "-C", str(VENDOR), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    out = {"checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "repository": "https://github.com/MelouxM/CAE",
           "file": PATH, "pinned_commit": commit, "vendored": describe((VENDOR / PATH).read_bytes())}
    out["upstream_at_pinned_commit"] = describe(get(f"https://raw.githubusercontent.com/MelouxM/CAE/{commit}/{PATH}"))
    try:
        out["upstream_default_branch"] = describe(get(f"https://raw.githubusercontent.com/MelouxM/CAE/HEAD/{PATH}"))
    except Exception as exc:
        out["upstream_default_branch"] = {"error": str(exc)}
    out["vendored_equals_upstream_at_commit"] = out["vendored"]["sha256"] == out["upstream_at_pinned_commit"]["sha256"]
    dest = STUDY / "posthoc/results"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "cae-upstream-check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
