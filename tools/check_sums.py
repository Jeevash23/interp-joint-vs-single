#!/usr/bin/env python3
"""Check this release's checksums (standard library only).

    python tools/check_sums.py

Checks every file listed in SHA256SUMS at the top level, and in small-networks/SHA256SUMS and gpt2/SHA256SUMS, the two
parts' own lists. Files that `make unpack` adds are not listed and are not checked here; tools/unpack.py in each part
checks them against that part's data/PACKED-FILES.json.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(folder: Path) -> tuple[int, list[str]]:
    bad, n = [], 0
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        path = folder / rel
        n += 1
        if not path.is_file():
            bad.append(f"missing: {path.relative_to(ROOT)}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(f"changed: {path.relative_to(ROOT)}")
    return n, bad


def main() -> int:
    problems, total = [], 0
    for folder in (ROOT, ROOT / "small-networks", ROOT / "gpt2"):
        n, bad = check(folder)
        total += n
        problems += bad
    for p in problems:
        print(p)
    if problems:
        return 1
    print(f"all {total} listed files match their SHA-256 (top level and both parts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
