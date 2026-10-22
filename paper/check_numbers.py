#!/usr/bin/env python3
"""Check that the combined paper's generated files come from the two parts' generated files (standard library only).

    python paper/check_numbers.py           # compare paper/generated/ with what the two parts give
    python paper/check_numbers.py --write   # rewrite paper/generated/ from the two parts

The paper prints no number of its own. Every number is a macro, and every macro and table body is taken unchanged from
small-networks/paper/generated/ (Studies 1-3) or gpt2/paper/generated/ (Study 4). Each part's `make quick-check` runs
its own paper/make_numbers.py --check, which recomputes those files from the part's saved outputs; this script checks
the step from the parts to the paper: the two macro files concatenated (no macro name in both), the table bodies byte
for byte except one wording change in Study 4's predictions table, and Study 4's copies of Study 2 numbers equal to
Study 2's own macros.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "small-networks/paper/generated"
GPT2 = ROOT / "gpt2/paper/generated"
OUT = ROOT / "paper/generated"
DEFINITION = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")
TABLES = [
    (SMALL, "table-compiled-candidates.tex", "table-compiled-candidates.tex", []),
    (SMALL, "table-compiled-panels.tex", "table-compiled-panels.tex", []),
    (SMALL, "table-trained-cells.tex", "table-trained-cells.tex", []),
    (SMALL, "table-multisite-cells.tex", "table-multisite-cells.tex", []),
    (SMALL, "table-strategies.tex", "table-strategies.tex", []),
    # Study 4's part calls Study 2 "the companion study"; in the combined paper it is Study 2.
    (GPT2, "table-predictions.tex", "table-gpt2-predictions.tex",
     [("above the companion study's 5.5\\%", "above Study~2's 5.5\\%")]),
]
CROSS = {
    "AiFiveContrast": "TrnContrastOneTwenty", "AiFiveJointOnly": "TrnPerfectJointFail",
    "AiFivePerfect": "TrnPerfect", "AiFiveTrainedNetworks": "TrnNetworks", "AiFiveFlagged": "TrnMbRuleFlagged",
    "DRefJointOnly": "TrnPerfectJointFail", "DRefPerfect": "TrnPerfect", "DRefShare": "TrnPerfectShare",
}


def macro_lines(path: Path) -> list[tuple[str, str, str]]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("%"):
            continue
        m = DEFINITION.match(line)
        if not m:
            sys.exit(f"unexpected line in {path.relative_to(ROOT)}: {line[:80]}")
        out.append((m.group(1), m.group(2), line))
    return out


def build() -> dict[str, bytes]:
    small, gpt2 = macro_lines(SMALL / "numbers.tex"), macro_lines(GPT2 / "numbers.tex")
    clash = sorted({n for n, _, _ in small} & {n for n, _, _ in gpt2})
    if clash:
        sys.exit(f"macro names in both parts: {clash}")
    values = {n: v for n, v, _ in small + gpt2}
    for copy, own in CROSS.items():
        if values[copy] != values[own]:
            sys.exit(f"Study 4's \\{copy} = {values[copy]} differs from Study 2's \\{own} = {values[own]}")
    lines = ["% Written by paper/check_numbers.py from the two parts' generated numbers. Do not edit.",
             "% Studies 1-3: small-networks/paper/generated/numbers.tex, unchanged."]
    lines += [line for _, _, line in small]
    lines += ["% Study 4: gpt2/paper/generated/numbers.tex, unchanged."]
    lines += [line for _, _, line in gpt2]
    files = {"numbers.tex": ("\n".join(lines) + "\n").encode()}
    for folder, source, target, subs in TABLES:
        text = (folder / source).read_text()
        for old, new in subs:
            if text.count(old) != 1:
                sys.exit(f"'{old}' does not occur exactly once in {folder.relative_to(ROOT)}/{source}")
            text = text.replace(old, new)
        files[target] = text.encode()
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    files = build()
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (OUT / name).write_bytes(data)
        print(f"wrote {len(files)} files in paper/generated/")
        return 0
    stale = [n for n, d in files.items() if not (OUT / n).exists() or (OUT / n).read_bytes() != d]
    if stale:
        print("paper/generated/ differs from the parts: " + ", ".join(stale))
        return 1
    n = len(re.findall(rb"\\newcommand", files["numbers.tex"]))
    print(f"paper/generated/ matches the two parts ({n} macros, {len(TABLES)} tables)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
