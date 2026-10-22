> Historical preliminary-pilot record. The completed V2 study is linked from `00-START-HERE.md`. Earlier feasibility/status claims below do not describe the completed study.

# Reproduction and provenance

From AI-05 root, with Python 3.14.7:

```
python3 -m venv .venv
.venv/bin/python 02-Code/study.py
.venv/bin/python 02-Code/verify_records.py
```

The experiment uses the Python standard library only; `02-Code/requirements.lock` is intentionally empty. It runs one CPU process and no numerical worker pool. The local virtual environment was created without package downloads. `04-Results/final.json` contains every target query, prediction, state and paired candidate score. `04-Results/independent-verification.json` recomputes all aggregates from the saved rows. The frozen protocol hash is embedded in final.json. See `06-Study-History/PROTOCOL-FROZEN-2026-09-29.md` before interpreting any outcome.

Published reproduction: `reproduce_grant()` in `02-Code/study.py` uses the printed 4x3 ReLU matrix, bias, natural representations and mean-difference vector from Grant et al., arXiv:2511.04638v5, section 4.2.1. This is a paper-equation reconstruction, not an execution of author code. It matches the reported natural scores 0.25, 0.5, 0, 0 and patched scores 0.5, 0.25 and verifies the dormant third unit activates only after the patch.

External source snapshots: paper version identifiers are pinned in `05-Notes/QUESTION-COMPARATOR-GAP.md`. Git access to GitHub failed with DNS resolution error, so no source commit can honestly be pinned and no external source code is run. The archived Apache-2.0 Tracr repository is a candidate for a future exact reproduction, not a dependency here. No AI-01–AI-04 environment or reserved data was touched.
