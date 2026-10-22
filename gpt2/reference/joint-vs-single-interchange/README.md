# Files of the companion study

The study builds on a companion study of joint and single interchange tests in compiled and small trained networks,
released separately. Its code imports that study's Study 2 network code, reads its candidate tables, a worked example and
its manuscript numbers, and arm d loads its Study 2 networks. Those files are copied here, in the companion study's own
layout, and the code points at this folder.

| Path | What it is | Used by |
|---|---|---|
| `studies/2-trained-networks/code/common.py` | Study 2's network code | arm d (`study/code/overlap_ai05.py`) |
| `studies/2-trained-networks/FROZEN-MANIFEST.json` | Study 2's freeze record | the study's freeze (`ai05_inputs`) |
| `studies/2-trained-networks/results/{final,development}/candidates.csv` | Study 2's candidate tables | arm d's analysis |
| `studies/2-trained-networks/runs/{development,final}/<network>/` | each network's weights, census, task and training record (232 networks), unpacked from `data/saved-outputs/companion-study2-networks.tar.xz` | arm d |
| `analysis/results/constructed-example.json` | the companion study's worked example | the unit tests |
| `paper/generated/numbers.tex` | the companion study's manuscript numbers | `paper/make_numbers.py` |

The study's freeze records the SHA-256 of five of these files as `ai05_inputs`; `make check-freezes` checks them.
