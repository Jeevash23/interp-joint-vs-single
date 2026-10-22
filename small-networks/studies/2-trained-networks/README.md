# Study 2: trained networks (frozen 6 October 2026)

200 small networks trained on the same two task families as Study 1 (25 fresh tables per family × four
architectures: transformers with model width 16 or 64, MLPs with hidden width 16 or 64), six alignments per
network, and the exhaustive single (1,458) and joint (19,683) interchange census of every alignment. The protocol,
code and final seeds were frozen at 01:16:32 UTC on 6 October 2026, after development on four tables per family
and before any final network was trained (`PROTOCOL.md`, `FROZEN-MANIFEST.json`); post-freeze changes are listed
in `AMENDMENTS.md`. The paper reports the study in Section 5 and Appendix B. The post hoc analyses were written after the
results were known and are labelled as such in the paper.

| Path | Contents |
|---|---|
| `PROTOCOL.md` | Question, what was looked at before the freeze (including the development results), systems, alignments, census, panels, primary and secondary quantities, verification |
| `FROZEN-MANIFEST.json` | Freeze time, final seeds, and SHA-256 of the protocol and code; `code/run.py final` refuses to run if any of them changed |
| `code/common.py` | Task generators (identical to Study 1's), the two architectures, training and competence gate, the alignment procedures, and the census |
| `code/run.py` | Trains, aligns and takes the census for every network of a split (resumable, two worker processes) |
| `code/analyze.py` | All prespecified quantities from the saved censuses, with the program-cluster bootstrap |
| `code/verify.py` | Independent NumPy re-implementation of both architectures; recomputes every census decision and recounts the per-alignment quantities |
| `code/freeze.py` | Writes the freeze manifest |
| `runs/development/`, `runs/final/` | One folder per network: `task.json`, `weights.npz`, `candidates.npz`, `census.npz` (decisions and off-natural distances) and `record.json` (training attempts, alignment losses, timings) |
| `results/development/`, `results/final/` | `analysis.json`, `candidates.csv` (one row per alignment) and `verification.json` |
| `posthoc/` | Post-freeze analyses of saved results: `joint_only.py` (the alignments that fail only jointly, and whether any single edit lands on its target), `detection_sensitivity.py` (the primary contrast under other cost conventions, and what extra joint or single queries add), `margin_bound.py` (the first-order and Lipschitz margin changes of Proposition 2 for every joint query of the full-domain DAS alignments and of the alignments that pass every single query: the unsigned score ρ, the signed score σ and the rule σ > 1, all defined before the script was first run) and `heldout_split.py` (the full-domain DAS alignments refitted on a random half of the single queries, with their census in `posthoc/heldout-split/`, and held-out single tests compared with joint tests) |
| `logs/` | Run logs |

Rerun the analysis with `../../.venv-v2/bin/python code/analyze.py final` and the verification with
`../../.venv-v2/bin/python code/verify.py final` (both read only the saved runs). `.venv-v2` is the project
environment; a storage cleanup removed the original on 8 October 2026 and it was rebuilt on 9 October 2026 with the
documented commands. `../../REPRODUCIBILITY.md` describes the environment and retraining.
