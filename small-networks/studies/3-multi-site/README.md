# Study 3: the two variables at different sites (predictions recorded and frozen 6 October 2026)

Lemma 1 of the paper needs both projectors at one site. Study 3 places A upstream of B, where a joint query reduces to
a single query only through A (Remark 2 of the paper), and measures how joint and single tests compare there. It uses
200 networks trained by Study 2's frozen code on 25 new tables per family and the same four architectures, five
alignments per network (DAS of rank 1, 2 and 4 at each site, conditional-mean contrasts and random subspaces), and the
exhaustive census of every single (1,458) and joint (19,683) query of every alignment.

- MLP: A at the first hidden layer after its ReLU, B at the second.
- Transformer: A at the residual stream of position 1 (the token a) before attention, B at the last position after
  attention.

The two predictions were written and hashed at 07:36:02 UTC on 6 October 2026, before any network of this study
existed (`PREDICTIONS-RECORD.json`). The protocol, code, final seeds and Study 2's `common.py` were frozen at 07:55:10
UTC, after development on four tables per family and before any final network was trained (`FROZEN-MANIFEST.json`).
The paper reports the study in Section 6 and Appendix C.

| Path | Contents |
|---|---|
| `PROTOCOL.md` | Question, systems, sites, alignments, census, quantities, the two predictions, and the development results appended before the freeze |
| `PREDICTIONS-RECORD.json` | Time and SHA-256 of the protocol as first written and of its predictions section, with the predictions' text |
| `FROZEN-MANIFEST.json` | Freeze time, final seeds, SHA-256 of the protocol, the prediction record, the code and Study 2's `common.py`; `code/run.py final` refuses to run if any of them changed |
| `code/sites.py` | The two sites, propagation from A's site to B's, the per-site DAS fit, the other alignments, and the cross-site census (imports Study 2's frozen `common.py` unchanged) |
| `code/run.py` | Trains, aligns and takes the census of every network of a split (resumable) |
| `code/analyze.py` | The prespecified quantities from the saved censuses, with the program-cluster bootstrap and the verdicts on the two predictions |
| `code/margins.py` | The exploratory first-order margin predictor (needs the saved models) |
| `code/verify.py` | Independent NumPy re-implementation of both architectures and the cross-site interventions; recomputes every census decision and recounts the per-alignment quantities |
| `code/freeze.py` | Writes the freeze manifest |
| `runs/development/`, `runs/final/` | One folder per network: `task.json`, `weights.npz`, `candidates.npz` (projectors at both sites), `census.npz` and `record.json` |
| `results/development/`, `results/final/` | `analysis.json`, `candidates.csv`, `margins.json`, `verification.json` |
| `posthoc/` | Written after the results were known: `mechanism.py` (the detection contrast for the alignments whose A-edits all land, and joint tests against single tests aimed at the weaker variable) and `signed_rule.py` (the signed margin rule, fixed on Study 2 before this study's protocol and predictions were recorded, applied unchanged: alignment-level ranking and query-level areas under the ROC curve by architecture) |
| `logs/` | Run logs and start and end times |

Rerun the analysis with `../../.venv-v2/bin/python code/analyze.py final` and the verification with
`../../.venv-v2/bin/python code/verify.py final` (both read only the saved runs). `.venv-v2` is the project
environment; a storage cleanup removed the original on 8 October 2026 and it was rebuilt on 9 October 2026 with the
documented commands (`../../REPRODUCIBILITY.md`).

## Results in brief

199 of 200 networks classified every input correctly with a logit margin of at least 1, the standard of Study 2 (one
lookup transformer of width 64 missed it at all three initialisations and is excluded). Prediction 1 held on its
estimate but not beyond sampling error: 3 of the 40 alignments that pass every single query fail a joint query (7.5%,
95% interval 1.6–20.4%, against Study 2's 5.5%), all three in MLPs on additive tasks. Prediction 2 held beyond
sampling error: joint minus single detection at 120 calls is +0.012 (95% interval +0.009 to +0.016, against Study 2's
−0.028). Post hoc, most of that gain comes from joint queries always exercising B, which fails more often than A in
504 of the 597 DAS alignments; against single tests aimed at the weaker variable the contrast is −0.004. The signed
margin rule, fixed on Study 2 before this study's protocol was recorded and applied post hoc without change, flags
exactly the three joint-only alignments, which it ranks first to third of the 40. The verifier agreed on all
21,035,295 decisions.
