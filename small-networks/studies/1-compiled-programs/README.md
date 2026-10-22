# Study 1: compiled transformers (frozen 29 September 2026)

80 Tracr programs (40 lookup, 40 additive), six alignments per program (480), exact validity labels from the
carrier equations, and eight sampled test panels at budgets of 12–120 model calls. The protocol, seeds and five
code files were frozen at 08:15:34 UTC on 29 September 2026, before the final programs were generated
(`06-Study-History/V2-FROZEN-MANIFEST.json`). The paper (`../../paper/manuscript.pdf`, Section 4 and Appendix A)
reports this study; the original V2 report in `01-Papers/` is the record as written at the time.

**Post-freeze analysis** (`posthoc/`, added 6 October 2026): exact computations on the saved networks and
alignments, not new data. `census.py` evaluates every single (1,458) and joint (19,683) interchange and every
frozen panel's whole query space for all 480 alignments; `summary.py` aggregates them and compares the frozen
panel decisions with their expected values; `worked_example.py` recomputes the lookup-10000 example;
`panel_simulation.py` replays each frozen panel's sampling rule 10,000 times on the census to give the expected
acceptance and its central 95% range; `cae_mediator.py` reruns the CAE controlled XOR experiment with the one-line
mediator fix. Results are in
`posthoc/results/`. Main facts: no alignment fails a joint query while passing every single query; the labels
equal the single-query census for 480/480 alignments; every valid alignment's single edits land on their targets
up to unread nuisance coordinates; all 28 frozen panel counts lie within the central 95% of the replays.

The internal folder names are those of the original layout and must stay: the frozen code computes its root
from its own location and reads `03-Data/v2`, `04-Results/v2`, `06-Study-History/V2-FROZEN-MANIFEST.json` and
`02-Code/vendor/...` relative to it.

| Path | Contents |
|---|---|
| `01-Papers/` | Original V2 report (`AI-05-REPORT-V2.md/.pdf`), literature record, and `sources/` (the five papers read, with SHA-256 download records, and the inspected OASR/DCD code) |
| `02-Code/` | Frozen study code (`v2_model.py`, `v2_candidates.py`, `v2_panel.py`, `v2_run.py`, `v2_resources.py`), analysis, verification, replay and official-reproduction scripts, requirements, and `vendor/` (Tracr, CAE and rep_divergence clones at pinned commits, unmodified) |
| `03-Data/v2/` | Development (4 per family) and final (80) programs: metadata, compiled weights, rotation, alignments |
| `04-Results/v2/` | Per-program records and compressed query records, analysis, independent verification, margin audit, clean-environment replay, official reproductions, figures, and every log, including failed development attempts |
| `05-Notes/` | Claim–evidence map, reproduction notes and the comparator memo as written for V2 (the memo is frozen) |
| `06-Study-History/` | Frozen protocol and manifest, the post-freeze verification amendment, and the source/environment manifest (529 files; check with `make verify`) |
| `posthoc/` | Post-freeze exact analyses and their results (above) |

Do not run `02-Code/v2_analyze.py`, `v2_verify.py`, `v2_figures.py`, `v2_clean_replay.py`, `v2_official_grant.py`
or `reproduce_official_v2.py` inside this folder: they overwrite hashed records in place. `make replay-study1`
reruns the whole study in a fresh temporary copy instead.
