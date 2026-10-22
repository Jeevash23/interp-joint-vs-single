# When can a joint intervention reject an alignment that single interventions accept? Code release

This folder holds the code, frozen inputs, study records and saved outputs behind the manuscript *When can a joint
intervention reject an alignment that single interventions accept?* Every number, generated table and plotted value
in the manuscript can be recomputed from the saved outputs on a CPU in about a minute, without network access.

## What the paper asks

Interchange interventions test whether a network's internal states carry the variables an explanation assigns to
them, usually one variable at a time. The paper asks when testing two variables at once adds anything, for
alignments that place two variables in orthogonal subspaces at one site. A joint edit is a single edit applied to
the result of the other single edit, so a joint test can reject what single tests accept only when both variables'
single edits give correct outputs without reproducing the natural states they simulate, and only where these misses
move the output's margin by more than the single queries' own margins. Three studies measure how often that happens.

## Studies

| Manuscript | Folder | What it did |
|---|---|---|
| Study 1 | `studies/1-compiled-programs` | 80 transformers compiled with Tracr, six alignments each, exact labels, eight frozen test panels, an independent verifier, and a post hoc census of every single and joint interchange of its 480 alignments |
| Study 2 | `studies/2-trained-networks` | 200 small trained transformers and MLPs (and 32 development networks): alignments fitted, every single and joint interchange taken, joint and single tests compared at prespecified costs; post hoc margin and held-out analyses |
| Study 3 | `studies/3-multi-site` | 200 networks (and 32 development networks) with the two variables at different sites; predictions recorded and protocol frozen before the final networks were trained |
| (pilot) | `studies/0-pilot` | A superseded pilot, not used in the manuscript, kept with its records |

Each study folder keeps its protocol and freeze manifest, its code and its results. Study 1 keeps the internal folder
names of its original layout (`02-Code`, `03-Data`, `04-Results`, `05-Notes`, `06-Study-History`), because its code
and its freeze manifest use paths relative to that folder. The freezes are local hash records written by the study
code, not public preregistrations. Changes after a freeze are listed in
`studies/1-compiled-programs/06-Study-History/V2-VERIFICATION-AMENDMENT.md` and `studies/2-trained-networks/AMENDMENTS.md`;
Study 3 has none.

## What is in this folder

| Path | Contents |
|---|---|
| `studies/` | The studies (above); after `make unpack`, also their saved outputs |
| `analysis/` | The constructed example of Section 3 and the post hoc comparison of test strategies (Section 7), with their results |
| `paper/make_numbers.py`, `paper/make_figures.py` | Every macro and generated table of the manuscript, and Figures 2 and 3, from the saved outputs; `paper/generated/` holds the values the manuscript was built with |
| `RESULTS.md` | The findings with their numbers |
| `data/saved-outputs/` | Saved outputs as compressed archives (see its README) |
| `tools/` | Unpacking, freeze checking and the replay of the saved-output analyses |
| `requirements.txt`, `ENVIRONMENT.md` | The exact environment (pinned packages, two libraries at fixed commits) |
| `RELEASE-CHANGES.md`, `release-changes.json` | Every file that differs from the original project, with the original's SHA-256, and every frozen file not shipped |

## Install

```sh
make setup        # .venv with CPython 3.12.14 and the pinned packages (see ENVIRONMENT.md)
```

This downloads the packages in `requirements.txt` (installed size about 1.3 GB, most of it PyTorch's CPU build and
jaxlib). The quick check needs only NumPy and SciPy; `ENVIRONMENT.md` gives a small route. Pass
`PYTHON=/path/to/python` to any `make` target to use another environment.

## Quick check (CPU, about a minute, no network)

```sh
make quick-check
```

It unpacks the saved outputs (checking every SHA-256), checks the three studies' freeze manifests against the files
they name, recomputes every macro and generated table of the manuscript (`paper/make_numbers.py --check`), and reruns
seven saved-output analyses in a temporary copy (Study 1's summary, the constructed example, the frozen analyses of
Studies 2 and 3, and three post hoc analyses), comparing each output with the shipped file byte for byte. Expected
end of the output:

```text
studies/1-compiled-programs/06-Study-History/V2-FROZEN-MANIFEST.json: 7 files: 7 unchanged
studies/2-trained-networks/FROZEN-MANIFEST.json: 6 files: 6 unchanged
studies/3-multi-site/FROZEN-MANIFEST.json: 9 files: 9 unchanged
total: 22 unchanged
...
7 analyses rerun from the saved records: every output identical to the shipped file
```

`paper/make_numbers.py --check` prints nothing when all 398 macros and the five tables match, and stops with
"out of date" otherwise.

`make verify` (full environment) adds Study 1's census, recomputed from the compiled networks and compared with the
saved one, and three more analyses: Study 1's worked example and panel simulation, and Study 2's held-out split
from its saved refits.

## More from the saved outputs

```sh
make numbers      # rewrite paper/generated/
make figures      # Figures 2 and 3 into paper/figures/
make vendor       # fetch the three third-party repositories Study 1 used (about 25 MB), for the CAE check
```

## What needs the trained networks

The trained weights and fitted alignments of Studies 2 and 3 (464 networks; about 110 MB compressed) are not
included because of the supplement's size limit. Their censuses are: every single and joint interchange decision of
every candidate alignment, which the frozen analyses, the joint-only and detection analyses and the manuscript's
numbers read. The steps that read the weights cannot run here: the NumPy verifiers of Studies 2 and 3
(`code/verify.py final`, which recompute every census decision; their saved results are
`results/final/verification.json`), Study 2's margin bound, Study 3's margin predictor and signed rule, and the
test-strategy comparison (`analysis/test_strategies.py`). Their saved results are in the studies' `results/` and
`posthoc/results/` folders and in `analysis/results/`. The networks can be trained again on a CPU:

```sh
cd studies/2-trained-networks && python code/run.py final   # about 25 minutes with 2 workers
cd studies/3-multi-site && python code/run.py final 3        # about 45 minutes with 3 workers
```

Both refuse to start if a frozen file has changed and skip networks that already exist. They train in float64 with
two CPU threads per worker; on another machine or library build the weights can differ in the last bits, so
retraining reproduces the distributions rather than every number. Study 1's compiled weights are bitwise
reproducible; its frozen pipeline reruns in a temporary copy with `cd studies/1-compiled-programs && python
02-Code/v2_fresh_run.py` (it needs at least 3 GiB of free memory and 10 GiB of free disk where the copy is made).

## Tables and figures

| Manuscript | Source |
|---|---|
| All numbers in the text | macros in `paper/generated/numbers.tex`, written by `paper/make_numbers.py` |
| Table 1 (compiled programs) | `paper/generated/table-compiled-candidates.tex` |
| Table 2 (Study 1 panels) | `paper/generated/table-compiled-panels.tex` |
| Table 3 (trained networks) | `paper/generated/table-trained-cells.tex` |
| Table 4 (two sites) | `paper/generated/table-multisite-cells.tex` |
| Table 5 (test strategies) | `paper/generated/table-strategies.tex` |
| Figure 1 | a diagram drawn in the manuscript's source |
| Figures 2 and 3 | `paper/make_figures.py` → `paper/figures/fig-compiled-panels.pdf`, `fig-trained.pdf` |

The values come from the studies' frozen analyses and verifications and from the post hoc analyses, which the
manuscript labels post hoc.

## What is not included

- The trained weights and fitted alignments of Studies 2 and 3 (above).
- The clones of Tracr, CAE and `rep_divergence` that Study 1 kept (fetch them with `make vendor`), the papers by
  other authors and the code excerpts the studies collected while reading, the original studies' written reports,
  notebooks and rendered figures, logs, and the original project's provenance records and manuscript source.
  Frozen files not shipped are listed with their SHA-256 in `RELEASE-CHANGES.md`.

## Changes from the original project

Files are copied byte for byte, except those listed in `RELEASE-CHANGES.md` with the original's SHA-256: records had
personal paths, other project folders' names and the names of software tools in working records replaced by placeholders (the assistants used are named in the top-level README.md, under "Use of AI assistants"), and local timestamps given in
UTC.

## Licence

MIT (`LICENSE`), covering the code and the studies' records and saved outputs. Tracr, CAE and `rep_divergence` are distributed under their own licences.
