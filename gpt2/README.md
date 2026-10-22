# Joint and single interchange tests in GPT-2: code release

This folder holds the code, frozen inputs, study records and saved outputs behind the manuscript *Joint interchange
tests find failures that single tests miss in GPT-2's indirect-object circuit*. Every number and generated table in the
manuscript can be recomputed from the saved outputs on a CPU in under a minute, without a model or network.

## What the paper asks

A companion study showed that, for two variables aligned to orthogonal subspaces at one site, a joint interchange edit
is one single edit applied after the other, so a joint test can reject an alignment that single tests accept only when
both single edits miss the natural states they simulate. It measured how often that happens, how well a first-order
margin rule predicts it, and how joint and single tests compare at equal budgets, but only in compiled and small trained
networks. This study asks the same questions in GPT-2 Small on indirect-object identification, with the two variables
of MIB's IOI model: the indirect object's position (which name repeats) and its identity (which names fill the first
clause). It adds MIB's cross-site head alignment and two arms on the overlap of separately fitted subspaces, one in
GPT-2 and one (arm d) in the companion study's own trained networks, and repeats the one-site arm in GPT-2 Medium on a
cloud GPU.

## The study

`study/` holds the protocol (`PROTOCOL.md`, with the approved text in `PROTOCOL-approved-2026-10-17.md`), the
predictions hashed before any model ran (`PREDICTIONS.md`, `PREDICTIONS-RECORD.json`), the bigger-model predictions of
Amendment 2 (`PREDICTIONS-BIGGER.md`, `AMENDMENT-2-RECORD.json`), the configuration, the prompt draws, the code, the
cloud code, the development notes (frozen), the freeze record (`freeze.json`: 54 files, the companion study's five
inputs and GPT-2 Small's weights digest) and everything after the freeze (`CONFIRMATION-NOTES.md`, with every
prediction scored). The freezes are local hash records written by the study code, not public preregistrations.

## What is in this folder

| Path | Contents |
|---|---|
| `study/` | The study (above); after `make unpack`, also its `results/` |
| `reference/joint-vs-single-interchange/` | The companion study's files that the code reads, and its Study 2 networks for arm d (see its README) |
| `posthoc/` | The post hoc NumPy float64 verifier and its results |
| `paper/make_numbers.py` | Every macro and generated table of the manuscript, from the saved outputs; `paper/generated/` holds the values the manuscript was built with |
| `RESULTS.md` | The findings, with every prediction scored |
| `provenance/Makefile-at-freeze` | The study's own Makefile, which the freeze lists (this release's Makefile replaces it) |
| `data/saved-outputs/` | Saved outputs as compressed archives (see its README) |
| `tools/`, `models/` | Unpacking, freeze checking, the replay of the analyses, and the model download helper |
| `pyproject.toml`, `uv.lock` | The exact environment of the runs (see `ENVIRONMENT.md`) |
| `RELEASE-CHANGES.md`, `release-changes.json` | Every file that differs from the original project, with the original's SHA-256, and every frozen file not shipped |

## Install

```sh
make setup        # uv sync --frozen --extra experiment --group dev: creates .venv from uv.lock
```

This downloads about 3.0 GB of packages (most of it PyTorch's CUDA libraries); the quick check needs only NumPy, SciPy and
tokenizers, and `ENVIRONMENT.md` gives a small CPU-only route. Pass `PYTHON=/path/to/python` to any `make` target to use
another environment.

## Quick check (CPU, under a minute, no model, no network)

```sh
make quick-check
```

It unpacks the saved outputs (checking every SHA-256), checks the freeze records against the files they name, recomputes
every macro and generated table of the manuscript (`paper/make_numbers.py --check`), and reruns the study's analysis for
GPT-2 Small's development and confirmation runs and for arm d's development and final networks in a temporary copy,
comparing each output with the shipped file byte for byte. Expected end of the output:

```text
total: 12 changed for the release (documented), 6 superseded by a recorded amendment, 68 unchanged
generated files match
...
4 analyses rerun from the saved records: every output identical to the shipped file
```

"Superseded by a recorded amendment" are six files that changed between the Amendment 2 record and the freeze, which
records their final digests.

```sh
make test         # the study's unit tests (PyTorch and transformers, make setup; GPT-2 Small's tokeniser:
                  # python tools/get_models.py --no-weights gpt2-small, a few MB)
make numbers      # rewrite paper/generated/
```

## What needs the per-task arrays or a model

Each GPT-2 task saved its subspace bases, margins and margin-rule scores as arrays (`arm-*.npz`, 162 MB in all). The
release keeps every task's JSON record, from which the analysis is computed, and reduced copies of the `arm-a.npz` files
with the four arrays the number script reads, but not the full arrays. The post hoc NumPy verifier (`posthoc/verify.py`),
which rebuilds GPT-2 Small in float64 from its weights and recomputes the margins of saved queries, needs the full arrays
and the checkpoint; its saved results are `posthoc/results/verify-development.json` and `verify-confirmation.json` (no
decision changed).

Model reruns:

```sh
make models           # GPT-2 Small at the pinned revision (about 0.55 GB)
```

Then follow the stages of `provenance/Makefile-at-freeze` and `study/PROTOCOL.md` (pilot, development, freeze,
confirmation, analysis); each stage refuses to overwrite its records, so run them in a copy of the release with the
stage's results removed. GPT-2 Small ran on one 6 GB NVIDIA GPU under a GPU lock file, now named by
`GPU_LOCK` (create it first, for example `touch gpu.lock`): the confirmation, 200 held-out draws through the competence
gate and 35 tasks, took about 10 hours. Arm d reruns on the CPU from the companion study's networks shipped here:

```sh
make overlap-final COPY=../arm-d-rerun     # about 4 hours of CPU (3.76 hours in the original run)
```

### The bigger-model arm on Modal (optional, your own account)

From `study/cloud/`, with the Modal CLI logged in to your own account, `run_stage.sh <stage> --model gpt2-medium` starts
each stage that `study/PROTOCOL.md` Section 16 lists (check, development, confirmation), and `watch_billing.sh` stops the
arm's apps at its cost limits. The image is built from this folder's `pyproject.toml` and `uv.lock`, the weights are
downloaded inside it and checked against their digests, and the records go to a Modal Volume created in your workspace;
`modal run ai10_modal.py::pull --stage <stage>` copies them back. The original run used H100 GPUs: GPT-2 Medium's development and confirmation (36 tasks,
four containers) cost $5.17 on Modal's billing report.

## Tables

| Manuscript | Source |
|---|---|
| All numbers in the text | macros in `paper/generated/numbers.tex`, written by `paper/make_numbers.py` |
| Table 1 (one site) | written in the manuscript's source from those macros |
| Table 2 (predictions) | `paper/generated/table-predictions.tex` |

The values come from the study's analyses (`study/results/analysis/`), the bigger-model arm's records
(`study/results/cloud/`), the post hoc verifier's results, the reduced `arm-a.npz` arrays and the companion study's
manuscript numbers. The post hoc analyses are labelled post hoc in the manuscript.

## What is not included

- The full per-task arrays (above), GPT-2's weights (download them; see `models/README.md`).
- Logs, the study's systemd service wrappers (the Makefile stages run the same commands), and the manuscript and its
  source. Frozen files not shipped are listed with their SHA-256 in `RELEASE-CHANGES.md`.

## Changes from the original project

Files are copied byte for byte, except those listed in `RELEASE-CHANGES.md` with the original's SHA-256: records had
personal paths, other project folders' names, cloud object ids and the names of software tools in working records replaced by placeholders (the assistants used are named in the top-level README.md, under "Use of AI assistants"), and local clock
times removed (UTC times are kept); the code reads the companion study from `reference/joint-vs-single-interchange`,
GPT-2 Small from `MODELS_DIR` and the GPU lock from `GPU_LOCK`; the cloud app builds its image from this folder; the freeze
check accepts the documented changes; the `arm-a.npz` files are reduced; and this release's Makefile replaces the
study's.

## Licence

MIT (`LICENSE`), covering the code and the studies' records and saved outputs. GPT-2 is distributed under its own licence (MIT).
