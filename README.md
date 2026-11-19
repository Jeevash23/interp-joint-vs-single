# When can a joint intervention reject an alignment that single interventions accept? From small networks to GPT-2

Code and results for the preprint *When can a joint intervention reject an alignment that single interventions accept? From small networks to GPT-2*, by Richard Jeevash (link to be added when it is posted).

This repository holds the code, frozen inputs, study records and saved outputs behind it, and the preprint's LaTeX
source; the saved outputs and the frozen protocols are attached to its v1.0 release (see [Release
files](#release-files)). Every number and generated table in the manuscript can be recomputed from the saved
outputs on a CPU, without a model or network access.

## Release files

The saved outputs and the frozen protocols are not in this repository's files; they are attached to its v1.0 release:

| File | Size | Contents |
|---|---|---|
| `per-run-outputs.tar.gz` | 69.8 MB | The saved outputs: the 6 compressed archives in `gpt2/data/saved-outputs/` and `small-networks/data/saved-outputs/`, which `make unpack` places in the study folders and checks file by file |
| `frozen-protocols.tar.gz` | 42 kB | The studies' protocols, predictions, amendments, approvals and freeze records (19 files), at their places in the folder, where the checks read them |
| `SHA256SUMS` | | Checksums of the two archives |

Download the three files into an empty folder, check them there, and unpack them into this one:

```sh
sha256sum -c SHA256SUMS
tar -xzf per-run-outputs.tar.gz -C /path/to/interp-joint-vs-single
tar -xzf frozen-protocols.tar.gz -C /path/to/interp-joint-vs-single
```

After unpacking, `sha256sum -c SHA256SUMS` in this folder checks every file that its `SHA256SUMS` lists, the
unpacked attachments included (each part keeps its own list), and the checks below run as described.

## What the paper asks

Interchange interventions test whether a network's internal states carry the variables an explanation assigns to them,
usually one variable at a time. For two variables aligned to orthogonal subspaces at one site, a joint edit is one
single edit applied after the other, so a joint test can reject what single tests accept only when both variables'
single edits give correct outputs without reproducing the natural states they simulate, and only where these misses
move the output's margin by more than the single queries' own margins. Studies 1–3 measure how often this happens in
compiled and small trained networks, and how joint and single tests compare at equal budgets. Study 4 asks the same
questions in GPT-2 Small and GPT-2 Medium on indirect-object identification, under a protocol whose predictions were
hashed before any model was run.

## Layout

| Folder | Studies | What it is |
|---|---|---|
| `small-networks/` | 1–3 | The complete release of the three small-network studies: 80 compiled Tracr programs, 200 trained transformers and MLPs, 200 networks with the variables at different sites, their protocols, freeze records, code and saved outputs. Its README and ENVIRONMENT.md describe it. |
| `gpt2/` | 4 | The complete release of the GPT-2 study: protocol, hashed predictions, amendments, prompts, code, cloud code, freeze record and saved outputs, plus the Study 2 networks that its arm d reuses (`gpt2/reference/`). Its README and ENVIRONMENT.md describe it. Inside it, "the companion study" means Studies 1–3 of the paper. |
| `paper/` | all | The preprint's LaTeX source (`main.tex`, `sections/`, `references.bib`) and its generated numbers and tables (`generated/`), with `check_numbers.py`, which checks them against the two parts. |
| `tools/check_sums.py` | | Checks the SHA-256 of every listed file, here and in both parts. |

The two parts were built and checked as separate releases; each keeps its own `SHA256SUMS`,
`RELEASE-CHANGES.md` and `make quick-check`. The top-level `SHA256SUMS` lists the files outside the two parts.
`gpt2/` ships Study 2's census and alignment records again under `gpt2/reference/` (it needs them next to the 232
network weight files that arm d reads), so about 9.5 MB of uncompressed records appear in both parts.

## Checking

First unpack the two release attachments into this folder (see [Release files](#release-files)): the checks read
the saved outputs and the freeze records they contain.

```sh
make setup          # each part's environment (see small-networks/ENVIRONMENT.md and gpt2/ENVIRONMENT.md)
make quick-check    # checksums, both parts' quick-checks, and the paper's generated files against the two parts
```

`make paper` draws the figures from `small-networks/` (needs matplotlib) and builds `paper/build/main.pdf` from the
preprint's LaTeX source in `paper/`.

`make quick-check` takes a few minutes on a CPU. Set `PYTHON=...` to use one existing Python for both parts, or
`PYTHON_SMALL=...` and `PYTHON_GPT2=...` for one each. The longer checks (Study 1's census recomputed from the compiled
networks, Study 4's unit tests, model downloads for reruns) are run inside each part; see their READMEs.

## Licence

MIT (`LICENSE`), covering the code and the studies' records and saved outputs; each part carries the same file.

## Study codes

The records and code name the studies by the codes they had while the work was done. Each code that occurs here,
and where that study is released:

| Code | Released in |
|---|---|
| AI-01 | `interp-answer-repair` |
| AI-03 | `interp-donor-pooling` |
| AI-04 | `interp-arithmetic-patching` |
| AI-05 | this repository |
| AI-06 | `interp-flyvis-first-order` |
| AI-08 | not public yet |
| AI-09 | `interp-flyvis-first-order` |
| AI-10 | this repository |

## Use of AI assistants

As the preprint states:

> Language-model assistants were used throughout this work, under the author's
> direction, to help design the studies, to write and run the code for the experiments and analyses, to draft and
> revise the text and to check the references against their online records. All reported results come from the
> saved study outputs, and the numbers in the text were generated from those outputs or checked against them.

In the working records shipped here, the names of software tools were replaced by placeholders such as
`[redacted]`.

## Citation

Please cite the preprint (the link will be added here when it is posted) and, for the code and results, this
repository; [CITATION.cff](CITATION.cff) has the details.

```bibtex
@software{interp_joint_vs_single_2026,
  author  = {Jeevash, Richard},
  title   = {{When can a joint intervention reject an alignment that single interventions accept? From small networks to GPT-2: code and results}},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/Jeevash23/interp-joint-vs-single}
}
```
