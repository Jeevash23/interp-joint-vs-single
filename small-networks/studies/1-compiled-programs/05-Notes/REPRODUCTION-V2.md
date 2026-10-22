# Reproduce AI-05 V2

## What was completed

80 final Tracr programs, 480 candidate explanations, 172,800 saved assay queries, an exact official Grant demonstration, a CAE qualification failure plus labelled adaptation, independent numerical verification, and a two-program clean-environment replay. The final report is `01-Papers/AI-05-REPORT-V2.pdf`. The older unsuffixed report and scripts are the preserved preliminary pilot.

## Inspect existing evidence without recomputing

- `04-Results/v2/primary-table.csv`: final-budget rates and actual call counts.
- `04-Results/v2/candidate-panel-results.csv`: all 80 programs, six procedures, eight panels and four budgets.
- `04-Results/v2/final/*.json`: native gates, discovery cost, independent reference-label witnesses, controls and panel summaries.
- `04-Results/v2/final/*-queries.jsonl.gz`: exact recipients, donors, targets, observed/expected outcomes and cost.
- `03-Data/v2/final/*/`: source-program metadata, compiled weights, adapter rotation and learned/control projectors.
- `04-Results/v2/independent-verification.json` and `margin-audit.json`: separate execution, reaggregation, numerical ambiguity and the exhaustive two-program composition checks.
- `04-Results/v2/clean-environment-replay.json`: fresh installation, recompilation and exact replay with saved candidates; **does not retrain DAS**.
- `06-Study-History/V2-FROZEN-MANIFEST.json`: source hashes and final seed allocation fixed before final generation.
- `06-Study-History/V2-VERIFICATION-AMENDMENT.md`: disclosed post-freeze margin audit. Original outcomes and frozen sources are unchanged.

All commands below run from the AI-05 project root. They only write within this project or an explicitly created temporary reproduction directory. Scientific jobs are serial, CPU only, affinity two cores with numerical thread caps of two. Admission requires 3 GiB available RAM and 10 GiB free disk; a worker stops below 2 GiB available RAM. Do not run the example jobs concurrently.

## Verify the saved run using the existing environment

```bash
.venv-v2/bin/python 02-Code/v2_analyze.py
.venv-v2/bin/python 02-Code/v2_verify.py
```

These regenerate only derived analysis/audit files; timings may vary. Exact categorical equality is deliberately **not** claimed for near ties: the independent verifier checks the tied-score gap and reports robust/indeterminate decisions. To repeat the saved clean replay:

```bash
.venv-v2-clean/bin/python 02-Code/v2_clean_replay.py
```

## Install an isolated full environment elsewhere

Use CPython 3.12.14 and `uv`. `requirements-v2.lock` is the original environment freeze and contains local source URLs; `requirements-v2-portable.txt` has those entries and Torch removed for portable installation. All remaining versions are exact. Install CPU Torch separately to prevent CUDA dependency downloads.

```bash
uv --no-cache venv --python python3.12 .venv-reproduction
uv --no-cache pip install --python .venv-reproduction/bin/python -r 02-Code/requirements-v2-portable.txt
uv --no-cache pip install --python .venv-reproduction/bin/python --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1+cpu'
uv --no-cache pip install --python .venv-reproduction/bin/python --no-deps --no-build-isolation 02-Code/vendor/tracr 02-Code/vendor/cae
```

The vendored source trees are pinned in `V2-SOURCE-ENVIRONMENT-MANIFEST.json`. If fetching them anew, use the recorded official repository URLs and exact commit IDs, retaining their licenses. The tested environment is Linux x86_64; other platforms are not qualified by these results.

## Repeat the full final study without overwriting original records

The frozen runner resumes completed programs, so running it in the original project skips existing results. The helper below creates a fresh temporary project containing frozen code, development model metadata and required history, then uses the installed packages to regenerate all final models, candidates, queries and analysis. The temporary output path is printed. This spends additional disk and compute; check available resources before running.

```bash
.venv-v2/bin/python 02-Code/v2_fresh_run.py
```

No old run is deleted. The fresh run uses the same exact source-hash gate and seeds. After it completes, compare its aggregate counts with the original; wall-clock profiles and near-tie category choices can differ across environments, so compare the disclosed margin-aware results as well as frozen binary counts.

## Official reproduction commands

```bash
.venv-v2/bin/python 02-Code/v2_official_grant.py
.venv-v2/bin/python 02-Code/reproduce_official_v2.py
```

The first runs the official section 4.2 sanity script unchanged. The second uses the official CAE controlled factory and metric implementations, with 20 runs of 500 samples rather than the source CLI's hardcoded 100 runs. Its separately labelled adaptation restricts the intervention domain to X; the original full-domain result is retained as a failed qualification gate. Both commands refresh their own reproduction records, so copy those records first if preserving identical runtime metadata matters.

## Rebuild figures/report

```bash
.venv-v2/bin/python 02-Code/v2_figures.py
```

The PDF builder `02-Code/v2_report_pdf.py` needs ReportLab and the installed Noto Sans font. The completed run used the app's bundled ReportLab 4.4.9 runtime; rendering used Poppler. The report source is Markdown. Re-render and visually inspect all pages after edits; scientific verification does not substitute for layout inspection.

## Known constraints and preserved failures

Haiku's optional-Flax stub can break Torch's introspection when imported first. `v2_candidates.py` imports Torch and its lazy optimizer machinery before Tracr; no source package was patched. Numerical aggregation of nonbinary values is unsupported by the pinned Tracr compiler, so additive programs select categorical tokens before mapping their values. The final optimizer uses a development-fixed convergence rule to prevent numerical drift from an exact solution. The exact-output independent replay failed in 257 near-tie cases; the post-freeze audit is mandatory, not optional cleanup.

This is local reproducibility evidence. It does not constitute external scientific review or a clean independent retraining of DAS.
