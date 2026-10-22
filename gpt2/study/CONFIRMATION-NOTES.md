# AI-10: confirmation notes

Everything after the freeze (`freeze.json`, 17 October 2026, 12:29:49 UTC) is recorded here; the frozen files,
including `DEVELOPMENT-NOTES.md`, are not edited again. Local clock times are replaced by [local time] in this copy; times marked UTC are kept.

## 17 October 2026

- **[local time], frozen** (54 files: protocol with Amendments 1–2 and the addendum of Section 17, both prediction files and
  their records, configuration, prompts, code, cloud code, development settings and analyses, GPT-2 Medium's
  development records and the cost decision; AI-05's five inputs; GPT-2 Small's weights).
- **[local time], GPT-2 Small confirmation started on the local machine** as the systemd user service `ai10-confirmation`
  (`code/confirmation_service.sh`: `make confirmation`, then `make analysis`; logs `results/logs/confirmation*.log`).
- **[local time], arm d final started** on the local machine's CPU as `ai10-overlap-final` (`code/overlap_final_service.sh`:
  `make overlap-final`; log `results/logs/overlap-final.log`).
- **[local time], GPT-2 Medium confirmation started on Modal** (36 tasks, 4 containers; `cloud/run_stage.sh confirmation
  --model gpt2-medium`). The billing guard read $1.95 for the arm and $14.05 for the month; the billing watchdog
  `ai10-billing-watch` was restarted.
- **[local time]–[local time], GPT-2 Medium confirmation finished on Modal**; records pulled to `results/cloud/gpt2-medium/`
  (`runs/confirmation/`, `analysis/confirmation/summary.json`, `confirmation-gate.json`, `orchestration/`). 36 of the
  first 47 held-out draws passed GPT-2 Medium's gate; all 36 tasks ran, none failed. Arm spend by the billing report
  afterwards: $5.17 (month $17.26). The freeze still verified after the pull.
  Scored against `PREDICTIONS-BIGGER.md` (GPT-2 Medium; 180 alignments):
  - Prediction 1, joint-only share ≤ 0.10: **does not hold, beyond sampling error.** 6 alignments pass every single
    query (all `das_r2`), and all 6 fail a joint query: share 1.0 (Clopper–Pearson 0.54 to 1.0).
  - Prediction 2, margin rule: σ > 1 flags all 6 joint-only alignments (at least half: holds). No alignment passes
    every single and joint query, so the false-flag rate and the area under the ROC curve cannot be computed; those two
    parts are not testable.
  - Prediction 3, joint minus full-domain single detection at 120 calls < 0: **does not hold, beyond sampling error**:
    +0.640 (0.615 to 0.660). Section 17's comparison against all single queries: +0.159 (0.123 to 0.195).
  - Descriptive: layer chosen 21, the latest candidate (inside); held-out gate pass rate 77% (inside 50–95%); mean
    full-domain single failure rate of the DAS alignments 0.088 (just outside 0.00–0.08); TF32 check agreement 1.0
    (inside). Irreducible share of joint-panel failures 0.66.
- **[local time]–[local time], post hoc verifier written and run on the development records** (`posthoc/verify.py`, labelled post
  hoc as Section 17 said; `posthoc/results/verify-development.json`). An independent NumPy float64 GPT-2 Small, with no
  shared model code, recomputed natural states from the tokens and the margins of 8,180 saved queries (arms a, a′, b and
  c; every query within 0.05 of the tie band, every query of any joint-only alignment, and 2,000 random queries): no
  decision changed; largest margin difference 6.6e-5. It will be run on the confirmation records when they finish.
- **[local time], manuscript started** in `paper/` (AI-05/AI-09 style: sections, references copied from AI-05's checked
  bibliography, numbers to come from `paper/make_numbers.py`).
- **[local time]–[local time], arm d final finished** (`results/overlap-ai05/final/`, `results/analysis/overlap-final/summary.json`;
  200 of AI-05 Study 2's networks, 800 separately fitted alignments, 3.76 hours of CPU at nice 5). Scored against
  `PREDICTIONS.md`:
  - Prediction 6a, share of joint-only alignments among separately fitted alignments that pass every single query above
    Study 2's 7 of 127 (5.5%): **holds beyond sampling error**: 18 of 109 (16.5%, Clopper–Pearson 10.1% to 24.8%).
  - Prediction 6b, area under the ROC curve of ‖P_A P_B‖ for joint-only status above 0.5: **holds beyond sampling
    error**: 0.69 (0.55 to 0.82).
  - Also: the margin rule flags 14 of the 18 joint-only alignments and none of the other 91 (area under the ROC curve
    of the largest σ, 0.95). By kind: raw contrast subspaces (median overlap 0.65) 10 of 15 single-passing alignments
    fail jointly; DAS rank 2 (0.12) 7 of 46; rank 4 (0.34) 1 of 48; rank 1 (0.09) none pass every single query.

## 18 October 2026

- **[local time], GPT-2 Small confirmation finished** (`results/runs/confirmation/`), and its analysis at [local time]
  (`results/analysis/confirmation/summary.json`; `make analysis` re-checked the freeze). Only 35 of the 200 held-out
  draws passed the competence gate (17.5%; 59% of the failures had GPT-2 preferring the wrong name on some prompt), so,
  as Section 3 prescribes, the analysis uses those 35 tasks (210 arm-a alignments). Scored against `PREDICTIONS.md`:
  - Prediction 1 (joint-only share ≤ 0.10 at one site): **not testable**: no alignment passes all 648 single queries
    (fewer than the 10 needed). DAS rank 2, the best fit, still fails at least 9 single queries on every task.
  - Prediction 2 (margin rule): **not testable**, for the same reason.
  - Prediction 3 (joint minus full-domain single detection at 120 calls < 0): **does not hold, beyond sampling error**:
    +0.613 (0.582 to 0.641). Section 17's comparison with all single queries: +0.052 (0.017 to 0.084).
  - Prediction 4 (arm b minus arm a > 0): **does not hold, beyond sampling error**: −0.613 (−0.641 to −0.581). Arm b's
    own contrast is zero to rounding (−1.1e-18; both tests detect every arm-b alignment), so its "≥ 0" part holds.
  - Prediction 5a (arm c median overlap < 0.2): **holds**: 0.17 (range 0.003 to 0.61).
  - Prediction 5b (Spearman of overlap with the joint failure rate > 0): **does not hold, beyond sampling error**:
    −0.31 (−0.40 to −0.20).
  - Prediction 5c (arm c joint-only share ≥ arm a's): **not testable** (no alignment passes every single query in
    either arm).
  - Predictions 6a and 6b: hold beyond sampling error (17 October, arm d).
  - Descriptive: layer chosen 10 (inside); mean full-domain single failure rate of arm a's DAS alignments 0.084 (just
    outside 0.00–0.08); share of those alignments passing every single query 0 (outside 0.2–0.8); irreducible share
    0.62 (outside ≤ 0.20); arm a′ token failure rate 0.52 and no alignment passing every single query (inside); arm b
    token 0.999 against position 0.28 (inside); held-out gate pass rate 17.5% (outside 40–90%).
  - Mechanism (descriptive, Section 17): DAS rank 2 and rank 4 fail no full-domain single query of either variable but
    fail 12.4% and 50.3% of the position queries whose donor has the recipient's position, and 15.2% and 32.9% of the
    joint panel.
- **[local time], correction to the GPT-2 Medium scoring of 17 October, [local time]** (appended, not edited): PROTOCOL.md
  Section 9 makes a prediction about joint-only alignments not testable when fewer than 10 alignments pass every
  single query, and in GPT-2 Medium 6 do. So GPT-2 Medium's predictions 1 and 2 are **not testable**, not "does not
  hold" and "holds in part". What was observed stays as description: all 6 of those alignments fail a joint query
  (share 1.0, Clopper–Pearson 0.54 to 1.0), and the margin rule flags all 6. Prediction 3 for GPT-2 Medium is unchanged:
  it does not hold, beyond sampling error.
- **[local time]–[local time], post hoc verification of the GPT-2 Small confirmation** (`posthoc/results/verify-confirmation.json`):
  31,447 queries across arms a, a′, b and c, recomputed by the independent NumPy float64 GPT-2 Small; no decision
  changed; largest margin difference 1.0e-4. (GPT-2 Medium's records were not re-verified: its arrays stay on the
  Modal Volume, and the cloud check stage reproduced GPT-2 Small's saved margins within 6.1e-5.)
- **[local time], `RESULTS.md` written** from the saved analyses; manuscript sections drafted in `paper/`.
- **[local time], rounding correction** (appended): GPT-2 Small's prediction-3 interval is +0.582 to +0.640 (upper bound
  0.64047), not +0.641 as written at [local time] above; RESULTS.md corrected.
- **[local time], rounding correction** (appended): GPT-2 Medium's comparison with all single queries is +0.158 (exact
  0.15850), not +0.159 as written on 17 October at [local time]; RESULTS.md corrected. The paper's numbers are generated from
  the saved analyses, not typed.
- **[local time], manuscript first draft** (`paper/manuscript.pdf`, 10 pages): sections in AI-05's style, every number and the
  predictions table generated by `paper/make_numbers.py` from the saved outputs; `make check-paper` passes (generated
  files current, macros and citations resolve, clean LaTeX log, 9.1 negations per 1,000 words, British spelling,
  typed literals listed). Every prediction is reported as frozen, with the not-testable cases and nulls stated, and the
  addendum's comparison labelled development-informed.
- **[local time], correction of the times above** (appended, not edited): the entries stamped [local time], [local time]–[local time], [local time], [local time],
  [local time] and [local time] were not read from the clock. File times show that all of that work fell between the confirmation's
  end at [local time] and [local time], when this file was last saved before this entry: `posthoc/results/verify-confirmation.json`
  was written at [local time], `RESULTS.md` at [local time], and `paper/generated/` was regenerated by [local time]. The contents
  of those entries stand.
- **[local time]–[local time], review copy of the manuscript** (`paper/` only; no frozen file, study output or result changed):
  blank author line; the set's shared AI-use statement, word for word, directly before the references; PDF metadata
  emptied (Author, Creator and Producer empty; no dates, time zone or trailer ID); the companion study cited as an
  anonymous manuscript in preparation; three references replaced by checked entries from a shared bibliography, and a
  URL or arXiv number added to two others; `make references` matched all 16 entries online. Wording: "Section A"
  became "Appendix", "approved as drafted" became "approved as written", "the local machine" became "the computer", and the
  "Findings" heading was kept on the page of its list. `make check-paper` passes (0 LaTeX or BibTeX warnings, 8.9
  negations per 1,000 words); all 10 pages were looked at. Byte copy:
  `REVIEW-COPIES/2026-10-17/joint-and-single-interchange-tests-in-gpt-2.pdf` (SHA-256 `29ef3745…0bc9`), listed in
  that folder's INDEX.md and MAP.md. Not pinned into any dependent document.
