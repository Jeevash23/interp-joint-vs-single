# AI-10: development notes

Every change to the code or settings between the author's approval and the freeze is listed here, with its reason
(PROTOCOL.md Section 11). Local clock times are replaced by [local time] in this copy; times marked UTC are kept.

## 17 October 2026

- **[local time], approval recorded.** The author approved the protocol as drafted, with 40 confirmation tasks and the sign rule
  primary; decision 5 (reading MIB online) was not answered, so nothing was read online. PROTOCOL.md's opening paragraph
  and Section 14 now say so; the approved text is kept read-only as `PROTOCOL-approved-2026-10-17.md`, and `freeze.py`
  hashes it at the freeze. `config.json`'s description line was updated to match. No other content changed.
- **02:34:58 UTC, predictions hashed** (`PREDICTIONS-RECORD.json`, read-only): PREDICTIONS.md, PROTOCOL.md,
  config.json and the three prompt files. No model had been run.
- **[local time], before the pilot: one code fix.** In `gpt2.py`, `HeadSites` stored head 8.10's live z as a view of the
  tensor it then overwrote, so in a joint query the stored value would have been the replaced one. It now stores a copy.
  The census and the margin rule took w from single A-queries, where nothing is replaced, so no planned quantity would
  have changed; the fix guards later uses. The safetensors header was read (key names and shapes only, nothing loaded)
  to confirm that all 160 weight names match `gpt2.py`. `make test`: 7 tests pass.
- **[local time], pilot queued.** `code/pilot_after_ai09.sh` runs as the systemd user service `ai10-pilot`: it waits until no
  process is using the GPU (AI-09's confirmation run was on its last models), then runs `make pilot` under the
  portfolio lock and a sleep inhibitor, retrying every 5 minutes while the GPU or RAM is busy. Log:
  `results/logs/pilot.log`.
- **[local time]–[local time], pilot** (`results/pilot.json`, log `results/logs/pilot.log`; GPU lock acquired with 5,852 MiB free).
  Peak GPU memory 1.57 GB. Draw 001 passed every check. Draw 000 missed one: our forward pass and transformers'
  GPT2LMHeadModel, both float32 on the GPU, differed by 1.37e-4 in a margin, against the protocol's 1e-4 (every logit
  was within 1.45e-4, inside the 1e-3 limit). Every null edit reproduced the natural margins within 3.1e-5 (one-site and
  final-position sites at layers 6, 8, 9 and 10, and the head sites). Measured DAS seconds per step (rank 2, 378
  queries, 100 steps): 0.268 at layer 6, 0.167 at 8, 0.118 at 9, 0.062 at 10; one census of 6,480 queries took 3.7 to
  5.5 s. No failure count was recorded.
- **[local time], diagnostic of the missed check** (`results/pilot-parity-float64.json`, `code/pilot_parity_float64.py`; CPU,
  the same two draws, no outcome). `gpt2.py` gained a `dtype` argument (default float32, so nothing else changed). In
  float64 the two implementations agree to 1.1e-13 in every logit and 5.7e-14 in every margin; in float32 on the CPU to
  5.3e-5 and 2.3e-5. The GPU gap is therefore float32 rounding (logits reach 113 in size), not a difference in the
  computation. The 1e-4 limit was too tight for two float32 implementations on the GPU. Proposed to the author: the
  parity gate becomes float64 agreement within 1e-8 on the CPU plus float32 agreement within 1e-3 (logits and margins)
  on the GPU. Not applied until the author decides; development has not started.
- **[local time], Amendment 1 approved and applied** (PROTOCOL.md Section 15): the parity gate is float64 agreement within
  1e-8 on the CPU plus float32 agreement within 1e-3 on the GPU, in every logit and margin; the null-edit limit stays
  1e-4. `pilot_checks.py`'s pass rule now applies the float32 limit. Evaluated on the saved pilot records, without
  rerunning anything, the amended gate passes (`results/pilot-gate-amendment-1.json`).
- **[local time]–[local time], smoke test before the long run** (outputs in the session's scratch folder, not in the project; 3 to 4
  DAS steps, so its alignments mean nothing and no outcome was read). The settings rules, every arm (a, c, a′, b), the
  analysis and arm d (one of AI-05's development networks, 21 s) ran end to end. Lemma 1's split held to 1.8e-15 and
  AI-05's form of u to 2.7e-14 at the GPT-2 site. Per task, everything except DAS training took about 135 s. One code
  fix: `choose_settings` now passes the configured step candidates to DAS as its checkpoints (before, it relied on the
  defaults matching the configuration). Nothing in AI-05 changed.
- **[local time], development started** as the systemd user service `ai10-development` (`code/development_service.sh`):
  `make development`, then `make analysis-development`, then `make overlap-development`. Logs in `results/logs/`.
- **[local time], status check after the desktop app crashed at about [local time]** (the run is a systemd service and was not
  affected; nothing was restarted). `run_study.py development` had run 2 h 19 min (1 h 59 min of CPU, 1.8 GB of GPU
  memory) and was still in the settings search, which writes `results/development-settings.json` only at its end. From
  the pilot's timings that search takes about 1.6 h if the steps rule picks 500 steps, 2.2 h at 1,000 and 2.9 h at
  1,500, so at [local time] it had picked 1,000 or 1,500. Gap to fix before the freeze, not during the run: the runner prints no
  progress during the settings search. The Makefile's header comment, which still said "draft", was updated.
- **[local time]–[local time], run paused for a power cut** at the author's request (`systemctl --user freeze`, then thaw); nothing was
  restarted, and the process resumed with the GPU active. At [local time], after about 2 h 46 min of active running, the
  settings search had still not written its file, so the steps rule must have picked 1,500 steps (1,000 would have
  finished by about [local time]). Since [local time] the run shares the CPU with a connectomics job and AI-03's CPU test.
- **About [local time], settings chosen** (`results/development-settings.json`, file time [local time] after the pause): tie band
  1e-3 (numerical noise 4.6e-5), learning rate 0.001, 1,500 steps, layer 10. All four candidate layers gave a mean
  full-domain single failure rate of 0 for rank-2 DAS, so the tie rule chose the latest. Two discrepancies with the
  protocol, found on reading the record, neither changing a rule's outcome as far as can be seen: (1) the learning-rate
  search ran at layer 9, because the code takes the third of the four candidates as the "middle" one, while Section 11
  says layer 8 (0.001 won by a wide margin: mean best loss 8.6e-5 against 1.3e-4 and 2.0e-4); (2) the file's
  `written_at_utc` (03:40:29 UTC) is when the search started, not when it ended, because the
  timestamp is evaluated before the search runs. Development tasks then took about 23 minutes each after the first.
- **[local time]–[local time], code for a possible bigger-model arm (not launched; nothing spent).** After a request relayed from the
  coordinating session to repeat the test in larger models on Modal, the following were written and tested locally.
  `run_study.py` gained options that default to the GPT-2 Small behaviour: `TRAIN_TF32` (off) with a `fitting` wrapper
  around every DAS fit, a choice of arms and of the held-out split in `run_task`/`run_phase`, candidate layers as an
  argument of `choose_settings`, and the settings timestamp now taken after the search. `gpt2.py`'s `from_snapshot`
  reads the head count from the model's configuration. The local run had loaded its code at [local time] and is unaffected;
  `make test` still passes (7 tests). New folder `cloud/`: configuration, `cloud_jobs.py`, the Modal app
  `ai10_modal.py`, `run_stage.sh`, `watch_billing.sh` and `test_cloud.py`, which ran every job on tiny random GPT-2
  models on the CPU and passed (`cloud/logs/test_cloud.log`). The protocol amendment and predictions for that arm are
  not written yet: the cloud spending needs the author's direct confirmation first.
- **About [local time], Amendment 2 confirmed by the author** (directly, in this session): run the bigger-model arm on Modal
  as designed, up to $6 for the arm, with hard stops at $6 for the arm and $27 for the account's month. PROTOCOL.md
  Section 16 and `PREDICTIONS-BIGGER.md` written; `freeze.py amendment2` and `make amendment2` added; the freeze now
  also covers the cloud code, each model's development records and the cost decision. Recorded and hashed in
  `AMENDMENT-2-RECORD.json` before any cloud computation and before any GPT-2 Small joint-query outcome was read.
- **Correction to the two entries above** (appended, because this file is hashed in `AMENDMENT-2-RECORD.json`): their
  times were estimates. The cloud code was written and tested between about [local time] and [local time], the author confirmed the
  arm at about [local time], and `AMENDMENT-2-RECORD.json` was written at 10:53:07 UTC.
- **[local time]–[local time], cloud check stage passed** (`results/cloud/check.json`; NVIDIA H100 80GB; about one minute of GPU
  time). (i) The 648 single-query margins of the local machine's saved `das_r2` alignment of development task 000 were
  reproduced within 6.1e-5 (limit 1e-3). (ii) Parity: GPT-2 Medium 3.4e-13 (float64) and 1.5e-4 (float32) in logits;
  GPT-2 Large 7.1e-14 and 4.0e-5; both pass Amendment 1's gate. (iii) TF32 training on the same alignment: best loss
  0.0018337 against the local machine's 0.0018340 in float32, and all 648 single decisions equal. (iv) Seconds per TF32 DAS
  step (rank 2) at the candidate layers: GPT-2 Medium 0.027, 0.017, 0.013, 0.008 (layers 13, 17, 19, 21); GPT-2 Large
  0.049, 0.029, 0.021, 0.012 (layers 20, 26, 29, 32); a census took 0.9–2.3 s. Arm spend after the check, including
  the image build: $0.12.
- **[local time], GPT-2 Medium development started on Modal** (`run_stage.sh development --model gpt2-medium`).
- **[local time]–[local time], GPT-2 Medium development on Modal** (`results/cloud/gpt2-medium/`). Settings (`development-settings.json`,
  search 943 s): tie band 1e-3 (noise 3.8e-5), learning rate 0.001, 1,500 steps, layer 21; as in GPT-2 Small, rank-2
  DAS failed none of the full-domain single queries at any candidate layer, so the latest layer was chosen. Arm a then
  ran on the first 8 development draws that passed GPT-2 Medium's gate, and was analysed in the cloud.
- **[local time], cost decision** (`results/cloud/cost-decision.json`, rule of Section 16): arm spend $1.95 by the billing
  report (month $10.56); GPT-2 Medium 64.9 s per task (1.56 times the check-stage projection). Forty confirmation tasks
  would bring the arm to about $6.0, above the $5.75 limit, so GPT-2 Medium gets 36, the largest number that fits
  (projected arm total $5.68); the shortfall of 4 tasks will be reported. GPT-2 Large is not run: its development and
  confirmation were projected at $3.18 and $5.39.
- **[local time], GPT-2 Small development finished** (8 tasks: draws 0, 1, 3–8; draw 2 failed the gate with a smallest
  margin of 0.12), and its analysis (`results/analysis/development/`). Main observations, development data only:
  - *Arm a.* No alignment passes all 648 single queries, so prediction 1 and the margin rule would not be testable on
    these tasks. Rank-2 and rank-4 DAS fail none of the full-domain single queries (the panel they are trained on, in
    which the swapped variable changes), and rank-2 DAS trained on half of them fails none of the other half; but they
    fail single position queries whose donor has the same position and other names (16% at rank 2, 50% at rank 4), and
    18.6% and 36.1% of the joint panel. So the position subspace also carries name information, which only queries
    outside the full-domain panel reveal. Joint minus full-domain single detection at 120 calls: +0.659 (bootstrap
    interval +0.646 to +0.666), against prediction 3's negative sign; against all 648 single queries instead
    (descriptive, not prespecified) +0.078. Irreducible share 0.63.
  - *Arm a′* (final position): DAS alignments fail 55% (position) and 52% (token) of full-domain single queries, as
    predicted. *Arm b* (MIB's heads): token queries at head 8.10 fail 100%, position queries 22%; joint panel 95%.
    *Arm c*: ‖P_A P_B‖ median 0.13 (0.004–0.60); Spearman with the joint failure rate −0.04 (−0.34 to 0.26).
- **GPT-2 Medium development, same observations** (`results/cloud/gpt2-medium/analysis/development/`): 2 alignments
  pass every single query (both `das_r2`), and both fail joint queries (63 and 531 of 5,832); σ > 1 flags both and no
  other such alignment. Joint minus full-domain single detection +0.598; against all single queries (descriptive) +0.146.
- **[local time]–[local time], arm d on AI-05's 32 development networks** (`results/overlap-ai05/development/`,
  `results/analysis/overlap-development/summary.json`; CPU, about 41 s per network, so the 200 final networks would take
  about 2.3 hours, not the 30–60 minutes of Section 10). Of 128 separately fitted alignments, 21 pass every single query
  and 2 of these fail a joint query (9.5%, interval 1.2% to 30%), against 1 of 24 for Study 2's own orthogonal
  alignments on the same networks. ‖P_A P_B‖ ranks the joint-only ones with an area under the ROC curve of 0.68 (0.39 to
  0.95). σ > 1 flags exactly the two joint-only alignments.
- **[local time], development complete.** Not frozen: the author sees these results first (Section 11, step 4).
- **About [local time], development-informed addendum approved by the author** (PROTOCOL.md Section 17): joint tests are also
  compared with single tests drawing from all single queries (a reported secondary comparison, no prediction).
  `interchange.panel_rates` now records each alignment's failure rate on all single queries and on those in which the
  swapped variable keeps its value; `analyse.detection` reports `contrast_all_single` with its interval when those rates
  are present. `make test` (7 tests) and the cloud jobs test passed. The NumPy verifier and the two simulated
  small-budget tests are deferred (post hoc if added later).
