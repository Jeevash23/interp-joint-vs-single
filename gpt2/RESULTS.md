# Results

Short account of what AI-10 found. Every number comes from the saved analyses: `study/results/analysis/confirmation/`
(GPT-2 Small, local machine), `study/results/cloud/gpt2-medium/analysis/confirmation/` (GPT-2 Medium, Modal),
`study/results/analysis/overlap-final/` (arm d), and the development analyses. Intervals are 95% bootstrap intervals
over tasks, or Clopper–Pearson intervals for shares. Each prediction's score is in `study/CONFIRMATION-NOTES.md`.

## The question

AI-05 showed that when two variables are aligned to orthogonal subspaces at one site, a joint interchange query is a
single query applied after another, so a joint test can reject an alignment that every single query accepts only when
both variables' single edits miss the states they simulate. In small trained networks that was rare (7 of 127
alignments), a first-order margin rule picked those cases out, and at equal cost joint tests detected fewer failing
alignments than single tests (−0.028 at 120 model calls). AI-10 asked whether this holds in a language model: GPT-2
Small (and, in a later amendment, GPT-2 Medium) on indirect-object identification, with MIB's two variables, the
indirect object's position (which name repeats) and its identity (which names fill the first clause), on held-out
names and templates.

## One site (arm a): the main result

| | GPT-2 Small | GPT-2 Medium |
|---|---|---|
| Confirmation tasks (held-out draws passing the gate) | 35 of 200 | 36 of 47 (cost rule: 36) |
| Alignments | 210 | 180 |
| Pass every single query | 0 | 6 |
| … of which fail a joint query (too few to test predictions 1–2) | — | 6 |
| Margin rule (σ > 1) flags | — | all 6 |
| Joint minus full-domain single detection, 120 calls | **+0.613** [0.582, 0.640] | **+0.640** [0.615, 0.660] |
| Joint minus all-single detection, 120 calls (addendum) | +0.052 [0.017, 0.084] | +0.158 [0.123, 0.195] |
| DAS: full-domain single failure rate, position / token | 0.07% / 16.6% | 1.8% / 15.8% |
| DAS: position queries with the recipient's position, failure rate | 20.9% | 21.0% |
| DAS: joint-panel failure rate | 36.2% | 41.1% |

- **Joint tests find failures that the prespecified single tests miss.** At equal cost (3 model calls per single
  query, 4 per joint query) joint tests detect far more failing alignments than single tests drawn from the
  full-domain panel, in both models (prediction 3, which expected a negative difference, does not hold, beyond
  sampling error). The difference is positive at every budget from 12 to 120 calls.
- **Why: DAS's position subspace also carries name information.** DAS of rank 2 and 4 fails none of the full-domain
  single queries (in which the swapped variable changes, the queries it is trained on), and rank 2 trained on half of
  them fails 0.1% of the other half (and 17.4% of the joint panel). But in GPT-2 Small it fails 12.4% (rank 2) and 50.3% (rank 4) of the position
  queries whose donor has the recipient's own position and other names, and 15.2% and 32.9% of the joint panel. A
  joint query always brings another task prompt's names through the position subspace, so it exposes the leak; a
  full-domain single query of the position variable changes the position and lets the leak pass unseen.
- **Against single tests drawn from all single queries** (the development-informed addendum, reported without a
  prediction), joint tests still detect more at 120 calls, by +0.052 in GPT-2 Small and +0.158 in GPT-2 Medium; at 12
  calls the difference is +0.271 in GPT-2 Small.
- **Joint-only failures.** In GPT-2 Small no alignment passes all 648 single queries. In GPT-2 Medium six do (all DAS
  of rank 2), fewer than the 10 that the protocol requires, so predictions 1 and 2 are not testable in either model.
  Described rather than scored: all six GPT-2 Medium alignments fail joint queries, 45 to 782 of 5,832 each, and the
  margin rule flags all six.

## The other arms (GPT-2 Small)

| Arm | Full-domain single failure, position / token | Joint panel | Joint minus single, 120 calls |
|---|---|---|---|
| a′, final position only | 50.6% / 52.0% | 65.7% | 0.000 |
| b, MIB's heads (7.3, 7.9, 8.6; 8.10) | 27.7% / 99.9% | 92.4% | 0.000 |
| c, subspaces fitted one variable at a time | 0.07% / 16.5% | 40.6% | +0.621 [0.592, 0.646] |

- **The final position cannot carry the token variable**, as predicted: about half of all single queries fail.
- **MIB's token head cannot change which name is said** under the counterfactual model: 99.9% of its token queries
  fail. Both tests detect every arm-b alignment, so arm b's contrast is zero, and prediction 4 (arm b's contrast above
  arm a's) does not hold: arm a's is the larger, by 0.613 [0.581, 0.641].
- **Overlap in GPT-2 is small and runs the other way.** Separately fitted subspaces at arm a's site overlap little
  (median ‖P_A P_B‖ 0.17; prediction 5a holds), and the joint failure rate falls rather than rises with overlap
  (Spearman −0.31 [−0.40, −0.20]; prediction 5b does not hold).

## Overlap in AI-05's networks (arm d)

On AI-05 Study 2's 200 trained networks, P_A and P_B fitted one variable at a time (800 alignments): 18 of the 109
that pass every single query fail a joint query (16.5% [10.1%, 24.8%]), against 7 of 127 (5.5%) for Study 2's own
orthogonal alignments, so prediction 6a holds beyond sampling error. ‖P_A P_B‖ separates the joint-only ones with an
area under the ROC curve of 0.69 [0.55, 0.82] (prediction 6b holds beyond sampling error). The margin rule flags 14 of
the 18 and none of the other 91.

## How the protocol changed

- **Amendment 1** (after the pilot, before development): the parity gate between our GPT-2 code and the transformers
  library became float64 agreement within 1e-8 plus float32 agreement within 1e-3; the original float32 limit of 1e-4
  was too tight for two implementations on a GPU (they agree to 1.1e-13 in float64).
- **Amendment 2** (during development, before any bigger-model computation; author-confirmed, $6): the one-site arm in
  GPT-2 Medium on Modal, with DAS trained in TF32 and every reported number in float32. The cost rule gave GPT-2 Medium
  36 confirmation tasks and ruled out GPT-2 Large. Spend: $5.17.
- **Development-informed addendum** (before the freeze): the all-single-queries comparison above, reported without a
  prediction. A float64 verifier planned for development was written after the freeze (post hoc; below); two simulated
  small-budget tests were not run.
- **Departures:** the learning-rate search ran at GPT-2 Small's layer 9, where the protocol said 8; the settings file's
  timestamp records the start of the search. Details in `study/DEVELOPMENT-NOTES.md`.

## Verification (post hoc)

`posthoc/verify.py`, an independent NumPy float64 GPT-2 Small, recomputed a sample of saved queries from the saved
subspaces: every query within 0.05 of the tie band, every query of any joint-only alignment, and 2,000 random
queries per phase, across arms a, a′, b and c. Development: 8,180 queries, no decision changed, largest margin
difference 6.6e-5. Confirmation: 31,447 queries, no decision changed, largest margin difference 1.0e-4. GPT-2 Medium's
arrays stay on the Modal Volume and were not re-verified; the cloud check stage reproduced 648 saved GPT-2 Small
margins within 6.1e-5 before any GPT-2 Medium computation.
