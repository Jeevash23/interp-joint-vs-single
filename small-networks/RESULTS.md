# Results

Short account of what AI-05 found. The manuscript (`paper/manuscript.pdf`) has the details, and every number
below is taken from the macros that `paper/make_numbers.py` generates from the saved outputs.

## The question

Interchange interventions are usually run one variable at a time, while a causal-abstraction claim covers
interventions on several variables at once. When can a joint intervention reject an alignment that single
interventions accept? We study alignments that map two variables to orthogonal subspaces at one site, on tasks
with inputs (a, b, n) ∈ {0,1,2}³, where every single (1,458) and joint (19,683) interchange can be evaluated.

## Theory (Section 3)

- A joint interchange edit equals a single edit applied to the result of the other single edit (Lemma 1).
- If either variable's single edits land on their targets (the natural states of the inputs they simulate), every
  joint query is an ordinary single query, so an alignment that passes all single queries passes all joint queries
  (Proposition 1; also up to directions the downstream computation ignores).
- A joint failure that does not trace to a failing single query needs both single edits to miss their targets
  (Corollary 1): their effect is a dormant behavioural change in the sense of Grant et al. A small explicit ReLU
  network meets the condition: 0 of 1,458 single queries fail and 24.7% of joint queries do; tilting only one
  variable's subspace gives no joint failures.
- Quantitative version (Proposition 2): a joint query's state is either reduced single query's state plus the part
  of the other edit's miss outside that query's subspace, so the joint query can fail only if, in both orderings,
  this lowers the margin by more than the single query's own margin. The first-order changes give two scores: the
  signed score σ, the smaller of the two predicted relative drops of margin (the main score, with the rule σ > 1),
  and the unsigned score ρ, the smaller of their absolute values. For MLP readouts a global Lipschitz constant gives a
  guarantee.
- For variables at different sites (Remark 2) the reduction works only through the upstream variable.

## Study 1: 80 compiled transformers, 480 alignments (360 distinct) with exact labels (Section 4)

- By construction every alignment is either exact or wrong on single queries, so none could fail only jointly; Study 1
  cannot test the condition, and its census explains the frozen null result. The census confirms the construction: valid alignments' single edits land on their targets up to unread nuisance coordinates
  (largest coordinate difference 1.9 × 10⁻¹⁴); invalid ones fail single queries; the exact labels coincide with
  "passes all 1,458 single queries" (480/480); no alignment fails only jointly.
- Replaying each frozen panel's sampling rule 10,000 times on the census puts all 28 frozen counts (seven panels,
  all but natural execution, at four budgets) within the central 95% of replays; Figure 2 shows six of them, raw state
  distance accepting no invalid alignment at any budget. Restricted singles accepted 81 of 160 invalid alignments (the 80 fitted on a, b ∈ {0,1},
  exact on exactly those queries, plus one random alignment); full-domain, held-out and joint queries, complement
  resampling and transported divergence each rejected all 160. The frozen joint-minus-single contrast of zero had
  probability at least 1 − 6.4 × 10⁻⁴ given the census.
- Raw state distance rejected the 80 valid alignments that move unread coordinates.

## Study 2: 200 trained transformers and MLPs, 1,200 alignments (Section 5; frozen 6 October 2026)

- Linear alignments are approximate: full-domain DAS alignments fail 32.4% of full-domain single queries and 29.3%
  of joint-panel queries on average; 127 of 1,200 alignments pass every single query, 110 of them in MLPs and 17 in
  transformers, all on additive tasks. No alignment of a lookup-task transformer passes every single query, so the
  joint-only result rests mainly on MLPs and says nothing about lookup-task transformers.
- No single edit that changes a variable lands on its target (smallest relative distance 5.4 × 10⁻⁵), so the
  condition holds for every joint-panel query (both variables change) of the alignments that pass all single
  queries. Even so, only 7 of the 127 fail a joint query (six MLPs, one transformer), on 2 to 101 of the 19,683
  joint queries each; six of the seven fail at most 0.75% of joint-panel queries, and the mean-contrast alignment of
  an additive-task MLP of width 64 fails only joint queries in which one variable keeps its value. A 120-call joint
  test would catch one with probability 5.7% on average.
- 8.8% of the DAS alignments' joint-panel failures are irreducible, all but 125 of them in alignments that also fail
  single queries.
- Post hoc, the signed margin score σ (the smaller of the two first-order predicted relative drops of margin) picks
  out the joint-only alignments: ranking the 127 alignments that pass every single query by their largest σ puts the
  seven at 1–6 and 14 (area under the ROC curve 0.99), and the rule σ > 1 flags six alignments, all joint-only,
  missing the rank-2 DAS alignment of an additive-task transformer of width 16, the only transformer among the seven.
  Each alignment's largest σ is taken over all 19,683 joint queries. The unsigned ρ ranks them 1, 3–5, 13, 15 and 60
  (0.91). Query by query the order reverses: σ 0.72, ρ 0.86, the off-target distances 0.68 and 0.70 (2,859,377
  queries, 136,057 failures); the first-order sign is right in 87% of cases but 76% of failures. ρ was specified
  first; σ was added before either was computed, both were computed in one run, and σ was chosen as the main score
  afterwards. For MLPs the Lipschitz bound certifies 6.9% of joint-panel queries without running them, none of which
  fails.
- At the prespecified costs, joint tests detect fewer failing alignments than single tests, slightly fewer at 120
  calls. Head to head, joint minus single detection probability at 120 calls is −0.028 (95% interval −0.037 to −0.020)
  with three calls per single and four per joint query, the prespecified primary contrast; it is −0.102 at 12 calls,
  negative at all four budgets computed (12, 24, 60 and 120 calls) and in all eight cells. Post hoc, with the same
  resampling, it is −0.018 (95% interval −0.025 to −0.012) at equal numbers of queries (30 each; positive in one of
  eight cells) and −0.005 (−0.011 to +0.001, within sampling error of zero) if natural states are cached (120 each;
  positive in three), and after a 120-call single test, 120 more calls of joint queries detect a further 1.1% of
  alignments, against 1.5% for more single queries. Minimal-pair singles detect 0.69 against 0.82 for full-domain
  singles at 120 calls.
- Held-out split (post hoc, exploratory): full-domain DAS refitted on a random half of the single queries fails
  31.8% of its training queries, 33.3% of the held-out ones and 29.5% of the joint panel. Against held-out single
  queries, joint minus single detection at 120 calls is −0.035 (95% interval −0.044 to −0.026; −0.021 against the
  training half), −0.025 at equal numbers of queries and −0.013 with cached states; after a held-out single test,
  joint queries add 1.0% against 1.7% for more held-out singles. Scoring singles on their training queries did not
  hide an advantage of joint tests.
- DAS fitted on the restricted inputs fails 10.8% of restricted and 60.6% of full-domain single queries.

## Study 3: the variables at different sites, 199 networks, 995 alignments (Section 6; predictions recorded and frozen 6 October 2026)

- Sites: MLP first vs second hidden layer; transformer position-1 residual before attention vs last-position residual
  after attention. Two predictions were hashed at 07:36 UTC, before any network existed; the protocol was frozen at
  07:55 UTC after development. One lookup transformer of width 64 never classified every input with a logit margin
  of at least 1 in its three initialisations and is excluded.
- Prediction 1 (more joint-only failures than Study 2's 5.5%) held on its estimate, not beyond sampling error: 3 of
  the 40 alignments that pass every single query fail a joint query (7.5%, 95% interval 1.6–20.4%), all in MLPs,
  on 1 to 25 of 19,683 joint queries. All 40 are on additive tasks.
- Prediction 2 (contrast above Study 2's −0.028) held beyond sampling error: +0.012 (95% interval +0.009 to +0.016),
  positive in every cell; +0.025 at equal queries, +0.003 with cached states; add-on 2.0% for joint against 1.7%
  for single queries. Post hoc, most of the gain is how the budget is spent: B, aligned nearer the output, fails more
  often than A in 504 of 597 DAS alignments, and every joint query exercises B. Where every A-edit lands (138
  transformer alignments) each joint query is a single B-query and the contrast is still +0.011; against single tests
  aimed at the weaker variable it is −0.004.
- Unexplained joint failures (single queries (r'', t) and (r, s) both pass) are 8.3% of joint-panel failures.
- Out of sample (applied post hoc, unchanged): the signed rule, fixed on Study 2 (07:23–07:25 UTC) before Study 3's
  protocol and predictions were recorded (07:36 UTC), with its one term, ranks the three joint-only alignments first
  to third of the 40 that pass every single query and flags exactly those three. Query by query, σ has area under the
  ROC curve 0.83 and 0.82 in transformers of width 16 and 64 and 0.91 in both MLPs (ρ: 0.98, 0.98, 0.87, 0.89; A-edit
  distance: 0.93, 0.94, 0.63, 0.74).
- The NumPy verifier agreed on all 21,035,295 decisions.

## Test strategies at small budgets (Section 7; post hoc, exploratory)

Expected share of the full-domain DAS alignments detected at 120 model calls (3 per single query, 4 per joint query,
1 per gradient); Table 5 has 12–120 calls.

| Test | Study 2 | Study 3 |
|---|---|---|
| Alternating single | 0.816 | 0.911 |
| Adaptive single, switching on observed failures (k = 4) | 0.816 (identical at every budget) | 0.911 |
| Adaptive single, switching on observed margins (k = 4) | 0.815 | 0.903 |
| Oracle-targeted single (weaker variable known) | 0.818 | 0.927 |
| Joint | 0.788 | 0.924 |
| Margin-guided joint | 0.799 | 0.901 |
| Margin-guided joint, candidates in random order | 0.796 | 0.897 |

- An adaptive test that switches on observed failures can only switch once the alignment is already detected, so it
  is the alternating test. Switching on margins helps in Study 3 at 24–60 calls and costs at 120.
- The margin-guided joint test scores 12–13 candidates and runs 6–7 joint queries at 120 calls; it beats random order
  on the joint-only alignments (3.6% vs 1.3% in Study 2, 2.6% vs 1.3% in Study 3) but a plain joint test does better
  still (6.6%, 4.6%): scoring a candidate costs more than running it. Scoring every joint query from a finished single
  census costs only the 1,458 single-query gradients and 39,366 dot products (Section 3).

## Reproductions (Appendix D)

- The released sanity script of Grant et al. (Section 4.2) passes all its assertions.
- The CAE controlled XOR experiment of Méloux et al. reproduces its comparative result (the faithfulness-aware
  metrics separate the valid from the backup-path model). In the released script the valid model's mediator (m in
  the paper, `a` in the code) does not respond to interventions (`systems/11_controlled_experiments.py`, line 107);
  with that line changed, every metric is exactly 0 on the valid model and only the faithfulness-aware metrics are
  non-zero on the backup-path model (0.072 and 0.063). The file is identical at the cited commit and on the
  repository's default branch (checked 6 October 2026).

## Limits

Orthogonal linear alignments at one site (two sites in Study 3); two variables, three values each, 27 inputs; small compiled and trained
networks. Approximate edits essentially never land on their targets, so the redundancy check settles the question
only for exact alignments. Full-domain DAS alignments in Study 2 are scored on the queries they were trained on.
Study 1's labels are exact for one precise claim; Studies 2 and 3 have no labels independent of the tests. The Study 1
census and the analyses marked post hoc in Studies 2 and 3 are computations on saved data added after the frozen
analyses. Study 3 was designed after Studies 1 and 2 were known, and its joint-only estimate rests on 40 alignments.
The advice to expect little from joint tests covers two variables at one site. Language-model alignments are often
cross-site: MIB's best whole-head alignment of its IOI variables in GPT-2 Small puts position at heads 7.3, 7.9 and
8.6 and token at head 8.10, closer to Study 3, where joint tests gained. Subspaces fitted one variable at a time can
overlap at one site, which takes them outside Lemma 1. A test in a language model is under way in a companion study
(AI-10); none of its results is used here.
