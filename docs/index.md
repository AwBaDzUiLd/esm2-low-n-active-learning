---
layout: default
title: "A pre-registered low-N protein engineering benchmark on a 6 GB laptop GPU"
---

# A pre-registered low-N protein engineering benchmark

By Abdul Wazid Shaik, independent protein design researcher. Code, data pipeline and full results available at https://github.com/AwBaDzUiLd/esm2-low-n-active-learning

## Abstract

Given a lab budget sufficient for measuring a few dozen variants, how much can a protein language model plus a simple model help, and should the model choose which variants to measure? I evaluated this on 16 publicly available deep mutational scanning assays from ProteinGym, using ESM-2 650M on a laptop with an RTX 3050 (6 GB). In the development runs, peak GPU memory stayed under 3 GB (1.3 to 2.8 GB).

The main results are:

- Labels help most where the zero-shot model is weak. With 192 labeled variants and a position-held-out split, four of eight assays gained 0.19 to 0.25 Spearman rank correlation over zero-shot ESM-2. The other four assays gained 0.03 or less.

- Performance is overstated with random train/test splits. For the best model at 192 labels, the Spearman rank correlation was 0.656 on a random split and 0.525 with test positions unseen during training.

- In simulated campaigns, model-guided selection finds about three times as many top-5% variants as random picking within 192 labels.

- Using the zero-shot score to choose part of the first mutations can speed up a campaign, especially with small numbers of labels. This held on 8 new assays chosen by a seeded rule under a plan committed before the test. The effect was modest, about 9 to 16% of greedy's enrichment.

## Methods

Data: Eight assays from ProteinGym were used for development, chosen by hand with a focus on binding. Most of them have binding or expression readouts. Eight more were selected for the hold-out set with a seeded random number generator, and none of them shares a protein with the development assays. The main experiments use single mutants. The mixed-pool experiments add up to 4,000 sampled double mutants per assay.

Models: Zero-shot scores are the log-probability of the mutant residue minus that of the wild-type residue, from ESM-2 650M. Supervised models use features from ESM-2 at the mutated position with ridge regression. The best low-N model was calibrated on the training-set labels to adjust the zero-shot score, then ridge regression was used on the residual error.

Evaluation: Training sizes of 24, 48, 96 and 192 variants, with 20 random seeds each. Two types of train/test splits were used: random, and position-held-out, where test variants sit at positions not seen during training. The position-held-out split is more realistic for protein design, since a designer often explores new positions during a campaign.

## Discovery 1: the value of labels depends on the zero-shot prior

Averaged across the eight development assays on the position-held-out split, the calibrated model had a Spearman rank correlation of 0.424 at 24 labels and 0.525 at 192, compared to 0.406 for the zero-shot model. This average hides two groups, split by how much the zero-shot model already knows.

| Assay (position-held-out, 192 labels) | Zero-shot | With labels | Gain |
|---|---|---|---|
| SPA stability | 0.280 | 0.531 | +0.251 |
| RASK abundance | 0.195 | 0.433 | +0.238 |
| SPG1 binding | 0.275 | 0.476 | +0.201 |
| Q53Z42 binding | 0.316 | 0.503 | +0.188 |
| Other four assays | 0.346 to 0.715 | n/a | +0.006 to +0.027 |

Where the zero-shot model was already strong (GRB2, RASK binding, Q53Z42 expression, CCR5), 192 labels added at most +0.027. Where it was weak (SPA stability, RASK abundance, SPG1, Q53Z42 binding), the same labels added +0.19 to +0.25. At 24 labels the supervised model mostly falls back to the prior, though it still beat zero-shot in 76% of held-out runs at 24 labels and 88% at 192.

## Discovery 2: beware the random splits

For the best model at 192 labels, the difference between the random and position-held-out splits was 0.13 in Spearman rank correlation. For a model that learns one weight per position, the difference was much larger. On small domains, a random split covers almost every position, so the model has already seen the positions it is tested on. Low-N results should be reported on a position-held-out split.

## Discovery 3: model-guided selection finds more top variants

I evaluated model-assisted active learning on single-mutant pools. The process was to pick a random set of 24 variants at the start, then 24 additional variants per round for seven rounds (192 labels in total). The metric was the enrichment of the true top 5% of variants in the pool, where 1.0 means no better than random picking.

| Strategy | Enrichment at 192 labels (development assays) |
|---|---|
| Random | 0.98 |
| Zero-shot score only | 1.99 |
| Prediction (greedy) | 3.00 |
| Prediction plus uncertainty bonus | 3.01 |
| Uncertainty only | 0.87 |

Greedy picking outperformed random picking in 96% of assay-by-seed campaigns (ties counted as half). Picking by uncertainty alone did worse than random on average: clearly worse in 3 of 8 assays, about equal in 3, and slightly better in 2 (GRB2 and RASK abundance). So exploring for its own sake did not help when the objective was to find high-scoring variants. The zero-shot ranking led at 48 labels, roughly tied with greedy by 72, and fell behind afterwards.

![Active learning: top-5% enrichment by strategy](https://awbadzuild.github.io/esm2-low-n-active-learning/assets/active_summary.png)

*Top-5% enrichment against number of labels for each strategy, on all eight development assays and on two groups defined by zero-shot strength. The grouping was defined after seeing the results.*

## Discovery 4: use the zero-shot prior for the first mutations (pre-registered)

The early lead of zero-shot suggested a hybrid approach. In hybrid1, the first model-guided batch (the 24 picks after the random initial 24) is chosen by zero-shot score, and the model takes over afterwards. A similar strategy, seeded_greedy, chooses 12 of the initial 24 variants by zero-shot score. Both performed well on the development assays: campaign-average enrichment (48 to 192 labels) was 2.84 for greedy, 3.08 for hybrid1 and 3.46 for seeded_greedy. Two prior-first rounds (hybrid2) gave no gain. But I designed these strategies after seeing those assays, so I did not trust that result. Before doing any analysis on new assays, I committed a plan (PLAN.md) to the repository.

The plan evaluated eight assays selected with a seeded random number generator, none sharing a protein with the development assays. The metric was campaign-average top-5% enrichment (mean over 48 to 192 labels). A strategy counted as replicated if its mean paired difference against greedy was positive and it beat greedy in more assays than it lost to it. The rule was permissive: there was no minimum effect size. The results were:

| Strategy | Campaign-average enrichment (48 to 192 labels) | Paired gain vs greedy | Assays better / same / worse |
|---|---|---|---|
| Random | 0.98 | | |
| Greedy | 2.98 | | |
| hybrid1 | 3.26 | +0.28 | 5 / 3 / 0 |
| seeded_greedy | 3.45 | +0.47 | 5 / 3 / 0 |

Both strategies met the success criterion. The improvements were modest, about 9 to 16% of greedy's enrichment, and they shrank with more labels. Averaged over the eight hold-out assays (1.0 = random):

| Labels | Random | Greedy | hybrid1 | seeded_greedy |
|---|---|---|---|---|
| 48 | 0.99 | 2.15 | 3.18 | 3.26 |
| 96 | 0.96 | 3.18 | 3.30 | 3.61 |
| 192 | 0.99 | 3.13 | 3.18 | 3.31 |

At 192 labels the paired differences were only +0.05 for hybrid1 and +0.18 for seeded_greedy. The benefit is speed early in a campaign, which matters when each round costs wet-lab money. One assay (CASP3) was borderline for hybrid1, about two standard errors below greedy. I did not run significance tests beyond these standard errors.

The criteria and the assay list were committed in 71e09a1 before any hold-out script or result was added to the repository, so the order is visible in the git history. In short, I did not trust the development results because the strategies were designed on those assays, and the hold-out result is the one to rely on.

## More data: double mutants

On pools containing singles plus up to 4,000 sampled double mutants (four assays), the trends were similar to the single-mutant case, with smaller differences. Guided strategies reached 3.1 to 3.5 times enrichment at 192 labels, and hybrid1 and seeded_greedy were again never clearly worse than greedy (campaign averages 2.96 and 3.15 against 2.74). SPA was left out of this experiment because the top 5% of its pool was almost entirely single mutants.

A model trained on N labeled single mutants can also rank double mutants. At 192 labeled singles, the Spearman correlation on doubles was 0.68 to 0.69 (excluding SPA) compared to 0.47 for the zero-shot model. Summing the measured single-mutant scores, which uses all the single-mutant data, reached 0.90. So at small N the limit is the accuracy of single-mutant predictions rather than epistasis. Some test doubles contain a single that was in the training set, which makes this transfer look somewhat easier than for a fully new double.

One assay, SPA, behaved strangely: zero-shot scores were negatively correlated with its double mutants (-0.23). The numbering is correct (0 of 1,070 doubles mismatched the sequence), and cysteine substitutions explain part of the problem but not all of it. Cysteine variants are 10% of the doubles but 52% of the top 54, and ESM-2 scores them lower (mean -10.6 against -8.2 for the rest) while the assay rewards them (mean score -0.60 against -1.23). The same pattern shows up in single mutants: zero-shot -6.4 against -4.2, and assay score +0.06 against -0.50, for cysteine against all other substitutions.

Removing every cysteine double moves the correlation from -0.23 to -0.09, which is near zero rather than positive. So the prior carries almost no signal for the other 959 doubles, even though it works for singles (+0.36 without cysteine). I do not know why. One candidate is that summing single-site scores is a poor proxy for pairs of nearby substitutions, which I have not tested. These subset analyses were exploratory, and many of the top doubles pair D34C with a substitution at position 36, so treat the cysteine result as specific to this domain. Zero-shot ESM-2 is not a safe default where your variants include chemistry the model dislikes.

## Limitations

- Eight development assays and eight hold-out assays are a small sample, and the development assays were chosen by hand with a binding focus. The hold-out set does not include any binder assays; the one assay categorized as Binding (CP2C9) is an enzyme with an activity readout. The hold-out set also had a stronger zero-shot prior (mean rank correlation 0.54 against 0.41), which may favour prior-based strategies.

- The success rule was permissive: it required only a positive mean difference and more assays better than worse, with no minimum effect size.

- Nine strategy configurations were compared on the same development assays, so best-of-nine results are optimistic and isolated per-assay differences may be chance.

- The standard errors reflect variation across random starting sets within each assay, not variation across proteins.

- The pools are near-complete single-mutant scans (plus sampled doubles in the mixed-pool experiments), so enrichment is optimistic compared with a real campaign, and the top 5% is defined within each pool.

- One language model (ESM-2 650M), ridge-style surrogates, untuned hyperparameters, and no wet-lab validation.

- Assay descriptions come from ProteinGym's metadata and should be checked against the original papers.

## Reproduce

The repository has a one-command pipeline. Only the steps that run ESM-2 need a GPU, and 6 GB was enough for me. The full pipeline takes hours, mostly in the embedding steps. Please file an issue if you cannot reproduce the results.

## What I want feedback on

If you are working on low-N engineering or active learning for proteins, I would like to hear whether Gaussian process or neural surrogates change the picture, and which binder datasets I should use for the next hold-out.

## Credits

ProteinGym (Notin et al., 2023), ESM-2 (Lin et al., 2023), and the original authors of each deep mutational scanning dataset. Check ProteinGym's license terms before redistributing any data.

A note on process: I used an AI assistant to help plan the analysis, write code and draft this post. All numbers come from the saved outputs in the repository (the results folder and the *_out.txt files).
