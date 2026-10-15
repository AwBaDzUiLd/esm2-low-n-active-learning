# Low-N protein fitness prediction with ESM-2: a benchmark on 8 DMS assays

Can a few labeled variants plus a protein language model rank variants and guide
which ones to test next? Benchmark on 8 ProteinGym assays (single mutants only),
run entirely on a 6 GB laptop GPU.

## Main findings
1. **Labels help most where the zero-shot prior is weak.** At N=192 (position-held-out),
   four assays gain +0.19 to +0.25 Spearman over zero-shot ESM-2 (SPA, SPG1,
   Q53Z42-binding, RASK-abundance); the other four gain at most +0.03.
2. **With 24 labels, the supervised model is roughly the prior.** Held-out average:
   0.424 vs 0.406 zero-shot. It beats zero-shot in 76% of runs at N=24 and 88% at N=192.
3. **Random splits overstate performance.** For ridge_pos+zs at N=192: 0.656 (random)
   vs 0.525 (held-out positions). Position-specific baselines lose far more.
4. **Active learning (pool-based, 192 labels):** model-guided greedy/UCB reach ~3.0x
   top-5% enrichment vs ~1.0 for random, winning in 96% of assay x seed campaigns.
   Zero-shot greedy leads early and then falls behind; the crossover came by ~48-72 labels on the
   development assays and ~96-144 on the hold-out, whose zero-shot prior was stronger on average.
5. **Using the prior only for the first batch speeds up the campaign.** Choosing round 1 by
   zero-shot score (`hybrid1`), or half of the initial 24 labels (`seeded_greedy`), raised
   campaign-average enrichment (48-192 labels) from 2.84 (greedy) to 3.08 and 3.46.
   At 96 labels `hybrid1` beat greedy in 4 of 8 assays and was worse in none; by 192 labels
   the difference was gone. Two prior-first rounds (`hybrid2`) gave no gain, and the UCB
   weight (0.5-2) made little difference. These strategies were designed after seeing
   the same 8 assays and need replication on new ones.
6. **Models trained on single mutants transfer to double mutants.** Excluding SPA, Spearman on
   doubles at N=192 labeled singles is 0.68 (direct) and 0.69 (summing per-single predictions)
   vs 0.47 for zero-shot; at N=24 the gain is small (0.51 vs 0.47). Summing measured
   single-mutant scores (a reference using all singles) reaches 0.90. Top-10% overlap is weaker:
   0.41 at N=192 vs 0.30 for zero-shot. In SPA, zero-shot is negatively correlated with doubles
   (-0.23); the cause is not established. Some test doubles contain a training single.
7. **Active learning on mixed pools (singles + 4,000 sampled doubles, 4 assays).** Guided
   strategies reach 3.1-3.5x top-5% enrichment at 192 labels vs 1.0 for random. `hybrid1` and
   `seeded_greedy` again were never worse than greedy beyond noise (campaign average 2.96 and
   3.15 vs 2.74), but most per-assay differences are within noise. Guided picks were mostly
   singles (5-21% doubles) even though random picks were 57-80% doubles.
8. **Pre-specified test on 8 new assays** (chosen by a seeded rule; criteria committed in PLAN.md
   before running; no tuning). Campaign-average top-5% enrichment (48-192 labels, averaged over
   assays): random 0.98, greedy 2.98, hybrid1 3.26, seeded_greedy 3.45. Paired against greedy,
   hybrid1 gained +0.28 (better in 5 assays, worse in 0) and seeded_greedy +0.47 (better in 5,
   worse in 0), meeting the pre-specified criteria. The effects are modest (about 10-16% of greedy's
   enrichment), shrink by 192 labels (+0.05 and +0.18), and the hold-out contains no binder
   assays (its one 'Binding' assay, CP2C9, is an enzyme with an activity readout)(criteria and assay list committed in 71e09a1, before any hold-out script or result was added to the repo).

## Reproduce
```bash
conda create -n lown python=3.10 -y && conda activate lown
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
bash run_all.sh        # GPU needed for baselines.py and embed.py only
```
Outputs go to `data/selected/` and `results/` (tables in `results/RESULTS.md`).

## Methods
- Data: 8 ProteinGym assays (single mutants only), chosen by me with a binding focus.
- Zero-shot: ESM-2 650M wild-type marginals. Embeddings: ESM-2 650M, fp16, at the mutated position.
- Models: ridge on embeddings; zero-shot score calibrated on training labels plus
  ridge on the residual (`ridge_pos+zs`); Bayesian ridge on 64 PCA components for active learning.
- Splits: random, and position-held-out (test positions never seen in training). 20 seeds.

## Limitations
- Single mutants only; pools are near-saturated libraries, so active-learning numbers
  are optimistic compared with real campaigns.
- Only 8 assays, hand-selected. The prior-strong/weak grouping was defined after seeing results.
- One PLM, ridge-style models only, untuned hyperparameters (beta=1, 64 PCA dims).
- No wet-lab validation. Assay readout descriptions come from ProteinGym metadata
  and should be checked against the original papers.
- SPG1 is supplied as a long sequence with a short mutated domain; zero-shot may be
  affected by this (untested).
- Nine strategy configurations were compared on the same 8 assays, and ~128 paired
  comparisons were made at a 2-SE threshold, so best-of-nine results are optimistic
  and isolated per-assay differences may be chance.
- Mixed-pool experiments use sampled doubles, and the top-5% target is defined within the pool,
  not the full assay. They reuse 4 of the 8 development assays, so they test robustness to
  pool composition, not independent replication.
- In the transfer experiment, some test doubles contain a single that was in the training set,
  which makes transfer look easier than for a fully new double.
- The hold-out test used the same model and settings as development, and the success rule needed
  only a positive mean difference and more assays better than worse (no minimum effect size).
  The hold-out's mean zero-shot Spearman (0.54) was higher than the development set's (0.41).

## Credits
ProteinGym (Notin et al., 2023), ESM-2 (Lin et al., 2023), and the original DMS
studies of each assay. Check ProteinGym's license terms before redistributing any data.