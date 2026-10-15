## Pre-specified test on fresh assays
- Assays: 8, from select_holdout.py (seed 2026), none sharing a protein with the development assays.
- Single-mutant pools. Strategies: random, zs_greedy, greedy, hybrid1, seeded_greedy; ucb1 as a sensitivity check. No tuning on the new assays.
- Primary metric: campaign-average top-5% enrichment (mean over 48-192 labels).
- Primary comparisons: hybrid1 vs greedy and seeded_greedy vs greedy, paired across 20 seeds.
- A strategy counts as replicated if its mean paired difference vs greedy is positive and it is better (beyond 2 SE) in more assays than it is worse.
## Outcome (added after running holdout_active.py; plan text above unchanged)
- All 8 assays processed; none skipped.
- hybrid1 vs greedy: +0.28 campaign-average enrichment; better in 5 assays, same in 3, worse in 0
  (CASP3 at -0.34 +/- 0.17 was right at the threshold). Criteria met.
- seeded_greedy vs greedy: +0.47; better in 5, same in 3, worse in 0. Criteria met.