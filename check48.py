# check48.py
import pandas as pd
r = pd.read_csv("data/selected/step12_active_results.csv")
x = r[(r.n_labels == 48) & r.strategy.isin(["zs_greedy", "hybrid1", "seeded_greedy"])]
print(x.groupby(["DMS_id", "strategy"])["hits"].mean().unstack().round(2))