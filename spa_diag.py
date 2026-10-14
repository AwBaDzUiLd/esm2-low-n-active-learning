import numpy as np, pandas as pd
from scipy.stats import spearmanr

a = "SPA_STAAU_Tsuboyama_2023_1LP1"
ref = pd.read_csv("data/DMS_substitutions.csv").set_index("DMS_id")
wt = ref.loc[a, "target_seq"]
pool = pd.read_csv(f"data/selected/{a}_pool.csv")
S = pool[pool.n_mut == 1].set_index("mutant")
D = pool[pool.n_mut == 2].copy()

p = D["mutant"].str.split(":", expand=True)
D["m1"], D["m2"] = p[0], p[1]
D["pos1"] = D["m1"].str[1:-1].astype(int)
D["pos2"] = D["m2"].str[1:-1].astype(int)

bad = sum(wt[r.pos1 - 1] != r.m1[0] or wt[r.pos2 - 1] != r.m2[0] for r in D.itertuples())
print("doubles whose WT residues do not match the sequence:", bad, "of", len(D))

D["s1"], D["s2"] = D["m1"].map(S["DMS_score"]), D["m2"].map(S["DMS_score"])
D["add"] = D["s1"] + D["s2"]
D["gap"] = (D["pos2"] - D["pos1"]).abs()
D["worst"] = D[["s1", "s2"]].min(axis=1)

q = [.05, .25, .5, .75, .95]
print("\nsingle score quantiles:", S["DMS_score"].quantile(q).round(2).tolist())
print("double score quantiles:", D["DMS_score"].quantile(q).round(2).tolist())
print("overall rho zs vs score:", round(spearmanr(D["zs"], D["DMS_score"]).correlation, 3))
print("overall rho measured-sum vs score:", round(spearmanr(D["add"], D["DMS_score"]).correlation, 3))


def table(col, bins=None):
    g = pd.cut(D[col], bins) if bins is not None else pd.qcut(D[col], 4, duplicates="drop")
    out = []
    for k, d in D.groupby(g, observed=True):
        out.append({"bin": str(k), "n": len(d),
                    "rho_zs": round(spearmanr(d["zs"], d["DMS_score"]).correlation, 3),
                    "rho_sum": round(spearmanr(d["add"], d["DMS_score"]).correlation, 3)})
    print(f"\nby {col}:\n", pd.DataFrame(out).to_string(index=False))


table("gap", [0, 1, 3, 7, 15, 100])
table("worst")
print("\nTop 10 doubles by score:")
print(D.nlargest(10, "DMS_score")[["mutant", "DMS_score", "s1", "s2", "zs"]].round(2).to_string(index=False))