import numpy as np, pandas as pd
from scipy.stats import spearmanr

a = "SPA_STAAU_Tsuboyama_2023_1LP1"
pool = pd.read_csv(f"data/selected/{a}_pool.csv")
S = pool[pool.n_mut == 1].copy()
D = pool[pool.n_mut == 2].copy()


def rho(d):
    return round(spearmanr(d.zs, d.DMS_score).correlation, 3) if len(d) > 10 else float("nan")


# ---- Singles: is the prior different for Cys substitutions? ----
S["to_cys"] = S["mutant"].str[-1] == "C"
print("=== SINGLES ===")
for name, d in [("to Cys", S[S.to_cys]), ("not to Cys", S[~S.to_cys])]:
    print(f"{name:11s} n={len(d):4d}  rho_zs={rho(d):+.3f}  "
          f"mean score={d.DMS_score.mean():+.2f}  mean zs={d.zs.mean():+.2f}")

# ---- Doubles ----
D["has_cys"] = D["mutant"].str.split(":").map(lambda ps: any(p[-1] == "C" for p in ps))
D["has_d34c"] = D["mutant"].str.contains("D34C")
print("\n=== DOUBLES ===")
print("doubles with a Cys substitution:", int(D.has_cys.sum()), "of", len(D))
for name, d in [("with Cys", D[D.has_cys]), ("without Cys", D[~D.has_cys])]:
    print(f"{name:11s} n={len(d):4d}  rho_zs={rho(d):+.3f}  "
          f"mean score={d.DMS_score.mean():+.2f}  mean zs={d.zs.mean():+.2f}")

d34 = D[D.has_d34c]
print(f"\nD34C doubles: {len(d34)}, mean score {d34.DMS_score.mean():+.2f}, "
      f"others {D[~D.has_d34c].DMS_score.mean():+.2f}")
print("rho_zs excluding D34C doubles:", rho(D[~D.has_d34c]))
print("rho_zs excluding all Cys doubles:", rho(D[~D.has_cys]))

top = D.nlargest(54, "DMS_score")        # about the top 5% of 1,070 doubles
print("\nshare of top-54 doubles with Cys: ", round(float(top.has_cys.mean()), 2))
print("share of top-54 doubles with D34C:", round(float(top.has_d34c.mean()), 2))
print("share of ALL doubles with Cys:    ", round(float(D.has_cys.mean()), 2))