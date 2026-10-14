import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)

SELECTED = [
    "SPG1_STRSG_Olson_2014",
    "RASK_HUMAN_Weng_2022_binding-DARPin_K55",
    "RASK_HUMAN_Weng_2022_abundance",
    "Q53Z42_HUMAN_McShan_2019_binding-TAPBPR",
    "Q53Z42_HUMAN_McShan_2019_expression",
    "CCR5_HUMAN_Gill_2023",
    "GRB2_HUMAN_Faure_2021",
    "SPA_STAAU_Tsuboyama_2023_1LP1",
]
ref = pd.read_csv("data/DMS_substitutions.csv").set_index("DMS_id")
res = pd.read_csv("data/selected/lowN_results.csv")

# =====================================================================
# PART A: summaries from saved results
# =====================================================================
print("=" * 70, "\nPART A: per-assay results\n", "=" * 70)
g = res.groupby(["DMS_id", "split", "model", "n"])["rho"].mean().reset_index()

for split in ["position_heldout", "random"]:
    for n in [24, 192]:
        p = g[(g.split == split) & (g.n == n)].pivot_table(
            index="DMS_id", columns="model", values="rho")
        p = p[["zero_shot", "additive_onehot", "ridge_pos", "ridge_pos+zs"]]
        p["gain_vs_zs"] = p["ridge_pos+zs"] - p["zero_shot"]
        p["type"] = ref.loc[p.index, "coarse_selection_type"].values
        print(f"\n--- {split}, N={n} (mean Spearman over 20 seeds) ---")
        print(p.round(3).to_string())

w = res.pivot_table(index=["DMS_id", "split", "seed", "n"],
                    columns="model", values="rho").reset_index()
w["win"] = w["ridge_pos+zs"] > w["zero_shot"]
print("\n--- Fraction of runs where ridge_pos+zs beats zero-shot ---")
print(w.groupby(["split", "n"])["win"].mean().unstack().round(2).to_string())
print("\n--- Same, per assay, N=24 ---")
print(w[w.n == 24].groupby(["DMS_id", "split"])["win"].mean().unstack().round(2).to_string())

print("\n--- Random minus held-out gap at N=192, per model ---")
gap = g[g.n == 192].pivot_table(index="model", columns="split", values="rho", aggfunc="mean")
gap["gap"] = gap["random"] - gap["position_heldout"]
print(gap.round(3).to_string())

print("\n--- Top-10% overlap (fraction of true top 10% recovered), mean over assays ---")
t10 = res.groupby(["split", "n", "model"])["top10"].mean().reset_index()
print(t10.pivot_table(index=["split", "n"], columns="model", values="top10").round(3).to_string())

# =====================================================================
# PART B: refit ridge_pos+zs, keep predictions, analyze errors
# =====================================================================
print("\n" + "=" * 70, "\nPART B: error analysis (N=96, seeds 0-4)\n", "=" * 70)

ALPHAS = np.logspace(0, 6, 13)
N_TRAIN, K = 96, 192
SEEDS = range(5)

KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
      'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
      'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}


def mt_class(a):
    if a == "P": return "to_Pro"
    if a == "G": return "to_Gly"
    if a in "DE": return "to_acidic"
    if a in "KR": return "to_basic"
    if a in "FWY": return "to_aromatic"
    if a in "AVILMC": return "to_aliphatic"
    return "to_polar"  # S T N Q H


def make_split(df, split, seed):
    rng = np.random.RandomState(seed)
    if split == "random":
        idx = rng.permutation(len(df))
        return idx[:K], idx[K:]
    positions = np.unique(df["pos"].values)
    rng.shuffle(positions)
    half = len(positions) // 2
    train_pos, test_pos = positions[:half], positions[half:]
    pool = np.where(np.isin(df["pos"].values, train_pos))[0]
    te = np.where(np.isin(df["pos"].values, test_pos))[0]
    return rng.permutation(pool)[:K], te


def run_preds(dms_id, split):
    df = pd.read_csv(f"data/selected/{dms_id}_singles_zs.csv")
    Xpos = StandardScaler().fit_transform(
        np.load(f"data/emb/{dms_id}_pos.npy").astype(np.float32))
    y, zs = df["DMS_score"].values, df["zs"].values

    pm = df.groupby("pos")["DMS_score"].mean()
    q = pd.qcut(pm.rank(method="first"), 4,
                labels=["Q1_most_sensitive", "Q2", "Q3", "Q4_most_tolerant"])
    out = []
    for seed in SEEDS:
        order_tr, te = make_split(df, split, seed)
        tr = order_tr[:N_TRAIN]
        lr = LinearRegression().fit(zs[tr].reshape(-1, 1), y[tr])
        resid = y[tr] - lr.predict(zs[tr].reshape(-1, 1))
        rg = RidgeCV(alphas=ALPHAS).fit(Xpos[tr], resid)
        pred = lr.predict(zs[te].reshape(-1, 1)) + rg.predict(Xpos[te])

        t = df.iloc[te][["mutant", "pos", "wt", "mt"]].copy()
        ry = rankdata(y[te]) / len(te)
        t["err_model"] = rankdata(pred) / len(te) - ry
        t["err_zs"] = rankdata(zs[te]) / len(te) - ry
        t["seed"], t["DMS_id"], t["split"] = seed, dms_id, split
        t["pos_sens"] = t["pos"].map(q).astype(str)
        out.append(t)
    return pd.concat(out)


frames = [run_preds(d, s) for d in SELECTED for s in ["position_heldout", "random"]]
allp = pd.concat(frames, ignore_index=True)
allp["type"] = ref.loc[allp["DMS_id"], "coarse_selection_type"].values
allp["mt_class"] = allp["mt"].map(mt_class)
allp["dKD"] = allp["mt"].map(KD) - allp["wt"].map(KD)
allp["dKD_bin"] = pd.cut(allp["dKD"], [-99, -4, -1, 1, 4, 99],
                         labels=["much_more_polar", "more_polar", "similar",
                                 "more_hydrophobic", "much_more_hydrophobic"]).astype(str)
allp.to_csv("data/selected/error_analysis_preds.csv", index=False)


def summarize(d, col, min_n=200):
    s = d.groupby(col).agg(
        n=("err_model", "size"),
        signed=("err_model", "mean"),
        abs_model=("err_model", lambda x: x.abs().mean()),
        abs_zs=("err_zs", lambda x: x.abs().mean()))
    s["model_minus_zs"] = s["abs_model"] - s["abs_zs"]  # negative = model better
    return s[s.n >= min_n].round(3)


print("\nColumns: signed = mean(pred rank - true rank), >0 means over-ranked.")
print("abs_model / abs_zs = mean absolute rank error (lower is better).")
print("model_minus_zs < 0 means the supervised model beats zero-shot in that group.")

for split in ["position_heldout", "random"]:
    d = allp[allp.split == split]
    for col in ["mt_class", "dKD_bin", "pos_sens", "type"]:
        print(f"\n--- {split} | by {col} ---")
        print(summarize(d, col).to_string())

d = allp[allp.split == "position_heldout"]
cell = d.groupby(["DMS_id", "mt_class"]).agg(
    n=("err_model", "size"), signed=("err_model", "mean")).reset_index()
cell = cell[cell.n >= 100].sort_values("signed")
print("\n--- Most UNDER-ranked (assay x mutant class), held-out ---")
print(cell.head(8).round(3).to_string(index=False))
print("\n--- Most OVER-ranked (assay x mutant class), held-out ---")
print(cell.tail(8).round(3).to_string(index=False))

print("\nSaved: data/selected/error_analysis_preds.csv")