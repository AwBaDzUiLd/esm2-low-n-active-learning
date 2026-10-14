import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
MODELS = ["zero_shot", "additive_onehot", "ridge_mean", "ridge_pos", "ridge_pos+zs"]
NS = [24, 48, 96, 192]
SEEDS = range(20)
SPLITS = ["random", "position_heldout"]
ALPHAS = np.logspace(0, 6, 13)
AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AAS)}


def additive_dense(df):
    L = int(df["pos"].max())
    n = len(df)
    X = np.zeros((n, L + 40), dtype=np.float32)
    X[np.arange(n), df["pos"].values - 1] = 1
    X[np.arange(n), L + df["mt"].map(AA_IDX).values] = 1
    X[np.arange(n), L + 20 + df["wt"].map(AA_IDX).values] = 1
    return X


def load(dms_id):
    df = pd.read_csv(f"data/selected/{dms_id}_singles_zs.csv")
    mean = np.load(f"data/emb/{dms_id}_mean.npy").astype(np.float32)
    pos = np.load(f"data/emb/{dms_id}_pos.npy").astype(np.float32)
    assert len(df) == len(mean) == len(pos), "embedding/data length mismatch"
    return {
        "df": df,
        "y": df["DMS_score"].values,
        "zs": df["zs"].values,
        "Xmean": StandardScaler().fit_transform(mean),
        "Xpos": StandardScaler().fit_transform(pos),
        "Xadd": additive_dense(df),
    }


def make_split(df, split, seed):
    rng = np.random.RandomState(seed)
    k = max(NS)
    if split == "random":
        idx = rng.permutation(len(df))
        return idx[:k], idx[k:]
    positions = np.unique(df["pos"].values)
    rng.shuffle(positions)
    half = len(positions) // 2
    train_pos, test_pos = positions[:half], positions[half:]
    pool = np.where(np.isin(df["pos"].values, train_pos))[0]
    te = np.where(np.isin(df["pos"].values, test_pos))[0]
    pool = rng.permutation(pool)
    return pool[:k], te


def fit_predict(model, tr, te, D):
    y, zs = D["y"], D["zs"]
    if model == "zero_shot":
        return zs[te]
    if model == "additive_onehot":
        return RidgeCV(alphas=ALPHAS).fit(D["Xadd"][tr], y[tr]).predict(D["Xadd"][te])
    if model == "ridge_mean":
        return RidgeCV(alphas=ALPHAS).fit(D["Xmean"][tr], y[tr]).predict(D["Xmean"][te])
    if model == "ridge_pos":
        return RidgeCV(alphas=ALPHAS).fit(D["Xpos"][tr], y[tr]).predict(D["Xpos"][te])
    if model == "ridge_pos+zs":
        lr = LinearRegression().fit(zs[tr].reshape(-1, 1), y[tr])
        resid = y[tr] - lr.predict(zs[tr].reshape(-1, 1))
        rg = RidgeCV(alphas=ALPHAS).fit(D["Xpos"][tr], resid)
        return lr.predict(zs[te].reshape(-1, 1)) + rg.predict(D["Xpos"][te])
    raise ValueError(model)


def top10_overlap(pred, y):
    k = max(1, int(0.1 * len(y)))
    a = set(np.argsort(-pred)[:k])
    b = set(np.argsort(-y)[:k])
    return len(a & b) / k


rows = []
for dms_id in SELECTED:
    D = load(dms_id)
    for split in SPLITS:
        for seed in tqdm(SEEDS, desc=f"{dms_id[:32]} | {split}", leave=False):
            order_tr, te = make_split(D["df"], split, seed)
            if len(te) < 50 or len(order_tr) < max(NS):
                print(f"!! skipping {dms_id} {split} seed {seed}: too few variants")
                continue
            for n in NS:
                tr = order_tr[:n]
                for m in MODELS:
                    pred = fit_predict(m, tr, te, D)
                    rows.append({
                        "DMS_id": dms_id, "split": split, "seed": seed, "n": n,
                        "model": m,
                        "rho": spearmanr(pred, D["y"][te]).correlation,
                        "top10": top10_overlap(pred, D["y"][te]),
                    })
    print(f"done: {dms_id}")

res = pd.DataFrame(rows)
res.to_csv("data/selected/lowN_results.csv", index=False)

# ---- Summary tables ----
g = res.groupby(["DMS_id", "split", "model", "n"])["rho"].mean().reset_index()
pd.set_option("display.width", 220)
for n in [24, 192]:
    p = g[g.n == n].pivot_table(index=["DMS_id", "split"], columns="model", values="rho")
    print(f"\n=== Spearman, N = {n} (mean of 20 seeds) ===")
    print(p[MODELS].round(3).to_string())

print("\n=== Average across all 8 assays ===")
avg = g.groupby(["split", "n", "model"])["rho"].mean().reset_index()
print(avg.pivot_table(index=["split", "n"], columns="model", values="rho")[MODELS].round(3).to_string())

# ---- Learning-curve plots ----
for split in SPLITS:
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)
    for ax, dms_id in zip(axes.ravel(), SELECTED):
        sub = res[(res.DMS_id == dms_id) & (res.split == split)]
        for m in MODELS:
            if m == "zero_shot":
                ax.axhline(sub[sub.model == m]["rho"].mean(), ls="--", color="gray", label=m)
                continue
            s = sub[sub.model == m].groupby("n")["rho"].agg(["mean", "std", "count"])
            se = s["std"] / np.sqrt(s["count"])
            ax.plot(s.index, s["mean"], marker="o", label=m)
            ax.fill_between(s.index, s["mean"] - se, s["mean"] + se, alpha=0.2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(NS); ax.set_xticklabels(NS)
        ax.set_title(dms_id[:34], fontsize=9)
        ax.set_xlabel("N training variants"); ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Spearman"); axes[1, 0].set_ylabel("Spearman")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Low-N learning curves ({split})")
    fig.tight_layout()
    fig.savefig(f"data/selected/lowN_{split}.png", dpi=130)
    plt.close(fig)

print("\nSaved: data/selected/lowN_results.csv, lowN_random.png, lowN_position_heldout.png")