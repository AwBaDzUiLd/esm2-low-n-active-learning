import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.width", 220)

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
STRONG = {"GRB2_HUMAN_Faure_2021",
          "RASK_HUMAN_Weng_2022_binding-DARPin_K55",
          "Q53Z42_HUMAN_McShan_2019_expression"}
STRATS = ["random", "zs_greedy", "greedy", "ucb", "uncertainty"]
N0, BATCH, ROUNDS = 24, 24, 7
FINAL = N0 + BATCH * ROUNDS          # 192 labels
SEEDS = range(20)
TOP_FRAC, PCA_DIMS, BETA = 0.05, 64, 1.0


def load(dms_id):
    df = pd.read_csv(f"data/selected/{dms_id}_singles_zs.csv")
    Xpos = StandardScaler().fit_transform(
        np.load(f"data/emb/{dms_id}_pos.npy").astype(np.float32))
    X = PCA(n_components=PCA_DIMS, random_state=0).fit_transform(Xpos)
    y, zs = df["DMS_score"].values, df["zs"].values
    n = len(y)
    k = max(1, int(np.ceil(TOP_FRAC * n)))
    is_top = np.zeros(n, bool)
    is_top[np.argsort(-y)[:k]] = True
    return dict(X=X, y=y, zs=zs, n=n, k=k, is_top=is_top, rank_pct=rankdata(y) / n)


def fit_predict(D, lab, cand):
    zs, y, X = D["zs"], D["y"], D["X"]
    lr = LinearRegression().fit(zs[lab].reshape(-1, 1), y[lab])
    resid = y[lab] - lr.predict(zs[lab].reshape(-1, 1))
    br = BayesianRidge().fit(X[lab], resid)
    mu_r, sd = br.predict(X[cand], return_std=True)
    return lr.predict(zs[cand].reshape(-1, 1)) + mu_r, sd


def record(rows, D, dms_id, strat, seed, lab):
    lab = np.asarray(lab)
    hits = int(D["is_top"][lab].sum())
    rows.append({
        "DMS_id": dms_id, "strategy": strat, "seed": seed, "n_labels": len(lab),
        "hits": hits,
        "enrichment": (hits / len(lab)) / (D["k"] / D["n"]),
        "best_pct": D["rank_pct"][lab].max(),
    })


def campaign(D, dms_id, strat, seed, rows):
    init = np.random.RandomState(seed).permutation(D["n"])[:N0]   # same for all strategies
    rng = np.random.RandomState(1000 + seed)
    lab = list(init)
    mask = np.ones(D["n"], bool)
    mask[init] = False
    record(rows, D, dms_id, strat, seed, lab)
    for _ in range(ROUNDS):
        cand = np.where(mask)[0]
        if strat == "random":
            pick = rng.choice(cand, BATCH, replace=False)
        elif strat == "zs_greedy":
            pick = cand[np.argsort(-D["zs"][cand])[:BATCH]]
        else:
            mu, sd = fit_predict(D, np.array(lab), cand)
            score = {"greedy": mu, "ucb": mu + BETA * sd, "uncertainty": sd}[strat]
            pick = cand[np.argsort(-score)[:BATCH]]
        lab.extend(pick.tolist())
        mask[pick] = False
        record(rows, D, dms_id, strat, seed, lab)


rows = []
for dms_id in SELECTED:
    D = load(dms_id)
    for seed in tqdm(SEEDS, desc=dms_id[:34], leave=False):
        for strat in STRATS:
            campaign(D, dms_id, strat, seed, rows)
    print(f"done: {dms_id}  (pool={D['n']}, top-5% size={D['k']})")

res = pd.DataFrame(rows)
res["group"] = np.where(res.DMS_id.isin(STRONG), "prior_strong", "prior_weak")
res.to_csv("data/selected/active_results.csv", index=False)
per = res.groupby(["DMS_id", "group", "strategy", "n_labels"])[
    ["enrichment", "best_pct"]].mean().reset_index()

for L in [96, FINAL]:
    p = per[per.n_labels == L].pivot_table(index="DMS_id", columns="strategy",
                                           values="enrichment")[STRATS]
    print(f"\n=== Top-5% enrichment at {L} labels (1.0 = random; mean of 20 seeds) ===")
    print(p.round(2).to_string())

print("\n=== Enrichment, average over all 8 assays ===")
print(per.pivot_table(index="n_labels", columns="strategy",
                      values="enrichment")[STRATS].round(2).to_string())

print("\n=== Enrichment by prior strength (exploratory grouping) ===")
print(per.pivot_table(index=["group", "n_labels"], columns="strategy",
                      values="enrichment")[STRATS].round(2).to_string())

print("\n=== Best percentile found, average over assays ===")
print(per.pivot_table(index="n_labels", columns="strategy",
                      values="best_pct")[STRATS].round(3).to_string())

print(f"\n=== Paired comparison vs random at {FINAL} labels (per assay x seed) ===")
w = res[res.n_labels == FINAL].pivot_table(index=["DMS_id", "seed"],
                                           columns="strategy", values="hits")
for s in STRATS[1:]:
    d = w[s] - w["random"]
    win = (d > 0).mean() + 0.5 * (d == 0).mean()
    print(f"{s:12s} mean extra hits {d.mean():+6.2f}   win-rate {win:.2f} (ties count half)")

# ---- Plots ----
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for ax, dms_id in zip(axes.ravel(), SELECTED):
    sub = res[res.DMS_id == dms_id]
    for s in STRATS:
        t = sub[sub.strategy == s].groupby("n_labels")["enrichment"].agg(["mean", "std", "count"])
        se = t["std"] / np.sqrt(t["count"])
        ax.plot(t.index, t["mean"], marker="o", label=s)
        ax.fill_between(t.index, t["mean"] - se, t["mean"] + se, alpha=0.15)
    ax.axhline(1, ls="--", color="gray")
    ax.set_title(dms_id[:34], fontsize=9)
    ax.set_xlabel("labels used"); ax.grid(alpha=0.3)
axes[0, 0].set_ylabel("top-5% enrichment"); axes[1, 0].set_ylabel("top-5% enrichment")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Active learning: enrichment of top-5% variants")
fig.tight_layout()
fig.savefig("data/selected/active_per_assay.png", dpi=130)
print("\nSaved: data/selected/active_results.csv, active_per_assay.png")