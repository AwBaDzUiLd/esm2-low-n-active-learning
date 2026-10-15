import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

pd.set_option("display.width", 220)
ids = [l.strip() for l in open("holdout_assays.txt") if l.strip()]
ASSAYS = [a for a in ids if os.path.exists(f"data/emb/{a}_pos.npy")
          and os.path.exists(f"data/selected/{a}_singles_zs.csv")]
missing = [a for a in ids if a not in ASSAYS]
if missing:
    print("!! not prepared (will be reported as skipped):", missing)

STRATS = {
    "random":        dict(init="random", mode=lambda r: "random"),
    "zs_greedy":     dict(init="random", mode=lambda r: "zs"),
    "greedy":        dict(init="random", mode=lambda r: "ucb0"),
    "ucb1":          dict(init="random", mode=lambda r: "ucb1"),
    "hybrid1":       dict(init="random", mode=lambda r: "zs" if r < 1 else "ucb0"),
    "seeded_greedy": dict(init="seeded", mode=lambda r: "ucb0"),
}
NAMES = list(STRATS)
N0, BATCH, ROUNDS, SEEDS = 24, 24, 7, range(20)
TOP_FRAC, PCA_DIMS = 0.05, 64


def load(a):
    df = pd.read_csv(f"data/selected/{a}_singles_zs.csv")
    Xs = StandardScaler().fit_transform(np.load(f"data/emb/{a}_pos.npy").astype(np.float32))
    X = PCA(n_components=PCA_DIMS, random_state=0).fit_transform(Xs)
    y, zs = df["DMS_score"].values, df["zs"].values
    n = len(y)
    k = int(np.ceil(TOP_FRAC * n))
    is_top = np.zeros(n, bool)
    is_top[np.argsort(-y)[:k]] = True
    return dict(X=X, y=y, zs=zs, n=n, k=k, is_top=is_top,
                rho_zs=spearmanr(zs, y).correlation)


def fit_predict(D, lab, cand):
    lr = LinearRegression().fit(D["zs"][lab].reshape(-1, 1), D["y"][lab])
    resid = D["y"][lab] - lr.predict(D["zs"][lab].reshape(-1, 1))
    br = BayesianRidge().fit(D["X"][lab], resid)
    mu_r, sd = br.predict(D["X"][cand], return_std=True)
    return lr.predict(D["zs"][cand].reshape(-1, 1)) + mu_r, sd


def record(rows, D, a, name, seed, lab):
    hits = int(D["is_top"][np.asarray(lab)].sum())
    rows.append(dict(DMS_id=a, strategy=name, seed=seed, n_labels=len(lab), hits=hits,
                     enrichment=(hits / len(lab)) / (D["k"] / D["n"])))


def campaign(D, a, name, cfg, seed, rows):
    perm = np.random.RandomState(seed).permutation(D["n"])
    if cfg["init"] == "random":
        init = perm[:N0]
    else:
        top = np.argsort(-D["zs"])[:N0 // 2]
        topset = set(top.tolist())
        init = np.concatenate([top, [i for i in perm if i not in topset][:N0 - len(top)]])
    rng = np.random.RandomState(1000 + seed)
    lab, mask = list(init), np.ones(D["n"], bool)
    mask[init] = False
    record(rows, D, a, name, seed, lab)
    for r in range(ROUNDS):
        cand = np.where(mask)[0]
        mode = cfg["mode"](r)
        if mode == "random":
            pick = rng.choice(cand, BATCH, replace=False)
        elif mode == "zs":
            pick = cand[np.argsort(-D["zs"][cand])[:BATCH]]
        else:
            mu, sd = fit_predict(D, np.array(lab), cand)
            pick = cand[np.argsort(-(mu + float(mode[3:]) * sd))[:BATCH]]
        lab.extend(pick.tolist())
        mask[pick] = False
        record(rows, D, a, name, seed, lab)


rows, info = [], []
for a in ASSAYS:
    D = load(a)
    info.append(dict(DMS_id=a, pool=D["n"], top5_size=D["k"], zero_shot_rho=round(D["rho_zs"], 3)))
    for seed in tqdm(SEEDS, desc=a[:34], leave=False):
        for name, cfg in STRATS.items():
            campaign(D, a, name, cfg, seed, rows)
    print(f"done: {a}")

res = pd.DataFrame(rows)
res.to_csv("data/selected/holdout_active_results.csv", index=False)
per = res.groupby(["DMS_id", "strategy", "n_labels"])["enrichment"].mean().reset_index()

print("\n=== Hold-out assays ===")
print(pd.DataFrame(info).to_string(index=False))
print("\n=== Enrichment, average over assays (SANITY: random ~1.0) ===")
print(per.pivot_table(index="n_labels", columns="strategy", values="enrichment")[NAMES]
      .loc[[48, 96, 144, 192]].round(2).to_string())
print("\n=== Enrichment at 192 labels, per assay ===")
print(per[per.n_labels == 192].pivot_table(index="DMS_id", columns="strategy",
                                           values="enrichment")[NAMES].round(2).to_string())

# ---- Pre-specified test (see PLAN.md): campaign-average enrichment, paired vs greedy ----
camp = (res[res.n_labels >= 48].groupby(["DMS_id", "strategy", "seed"])["enrichment"]
        .mean().reset_index().pivot_table(index=["DMS_id", "seed"], columns="strategy",
                                          values="enrichment"))
print("\n=== Campaign-average enrichment (48..192 labels), average over assays ===")
print(camp.groupby("DMS_id").mean().mean()[NAMES].round(2).to_string())

tab, summ = {}, []
for s in [x for x in NAMES if x != "greedy"]:
    d = (camp[s] - camp["greedy"]).groupby("DMS_id").agg(["mean", "std", "count"])
    se = d["std"] / np.sqrt(d["count"])
    tab[s] = d.apply(lambda r: f"{r['mean']:+.2f}±{r['std']/np.sqrt(r['count']):.2f}", axis=1)
    b, w = int((d["mean"] > 2 * se).sum()), int((d["mean"] < -2 * se).sum())
    summ.append(dict(strategy=s, mean_diff=round(d["mean"].mean(), 3), better=b,
                     same=len(d) - b - w, worse=w,
                     verdict=("REPLICATED" if (d["mean"].mean() > 0 and b > w) else "NOT replicated")
                     if s in ("hybrid1", "seeded_greedy") else "(reference only)"))
print("\n=== Paired difference in campaign-average enrichment vs greedy (mean ± SE over 20 seeds) ===")
print(pd.DataFrame(tab).to_string())
print()
print(pd.DataFrame(summ).to_string(index=False))
print(f"\nAssays evaluated: {len(ASSAYS)} of {len(ids)}" + (f"; skipped: {missing}" if missing else ""))