import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

pd.set_option("display.width", 220)
ASSAYS = [
    "SPG1_STRSG_Olson_2014",
    "RASK_HUMAN_Weng_2022_binding-DARPin_K55",
    "RASK_HUMAN_Weng_2022_abundance",
    "GRB2_HUMAN_Faure_2021",
]
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
    pool = pd.read_csv(f"data/selected/{a}_pool.csv")
    Xs = StandardScaler().fit_transform(np.load(f"data/emb/{a}_pool_pos.npy").astype(np.float32))
    X = PCA(n_components=PCA_DIMS, random_state=0).fit_transform(Xs)
    y, zs = pool["DMS_score"].values, pool["zs"].values
    n = len(y)
    k = int(np.ceil(TOP_FRAC * n))
    is_top = np.zeros(n, bool)
    is_top[np.argsort(-y)[:k]] = True
    return dict(X=X, y=y, zs=zs, n=n, k=k, is_top=is_top, multi=(pool["n_mut"].values > 1))


def fit_predict(D, lab, cand):
    lr = LinearRegression().fit(D["zs"][lab].reshape(-1, 1), D["y"][lab])
    resid = D["y"][lab] - lr.predict(D["zs"][lab].reshape(-1, 1))
    br = BayesianRidge().fit(D["X"][lab], resid)
    mu_r, sd = br.predict(D["X"][cand], return_std=True)
    return lr.predict(D["zs"][cand].reshape(-1, 1)) + mu_r, sd


def record(rows, D, a, name, seed, lab):
    lab = np.asarray(lab)
    hits = int(D["is_top"][lab].sum())
    rows.append(dict(DMS_id=a, strategy=name, seed=seed, n_labels=len(lab), hits=hits,
                     enrichment=(hits / len(lab)) / (D["k"] / D["n"]),
                     frac_multi=float(D["multi"][lab].mean())))


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


rows = []
for a in ASSAYS:
    D = load(a)
    print(f"{a}: pool={D['n']}, top-5% size={D['k']}, top-5% that are doubles={D['multi'][D['is_top']].mean():.2f}")
    for seed in tqdm(SEEDS, desc=a[:34], leave=False):
        for name, cfg in STRATS.items():
            campaign(D, a, name, cfg, seed, rows)

res = pd.DataFrame(rows)
res.to_csv("data/selected/step12_active_results.csv", index=False)
per = res.groupby(["DMS_id", "strategy", "n_labels"])[["enrichment", "frac_multi"]].mean().reset_index()

print("\n=== Enrichment, average over 4 assays (SANITY: random ~1.0) ===")
print(per.pivot_table(index="n_labels", columns="strategy", values="enrichment")[NAMES]
      .loc[[48, 96, 144, 192]].round(2).to_string())
print("\n=== Enrichment at 192 labels, per assay ===")
print(per[per.n_labels == 192].pivot_table(index="DMS_id", columns="strategy",
                                           values="enrichment")[NAMES].round(2).to_string())
print("\n=== Campaign-average enrichment (48..192 labels) ===")
auc = res[res.n_labels >= 48].groupby(["DMS_id", "strategy", "seed"])["enrichment"].mean().reset_index()
print(auc.groupby("strategy")["enrichment"].mean()[NAMES].round(2).to_string())
print("\n=== Fraction of labeled variants that are doubles, at 192 labels, per assay ===")
print(per[per.n_labels == 192].pivot_table(index="DMS_id", columns="strategy",
                                           values="frac_multi")[NAMES].round(2).to_string())

for L in [96, 192]:
    w = res[res.n_labels == L].pivot_table(index=["DMS_id", "seed"], columns="strategy", values="enrichment")
    tab, summ = {}, []
    for s in NAMES:
        if s == "greedy":
            continue
        d = (w[s] - w["greedy"]).groupby("DMS_id").agg(["mean", "std", "count"])
        se = d["std"] / np.sqrt(d["count"])
        tab[s] = d.apply(lambda r: f"{r['mean']:+.2f}±{r['std']/np.sqrt(r['count']):.2f}", axis=1)
        b, wr = int((d["mean"] > 2 * se).sum()), int((d["mean"] < -2 * se).sum())
        summ.append(dict(strategy=s, mean_diff=round(d["mean"].mean(), 3), better=b, same=len(d) - b - wr, worse=wr))
    print(f"\n=== Paired difference vs greedy at {L} labels (mean ± SE over 20 seeds) ===")
    print(pd.DataFrame(tab).to_string())
    print(pd.DataFrame(summ).to_string(index=False))