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

# mode per round: "random", "zs", or "ucb<beta>"  (ucb0 = pure greedy)
STRATS = {
    "random":        dict(init="random", mode=lambda r: "random"),
    "zs_greedy":     dict(init="random", mode=lambda r: "zs"),
    "greedy":        dict(init="random", mode=lambda r: "ucb0"),
    "ucb0.5":        dict(init="random", mode=lambda r: "ucb0.5"),
    "ucb1":          dict(init="random", mode=lambda r: "ucb1"),
    "ucb2":          dict(init="random", mode=lambda r: "ucb2"),
    "hybrid1":       dict(init="random", mode=lambda r: "zs" if r < 1 else "ucb0"),
    "hybrid2":       dict(init="random", mode=lambda r: "zs" if r < 2 else "ucb0"),
    "seeded_greedy": dict(init="seeded", mode=lambda r: "ucb0"),
}
NAMES = list(STRATS)
N0, BATCH, ROUNDS = 24, 24, 7
FINAL = N0 + BATCH * ROUNDS
SEEDS = range(20)
TOP_FRAC, PCA_DIMS = 0.05, 64


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
    return dict(X=X, y=y, zs=zs, n=n, k=k, is_top=is_top)


def fit_predict(D, lab, cand):
    zs, y, X = D["zs"], D["y"], D["X"]
    lr = LinearRegression().fit(zs[lab].reshape(-1, 1), y[lab])
    resid = y[lab] - lr.predict(zs[lab].reshape(-1, 1))
    br = BayesianRidge().fit(X[lab], resid)
    mu_r, sd = br.predict(X[cand], return_std=True)
    return lr.predict(zs[cand].reshape(-1, 1)) + mu_r, sd


def record(rows, D, dms_id, name, seed, lab):
    hits = int(D["is_top"][np.asarray(lab)].sum())
    rows.append({"DMS_id": dms_id, "strategy": name, "seed": seed,
                 "n_labels": len(lab), "hits": hits,
                 "enrichment": (hits / len(lab)) / (D["k"] / D["n"])})


def campaign(D, dms_id, name, cfg, seed, rows):
    perm = np.random.RandomState(seed).permutation(D["n"])
    if cfg["init"] == "random":
        init = perm[:N0]
    else:  # 12 best by zero-shot + 12 random others
        top = np.argsort(-D["zs"])[:N0 // 2]
        rest = [i for i in perm if i not in set(top)][:N0 - len(top)]
        init = np.concatenate([top, rest])
    rng = np.random.RandomState(1000 + seed)
    lab = list(init)
    mask = np.ones(D["n"], bool)
    mask[init] = False
    record(rows, D, dms_id, name, seed, lab)
    for r in range(ROUNDS):
        cand = np.where(mask)[0]
        mode = cfg["mode"](r)
        if mode == "random":
            pick = rng.choice(cand, BATCH, replace=False)
        elif mode == "zs":
            pick = cand[np.argsort(-D["zs"][cand])[:BATCH]]
        else:
            beta = float(mode[3:])
            mu, sd = fit_predict(D, np.array(lab), cand)
            pick = cand[np.argsort(-(mu + beta * sd))[:BATCH]]
        lab.extend(pick.tolist())
        mask[pick] = False
        record(rows, D, dms_id, name, seed, lab)


rows = []
for dms_id in SELECTED:
    D = load(dms_id)
    for seed in tqdm(SEEDS, desc=dms_id[:34], leave=False):
        for name, cfg in STRATS.items():
            campaign(D, dms_id, name, cfg, seed, rows)
    print(f"done: {dms_id}")

res = pd.DataFrame(rows)
res["group"] = np.where(res.DMS_id.isin(STRONG), "prior_strong", "prior_weak")
res.to_csv("data/selected/active_hybrid_results.csv", index=False)

per = res.groupby(["DMS_id", "group", "strategy", "n_labels"])["enrichment"].mean().reset_index()

print("\n=== Enrichment, average over 8 assays (SANITY: random~0.98, zs_greedy~1.99, greedy~3.00 at 192) ===")
t = per.pivot_table(index="n_labels", columns="strategy", values="enrichment")[NAMES]
print(t.loc[[48, 72, 96, 144, 192]].round(2).to_string())

print("\n=== By prior-strength group (exploratory grouping) ===")
t = per.pivot_table(index=["group", "n_labels"], columns="strategy", values="enrichment")[NAMES]
print(t[t.index.get_level_values("n_labels").isin([48, 96, 192])].round(2).to_string())

# Mean enrichment over rounds 48..192 = how good the whole campaign was, not just the end
auc = res[res.n_labels >= 48].groupby(["DMS_id", "strategy", "seed"])["enrichment"].mean().reset_index()
print("\n=== Campaign-average enrichment (mean over 48..192 labels), average over assays ===")
print(auc.groupby("strategy")["enrichment"].mean()[NAMES].round(2).to_string())

# Paired differences vs greedy (same seeds, same starting set except seeded_greedy)
print("\n=== Paired difference vs greedy, per assay (mean +/- SE over 20 seeds) ===")
print("Verdict counts: better / worse = difference beyond 2 SE; same = within 2 SE.")
for L in [96, FINAL]:
    w = res[res.n_labels == L].pivot_table(index=["DMS_id", "seed"],
                                           columns="strategy", values="enrichment")
    tab, summ = {}, []
    for s in NAMES:
        if s == "greedy":
            continue
        d = (w[s] - w["greedy"]).groupby("DMS_id").agg(["mean", "std", "count"])
        se = d["std"] / np.sqrt(d["count"])
        tab[s] = d.apply(lambda r: f"{r['mean']:+.2f}±{r['std']/np.sqrt(r['count']):.2f}", axis=1)
        better = int((d["mean"] > 2 * se).sum())
        worse = int((d["mean"] < -2 * se).sum())
        summ.append({"strategy": s, "mean_diff_over_assays": round(d["mean"].mean(), 3),
                     "better": better, "same": len(d) - better - worse, "worse": worse})
    print(f"\n--- at {L} labels ---")
    print(pd.DataFrame(tab).to_string())
    print(pd.DataFrame(summ).to_string(index=False))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
panels = [("all 8 assays", per),
          ("prior-strong (3)", per[per.group == "prior_strong"]),
          ("prior-weak (5)", per[per.group == "prior_weak"])]
for ax, (title, d) in zip(axes, panels):
    c = d.groupby(["strategy", "n_labels"])["enrichment"].mean().unstack("strategy")
    for s in NAMES:
        ax.plot(c.index, c[s], marker="o", ms=3, label=s)
    ax.axhline(1, ls="--", color="gray")
    ax.set_title(title); ax.set_xlabel("labels used"); ax.grid(alpha=0.3)
axes[0].set_ylabel("top-5% enrichment (1 = random)")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig("data/selected/active_hybrid.png", dpi=140)
print("\nSaved: data/selected/active_hybrid_results.csv, active_hybrid.png")