import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

pd.set_option("display.width", 220)
ASSAYS = [
    "SPG1_STRSG_Olson_2014",
    "RASK_HUMAN_Weng_2022_binding-DARPin_K55",
    "RASK_HUMAN_Weng_2022_abundance",
    "GRB2_HUMAN_Faure_2021",
    "SPA_STAAU_Tsuboyama_2023_1LP1",
]
NS, SEEDS = [24, 96, 192], range(20)
ALPHAS = np.logspace(0, 6, 13)
MODELS = ["zero_shot", "ridge_direct", "ridge_additive", "additive_measured"]


def top10(pred, y):
    k = max(1, int(0.1 * len(y)))
    return len(set(np.argsort(-pred)[:k]) & set(np.argsort(-y)[:k])) / k


rows = []
for a in ASSAYS:
    pool = pd.read_csv(f"data/selected/{a}_pool.csv")
    X = StandardScaler().fit_transform(np.load(f"data/emb/{a}_pool_pos.npy").astype(np.float32))
    is1 = (pool["n_mut"] == 1).values
    S, Xs = pool[is1].reset_index(drop=True), X[is1]
    Dd, Xd = pool[~is1].reset_index(drop=True), X[~is1]

    idx = {m: i for i, m in enumerate(S["mutant"])}
    parts = Dd["mutant"].str.split(":", expand=True)
    i1, i2 = parts[0].map(idx), parts[1].map(idx)
    ok = (i1.notna() & i2.notna()).values
    Dd, Xd = Dd[ok].reset_index(drop=True), Xd[ok]
    i1, i2 = i1[ok].astype(int).values, i2[ok].astype(int).values

    y_s, zs_s = S["DMS_score"].values, S["zs"].values
    y_d, zs_d = Dd["DMS_score"].values, Dd["zs"].values
    add_meas = y_s[i1] + y_s[i2]
    print(f"{a}: {len(S)} singles, {len(Dd)} doubles evaluated")

    def evaluate(tr, n_label, seed):
        lr = LinearRegression().fit(zs_s[tr].reshape(-1, 1), y_s[tr])
        resid = y_s[tr] - lr.predict(zs_s[tr].reshape(-1, 1))
        rg = RidgeCV(alphas=ALPHAS).fit(Xs[tr], resid)
        f = lambda z, x: lr.predict(z.reshape(-1, 1)) + rg.predict(x)
        P = f(zs_s, Xs)
        preds = {"zero_shot": zs_d, "ridge_direct": f(zs_d, Xd),
                 "ridge_additive": P[i1] + P[i2], "additive_measured": add_meas}
        for m, p in preds.items():
            rows.append(dict(DMS_id=a, n=n_label, seed=seed, model=m,
                             rho=spearmanr(p, y_d).correlation, top10=top10(p, y_d)))

    for seed in tqdm(SEEDS, desc=a[:34], leave=False):
        perm = np.random.RandomState(seed).permutation(len(S))
        for n in NS:
            evaluate(perm[:n], n, seed)
    evaluate(np.arange(len(S)), len(S), 0)       # all singles (one fit, no seed variation)

res = pd.DataFrame(rows)
res.to_csv("data/selected/step12_transfer_results.csv", index=False)
res["n_label"] = res["n"].where(res["n"].isin(NS), -1)       # -1 = all singles

for metric in ["rho", "top10"]:
    t = res.groupby(["DMS_id", "n_label", "model"])[metric].mean().unstack("model")[MODELS]
    print(f"\n=== {metric}: singles -> doubles (mean over 20 seeds; n_label -1 = all singles) ===")
    print(t.round(3).to_string())
    for name, keep in [("average over 5 assays", ASSAYS), ("average excluding SPA", ASSAYS[:4])]:
        s = t.reset_index()
        s = s[s.DMS_id.isin(keep)].groupby("n_label")[MODELS].mean()
        print(f"\n{metric}, {name}:")
        print(s.round(3).to_string())