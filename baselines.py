import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from scipy import sparse
from sklearn.linear_model import Ridge
from transformers import AutoTokenizer, EsmForMaskedLM

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
AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AAS)}

ref = pd.read_csv("data/DMS_substitutions.csv").set_index("DMS_id")

name = "facebook/esm2_t33_650M_UR50D"
tok = AutoTokenizer.from_pretrained(name)
model = EsmForMaskedLM.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
aa_token_ids = {a: tok.convert_tokens_to_ids(a) for a in AAS}


def parse_singles(df):
    df = df[~df["mutant"].str.contains(":")].copy()
    df["wt"] = df["mutant"].str[0]
    df["mt"] = df["mutant"].str[-1]
    df["pos"] = df["mutant"].str[1:-1].astype(int)  # 1-indexed
    return df.reset_index(drop=True)


@torch.no_grad()
def wt_marginals(wt_seq):
    """Log-probs at every position from one forward pass of the WT sequence."""
    inputs = tok(wt_seq, return_tensors="pt", truncation=True, max_length=1022).to("cuda")
    logits = model(**inputs).logits[0].float()      # (L+2, vocab)
    return torch.log_softmax(logits, dim=-1)[1:-1].cpu()   # drop CLS/EOS -> (L, vocab)


def zero_shot_scores(df, wt_seq):
    logp = wt_marginals(wt_seq)
    scores = []
    for wt, mt, pos in zip(df["wt"], df["mt"], df["pos"]):
        i = pos - 1
        scores.append((logp[i, aa_token_ids[mt]] - logp[i, aa_token_ids[wt]]).item())
    return np.array(scores)


def additive_features(df, L):
    n = len(df)
    pos_oh = sparse.csr_matrix((np.ones(n), (np.arange(n), df["pos"].values - 1)), shape=(n, L))
    mt_oh = sparse.csr_matrix((np.ones(n), (np.arange(n), df["mt"].map(AA_IDX).values)), shape=(n, 20))
    wt_oh = sparse.csr_matrix((np.ones(n), (np.arange(n), df["wt"].map(AA_IDX).values)), shape=(n, 20))
    return sparse.hstack([pos_oh, mt_oh, wt_oh]).tocsr()


results = []
for dms_id in SELECTED:
    wt_seq = ref.loc[dms_id, "target_seq"]
    L = len(wt_seq)
    df = parse_singles(pd.read_csv(f"data/selected/{dms_id}.csv"))

    # Sanity check: does the WT residue in the mutant name match the sequence?
    match = np.mean([wt_seq[p - 1] == w for p, w in zip(df["pos"], df["wt"])
                     if 1 <= p <= L])
    if match < 0.99:
        print(f"!! {dms_id}: WT match only {match:.2f} (possible position offset)")

    # Zero-shot
    df["zs"] = zero_shot_scores(df, wt_seq)
    zs_rho = spearmanr(df["zs"], df["DMS_score"]).correlation

    # Additive ridge, random 80/20, 5 seeds
    X = additive_features(df, L)
    y = df["DMS_score"].values
    rhos = []
    for seed in range(5):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(df))
        cut = int(0.8 * len(df))
        tr, te = idx[:cut], idx[cut:]
        m = Ridge(alpha=1.0).fit(X[tr], y[tr])
        rhos.append(spearmanr(m.predict(X[te]), y[te]).correlation)

    df.to_csv(f"data/selected/{dms_id}_singles_zs.csv", index=False)
    results.append({
        "DMS_id": dms_id, "n_single": len(df), "wt_match": round(match, 3),
        "zero_shot_rho": round(zs_rho, 3),
        "ridge_80pct_rho": round(np.mean(rhos), 3),
        "ridge_std": round(np.std(rhos), 3),
    })
    print(f"done: {dms_id}")

out = pd.DataFrame(results)
pd.set_option("display.width", 200)
print("\n", out.to_string(index=False))
out.to_csv("data/selected/baselines.csv", index=False)
print("\nPeak VRAM (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))