import os, json
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM

pd.set_option("display.width", 220)
ASSAYS = [
    "SPG1_STRSG_Olson_2014",
    "RASK_HUMAN_Weng_2022_binding-DARPin_K55",
    "RASK_HUMAN_Weng_2022_abundance",
    "GRB2_HUMAN_Faure_2021",
    "SPA_STAAU_Tsuboyama_2023_1LP1",
]
N_MULTI = 4000
AAS = "ACDEFGHIKLMNPQRSTVWY"

ref = pd.read_csv("data/DMS_substitutions.csv").set_index("DMS_id")
name = "facebook/esm2_t33_650M_UR50D"
tok = AutoTokenizer.from_pretrained(name)
aa_id = {a: tok.convert_tokens_to_ids(a) for a in AAS}


def parse(m):
    return [(p[0], int(p[1:-1]), p[-1]) for p in m.split(":")]


def diag_path(a):
    return f"data/selected/{a}_pool_diag.json"


todo = [a for a in ASSAYS if not os.path.exists(diag_path(a))]

# Stage 1: wild-type log-probs (masked-LM head), then free that model
logps = {}
if todo:
    mlm = EsmForMaskedLM.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        for a in todo:
            inp = tok(ref.loc[a, "target_seq"], return_tensors="pt",
                      truncation=True, max_length=1022).to("cuda")
            lg = mlm(**inp).logits[0].float()
            logps[a] = torch.log_softmax(lg, dim=-1)[1:-1].cpu()
    del mlm
    torch.cuda.empty_cache()
    enc = EsmModel.from_pretrained(name, torch_dtype=torch.float16,
                                   add_pooling_layer=False).cuda().eval()

# Stage 2: build each pool
for a in todo:
    logp = logps[a]
    wt_seq = ref.loc[a, "target_seq"]

    def add_zs(m):
        return sum((logp[p - 1, aa_id[mt]] - logp[p - 1, aa_id[wt]]).item()
                   for wt, p, mt in parse(m))

    singles = pd.read_csv(f"data/selected/{a}_singles_zs.csv")
    full = pd.read_csv(f"data/selected/{a}.csv", usecols=["mutant", "mutated_sequence", "DMS_score"])
    multi_all = full[full["mutant"].str.contains(":")]
    n_multi_total = len(multi_all)
    nmut_dist = (multi_all["mutant"].str.count(":") + 1).value_counts().sort_index().to_dict()
    multi = multi_all.sample(min(N_MULTI, n_multi_total), random_state=0).reset_index(drop=True)
    del full, multi_all

    multi["zs"] = multi["mutant"].map(add_zs)
    zs_check = float(np.abs(singles["mutant"].map(add_zs).values - singles["zs"].values).max())

    # Embeddings: mean hidden state over the mutated positions
    bs = 16 if len(wt_seq) <= 250 else 8
    seqs = multi["mutated_sequence"].tolist()
    posl = [[p for _, p, _ in parse(m)] for m in multi["mutant"]]
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), bs), desc=a[:34], leave=False):
            inp = tok(seqs[i:i + bs], return_tensors="pt", padding=True).to("cuda")
            h = enc(**inp).last_hidden_state                 # token index = position (CLS is 0)
            for j, ps in enumerate(posl[i:i + bs]):
                embs.append(h[j, ps].float().mean(0).cpu().numpy().astype(np.float16))

    pool = pd.concat([
        singles[["mutant", "DMS_score", "zs"]].assign(n_mut=1),
        multi[["mutant", "DMS_score", "zs"]].assign(n_mut=multi["mutant"].str.count(":") + 1),
    ], ignore_index=True)
    pool_emb = np.concatenate([np.load(f"data/emb/{a}_pos.npy"), np.stack(embs)])
    assert len(pool) == len(pool_emb)

    # Diagnostics
    sdict = dict(zip(singles["mutant"], singles["DMS_score"]))

    def add_true(m):
        parts = m.split(":")
        return sum(sdict[p] for p in parts) if all(p in sdict for p in parts) else np.nan

    multi["add_true"] = multi["mutant"].map(add_true)
    ok = multi["add_true"].notna()
    k = int(np.ceil(0.05 * len(pool)))
    top = pool.nlargest(k, "DMS_score")
    diag = dict(
        DMS_id=a, n_singles=len(singles), n_multi_total=int(n_multi_total),
        n_multi_pool=len(multi),
        nmut_dist={int(x): int(y) for x, y in nmut_dist.items()},
        zs_check_maxdiff=zs_check,
        mean_score_singles=float(singles["DMS_score"].mean()),
        mean_score_multi=float(multi["DMS_score"].mean()),
        rho_zs_singles=float(spearmanr(singles["zs"], singles["DMS_score"]).correlation),
        rho_zs_multi=float(spearmanr(multi["zs"], multi["DMS_score"]).correlation),
        frac_multi_with_all_singles=float(ok.mean()),
        rho_additive_measured=(float(spearmanr(multi.loc[ok, "add_true"],
                                               multi.loc[ok, "DMS_score"]).correlation)
                               if ok.sum() > 50 else None),
        top5_frac_multi=float((top["n_mut"] > 1).mean()),
    )
    pool.to_csv(f"data/selected/{a}_pool.csv", index=False)
    np.save(f"data/emb/{a}_pool_pos.npy", pool_emb)
    with open(diag_path(a), "w") as fh:
        json.dump(diag, fh)
    print(f"saved pool: {a} ({len(singles)} singles + {len(multi)} multi)")
    torch.cuda.empty_cache()

rows = [json.load(open(diag_path(a))) for a in ASSAYS]
d = pd.DataFrame(rows).set_index("DMS_id")
print("\n=== Pool diagnostics ===")
print(d.drop(columns="nmut_dist").round(3).T.to_string())
print("\nMutations per multi-variant (count of variants in the full assay):")
print(d["nmut_dist"].to_string())
print("\nPeak VRAM (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))