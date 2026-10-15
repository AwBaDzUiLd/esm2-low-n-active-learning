import glob, os
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM

AAS = "ACDEFGHIKLMNPQRSTVWY"
ids = [l.strip() for l in open("holdout_assays.txt") if l.strip()]
ref = pd.read_csv("data/DMS_substitutions.csv").set_index("DMS_id")
os.makedirs("data/selected", exist_ok=True)
os.makedirs("data/emb", exist_ok=True)

name = "facebook/esm2_t33_650M_UR50D"
tok = AutoTokenizer.from_pretrained(name)
aa_id = {a: tok.convert_tokens_to_ids(a) for a in AAS}

# Stage 0: load singles and check numbering (CPU)
dfs = {}
for a in ids:
    if os.path.exists(f"data/emb/{a}_pos.npy") and os.path.exists(f"data/selected/{a}_singles_zs.csv"):
        print(f"skip (already done): {a}")
        continue
    hits = glob.glob(f"data/dms_assays/**/{ref.loc[a, 'DMS_filename']}", recursive=True)
    if not hits:
        print(f"!! file not found: {a}")
        continue
    df = pd.read_csv(hits[0])
    df = df[~df["mutant"].str.contains(":")].dropna(subset=["DMS_score"]).copy()
    df["wt"], df["mt"] = df["mutant"].str[0], df["mutant"].str[-1]
    df["pos"] = df["mutant"].str[1:-1].astype(int)
    n_before = len(df)
    df = df[df.wt.isin(list(AAS)) & df.mt.isin(list(AAS))].reset_index(drop=True)
    if len(df) < n_before:
        print(f"note: {a}: dropped {n_before - len(df)} variants with non-standard residues")
    wt_seq = ref.loc[a, "target_seq"]
    match = np.mean([1 <= p <= len(wt_seq) and wt_seq[p - 1] == w for p, w in zip(df.pos, df.wt)])
    if match < 0.99 or df["mutated_sequence"].str.len().nunique() != 1:
        print(f"!! {a}: numbering/length check failed (WT match {match:.3f}); not processed")
        continue
    dfs[a] = df
    print(f"loaded {a}: {len(df)} singles, length {len(wt_seq)}, WT match {match:.3f}")

todo = list(dfs)
if todo:
    # Stage 1: zero-shot scores from wild-type marginals
    mlm = EsmForMaskedLM.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        for a in todo:
            inp = tok(ref.loc[a, "target_seq"], return_tensors="pt",
                      truncation=True, max_length=1022).to("cuda")
            logp = torch.log_softmax(mlm(**inp).logits[0].float(), dim=-1)[1:-1].cpu()
            df = dfs[a]
            df["zs"] = [(logp[p - 1, aa_id[m]] - logp[p - 1, aa_id[w]]).item()
                        for w, m, p in zip(df.wt, df.mt, df.pos)]
    del mlm
    torch.cuda.empty_cache()

    # Stage 2: embeddings at the mutated position
    enc = EsmModel.from_pretrained(name, torch_dtype=torch.float16, add_pooling_layer=False).cuda().eval()
    for a in todo:
        df = dfs[a]
        seqs, positions = df["mutated_sequence"].tolist(), df["pos"].values
        bs = 16 if len(seqs[0]) <= 250 else 8
        out = []
        with torch.no_grad():
            for i in tqdm(range(0, len(seqs), bs), desc=a[:34]):
                inp = tok(seqs[i:i + bs], return_tensors="pt", padding=True).to("cuda")
                h = enc(**inp).last_hidden_state
                idx = torch.tensor(positions[i:i + bs], device="cuda")
                out.append(h[torch.arange(len(idx)), idx].cpu().numpy().astype(np.float16))
        df[["mutant", "mutated_sequence", "DMS_score", "wt", "mt", "pos", "zs"]].to_csv(
            f"data/selected/{a}_singles_zs.csv", index=False)
        np.save(f"data/emb/{a}_pos.npy", np.concatenate(out))
        rho = spearmanr(df["zs"], df["DMS_score"]).correlation
        print(f"saved {a} ({len(df)} variants, zero-shot rho {rho:.3f})")
        torch.cuda.empty_cache()
    print("\nPeak VRAM (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))