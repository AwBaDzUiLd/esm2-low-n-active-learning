import os, time
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

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

os.makedirs("data/emb", exist_ok=True)
name = "facebook/esm2_t33_650M_UR50D"
tok = AutoTokenizer.from_pretrained(name)
model = EsmModel.from_pretrained(name, torch_dtype=torch.float16, add_pooling_layer=False).cuda().eval()

for dms_id in SELECTED:
    mean_path = f"data/emb/{dms_id}_mean.npy"
    pos_path = f"data/emb/{dms_id}_pos.npy"
    if os.path.exists(mean_path) and os.path.exists(pos_path):
        print(f"skip (already done): {dms_id}")
        continue

    df = pd.read_csv(f"data/selected/{dms_id}_singles_zs.csv")
    seqs = df["mutated_sequence"].tolist()
    positions = df["pos"].values            # 1-indexed; token index = pos (CLS is token 0)
    L = len(seqs[0])
    bs = 16 if L <= 250 else 8

    means, poss = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), bs), desc=dms_id):
            batch = seqs[i:i + bs]
            inputs = tok(batch, return_tensors="pt", padding=True).to("cuda")
            h = model(**inputs).last_hidden_state            # (B, L+2, 1280)
            means.append(h[:, 1:-1].mean(dim=1).cpu().numpy().astype(np.float16))
            idx = torch.tensor(positions[i:i + bs], device="cuda")
            poss.append(h[torch.arange(len(batch)), idx].cpu().numpy().astype(np.float16))

    np.save(mean_path, np.concatenate(means))
    np.save(pos_path, np.concatenate(poss))
    print(f"saved {dms_id} ({len(seqs)} variants, {time.time()-t0:.0f}s)")
    torch.cuda.empty_cache()

print("\nPeak VRAM (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))