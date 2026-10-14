# select_holdout.py
import numpy as np, pandas as pd

USED = [
    "SPG1_STRSG_Olson_2014", "RASK_HUMAN_Weng_2022_binding-DARPin_K55",
    "RASK_HUMAN_Weng_2022_abundance", "Q53Z42_HUMAN_McShan_2019_binding-TAPBPR",
    "Q53Z42_HUMAN_McShan_2019_expression", "CCR5_HUMAN_Gill_2023",
    "GRB2_HUMAN_Faure_2021", "SPA_STAAU_Tsuboyama_2023_1LP1",
]
QUOTA = {"Binding": 1, "Activity": 2, "Expression": 2, "Stability": 1, "OrganismalFitness": 2}

ref = pd.read_csv("data/DMS_substitutions.csv")
used_proteins = set(ref.loc[ref.DMS_id.isin(USED), "UniProt_ID"])
cand = ref[(ref.seq_len <= 500) & (ref.DMS_number_single_mutants >= 1000)
           & ~ref.UniProt_ID.isin(used_proteins)]

rng = np.random.RandomState(2026)
picked, taken = [], set()
for t, q in QUOTA.items():
    pool = cand[cand.coarse_selection_type == t].sample(frac=1, random_state=rng)
    n = 0
    for r in pool.itertuples():
        if r.UniProt_ID in taken:
            continue
        picked.append(r); taken.add(r.UniProt_ID); n += 1
        if n == q:
            break
    if n < q:
        print(f"!! only {n} of {q} available for {t}")

out = pd.DataFrame(picked)[["DMS_id", "coarse_selection_type", "seq_len",
                            "DMS_number_single_mutants", "UniProt_ID", "taxon"]]
print(out.to_string(index=False))
out["DMS_id"].to_csv("holdout_assays.txt", index=False, header=False)