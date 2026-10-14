import glob, os
import pandas as pd

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

ref = pd.read_csv("data/DMS_substitutions.csv").set_index("DMS_id")
rows = []

for dms_id in SELECTED:
    fname = ref.loc[dms_id, "DMS_filename"]
    hits = glob.glob(f"data/dms_assays/**/{fname}", recursive=True)
    if not hits:
        print(f"!! File not found for {dms_id} ({fname})")
        continue

    df = pd.read_csv(hits[0])
    n_muts = df["mutant"].str.count(":") + 1
    wt = ref.loc[dms_id, "target_seq"]

    rows.append({
        "DMS_id": dms_id,
        "seq_len": len(wt),
        "n_variants": len(df),
        "n_single": int((n_muts == 1).sum()),
        "n_multi": int((n_muts > 1).sum()),
        "score_mean": round(df["DMS_score"].mean(), 3),
        "score_std": round(df["DMS_score"].std(), 3),
        "n_missing": int(df["DMS_score"].isna().sum()),
        "columns": list(df.columns),
    })

    # Save a tidy copy for later steps
    os.makedirs("data/selected", exist_ok=True)
    df.to_csv(f"data/selected/{dms_id}.csv", index=False)

summary = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(summary.drop(columns="columns").to_string(index=False))
print("\nColumns in first file:", rows[0]["columns"] if rows else "none")
summary.drop(columns="columns").to_csv("data/selected/summary.csv", index=False)