import pandas as pd

ref = pd.read_csv("data/DMS_substitutions.csv")
print("Columns:", list(ref.columns))
print("Total assays:", len(ref))

# Keep short proteins with enough single mutants for low-N experiments
cand = ref[(ref["seq_len"] <= 500) & (ref["DMS_number_single_mutants"] >= 1000)]
print("\nAssays after filtering:", len(cand))

cols = ["DMS_id", "seq_len", "DMS_number_single_mutants",
        "coarse_selection_type", "taxon"]
cols = [c for c in cols if c in cand.columns]

print("\nCounts by selection type:")
print(cand["coarse_selection_type"].value_counts())

print("\nCandidate assays:")
print(cand[cols].sort_values("coarse_selection_type").to_string(index=False))