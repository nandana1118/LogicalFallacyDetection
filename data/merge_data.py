import os
import pandas as pd

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "raw")

original_path   = os.path.join(RAW_DIR, "edu_train.csv")
full_batch_path = os.path.join(RAW_DIR, "augmented_full_batch.csv")
remaining_path  = os.path.join(RAW_DIR, "augmented_remaining_batch.csv")
output_path     = os.path.join(RAW_DIR, "edu_train_augmented.csv")

# =========================
# LOAD ALL THREE FILES
# =========================
print("Loading files...")

original  = pd.read_csv(original_path)[['source_article', 'updated_label']]
full      = pd.read_csv(full_batch_path)[['source_article', 'updated_label']]
remaining = pd.read_csv(remaining_path)[['source_article', 'updated_label']]

print(f"  Original training data : {len(original)} rows")
print(f"  Full batch augmented   : {len(full)} rows")
print(f"  Remaining batch        : {len(remaining)} rows")

# =========================
# MERGE
# =========================
merged = pd.concat([original, full, remaining], ignore_index=True)

print(f"\nTotal after merge: {len(merged)} rows")

# =========================
# REMOVE DUPLICATES
# =========================
before = len(merged)
merged = merged.drop_duplicates(subset=['source_article'])
after  = len(merged)

print(f"Duplicates removed: {before - after}")
print(f"Final total: {after} rows")

# =========================
# SANITY CHECK
# =========================
print("\nFinal class distribution:")
print("=" * 45)
counts = merged['updated_label'].value_counts()
for label, count in counts.items():
    print(f"  {label:<30} : {count}")

print(f"\nTotal classes: {merged['updated_label'].nunique()}")

# Check for any unexpected labels
original_labels = set(pd.read_csv(original_path)['updated_label'].unique())
merged_labels   = set(merged['updated_label'].unique())
unexpected      = merged_labels - original_labels

if unexpected:
    print(f"\nWARNING: Unexpected labels found: {unexpected}")
    print("These will cause errors during training. Fix before proceeding.")
else:
    print("\nAll labels match original training data. Safe to proceed.")

# =========================
# SAVE
# =========================
merged.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
print("Done!")
