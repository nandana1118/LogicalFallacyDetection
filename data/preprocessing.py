import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder

# =========================
# LOAD TRAINING DATA
# =========================
df = pd.read_csv("raw/edu_train.csv")

# --- Initial inspection (kept for reference, not needed later) ---
print("Columns in dataset:")
print(df.columns)

print("\nNumber of rows:", len(df))

# print("\nFirst 5 rows:")
# print(df.head())
# (Commented: useful only for early inspection)

# =========================
# CLEAN DATA
# =========================
# Keep only columns needed for modeling
df = df[['source_article', 'updated_label']]

print("\nCleaned columns:")
print(df.columns)

print("\nSample cleaned rows:")
print(df.head())

# =========================
# DATASET ANALYSIS 
# =========================
# Count number of samples per fallacy
label_counts = df['updated_label'].value_counts()

print("\nNumber of fallacy types:", df['updated_label'].nunique())
print("\nSamples per fallacy:")
print(label_counts)

# =========================
# LABEL ENCODING
# =========================
# Convert textual fallacy labels into numeric form
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['updated_label'])

# Save the trained label encoder for later use
joblib.dump(label_encoder, "../model/label_encoder.pkl")
print("\nLabel encoder saved to model/label_encoder.pkl")


print("\nLabel encoding completed.")
print("\nLabel mapping:")

for i, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {i}")

print("\nSample encoded rows:")
print(df.head())

# =========================
# NOTE:
# The same label_encoder MUST be reused
# for dev and test datasets using .transform()
# =========================


# =========================
# LOAD & PREPROCESS DEV DATA
# =========================
dev_df = pd.read_csv("raw/edu_dev.csv")

# Keep only required columns
dev_df = dev_df[['source_article', 'updated_label']]

# Encode labels using the SAME encoder
dev_df['label_encoded'] = label_encoder.transform(dev_df['updated_label'])

print("\nDEV DATA")
print("Shape:", dev_df.shape)
print(dev_df.head())


# =========================
# LOAD & PREPROCESS TEST DATA
# =========================
test_df = pd.read_csv("raw/edu_test.csv")

# Keep only required columns
test_df = test_df[['source_article', 'updated_label']]

# Encode labels using the SAME encoder
test_df['label_encoded'] = label_encoder.transform(test_df['updated_label'])

print("\nTEST DATA")
print("Shape:", test_df.shape)
print(test_df.head())
