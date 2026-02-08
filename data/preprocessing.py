import pandas as pd

# Load training dataset
df = pd.read_csv("raw/edu_train.csv")

print("Columns in dataset:")
print(df.columns)

#print("\nFirst 5 rows:")
#print(df.head())

print("\nNumber of rows:", len(df))

# Keep only useful columns
df = df[['source_article', 'updated_label']]

print("\nCleaned columns:")
print(df.columns)

print("\nSample cleaned rows:")
print(df.head())


# Count number of samples per fallacy
label_counts = df['updated_label'].value_counts()

print("\nNumber of fallacy types:", df['updated_label'].nunique())
print("\nSamples per fallacy:")
print(label_counts)
