import pandas as pd

df = pd.read_csv("data/spam.csv")

print(df.head())

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nLabel distribution:")
print(df["target"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
