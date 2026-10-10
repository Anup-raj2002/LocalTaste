import pandas as pd

df = pd.read_csv("zomato.csv")

print("SHAPE:", df.shape)
print("\nCOLUMNS:", df.columns.tolist())
print("\nDTYPES:\n", df.dtypes)
print("\nNULLS:\n", df.isnull().sum())
print("\nSAMPLE ROW:\n", df.iloc[0])