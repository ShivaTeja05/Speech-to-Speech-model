import pandas as pd
import os

parquet_file = "DATASET FOR TAMIL/train-00000-of-00017.parquet"

if os.path.exists(parquet_file):
    try:
        df = pd.read_parquet(parquet_file)
        print("Columns:", df.columns.tolist())
        print("\nFirst row sample:")
        print(df.iloc[0])
    except Exception as e:
        print(f"Error reading parquet: {e}")
else:
    print(f"File not found: {parquet_file}")
