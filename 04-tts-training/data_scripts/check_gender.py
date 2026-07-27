import pandas as pd
import os

parquet_file = "DATASET FOR TAMIL/train-00000-of-00017.parquet"
df = pd.read_parquet(parquet_file)
print("Unique genders:", df['gender'].unique())
