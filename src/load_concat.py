# src/load_concat.py
import pandas as pd, numpy as np, glob, os

RAW = "data/raw"
OUT = "data/processed/merged.parquet"

def read_one(path):
    df = pd.read_csv(path, low_memory=False)
    # standardize column names
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    return df

def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    parts = [read_one(f) for f in files]
    df = pd.concat(parts, ignore_index=True)

    # downcast numeric to save RAM
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast='integer')
    df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast='float')

    # unify label name
    if 'Label' in df.columns:
        df.rename(columns={'Label':'label'}, inplace=True)

    df.to_parquet(OUT, index=False)
    print(f"Saved {OUT} with {df.shape[0]:,} rows & {df.shape[1]} cols")

if __name__ == "__main__":
    main()
