# src/clean.py
import pandas as pd, numpy as np

INP = "data/processed/merged.parquet"
OUT = "data/processed/clean.parquet"

DROP_IF_PRESENT = {
    'Flow_ID','Timestamp','Source_IP','Destination_IP',
    'Src_IP','Dst_IP','Src_Port','Dst_Port','SimillarHTTP'
}

def main():
    df = pd.read_parquet(INP)

    drop_cols = [c for c in DROP_IF_PRESENT if c in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')

    for c in df.columns:
        if df[c].dtype == object and c != 'label':
            df[c] = pd.to_numeric(
                df[c].replace({'Infinity': np.inf, 'inf': np.inf, 'NaN': np.nan}),
                errors='ignore'
            )

    label_col = 'label'
    num_cols = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
    df = df[num_cols + [label_col]] if label_col in df.columns else df[num_cols]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    df.to_parquet(OUT, index=False)
    print("Saved", OUT, "shape:", df.shape)

if __name__ == "__main__":
    main()

