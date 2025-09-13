# src/feature_selector.py
import pandas as pd, numpy as np, json
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif

INP = "data/processed/clean.parquet"
OUT = "data/processed/selected_features.json"
LABEL = "label"

K = 60
CORR_THRESH = 0.95

def drop_correlated(X, thr=CORR_THRESH):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] >= thr)]
    return X.drop(columns=to_drop), to_drop

def main():
    df = pd.read_parquet(INP)
    if LABEL not in df.columns:
        raise SystemExit(f"'{LABEL}' column not found; ensure your data has labels.")
    y = df[LABEL].astype('category').cat.codes
    X = df.drop(columns=[LABEL])

    vt = VarianceThreshold(threshold=0.0)
    X_vt = pd.DataFrame(vt.fit_transform(X), columns=X.columns[vt.get_support()])

    X_nc, dropped = drop_correlated(X_vt)

    k = min(K, X_nc.shape[1])
    skb = SelectKBest(mutual_info_classif, k=k)
    skb.fit(X_nc, y)
    keep = list(X_nc.columns[skb.get_support()])

    with open(OUT, "w") as f:
        json.dump({"kept": keep, "dropped_correlated": dropped}, f, indent=2)

    print(f"Selected {len(keep)} features → {OUT}")
    print("Top 15:", keep[:15])

if __name__ == "__main__":
    main()

