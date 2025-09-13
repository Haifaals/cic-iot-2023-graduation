# src/train.py
import json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, mutual_info_classif

DATA = "data/processed/clean.parquet"
SEL  = "data/processed/selected_features.json"
MODEL_OUT = "models/model.joblib"
LABEL = "label"

def main():
    df = pd.read_parquet(DATA)
    with open(SEL) as f: keep = json.load(f)["kept"]

    X = df[keep]
    y = df[LABEL].astype('category').cat.codes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(steps=[
        ("scale", RobustScaler(with_centering=False)),
        ("smote", SMOTE(k_neighbors=5, random_state=42)),
        ("select", SelectKBest(mutual_info_classif, k=min(len(keep), 60))),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=None, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring=["f1_macro", "roc_auc_ovr_weighted", "accuracy"],
        n_jobs=-1, return_train_score=False
    )
    print("CV F1_macro:", scores["test_f1_macro"].mean())
    print("CV ROC_AUC :", scores["test_roc_auc_ovr_weighted"].mean())
    print("CV Acc     :", scores["test_accuracy"].mean())

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("\nHold-out performance:\n", classification_report(y_test, y_pred, digits=4))

    joblib.dump({"pipeline": pipe, "features": keep}, MODEL_OUT)
    print("Saved model →", MODEL_OUT)

if __name__ == "__main__":
    main()

