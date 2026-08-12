"""
Train an XGBoost classifier on paysim.csv (PaySim synthetic mobile-money
fraud detection dataset: step, type, amount, nameOrig, oldbalanceOrg,
newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud,
isFlaggedFraud).

Usage:
    python train_xgboost.py [path_to_csv] [--realistic]

--realistic drops newbalanceOrig/newbalanceDest (and any engineered
feature derived from them). Those fields only exist AFTER a transaction
has posted, so a live fraud-prevention system deciding whether to allow
a transaction would not have them available yet. This mode reflects
what the model can actually see at decision time.

Defaults to ~/Downloads/paysim.csv if no path is given.
"""
import argparse
import os
import time
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

DEFAULT_PATH = os.path.expanduser("~/Downloads/paysim.csv")
TARGET_CANDIDATES = ["isFraud", "isfraud", "fraud", "Fraud", "Class", "class", "target", "label", "Label"]
ID_LIKE_COLUMNS = ["nameOrig", "nameDest", "id", "ID", "Id", "index"]
POST_TRANSACTION_COLUMNS = ["newbalanceOrig", "newbalanceDest"]
ONE_HOT_MAX_CARDINALITY = 15


def load_data(path):
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def find_target_column(df):
    for cand in TARGET_CANDIDATES:
        if cand in df.columns:
            return cand
    for col in reversed(df.columns):
        if df[col].nunique() == 2:
            return col
    raise ValueError(
        "Could not auto-detect a target column. "
        f"Available columns: {list(df.columns)}. "
        "Edit TARGET_CANDIDATES in this script or pass the column name manually."
    )


def engineer_paysim_features(df):
    """PaySim-specific balance-error features: fraud transactions almost
    always leave the origin/destination balances inconsistent with the
    transferred amount, which is one of the strongest signals in this
    dataset. NOTE: both rely on newbalanceOrig/newbalanceDest, i.e. the
    POST-transaction balance, so they are skipped entirely in
    --realistic mode."""
    if {"oldbalanceOrg", "newbalanceOrig", "amount"}.issubset(df.columns):
        df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    if {"oldbalanceDest", "newbalanceDest", "amount"}.issubset(df.columns):
        df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return df


def preprocess(df, target_col, realistic=False):
    df = df.copy()

    drop_cols = [c for c in ID_LIKE_COLUMNS if c in df.columns and c != target_col]
    if drop_cols:
        print(f"Dropping identifier columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    if realistic:
        post_cols = [c for c in POST_TRANSACTION_COLUMNS if c in df.columns]
        if post_cols:
            print(f"[realistic mode] Dropping post-transaction columns: {post_cols}")
            df = df.drop(columns=post_cols)
    else:
        df = engineer_paysim_features(df)

    y = df[target_col]
    X = df.drop(columns=[target_col])
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    low_card = [c for c in cat_cols if X[c].nunique() <= ONE_HOT_MAX_CARDINALITY]
    high_card = [c for c in cat_cols if c not in low_card]

    if low_card:
        print(f"One-hot encoded categorical columns: {low_card}")
        X = pd.get_dummies(X, columns=low_card)
    for col in high_card:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if high_card:
        print(f"Label-encoded categorical columns: {high_card}")

    if y.dtype == "object" or str(y.dtype) == "category":
        y = LabelEncoder().fit_transform(y.astype(str))

    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_PATH)
    parser.add_argument(
        "--realistic", action="store_true",
        help="Drop post-transaction balance columns (newbalanceOrig/newbalanceDest) "
             "to simulate what's actually available at decision time.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"ERROR: file not found at {args.csv_path}")
        raise SystemExit(1)

    t_load = time.time()
    df = load_data(args.csv_path)
    print(f"Load took {time.time() - t_load:.1f}s")
    target_col = find_target_column(df)
    print(f"Using target column: '{target_col}'")
    print(f"Class balance:\n{df[target_col].value_counts()}\n")
    if args.realistic:
        print("Running in --realistic mode (pre-transaction features only)\n")

    X, y = preprocess(df, target_col, realistic=args.realistic)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    print(f"Training XGBoost model on {len(X_train):,} rows / {X_train.shape[1]} features...")
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    print(f"Training took {time.time() - t0:.1f}s")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Evaluation ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\n=== Top 15 feature importances ===")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances.head(15))

    model_name = "xgb_model_realistic.json" if args.realistic else "xgb_model.json"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
    model.save_model(out_path)
    print(f"\nModel saved to: {out_path}")


if __name__ == "__main__":
    main()
