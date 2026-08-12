import argparse
import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
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
from pytorch_tabnet.pretraining import TabNetPretrainer
import warnings
warnings.filterwarnings('ignore')

DEFAULT_PATH = os.path.expanduser("~/Downloads/paysim.csv")
TARGET_CANDIDATES = ["isFraud", "isfraud", "fraud", "Fraud", "Class", "class", "target", "label", "Label"]
ID_LIKE_COLUMNS = ["nameOrig", "nameDest", "id", "ID", "Id", "index"]
POST_TRANSACTION_COLUMNS = ["newbalanceOrig", "newbalanceDest"]


def load_data(path):
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    return df


def find_target_column(df):
    for cand in TARGET_CANDIDATES:
        if cand in df.columns:
            return cand
    for col in reversed(df.columns):
        if df[col].nunique() == 2:
            return col
    raise ValueError("Could not auto-detect a target column.")


def engineer_paysim_features(df):
    if {"oldbalanceOrg", "newbalanceOrig", "amount"}.issubset(df.columns):
        df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    if {"oldbalanceDest", "newbalanceDest", "amount"}.issubset(df.columns):
        df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return df


def preprocess_for_tabnet(df, target_col, realistic=False):
    """
    Preprocess data specifically for TabNet.
    TabNet requires all categorical variables to be Label Encoded (not one-hot).
    """
    df = df.copy()

    drop_cols = [c for c in ID_LIKE_COLUMNS if c in df.columns and c != target_col]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if realistic:
        post_cols = [c for c in POST_TRANSACTION_COLUMNS if c in df.columns]
        if post_cols:
            df = df.drop(columns=post_cols)
    else:
        df = engineer_paysim_features(df)

    y = df[target_col].values
    X_df = df.drop(columns=[target_col])
    
    # Label encode ALL categorical columns for TabNet
    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_idxs = []
    cat_dims = []
    
    for i, col in enumerate(X_df.columns):
        if col in cat_cols:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
            cat_idxs.append(i)
            cat_dims.append(len(le.classes_))
            
    X = X_df.values
    return X, y, X_df.columns.tolist(), cat_idxs, cat_dims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_PATH)
    parser.add_argument("--realistic", action="store_true", help="Drop post-transaction balance columns")
    parser.add_argument("--epochs", type=int, default=10, help="Number of pre-training epochs")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"ERROR: file not found at {args.csv_path}")
        raise SystemExit(1)

    t_load = time.time()
    df = load_data(args.csv_path)
    print(f"Load took {time.time() - t_load:.1f}s")
    target_col = find_target_column(df)
    
    # Preprocess
    print("Preprocessing data for TabNet...")
    X, y, feature_names, cat_idxs, cat_dims = preprocess_for_tabnet(df, target_col, realistic=args.realistic)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- PHASE 1: TabNet Pre-training (Self-Supervised) ---
    print("\n=== PHASE 1: TabNet Self-Supervised Pre-training ===")
    print("Training TabNet to reconstruct masked features...")
    
    X_pretrain, X_valid = train_test_split(X_train, test_size=0.1, random_state=42)
    
    pretrainer = TabNetPretrainer(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax'
    )

    t0 = time.time()
    pretrainer.fit(
        X_train=X_pretrain,
        eval_set=[X_valid],
        max_epochs=args.epochs,
        patience=3,
        batch_size=16384, 
        virtual_batch_size=4096,
        num_workers=0,
        drop_last=False,
        pretraining_ratio=0.5 
    )
    print(f"Pre-training took {time.time() - t0:.1f}s")
    
    pretrainer_path = "tabnet_pretrainer.zip"
    pretrainer.save_model(pretrainer_path)
    print(f"Saved TabNet Pretrainer to {pretrainer_path}")

    # --- PHASE 2: Extract Embeddings ---
    print("\n=== PHASE 2: Extracting Latent Embeddings ===")
    print("Passing data through TabNet to get deep representations...")
    t1 = time.time()
    
   
    train_embeddings, _ = pretrainer.predict(X_train)
    test_embeddings, _ = pretrainer.predict(X_test)
    
    X_train_hybrid = np.hstack((X_train, train_embeddings))
    X_test_hybrid = np.hstack((X_test, test_embeddings))
    print(f"Extraction took {time.time() - t1:.1f}s. New feature space size: {X_train_hybrid.shape[1]}")

    # --- PHASE 3: XGBoost Training ---
    print("\n=== PHASE 3: Training Hybrid XGBoost Model ===")
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()

    scale_pos_weight = min(50.0, neg / pos if pos > 0 else 1.0)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        min_child_weight=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        tree_method="hist", 
        device="cuda" if torch.cuda.is_available() else "cpu", # Automatically use GPU if available
        random_state=42,
    )

    t2 = time.time()
    model.fit(
        X_train_hybrid, y_train,
        eval_set=[(X_test_hybrid, y_test)],
        verbose=False,
    )
    print(f"XGBoost training took {time.time() - t2:.1f}s")

    # --- Evaluation (With Threshold Tuning) ---
    y_proba = model.predict_proba(X_test_hybrid)[:, 1]

    print("\n=== Tuning Probability Threshold for best F1-Score ===")
    best_threshold = 0.5
    best_f1 = 0.0
    
    for thresh in np.arange(0.1, 0.95, 0.05):
        y_pred_thresh = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred_thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    print(f"Optimal Threshold found: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    # Use optimal threshold for final predictions
    y_pred_optimal = (y_proba >= best_threshold).astype(int)

    print("\n=== Final Evaluation ===")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred_optimal):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_optimal):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_optimal):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred_optimal))

    model_name = "hybrid_xgb_model_realistic.json" if args.realistic else "hybrid_xgb_model.json"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
    model.save_model(out_path)
    print(f"\nHybrid XGBoost Model saved to: {out_path}")


if __name__ == "__main__":
    main()
