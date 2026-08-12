"""
The "Fast Path": a pre-trained Hybrid TabNet + XGBoost model scores a transaction in
milliseconds.
This requires the PyTorch-TabNet model (`tabnet_pretrainer.zip`) to extract embeddings
and the XGBoost model (`hybrid_xgb_model_realistic.json`) for the final score.
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

# Suppress PyTorch/TabNet warnings during fast API calls
warnings.filterwarnings('ignore')

try:
    import torch
    from pytorch_tabnet.pretraining import TabNetPretrainer
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_xgb_model_realistic.json")
TABNET_MODEL_PATH = os.path.join(MODEL_DIR, "tabnet_pretrainer.zip")

# Default threshold tuned to 0.70 to minimize false positives while preserving fraud detection
FLAG_THRESHOLD = float(os.getenv("FLAG_THRESHOLD", "0.85"))

_xgb_model = None
_tabnet_model = None

# Mapping for the 'type' categorical column based on LabelEncoder alphabetical sorting
TYPE_MAP = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}


def load_models():
    global _xgb_model, _tabnet_model
    
    if _xgb_model is None:
        _xgb_model = xgb.XGBClassifier()
        if os.path.exists(XGB_MODEL_PATH):
            _xgb_model.load_model(XGB_MODEL_PATH)
            
    if _tabnet_model is None and HAS_TABNET:
        if os.path.exists(TABNET_MODEL_PATH):
            _tabnet_model = TabNetPretrainer()
            _tabnet_model.load_model(TABNET_MODEL_PATH)
            
    return _xgb_model, _tabnet_model


def _vectorize(transaction: dict) -> np.ndarray:
    """
    Preprocess the transaction into the exact format expected by the trained model.
    Columns must be in the exact order as the PaySim training dataset after drops:
    ['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest', 'isFlaggedFraud']
    """
    t_type = transaction.get("type", "PAYMENT")
    type_val = TYPE_MAP.get(t_type, 3) # Default to PAYMENT if unknown
    
    row = [
        float(transaction.get("step", 0)),
        float(type_val),
        float(transaction.get("amount", 0.0)),
        float(transaction.get("oldbalanceOrg", 0.0)),
        float(transaction.get("oldbalanceDest", 0.0)),
        float(transaction.get("isFlaggedFraud", 0))
    ]
    return np.array([row])


def predict_risk(transaction: dict) -> dict:
    xgb_m, tabnet_m = load_models()
    
    # Never turn an unavailable model into a legitimate-looking 0% risk score.
    # Failing closed prevents the API from silently approving every transaction
    # when a model artifact or runtime dependency is absent.
    missing = []
    if not os.path.exists(XGB_MODEL_PATH):
        missing.append(XGB_MODEL_PATH)
    if not os.path.exists(TABNET_MODEL_PATH):
        missing.append(TABNET_MODEL_PATH)
    if not HAS_TABNET:
        missing.append("PyTorch/pytorch-tabnet runtime")
    if xgb_m is None or tabnet_m is None:
        missing.append("loaded model instance")
    if missing:
        raise RuntimeError(
            "Fraud model is unavailable; missing: " + ", ".join(missing)
        )
        
    try:
        # 1. Preprocess
        X_raw = _vectorize(transaction)
        
        # 2. Extract TabNet Embeddings
        embeddings, _ = tabnet_m.predict(X_raw)
        
        # 3. Concatenate original features with embeddings
        X_hybrid = np.hstack((X_raw, embeddings))
        
        # 4. Predict
        proba = float(xgb_m.predict_proba(X_hybrid)[0, 1])
        
        return {
            "risk_score": round(proba * 100, 2),
            "is_flagged": proba >= FLAG_THRESHOLD,
        }
    except Exception as e:
        raise RuntimeError(f"Fraud model inference failed: {e}") from e


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "aligned_transactions.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        for _, row in df.sample(5, random_state=1).iterrows():
            result = predict_risk(row.to_dict())
            print(f"customer={row['customer_id']} type={row['type']} amount={row['amount']:.2f} "
                  f"actual_fraud={row['isFraud']} -> risk={result['risk_score']}% flagged={result['is_flagged']}")
    else:
        print(f"Could not find test data at {data_path}")
