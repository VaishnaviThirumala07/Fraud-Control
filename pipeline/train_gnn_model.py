import os
import json
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import xgboost as xgb
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GRAPH_DIR = os.path.join(DATA_DIR, "graph")

def main():
    print("1. Loading graph data...")
    has_device = pd.read_csv(os.path.join(GRAPH_DIR, "edges_has_device.csv"))
    transactions = pd.read_csv(os.path.join(GRAPH_DIR, "edges_transactions.csv"))

    print("2. Building NetworkX Graph...")
    G = nx.Graph()
    
    for _, row in has_device.iterrows():
        G.add_edge(row["user_id"], row["device_id"], weight=1.0)
        
    for _, row in transactions.iterrows():
        G.add_edge(row["user_id"], row["merchant_id"], weight=1.0)

    print(f"Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    print("3. Training Node2Vec GNN model...")
    node2vec = Node2Vec(G, dimensions=2, walk_length=10, num_walks=10, workers=4, quiet=False)
    gnn_model = node2vec.fit(window=5, min_count=1, batch_words=4)
    
    print("4. Extracting Embeddings for Users...")
    users = pd.read_csv(os.path.join(GRAPH_DIR, "nodes_users.csv"))
    embeddings = {}
    for user_id in users["user_id"]:
        if user_id in gnn_model.wv:
            vec = gnn_model.wv[user_id].tolist()
            embeddings[user_id] = {"gnn_emb_0": vec[0], "gnn_emb_1": vec[1]}
        else:
            embeddings[user_id] = {"gnn_emb_0": 0.0, "gnn_emb_1": 0.0}
            
    
    emb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnn_embeddings.json")
    with open(emb_path, 'w') as f:
        json.dump(embeddings, f)
    print(f"Saved GNN embeddings to {emb_path}")

    print("5. Augmenting Training Data with GNN Embeddings...")
    tx_df = pd.read_csv(os.path.join(DATA_DIR, "aligned_transactions.csv"))
    
    emb_0_col = []
    emb_1_col = []
    for _, row in tx_df.iterrows():
        user_id = row["customer_id"]
        if user_id in embeddings:
            emb_0_col.append(embeddings[user_id]["gnn_emb_0"])
            emb_1_col.append(embeddings[user_id]["gnn_emb_1"])
        else:
            emb_0_col.append(0.0)
            emb_1_col.append(0.0)
            
    tx_df["gnn_emb_0"] = emb_0_col
    tx_df["gnn_emb_1"] = emb_1_col
    
    
    print("6. Training XGBoost Fast Path Model with GNN Awareness...")
    
    
    type_dummies = pd.get_dummies(tx_df['type'], prefix='type')
    X = pd.concat([tx_df[['step', 'amount', 'oldbalanceOrg', 'oldbalanceDest']], type_dummies], axis=1)
    
    for c in ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]:
        if c not in X.columns:
            X[c] = 0
            
    X["gnn_emb_0"] = tx_df["gnn_emb_0"]
    X["gnn_emb_1"] = tx_df["gnn_emb_1"]
    
    X = X.reindex(sorted(X.columns), axis=1)
    
    y = tx_df["isFraud"]
    
    clf = XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1, 
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42
    )
    clf.fit(X, y)
    
    model_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xgb_model_gnn.json")
    clf.save_model(model_out)
    print(f"✅ Successfully trained and saved GNN-augmented XGBoost to {model_out}")

if __name__ == "__main__":
    main()
