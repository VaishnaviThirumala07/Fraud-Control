# Fraud Detection MVP — Multi-Agent Pipeline

Implements the "2-dataset, small multi-agent" MVP scoped in the design chat:
PaySim (real transactional data) as the Fast Path, synthetic KYC profiles
as the Slow Path's second dataset, and a 3-agent LlamaIndex Workflow
(Transaction Agent -> Customer Agent -> Reasoning Agent) grounded in a
mock AML policy via RAG.

```
Real PaySim Transaction
        |
        v
  Fast Path: XGBoost (ml_engine.py) ---- risk < 60% ----> Approve
        |
     risk >= 60%
        v
  Slow Path: LlamaIndex Workflow (agent_engine.py)
    Transaction Agent -> Customer Agent -> Reasoning Agent (+ AML policy RAG)
        |
        v
  Streamlit Dashboard (app.py)
```

## Setup

```bash
pip install -r requirements.txt   # or see the package list below
cp .env.example .env              # then add your GEMINI_API_KEY
```

Get a free Gemini API key at https://aistudio.google.com/apikey

## Run order

1. `python align_datasets.py` — samples PaySim transactions (oversampling
   fraud) and generates matching synthetic customer profiles, keyed by
   real PaySim `nameOrig` IDs. Writes `data/aligned_transactions.csv`
   and `data/customer_profiles.csv`.
2. `python ml_engine.py` — smoke-tests the fast-path model against a few
   sampled rows.
3. `python agent_engine.py` — runs one hardcoded transaction through the
   full 3-agent workflow and prints the JSON investigation report.
   Requires `GEMINI_API_KEY`.
4. `streamlit run app.py` — the full interactive dashboard.
5. `python graph_data_generator.py` — optional bonus: generates a
   standalone graph dataset (nodes/edges CSVs) with an injected 5-account
   fraud ring, ready to load into Neo4j/NetworkX. Not wired into the
   dashboard for the MVP (out of scope per the "what to skip" list).

## Design decisions worth knowing

- **The fast-path model is `xgb_model_realistic.json`**, not the original
  99.99%-accuracy one. The original model leaked post-transaction balance
  fields that wouldn't exist yet at real-time decision time — see
  `../results_realistic.txt` for the full analysis. Using the realistic
  model means the demo's fast-path numbers (97% recall, ~21% precision at
  a 0.5 threshold) are the honest ones for this framing.
- **The 3-agent workflow is sequential, not the full 10-agent design**
  from the chat (no Graph Agent, Regulatory Agent is folded into the
  Reasoning Agent's RAG call, no Mitigation Agent). This matches the
  chat's own advice: "a multi-agent DAG is overkill for an MVP."
- **Customer profiles are keyed to real PaySim `nameOrig` IDs**, not
  arbitrary `CUST_i` indices, so the two datasets genuinely join on a
  real field rather than being independently generated and stapled
  together by row position.
- **Skipped per the MVP scope**: Kafka/Pub-Sub, Neo4j (graph data is
  generated but not loaded into a live graph DB), Docker/Kubernetes,
  the full 10-step agent DAG, and the MERN analyst dashboard (Streamlit
  substitutes for it).

## Files

| File | Role |
|---|---|
| `align_datasets.py` | Bridges PaySim + synthetic profiles (Dataset A + B) |
| `ml_engine.py` | Fast path: XGBoost risk scoring |
| `agent_engine.py` | Slow path: 3-agent LlamaIndex Workflow + AML RAG |
| `knowledge_base/aml_policy.txt` | Mock AML policy the Reasoning Agent cites |
| `app.py` | Streamlit investigator dashboard |
| `graph_data_generator.py` | Bonus: synthetic fraud-ring graph dataset |
| `data/` | Generated CSVs (gitignore-worthy, regenerate via scripts above) |
