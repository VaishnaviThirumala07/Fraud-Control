# 🛡️ FraudControl: Next-Gen Autonomous AI Fraud Investigation Platform (v2 Architecture)

FraudControl is an advanced, production-grade fraud detection and investigation platform. It combines the millisecond speed of traditional Machine Learning (XGBoost + GNN Node2Vec Embeddings) for real-time inference with the deep, nuanced reasoning capabilities of Large Language Models (LLMs) orchestrated in a 4-agent parallel workflow with autonomous self-reflection and action execution.

---

## 🚀 Quick Start (Running the Application)

To bring the entire platform online locally or in a container environment, follow these steps:

### Option A: Local Docker Compose (Recommended)
```bash
# 1. Start all containerized services (Kafka/Redpanda, Neo4j, FastAPI Backend, React Frontend)
docker compose up -d --build

# 2. Start the Real-Time Transaction Producer Stream
docker compose exec dev sh -lc 'cd /app/pipeline && uv run python producer.py'
```
Then navigate to **http://localhost:5173** to access the live dashboard!

### Option B: Terminal-by-Terminal Local Execution
```bash
# Terminal 1: Infrastructure (Kafka + Neo4j)
docker start redpanda neo4j

# Terminal 2: FastAPI Backend API & AI Agents
cd pipeline
uvicorn api:app --reload --port 8000

# Terminal 3: Real-Time Producer Stream (Streams 1 transaction every 25s)
cd pipeline
python producer.py

# Terminal 4: React Dashboard
cd frontend
npm run dev
```

---

## 🏗️ System Architecture Flowchart

The system operates on an event-driven architecture powered by Kafka / RabbitMQ and real-time WebSockets streaming.

```mermaid
flowchart TD
    subgraph Stream [Event-Driven Stream]
        Producer(producer.py) -->|Publishes every 25s| Queue[Kafka / RabbitMQ Topic]
    end

    subgraph FastPath [Fast Path: Millisecond Inference]
        Consumer(api.py Consumer)
        Queue --> Consumer
        XGBoost{XGBoost Classifier + GNN Embeddings}
        Consumer -->|Evaluate Risk| XGBoost
        Consumer -->|Async Cypher MERGE| Neo4j[(Neo4j Graph)]
    end

    subgraph SlowPath [Slow Path: LlamaIndex Multi-Agent Workflow v2]
        direction TB
        subgraph Agents [Parallel Specialist Agents]
            TxAgent[1. Transaction Agent]
            CustAgent[2. Customer Agent]
            GraphAgent[3. Graph Agent]
            MemAgent[4. Memory Agent SQLite]
        end
        
        XGBoost -- Flagged > 60% --> Agents
        
        RAG[(ChromaDB RAG Policy Base)]
        
        Agents --> Supervisor[Supervisor Agent]
        RAG <-->|Policy Search| Supervisor
        
        Supervisor --> SelfReflection{Confidence < 75%?}
        SelfReflection -- Yes -->|Refine RAG Query & Re-reason| Supervisor
    end

    subgraph Execution [Autonomous Action & Persistence]
        ActionEngine[Action Agent]
        SelfReflection -- Final Decision --> ActionEngine
        ActionEngine -->|Execute Block / Hold / Approve| AuditLog[(SQLite Audit & Memory)]
    end

    subgraph UI [React Frontend Dashboard]
        WebSockets((WebSockets Manager))
        Dashboard[Live Compliance Dashboard]
        
        Consumer -- FAST_PATH Event --> WebSockets
        Agents -- THINKING_UPDATE Steps --> WebSockets
        Supervisor -- SLOW_PATH SAR Report --> WebSockets
        ActionEngine -- ACTION_TAKEN Event --> WebSockets
        WebSockets --> Dashboard
    end
```

---

## 🧠 Machine Learning (The "Fast Path")

The first line of defense is a highly optimized XGBoost classifier trained to instantly evaluate incoming transactions.

- **Algorithms**: XGBoost (`xgboost`) + TabNet + Graph Neural Network structural embeddings (Node2Vec).
- **Graph Neural Network (GNN) Integration**: Ingests Neo4j topology to generate structural embeddings (`gnn_emb_0`, `gnn_emb_1`) so the Fast Path has graph-relational awareness in milliseconds without needing a live database traversal for unflagged transactions.
- **Core Features**:
  - `amount`, `oldbalanceOrg`, `oldbalanceDest`, `newbalanceOrig`, `newbalanceDest`.
  - Identity & session features (`account_age_days`, `shared_ip_count`, `recent_failed_logins`, `session_velocity_seconds`).
  - GNN structural embedding vectors.
- **Outcome**: Assigns a `risk_score` (0-100%). Transactions scoring > 60% are **flagged** and routed to the Agentic Slow Path for deep investigation.

---

## 🤖 The Multi-Agent Workflow (The "Slow Path v2")

When a transaction is flagged, **four specialist AI agents** execute in parallel via `asyncio.gather` using **LlamaIndex Workflows** and **OpenAI (GPT-4o / GPT-4o-mini)** models.

### 1. Transaction Analysis Agent
- **Role**: Analyzes financial structure, amount velocity, balance draining behavior, and anomaly thresholds.

### 2. Customer Intelligence Agent
- **Role**: Evaluates identity signals (KYC status, PEP flags, account age, failed login clusters, sub-5s session velocity for Account Takeover detection).

### 3. Graph Network Analysis Agent
- **Role**: Connects to **Neo4j** via live Cypher queries to detect shared device clusters, IP overlapping, and multi-user money mule rings.

### 4. Memory Agent (Cross-Transaction Persistence)
- **Role**: Queries a local **SQLite Memory Store** (`storage/memory.db`) to identify temporal patterns across multiple transactions over time (e.g., repeat holds, escalation after previous blocks).

### 5. Supervisor Agent with Self-Reflection Loop
- **Role**: Collects findings from all 4 agents and queries an internal **AML Policy Knowledge Base** (ChromaDB vector index).
- **Self-Reflection Mechanism**:
  - Produces an initial decision and a `CONFIDENCE` score (0-100).
  - If `CONFIDENCE < 75%`, the supervisor **autonomously triggers a self-reflection pass**: it refines its RAG policy query, re-queries ChromaDB for deeper compliance context, and re-evaluates before finalizing the Suspicious Activity Report (SAR).

---

## ⚡ Autonomous Action Engine (`action_engine.py`)

Once the investigation completes, the Action Agent autonomously executes the decision:
- **Block**: Instantly freezes outbound transfers and registers the account in the live `blocked_accounts` registry.
- **Hold for Review**: Places the transaction on the compliance alert feed and triggers optional customer OTP step-up verification.
- **Approve**: Releases funds if co-occurring risk indicators are absent.
- **Human-In-The-Loop (HITL) Overrides**: Compliance officers can inspect live reasoning streams and issue manual overrides (`/api/hitl-override`).

---

## 📜 Configuration & Environment Settings

The pipeline configuration is managed via `pipeline/.env`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | Required | API key for GPT-4o-mini reasoning & embeddings |
| `MOCK_LLM` | `false` | Set to `true` during dev to use \$0 mock investigations |
| `MOCK_LLM_INVESTIGATIONS` | `false` | Alternative flag supported in deployment environments |
| `TRANSACTION_INTERVAL` | `25` | Delay in seconds between streamed transactions |
| `KAFKA_BROKER` | `localhost:9092` | Kafka / Redpanda broker URI (`redpanda:9092` in Docker) |
| `CLOUDAMQP_URL` | Optional | RabbitMQ / CloudAMQP connection URL |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection URI (`bolt://neo4j:7687` in Docker) |

---

## 🛠️ Daytona CDE & Deployment Commands

To run FraudControl safely in a **Daytona Cloud Development Environment**:

1. **Start Docker Daemon & Containers**:
   ```bash
   dockerd > /tmp/dockerd.log 2>&1 & sleep 5 && docker info
   docker compose up -d --build
   ```

2. **Expose Daytona Preview Port**:
   ```bash
   daytona preview-url <workspace-id> --port 5173
   ```

3. **Launch Transaction Stream**:
   ```bash
   docker compose exec dev sh -lc 'cd /app/pipeline && uv run python producer.py'
   ```