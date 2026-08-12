"""
The "Slow Path": A true parallel multi-agent workflow powered by LlamaIndex.

Architecture (v2 — Hackathon Edition):
  - FOUR specialist agents fire simultaneously via asyncio.gather:
      1. Transaction Agent   — analyzes tx structure and anomalies
      2. Customer Agent      — analyzes KYC, PEP, behavioral signals
      3. Graph Agent         — live Neo4j Cypher queries for fraud ring detection
      4. Memory Agent        — queries SQLite for cross-transaction customer history
  - Each agent emits its findings as a LlamaIndex workflow Event.
  - The Supervisor Agent waits for all 4, retrieves RAG policy context from ChromaDB,
    then applies a SELF-REFLECTION LOOP:
        → Asks LLM to produce decision + CONFIDENCE score (0-100)
        → If CONFIDENCE < 75, it automatically re-queries RAG with a refined
          query and re-reasons before finalizing — autonomous iteration!
  - Every agent step emits a THINKING_UPDATE event for real-time UI streaming.
  - Final InvestigationReport includes full reasoning_trace for dashboard display.
"""
from __future__ import annotations
import os
import asyncio
from typing import List, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from neo4j import AsyncGraphDatabase

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "fraudcontrol")


# ── LLM/Embedding Setup ──────────────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("No OpenAI API key found. Set OPENAI_API_KEY in your .env file.")
    return key

def configure_llm():
    """Configure LLM and embedding settings once at module load."""
    api_key = _get_api_key()
    Settings.llm = OpenAI(model=OPENAI_MODEL, api_key=api_key, temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model_name=OPENAI_EMBED_MODEL, api_key=api_key)

configure_llm()


# ── RAG / ChromaDB Setup ──────────────────────────────────────────────────────

_policy_index: Optional[VectorStoreIndex] = None

def _init_policy_index():
    """Initialize or load ChromaDB vector index. Called lazily on first use."""
    global _policy_index
    if _policy_index is not None:
        return

    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core.node_parser import SentenceSplitter

    persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
    chroma_db_dir = os.path.join(persist_dir, "chroma")

    db = chromadb.PersistentClient(path=chroma_db_dir)
    chroma_collection = db.get_or_create_collection("fraud_policies")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if not os.path.exists(chroma_db_dir) or not chroma_collection.count():
        print("[RAG] Building new index from knowledge_base PDFs (one-time, may take ~2 min)...")
        docs = SimpleDirectoryReader(KB_DIR).load_data()
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        _policy_index = VectorStoreIndex.from_documents(
            docs, storage_context=storage_context, transformations=[splitter]
        )
        print(f"[RAG] Index built with {len(docs)} document chunks.")
    else:
        print("[RAG] Loading pre-built index from ChromaDB...")
        _policy_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        print("[RAG] Index loaded.")


async def initialize_policy_index() -> None:
    """Warm the policy index without blocking the application's event loop."""
    await asyncio.to_thread(_init_policy_index)


async def _retrieve_policy_context(query: str) -> tuple[str, str]:
    """
    Retrieve raw text chunks from ChromaDB without a LLM synthesis call.

    Strategy: Retrieve top-12 candidates then apply source-diversity selection —
    keep the top-2 chunks per unique source document. This prevents a large
    document (INTERPOL, 2435 chunks) from monopolising all 3 citation slots
    simply because it has proportionally more chunks in the index.
    """
    _init_policy_index()
    retriever = _policy_index.as_retriever(similarity_top_k=12)
    nodes = await retriever.aretrieve(query)

    seen_sources: dict[str, int] = {}
    diverse_nodes = []
    MAX_PER_SOURCE = 2
    for node in nodes:  # nodes are already ranked by similarity score
        src = node.node.metadata.get("file_name", "Unknown")
        if seen_sources.get(src, 0) < MAX_PER_SOURCE:
            diverse_nodes.append(node)
            seen_sources[src] = seen_sources.get(src, 0) + 1
        if len(diverse_nodes) >= 6:  # cap at 6 chunks total
            break

    citations = list(dict.fromkeys(
        node.node.metadata.get("file_name", "Unknown") for node in diverse_nodes
    ))
    raw_chunks = [node.node.text for node in diverse_nodes]

    citations_str = "\n".join(f"  [{i+1}] {c}" for i, c in enumerate(citations))
    policy_context = "\n---\n".join(raw_chunks) if raw_chunks else "No relevant policy found."
    return policy_context, citations_str


# ── Pydantic Data Models ──────────────────────────────────────────────────────

class TransactionData(BaseModel):
    customer_id: str
    type: str
    amount: float
    oldbalanceOrg: float
    oldbalanceDest: float
    risk_score: float
    ml_flagged: bool

class CustomerData(BaseModel):
    customer_id: str
    kyc_status: str
    pep_status: bool
    risk_tier: str
    account_age_days: int
    unique_counterparties_30d: int
    shared_ip_count: int
    recent_failed_logins: int
    session_velocity_seconds: int
    historical_avg_tx_amount: float

class TransactionFindings(BaseModel):
    analysis: str
    deviation_flags: List[str]

class CustomerFindings(BaseModel):
    analysis: str
    risk_flags: List[str]

class GraphFindings(BaseModel):
    analysis: str
    ring_flags: List[str]
    raw_graph_data: dict

class MemoryFindings(BaseModel):
    analysis: str
    memory_flags: List[str]
    prior_count: int
    was_previously_blocked: bool

class InvestigationReport(BaseModel):
    risk_score: float
    recommended_action: str
    reasons: List[str]
    sar_explanation: str
    policy_citations: str
    reasoning_trace: List[str] = Field(default_factory=list)
    reflection_triggered: bool = False
    reflection_reason: str = ""
    confidence_score: float = 0.0


# ── Workflow Events ───────────────────────────────────────────────────────────

class ThinkingEvent(Event):
    """Emitted whenever an agent completes a reasoning step — streamed to UI."""
    agent_name: str
    step_message: str
    step_index: int

class TransactionAnalyzedEvent(Event):
    transaction: TransactionData
    findings: TransactionFindings

class CustomerAnalyzedEvent(Event):
    customer: CustomerData
    findings: CustomerFindings

class GraphAnalyzedEvent(Event):
    findings: GraphFindings

class MemoryAnalyzedEvent(Event):
    findings: MemoryFindings


# ── Specialist Agent Functions ────────────────────────────────────────────────

async def _run_transaction_agent(tx: TransactionData) -> TransactionAnalyzedEvent:
    """
    Transaction Analysis Agent: reasons about the transaction's structure,
    amount, type, and balance changes to surface anomalies.
    """
    flags = []
    if tx.type in ("CASH_OUT", "TRANSFER") and tx.amount > 50_000:
        flags.append(f"High-risk transaction type with elevated value: {tx.type} (${tx.amount:,.2f})")
    if tx.amount > 20_000:
        flags.append(f"High-value transaction: ${tx.amount:,.2f} (exceeds $20k reporting threshold)")
    if tx.oldbalanceOrg > 0 and tx.amount >= tx.oldbalanceOrg * 0.95:
        flags.append("Account drain: transaction consumes ≥95% of origin balance")
    if tx.amount > tx.oldbalanceOrg * 5 and tx.oldbalanceDest == 0 and tx.amount > 5_000:
        flags.append("Destination account has zero balance — possible mule account")

    prompt = f"""You are a specialist Transaction Analysis Agent inside an AML fraud investigation system.
Your job is to examine a single transaction and produce a concise, evidence-based analysis for a human investigator.
Do NOT produce a recommendation — only analyze the transaction data.

Transaction Data:
  - Type: {tx.type}
  - Amount: ${tx.amount:,.2f}
  - Origin balance before: ${tx.oldbalanceOrg:,.2f}
  - Destination balance before: ${tx.oldbalanceDest:,.2f}
  - ML model risk score: {tx.risk_score:.1f}%
  - ML flagged: {tx.ml_flagged}

Pre-computed deviation signals: {flags if flags else 'None'}

Write 2-3 sentences. Be specific. Reference the numbers. Highlight the most suspicious aspect."""

    response = await Settings.llm.acomplete(prompt)
    return TransactionAnalyzedEvent(
        transaction=tx,
        findings=TransactionFindings(analysis=str(response).strip(), deviation_flags=flags),
    )


async def _run_customer_agent(cust: CustomerData) -> CustomerAnalyzedEvent:
    """
    Customer Intelligence Agent: reasons about identity risk signals —
    KYC status, PEP flags, account age, device sharing, and session anomalies.
    """
    flags = []
    if cust.kyc_status in ("Failed", "Rejected"):
        flags.append(f"KYC status '{cust.kyc_status}' — identity validation failed")
    elif cust.kyc_status == "Pending":
        flags.append("KYC status 'Pending' — identity verification incomplete")
    if cust.pep_status:
        flags.append("Politically Exposed Person (PEP) — requires Enhanced Due Diligence")
    if cust.risk_tier in ("High", "Critical", "High (Network Risk)"):
        flags.append(f"Pre-existing risk classification: {cust.risk_tier}")
    if cust.account_age_days < 30:
        flags.append(f"Very new account: only {cust.account_age_days} days old — elevated mule risk")
    if cust.shared_ip_count >= 5:
        flags.append(f"Device/IP shared with {cust.shared_ip_count} other accounts — synthetic identity signal")
    if cust.unique_counterparties_30d > 40:
        flags.append(f"High counterparty velocity: {cust.unique_counterparties_30d} unique recipients in 30 days")
    if cust.recent_failed_logins >= 3 and cust.session_velocity_seconds < 10:
        flags.append(
            f"Account Takeover (ATO) pattern: {cust.recent_failed_logins} failed logins + "
            f"{cust.session_velocity_seconds}s session velocity"
        )

    prompt = f"""You are a specialist Customer Intelligence Agent inside an AML fraud investigation system.
Your role is to assess a customer's identity and behavioral risk profile and produce an analysis for a human investigator.
Do NOT produce a recommendation — only analyze the customer.

Customer Profile:
  - KYC Status: {cust.kyc_status}
  - PEP Status: {cust.pep_status}
  - Risk Tier: {cust.risk_tier}
  - Account Age: {cust.account_age_days} days
  - Unique counterparties (30d): {cust.unique_counterparties_30d}
  - Accounts sharing same IP/device: {cust.shared_ip_count}
  - Recent failed logins: {cust.recent_failed_logins}
  - Session velocity: {cust.session_velocity_seconds} seconds
  - Historical average transaction: ${cust.historical_avg_tx_amount:,.2f}

Pre-computed risk signals: {flags if flags else 'None — customer profile appears clean'}

Write 2-3 sentences. Reason about whether the combination of signals suggests a specific fraud typology 
(e.g., Account Takeover, Synthetic Identity, Money Mule, PEP abuse). Be specific."""

    response = await Settings.llm.acomplete(prompt)
    return CustomerAnalyzedEvent(
        customer=cust,
        findings=CustomerFindings(analysis=str(response).strip(), risk_flags=flags),
    )


async def _run_graph_agent(customer_id: str) -> GraphAnalyzedEvent:
    """
    Graph Traversal Agent: actively queries Neo4j to explore the customer's
    network, then uses an LLM to interpret what the topology means for fraud.
    """
    raw_graph_data = {}
    flags = []

    try:
        driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        async with driver.session() as session:
            # Query 1: Shared device ring detection
            result = await session.run(
                "MATCH (u:User {id: $cust_id})-[:HAS_DEVICE]->(d:Device)<-[:HAS_DEVICE]-(other:User) "
                "RETURN count(DISTINCT other) AS shared_users, collect(DISTINCT d.id) AS devices",
                cust_id=customer_id
            )
            record = await result.single()
            if record:
                raw_graph_data["shared_device_users"] = record["shared_users"]
                raw_graph_data["shared_devices"] = record["devices"]
                if record["shared_users"] > 0:
                    flags.append(
                        f"Device ring: customer shares {len(record['devices'])} device(s) "
                        f"with {record['shared_users']} other account(s)"
                    )

            # Query 2: Transaction history depth
            result2 = await session.run(
                "MATCH (u:User {id: $cust_id})-[:PERFORMED]->(t:TransactionNode) "
                "RETURN count(t) AS tx_count, max(t.amount) AS max_tx",
                cust_id=customer_id
            )
            record2 = await result2.single()
            if record2:
                raw_graph_data["graph_tx_count"] = record2["tx_count"]
                raw_graph_data["graph_max_tx"] = record2["max_tx"]

        await driver.close()
    except Exception as e:
        raw_graph_data["error"] = str(e)
        flags.append(f"Graph query unavailable: {str(e)[:100]}")

    # Let the LLM interpret the graph topology
    graph_context = raw_graph_data if raw_graph_data else {"note": "No graph data found for this customer"}
    prompt = f"""You are a specialist Graph Network Analysis Agent inside an AML fraud investigation system.
You have just executed live Cypher queries against a Neo4j graph database and retrieved raw network data.
Your role is to interpret this graph topology for a human investigator.
Do NOT make a final recommendation — only analyze what the network data reveals.

Customer ID: {customer_id}
Raw Neo4j Graph Data: {graph_context}
Pre-computed network signals: {flags if flags else 'No network anomalies detected'}

Write 2-3 sentences. Focus on what the network connections reveal about fraud ring membership,
money mule patterns, or coordinated account abuse. If the graph is clean, state that clearly."""

    response = await Settings.llm.acomplete(prompt)
    return GraphAnalyzedEvent(
        findings=GraphFindings(
            analysis=str(response).strip(),
            ring_flags=flags,
            raw_graph_data=raw_graph_data,
        )
    )


async def _run_memory_agent(customer_id: str) -> MemoryAnalyzedEvent:
    """
    Memory Agent: queries SQLite for this customer's prior investigation history,
    then generates an LLM-authored temporal pattern analysis.
    This is unique to v2 — the system now has persistent, cross-transaction memory.
    """
    from memory_engine import run_memory_agent, format_history_for_prompt

    memory_data = await run_memory_agent(customer_id)
    history = memory_data["history"]
    formatted = memory_data["formatted_context"]
    flags = []

    if memory_data["was_previously_blocked"]:
        flags.append("CRITICAL: Customer was BLOCKED in a prior investigation — re-appearance is a strong fraud signal")
    if memory_data["repeated_holds"] >= 2:
        flags.append(f"Pattern: Customer has been HELD {memory_data['repeated_holds']} times previously — escalation warranted")
    if memory_data["prior_investigation_count"] > 3:
        flags.append(f"High investigation frequency: {memory_data['prior_investigation_count']} prior reviews on record")

    prompt = f"""You are a specialist Memory Agent inside an AML fraud investigation system.
You have queried the persistent investigation database and retrieved this customer's historical fraud investigation record.
Your role is to identify temporal patterns and escalation signals based on the history.
Do NOT make a final recommendation — only analyze the historical pattern.

{formatted}

Pre-computed memory signals: {flags if flags else 'No prior history — first investigation for this customer'}

Write 2-3 sentences. Focus on: (1) whether the current investigation follows a pattern,
(2) whether prior decisions were escalating or de-escalating, (3) any re-appearance-after-block anomalies."""

    response = await Settings.llm.acomplete(prompt)
    return MemoryAnalyzedEvent(
        findings=MemoryFindings(
            analysis=str(response).strip(),
            memory_flags=flags,
            prior_count=memory_data["prior_investigation_count"],
            was_previously_blocked=memory_data["was_previously_blocked"],
        )
    )


# ── Main Workflow ─────────────────────────────────────────────────────────────

# Shared callback for broadcasting thinking events (set by api.py)
_thinking_callback = None

def set_thinking_callback(callback):
    """Register an async callback to broadcast THINKING_UPDATE events to WebSocket."""
    global _thinking_callback
    _thinking_callback = callback


async def _emit_thinking(agent: str, message: str, step_idx: int, customer_id: str = None):
    """Emit a thinking step to the WebSocket if a callback is registered."""
    if _thinking_callback:
        try:
            await _thinking_callback(agent, message, step_idx, customer_id)
        except Exception:
            pass


class FraudInvestigationWorkflow(Workflow):
    """
    Parallel multi-agent fraud investigation workflow (v2).

    Step 1 (start):        Fan out to 4 specialist agents via asyncio.gather.
    Step 2 (fan-in x4):    Each agent emits its event independently.
    Step 3 (supervisor):   collect_events waits for all 4, then:
                             a) Retrieves RAG policy context
                             b) Runs supervisor LLM call with CONFIDENCE scoring
                             c) If CONFIDENCE < 75 → self-reflection loop (re-query + re-reason)
                             d) Emits final InvestigationReport with reasoning_trace
    """

    @step
    async def dispatch_agents(
        self, ctx: Context, ev: StartEvent
    ) -> Union[TransactionAnalyzedEvent, CustomerAnalyzedEvent, GraphAnalyzedEvent, MemoryAnalyzedEvent]:
        """Fan out: run all 4 specialist agents in parallel."""
        tx: TransactionData = ev.transaction
        cust: CustomerData = ev.customer

        print(f"[Workflow] Dispatching 4 specialist agents in parallel for customer {cust.customer_id}...")

        await _emit_thinking("Orchestrator", f"Dispatching 4 specialist agents in parallel for {cust.customer_id}", 0, cust.customer_id)

        tx_result, cust_result, graph_result, memory_result = await asyncio.gather(
            _run_transaction_agent(tx),
            _run_customer_agent(cust),
            _run_graph_agent(cust.customer_id),
            _run_memory_agent(cust.customer_id),
        )

        await _emit_thinking("Transaction Agent", f"✓ Flagged {len(tx_result.findings.deviation_flags)} deviation(s): {tx_result.findings.deviation_flags[:2]}", 1, cust.customer_id)
        await _emit_thinking("Customer Agent", f"✓ Surfaced {len(cust_result.findings.risk_flags)} risk signal(s): {cust_result.findings.risk_flags[:2]}", 2, cust.customer_id)
        await _emit_thinking("Graph Agent", f"✓ Network analysis complete — {len(graph_result.findings.ring_flags)} ring flag(s) found", 3, cust.customer_id)
        await _emit_thinking("Memory Agent", f"✓ Retrieved {memory_result.findings.prior_count} prior investigation(s) for this customer", 4, cust.customer_id)

        print("[Workflow] All 4 agents completed. Forwarding to Supervisor...")

        ctx.send_event(tx_result)
        ctx.send_event(cust_result)
        ctx.send_event(graph_result)
        ctx.send_event(memory_result)

    @step
    async def supervisor_reasoning(
        self,
        ctx: Context,
        ev: Union[TransactionAnalyzedEvent, CustomerAnalyzedEvent, GraphAnalyzedEvent, MemoryAnalyzedEvent],
    ) -> Optional[StopEvent]:
        """
        Supervisor / Reasoning Agent with Self-Reflection.
        Waits for all 4 agent findings, retrieves RAG policy context,
        then iterates with a confidence check before finalizing.
        """
        ready = ctx.collect_events(
            ev,
            [TransactionAnalyzedEvent, CustomerAnalyzedEvent, GraphAnalyzedEvent, MemoryAnalyzedEvent],
        )
        if ready is None:
            return None  # Still waiting for agents

        tx_ev, cust_ev, graph_ev, mem_ev = ready
        all_flags = (
            tx_ev.findings.deviation_flags
            + cust_ev.findings.risk_flags
            + graph_ev.findings.ring_flags
            + mem_ev.findings.memory_flags
        )

        reasoning_trace = []
        print(f"[Supervisor] All 4 agent findings received. Total risk signals: {len(all_flags)}. Querying RAG...")

        cust_id = cust_ev.customer.customer_id
        await _emit_thinking("Supervisor", f"All 4 agents reported. Aggregated {len(all_flags)} total risk signals. Querying RAG policy knowledge base...", 5, cust_id)

        # Retrieve AML policy context
        rag_query = (
            f"AML compliance rules for {tx_ev.transaction.type} transactions, "
            f"KYC status '{cust_ev.customer.kyc_status}', "
            f"PEP status {cust_ev.customer.pep_status}, "
            f"risk tier '{cust_ev.customer.risk_tier}'. "
            f"Relevant flags: {all_flags[:5]}"
        )
        policy_context, citations_str = await _retrieve_policy_context(rag_query)
        await _emit_thinking("RAG Engine", f"✓ Retrieved policy context from knowledge base. Sources: {citations_str[:80]}...", 6, cust_id)

        reasoning_trace.append(f"Agents completed: TX({len(tx_ev.findings.deviation_flags)} flags), Customer({len(cust_ev.findings.risk_flags)} flags), Graph({len(graph_ev.findings.ring_flags)} flags), Memory({mem_ev.findings.prior_count} prior records)")
        reasoning_trace.append(f"RAG query retrieved policy context from: {citations_str[:120]}")

        # ── First Supervisor Pass ─────────────────────────────────────────────
        supervisor_prompt = _build_supervisor_prompt(
            tx_ev, cust_ev, graph_ev, mem_ev, all_flags, policy_context, citations_str, reflection_pass=False
        )

        await _emit_thinking("Supervisor", "Running primary reasoning pass — synthesizing all agent findings against AML policy...", 7, cust_id)
        response = await Settings.llm.acomplete(supervisor_prompt)
        text = str(response).strip()

        action, explanation, confidence = _parse_supervisor_response(text)
        reasoning_trace.append(f"Primary decision: {action} (confidence: {confidence}%)")

        reflection_triggered = False
        reflection_reason = ""

        # ── Self-Reflection Loop ──────────────────────────────────────────────
        if confidence < 75:
            reflection_triggered = True
            reflection_reason = (
                f"Initial confidence of {confidence}% fell below the 75% threshold. "
                f"Agent autonomously re-querying with a more specific policy search."
            )
            await _emit_thinking(
                "Supervisor",
                f"⚠️ Confidence {confidence}% < 75% threshold. Triggering self-reflection — refining RAG query and re-reasoning...",
                8, cust_id
            )
            reasoning_trace.append(reflection_reason)

            # Refine RAG query based on the primary flags
            refined_query = (
                f"Specific AML escalation rules for {action} decision: "
                f"flags include {', '.join(all_flags[:3])}. "
                f"Customer type: {'PEP' if cust_ev.customer.pep_status else 'Non-PEP'}, "
                f"KYC: {cust_ev.customer.kyc_status}. "
                f"What additional policy sections apply?"
            )
            refined_context, refined_citations = await _retrieve_policy_context(refined_query)
            await _emit_thinking("RAG Engine", f"✓ Reflection RAG query complete. Retrieved additional policy context.", 9, cust_id)

            # Second supervisor pass with enriched context
            reflection_prompt = _build_supervisor_prompt(
                tx_ev, cust_ev, graph_ev, mem_ev, all_flags,
                policy_context + "\n\n=== ADDITIONAL CONTEXT (self-reflection pass) ===\n" + refined_context,
                citations_str + "\n" + refined_citations,
                reflection_pass=True,
                original_decision=action,
                original_confidence=confidence,
            )
            await _emit_thinking("Supervisor", "Running reflection pass — re-evaluating with expanded policy context...", 10, cust_id)
            response2 = await Settings.llm.acomplete(reflection_prompt)
            text2 = str(response2).strip()
            action, explanation, confidence = _parse_supervisor_response(text2)
            reasoning_trace.append(f"Reflection decision: {action} (confidence: {confidence}%)")

        await _emit_thinking("Supervisor", f"✅ Final decision: {action} (confidence: {confidence}%). SAR generated.", 11 if reflection_triggered else 8, cust_id)
        print(f"[Supervisor] SAR complete. Decision: {action} | Confidence: {confidence}% | Reflection: {reflection_triggered}")

        return StopEvent(
            result=InvestigationReport(
                risk_score=tx_ev.transaction.risk_score,
                recommended_action=action,
                reasons=all_flags,
                sar_explanation=explanation,
                policy_citations=citations_str,
                reasoning_trace=reasoning_trace,
                reflection_triggered=reflection_triggered,
                reflection_reason=reflection_reason,
                confidence_score=float(confidence),
            )
        )


# ── Prompt Builders ───────────────────────────────────────────────────────────

def _build_supervisor_prompt(
    tx_ev, cust_ev, graph_ev, mem_ev, all_flags,
    policy_context, citations_str,
    reflection_pass: bool = False,
    original_decision: str = "",
    original_confidence: int = 0,
) -> str:
    reflection_note = ""
    if reflection_pass:
        reflection_note = f"""
⚠️ SELF-REFLECTION PASS: Your initial decision was "{original_decision}" with {original_confidence}% confidence,
which was below the required 75% threshold. You have been provided with additional policy context.
Re-evaluate carefully. Your final decision may confirm or change the initial assessment.
"""

    return f"""You are the Supervisor Reasoning Agent — the final decision-maker in a parallel multi-agent AML investigation.
{reflection_note}
You have received independent analyses from FOUR specialist sub-agents. Your job is to:
1. Deliberate on the combined evidence across all four dimensions (transaction, customer identity, network graph, investigation memory).
2. Triangulate whether the flags are mutually corroborating or isolated.
   * FALSE POSITIVE REDUCTION RULE: High transaction amounts or routine transfers are common in legitimate banking. Do NOT recommend 'Block' or 'Hold for Review' unless there are MULTIPLE co-occurring risk flags across at least 2 distinct agent domains (e.g. Transaction + Customer ATO, or Customer + Fraud Ring Graph). If an account is Verified with clean history and isolated flags, recommend 'Approve'.
3. Ground your decision in the retrieved AML policy excerpts below.
4. Produce a clear, cited SAR decision.
5. Provide a CONFIDENCE score (0-100) reflecting how certain you are of the decision.

═══════════════════════════════════════════════
SPECIALIST AGENT FINDINGS
═══════════════════════════════════════════════

[Transaction Analysis Agent]
{tx_ev.findings.analysis}
Deviation flags: {tx_ev.findings.deviation_flags}

[Customer Intelligence Agent]
{cust_ev.findings.analysis}
Risk flags: {cust_ev.findings.risk_flags}

[Graph Network Analysis Agent]
{graph_ev.findings.analysis}
Network flags: {graph_ev.findings.ring_flags}

[Memory Agent — Cross-Transaction History]
{mem_ev.findings.analysis}
Memory flags: {mem_ev.findings.memory_flags}
Prior investigations: {mem_ev.findings.prior_count} | Previously blocked: {mem_ev.findings.was_previously_blocked}

═══════════════════════════════════════════════
QUANTITATIVE SIGNALS
═══════════════════════════════════════════════
ML Fast-Path Risk Score: {tx_ev.transaction.risk_score:.1f}%
Total combined risk signals: {len(all_flags)}

═══════════════════════════════════════════════
AML POLICY CONTEXT (from knowledge base)
═══════════════════════════════════════════════
{policy_context}

Knowledge base sources:
{citations_str}

═══════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════
Respond in EXACTLY this format (no additional text before or after):

RECOMMENDED_ACTION: <Approve | Hold for Review | Block>

CONFIDENCE: <integer 0-100>

EXPLANATION: <Write 4-5 sentences for a human compliance investigator. 
Triangulate the agent findings. Cite the specific AML policy sections and 
document names from the sources above. State clearly what the combined 
evidence suggests about the fraud typology (e.g., ATO, money mule, PEP abuse, fraud ring).>"""


def _parse_supervisor_response(text: str) -> tuple[str, str, int]:
    """Parse action, explanation, and confidence from the supervisor LLM response."""
    action = "Hold for Review"
    explanation = text
    confidence = 70  # Default if not parsed

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("RECOMMENDED_ACTION:"):
            action = stripped.split(":", 1)[1].strip()
        elif stripped.upper().startswith("CONFIDENCE:"):
            try:
                confidence = int(stripped.split(":", 1)[1].strip())
                confidence = max(0, min(100, confidence))
            except ValueError:
                pass
        elif stripped.upper().startswith("EXPLANATION:"):
            explanation = stripped.split(":", 1)[1].strip()

    return action, explanation, confidence


# ── Public API ────────────────────────────────────────────────────────────────

async def investigate(
    transaction: TransactionData, customer: CustomerData
) -> InvestigationReport:
    """Entry point: run the parallel multi-agent investigation workflow."""
    if os.environ.get("MOCK_LLM", "").lower() in ("true", "1"):
        cust_id = customer.customer_id
        await _emit_thinking("Orchestrator", f"[MOCK] Dispatching 4 specialist agents for {cust_id}", 0, cust_id)
        await asyncio.sleep(0.2)
        await _emit_thinking("Transaction Agent", f"[MOCK] ✓ Flagged deviations: High value ${transaction.amount:,.2f}, balance drain", 1, cust_id)
        await _emit_thinking("Customer Agent", f"[MOCK] ✓ Surfaced risk signals: KYC {customer.kyc_status}, Shared IPs {customer.shared_ip_count}", 2, cust_id)
        await _emit_thinking("Graph Agent", f"[MOCK] ✓ Network graph complete — shared device cluster detected", 3, cust_id)
        await _emit_thinking("Memory Agent", f"[MOCK] ✓ Query complete — prior history checked", 4, cust_id)
        await _emit_thinking("Supervisor", "[MOCK] Synthesizing findings against AML policy...", 5, cust_id)
        await _emit_thinking("RAG Engine", "[MOCK] ✓ Retrieved policy context (AML Policy Sec. 2.4 & 4.1)", 6, cust_id)
        
        action = "Block" if transaction.risk_score > 85 else "Hold for Review"
        await _emit_thinking("Supervisor", f"✅ Final decision: {action} (confidence: 94%). SAR generated.", 7, cust_id)

        return InvestigationReport(
            risk_score=transaction.risk_score,
            recommended_action=action,
            reasons=[
                f"High-value transaction (${transaction.amount:,.2f}) draining origin balance",
                f"Identity signal: KYC {customer.kyc_status}, shared IP count: {customer.shared_ip_count}",
                f"Graph cluster: account linked to multi-user device ring"
            ],
            sar_explanation=f"MOCK MODE ACTIVE: Transaction for customer {cust_id} exhibits multi-vector risk indicators including high velocity, shared device topology, and balance draining behavior under AML Policy Section 2.4.",
            policy_citations="  [1] aml_policy.txt\n  [2] interpol_guidelines.pdf",
            reasoning_trace=["Mock investigation generated without LLM API call (MOCK_LLM=true)"],
            confidence_score=94.0
        )

    # Safety net for CLI/tests that call investigate() without FastAPI startup.
    _init_policy_index()
    workflow = FraudInvestigationWorkflow(timeout=300)
    result = await workflow.run(transaction=transaction, customer=customer)
    return result


# ── Demo / CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_tx = TransactionData(
        customer_id="C1539299608", type="TRANSFER", amount=1_241_118.86,
        oldbalanceOrg=1_241_118.86, oldbalanceDest=0.0, risk_score=97.4, ml_flagged=True,
    )
    demo_cust = CustomerData(
        customer_id="C1539299608", kyc_status="Pending", pep_status=False,
        risk_tier="High (Network Risk)", account_age_days=42,
        unique_counterparties_30d=38, shared_ip_count=20,
        recent_failed_logins=4, session_velocity_seconds=3,
        historical_avg_tx_amount=890.50,
    )

    print("=" * 60)
    print("Running Parallel Multi-Agent Fraud Investigation (v2)...")
    print("=" * 60)
    report = asyncio.run(investigate(demo_tx, demo_cust))
    print("\n" + "=" * 60)
    print("INVESTIGATION REPORT")
    print("=" * 60)
    print(report.model_dump_json(indent=2))
