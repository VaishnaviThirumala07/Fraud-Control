import { useState, useEffect, useRef, useCallback } from 'react';
import {
  ShieldAlert, ShieldCheck, ShieldX, Bot, ArrowRightLeft, Radio,
  Activity, Zap, Brain, Network, Database, Eye, GitBranch,
  AlertTriangle, CheckCircle2, Clock, Lock,
  ChevronDown, ChevronUp, ChevronLeft, ChevronRight, Play, Pause, X, FileText, Cpu, Key, UserCheck, UserX,
  Sun, Moon, Shield, Search, Wrench, HardDrive, BookOpen, Siren,
} from 'lucide-react';
import './index.css';

// ── Types ──────────────────────────────────────────────────────────────────

interface Transaction {
  customer_id: string;
  type: string;
  amount: number;
  oldbalanceOrg: number;
  oldbalanceDest: number;
  isFraud?: number;
}

interface Profile {
  customer_id: string;
  kyc_status: string;
  pep_status: boolean | number;
  risk_tier: string;
  account_age_days: number;
  unique_counterparties_30d: number;
  shared_ip_count: number;
  recent_failed_logins: number;
  session_velocity_seconds: number;
  historical_avg_tx_amount: number;
}

interface FastPathResult {
  risk_score: number;
  is_flagged: boolean;
  error?: string;
}

interface Report {
  recommended_action: string;
  sar_explanation: string;
  reasons: string[];
  policy_citations: string;
  reasoning_trace: string[];
  reflection_triggered: boolean;
  reflection_reason: string;
  confidence_score: number;
}

interface OTPChallenge {
  code: string;
  status: string;
  message: string;
}

interface ActionResult {
  action_taken: string;
  action_description: string;
  alert_id?: string;
  otp_challenge?: OTPChallenge;
}

interface ThinkingLine {
  agent: string;
  message: string;
  step_index: number;
}

interface BlockedAccount {
  customer_id: string;
  blocked_at: string;
  risk_score: number;
  transaction_amount: number;
  transaction_type: string;
  primary_reason: string;
}

interface AlertItem {
  id: string;
  severity: 'CRITICAL' | 'HIGH';
  type: string;
  customer_id: string;
  timestamp: string;
  message: string;
}

interface Stats {
  total_processed: number;
  blocked: number;
  held: number;
  approved: number;
  fraud_rate_pct: number;
}

interface QueuedEvent {
  id: string;
  transaction: Transaction;
  profile: Profile;
  fastPathResult: FastPathResult;
  report?: Report;
  actionResult?: ActionResult;
  manual?: boolean;
}

// ── Agent Timeline Steps ───────────────────────────────────────────────────

const AGENT_STEPS = [
  { key: 'orchestrator', name: 'Orchestrator', icon: GitBranch,  desc: 'Dispatching 4 parallel agents' },
  { key: 'transaction',  name: 'Transaction Agent',  icon: ArrowRightLeft, desc: 'Analyzing amount, type, balance' },
  { key: 'customer',     name: 'Customer Agent',     icon: Eye,    desc: 'Evaluating KYC, PEP, behavior' },
  { key: 'graph',        name: 'Graph Agent',        icon: Network, desc: 'Neo4j Cypher ring detection' },
  { key: 'memory',       name: 'Memory Agent',       icon: Database, desc: 'Querying SQLite history' },
  { key: 'rag',          name: 'RAG Engine',         icon: Brain,  desc: 'Retrieving AML policy context' },
  { key: 'supervisor',   name: 'Supervisor Agent',   icon: Cpu,    desc: 'Synthesizing all findings' },
];

const STEP_INDEX_TO_KEY: Record<number, string> = {
  0: 'orchestrator', 1: 'transaction', 2: 'customer', 3: 'graph', 4: 'memory',
  5: 'supervisor', 6: 'rag', 7: 'supervisor', 8: 'supervisor', 9: 'rag',
  10: 'supervisor', 11: 'supervisor',
};

// ── Dynamic URL Resolution (Works locally & on Daytona CDE) ────────────────
const getWsUrl = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/stream`;
};

const WS_URL = getWsUrl();
const API_URL = WS_URL.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws/stream', '');

const fmt = (v: number) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(v);
const fmtDate = (iso: string) => new Date(iso).toLocaleTimeString();

const getRiskClass = (score: number) =>
  score >= 85 ? 'high' : score >= 60 ? 'medium' : 'low';

const getActionClass = (action: string) => {
  const a = action.toLowerCase();
  if (a.includes('block')) return 'block';
  if (a.includes('hold') || a.includes('review')) return 'hold';
  return 'approve';
};

const getActionTakenClass = (taken: string) => {
  if (taken.includes('BLOCK')) return 'blocked';
  if (taken.includes('HELD')) return 'held';
  return 'approved';
};

const DEFAULT_FORM = {
  customer_id: 'C1539299608',
  type: 'TRANSFER',
  amount: '1241118.86',
  oldbalanceOrg: '1241118.86',
  oldbalanceDest: '0.0',
  kyc_status: 'Pending',
  pep_status: 'false',
  risk_tier: 'High (Network Risk)',
  account_age_days: '42',
  unique_counterparties_30d: '38',
  shared_ip_count: '20',
  recent_failed_logins: '4',
  session_velocity_seconds: '3',
  historical_avg_tx_amount: '890.50',
};

export default function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('fraudcontrol-theme');
    if (saved === 'light' || saved === 'dark') return saved;
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  });
  const [queue, setQueue] = useState<QueuedEvent[]>([]);
  const [historyList, setHistoryList] = useState<QueuedEvent[]>([]);
  const [historySidebarCollapsed, setHistorySidebarCollapsed] = useState(false);
  const [metricsExpanded, setMetricsExpanded] = useState(false);
  const [activeEvent, setActiveEvent] = useState<QueuedEvent | null>(null);
  const [inspectedEvent, setInspectedEvent] = useState<QueuedEvent | null>(null);
  const [isInvestigating, setIsInvestigating] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isPaused, setIsPaused] = useState(false); // STREAM CONTROL: Pause auto-advance to allow deep inspection
  const [stats, setStats] = useState<Stats>({ total_processed: 0, blocked: 0, held: 0, approved: 0, fraud_rate_pct: 0 });
  const [blockedAccounts, setBlockedAccounts] = useState<BlockedAccount[]>([]);
  const [alertItems, setAlertItems] = useState<AlertItem[]>([]);
  const [thinkingLines, setThinkingLines] = useState<ThinkingLine[]>([]);
  const [completedSteps, setCompletedSteps] = useState<Set<string>>(new Set());
  const [activeStep, setActiveStep] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState(DEFAULT_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [profileExpanded, setProfileExpanded] = useState(false);
  const [policyExpanded, setPolicyExpanded] = useState(false);
  const [wasPaused, setWasPaused] = useState(false);

  // OTP & HITL local states
  const [otpInput, setOtpInput] = useState('');
  const [otpStatusMsg, setOtpStatusMsg] = useState<string | null>(null);
  const [hitlNotes, setHitlNotes] = useState('');
  const [hitlStatusMsg, setHitlStatusMsg] = useState<string | null>(null);
  const [hitlSubmitting, setHitlSubmitting] = useState(false);

  const activeEventRef = useRef(activeEvent);
  const queueRef = useRef(queue);
  const isPausedRef = useRef(isPaused);
  const thinkingRef = useRef<HTMLDivElement>(null);

  useEffect(() => { activeEventRef.current = activeEvent; }, [activeEvent]);
  useEffect(() => { queueRef.current = queue; }, [queue]);
  useEffect(() => { isPausedRef.current = isPaused; }, [isPaused]);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('fraudcontrol-theme', theme);
  }, [theme]);

  useEffect(() => {
    if (thinkingRef.current) {
      thinkingRef.current.scrollTop = thinkingRef.current.scrollHeight;
    }
  }, [thinkingLines]);

  const resetAgentState = () => {
    setThinkingLines([]);
    setCompletedSteps(new Set());
    setActiveStep(null);
    setOtpInput('');
    setOtpStatusMsg(null);
    setHitlNotes('');
    setHitlStatusMsg(null);
  };

  // Helper to add finished event to history
  const addToHistory = (ev: QueuedEvent) => {
    setHistoryList(prev => {
      const exists = prev.some(item => item.id === ev.id);
      if (exists) {
        return prev.map(item => item.id === ev.id
          ? {
              ...item,
              ...ev,
              // SLOW_PATH and ACTION_TAKEN can arrive back-to-back. Preserve
              // whichever half of the completed investigation is already stored.
              report: ev.report ?? item.report,
              actionResult: ev.actionResult ?? item.actionResult,
            }
          : item);
      }
      return [ev, ...prev];
    });
  };

  const processNextInQueue = useCallback(() => {
    if (isPausedRef.current) return; // If stream is paused by user, freeze current inspection card!

    if (queueRef.current.length > 0) {
      const next = queueRef.current[0];
      setQueue(prev => prev.slice(1));
      activeEventRef.current = next;
      setActiveEvent(next);
      setProfileExpanded(false);
      setPolicyExpanded(false);
      resetAgentState();
      if (next.fastPathResult.is_flagged && !next.report) {
        setIsInvestigating(true);
        setActiveStep('orchestrator');
      } else {
        setIsInvestigating(false);
        addToHistory(next);
        setTimeout(() => processNextInQueue(), 6000);
      }
    } else {
      activeEventRef.current = null;
      setActiveEvent(null);
    }
  }, []);

  useEffect(() => {
    if (wasPaused && !isPaused) {
      if (!isInvestigating && (queue.length > 0 || activeEvent)) {
        processNextInQueue();
      }
    }
    setWasPaused(isPaused);
  }, [isPaused, wasPaused, isInvestigating, queue.length, activeEvent, processNextInQueue]);

  // WebSocket
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => setIsConnected(true);
      ws.onerror = () => setIsConnected(false);
      ws.onclose = () => { setIsConnected(false); setTimeout(connect, 3000); };

      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);

        if (data.type === 'INIT') {
          if (data.stats) setStats(data.stats);
          if (data.blocked_accounts) setBlockedAccounts(data.blocked_accounts);
          if (data.alerts) setAlertItems(data.alerts);
          if (data.transactions) {
            const restored: QueuedEvent[] = data.transactions;
            setHistoryList(restored);
            if (!activeEventRef.current && restored.length > 0) {
              const latest = restored[0];
              activeEventRef.current = latest;
              setActiveEvent(latest);
              setIsInvestigating(latest.fastPathResult.is_flagged && !latest.report);
            }
          }
          return;
        }

        if (data.type === 'FAST_PATH') {
          if (data.stats) setStats(data.stats);
          if (data.blocked_accounts) setBlockedAccounts(data.blocked_accounts);
          if (data.alerts) setAlertItems(data.alerts);
          const newEvent: QueuedEvent = {
            id: data.transaction_id || `${data.transaction.customer_id}:${Date.now()}`,
            transaction: data.transaction,
            profile: data.profile,
            fastPathResult: data.fast_path_result,
            manual: data.manual_trigger,
          };
          // The sidebar represents received transactions, not only completed
          // investigations. SLOW_PATH and ACTION_TAKEN enrich this same record.
          addToHistory(newEvent);
          if (!activeEventRef.current) {
            activeEventRef.current = newEvent;
            setActiveEvent(newEvent);
            setProfileExpanded(false);
            setPolicyExpanded(false);
            resetAgentState();
            if (newEvent.fastPathResult.is_flagged) {
              setIsInvestigating(true);
              setActiveStep('orchestrator');
            } else {
              setIsInvestigating(false);
              addToHistory(newEvent);
              if (!isPausedRef.current) {
                setTimeout(() => processNextInQueue(), 6000);
              }
            }
          } else {
            setQueue(prev => [...prev, newEvent]);
          }
          return;
        }

        if (data.type === 'THINKING_UPDATE') {
          const { agent, message, step_index, customer_id } = data;
          
          // Only append thinking lines if they belong to the active event being investigated
          if (customer_id && activeEventRef.current && activeEventRef.current.transaction.customer_id !== customer_id) {
            return; 
          }
          
          setThinkingLines(prev => {
            const exists = prev.some(l => l.agent === agent && l.message === message && l.step_index === step_index);
            if (exists) return prev;
            return [...prev.slice(-29), { agent, message, step_index }];
          });
          const stepKey = STEP_INDEX_TO_KEY[step_index] || 'supervisor';
          setActiveStep(stepKey);
          setCompletedSteps(prev => {
            const next = new Set(prev);
            const keys = AGENT_STEPS.map(s => s.key);
            const idx = keys.indexOf(stepKey);
            for (let i = 0; i < idx; i++) next.add(keys[i]);
            return next;
          });
          return;
        }

        if (data.type === 'SLOW_PATH') {
          const report: Report = data.report;

          if (activeEventRef.current && (data.transaction_id
            ? activeEventRef.current.id === data.transaction_id
            : activeEventRef.current.transaction.customer_id === data.customer_id)) {
            setCompletedSteps(new Set(AGENT_STEPS.map(s => s.key)));
            setActiveStep(null);
            const updated = { ...activeEventRef.current!, report };
            activeEventRef.current = updated;
            setActiveEvent(updated);
            addToHistory(updated);
            setIsInvestigating(false);
            if (!isPausedRef.current) {
              setTimeout(() => processNextInQueue(), 10000);
            }
          } else {
            setQueue(prev => prev.map(item =>
              (data.transaction_id ? item.id === data.transaction_id : item.transaction.customer_id === data.customer_id)
                ? { ...item, report }
                : item
            ));
          }
          return;
        }

        if (data.type === 'ACTION_TAKEN') {
          if (data.stats) setStats(data.stats);
          if (data.blocked_accounts) setBlockedAccounts(data.blocked_accounts);
          if (data.alerts) setAlertItems(data.alerts);
          const actionResult: ActionResult = data.action_result;
          if (!data.blocked_accounts && actionResult.action_taken.includes('BLOCK')) {
            setBlockedAccounts(prev => [
              { customer_id: data.customer_id, blocked_at: new Date().toISOString(), risk_score: 0, transaction_amount: 0, transaction_type: '', primary_reason: '' },
              ...prev.slice(0, 19)
            ]);
          }
          if (activeEventRef.current && (data.transaction_id
            ? activeEventRef.current.id === data.transaction_id
            : activeEventRef.current.transaction.customer_id === data.customer_id)) {
            const updated = { ...activeEventRef.current!, actionResult };
            activeEventRef.current = updated;
            setActiveEvent(updated);
            addToHistory(updated);
          } else {
            setQueue(prev => prev.map(item =>
              (data.transaction_id ? item.id === data.transaction_id : item.transaction.customer_id === data.customer_id)
                ? { ...item, actionResult }
                : item
            ));
          }
          return;
        }
      };
      return ws;
    };
    const ws = connect();
    return () => {
      ws.onclose = null;
      ws.close();
    };
  }, [processNextInQueue]);

  // OTP Verification Submission
  const handleVerifyOTP = async (codeToVerify?: string) => {
    const targetEvent = inspectedEvent ?? activeEvent;
    if (!targetEvent) return;
    const code = codeToVerify || otpInput;
    if (!code) return;
    try {
      const res = await fetch(`${API_URL}/api/verify-otp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ customer_id: targetEvent.transaction.customer_id, code }),
      });
      const data = await res.json();
      if (data.success) {
        setOtpStatusMsg('OTP verified successfully. Transaction released.');
        if (data.action_result) {
          setActiveEvent(prev => prev?.id === targetEvent.id ? { ...prev, actionResult: data.action_result } : prev);
          setInspectedEvent(prev => prev?.id === targetEvent.id ? { ...prev, actionResult: data.action_result } : prev);
          setHistoryList(prev => prev.map(item => item.id === targetEvent.id ? { ...item, actionResult: data.action_result } : item));
        }
      } else {
        setOtpStatusMsg(data.message);
      }
    } catch (err) {
      setOtpStatusMsg('Failed to verify OTP.');
    }
  };

  // HITL Override Submission
  const handleHITLOverride = async (action: 'Approve' | 'Block') => {
    const targetEvent = inspectedEvent ?? activeEvent;
    if (!targetEvent || hitlSubmitting) return;
    setHitlSubmitting(true);
    try {
      const res = await fetch(`${API_URL}/api/hitl-override`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          customer_id: targetEvent.transaction.customer_id,
          action,
          notes: hitlNotes || 'Investigator live override from dashboard UI',
        }),
      });
      const data = await res.json();
      if (data.success) {
        setHitlStatusMsg(`Investigator override: ${action.toUpperCase()} applied.`);
        if (data.action_result) {
          setActiveEvent(prev => prev?.id === targetEvent.id ? { ...prev, actionResult: data.action_result } : prev);
          setInspectedEvent(prev => prev?.id === targetEvent.id ? { ...prev, actionResult: data.action_result } : prev);
          setHistoryList(prev => prev.map(item => item.id === targetEvent.id ? { ...item, actionResult: data.action_result } : item));
        }
      }
    } catch (err) {
      setHitlStatusMsg('Override action failed.');
    } finally {
      setHitlSubmitting(false);
    }
  };

  // Manual trigger submit
  const handleTrigger = async () => {
    setSubmitting(true);
    setShowModal(false);
    try {
      await fetch(`${API_URL}/api/investigate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          amount: parseFloat(form.amount),
          oldbalanceOrg: parseFloat(form.oldbalanceOrg),
          oldbalanceDest: parseFloat(form.oldbalanceDest),
          account_age_days: parseInt(form.account_age_days),
          unique_counterparties_30d: parseInt(form.unique_counterparties_30d),
          shared_ip_count: parseInt(form.shared_ip_count),
          recent_failed_logins: parseInt(form.recent_failed_logins),
          session_velocity_seconds: parseInt(form.session_velocity_seconds),
          historical_avg_tx_amount: parseFloat(form.historical_avg_tx_amount),
          pep_status: form.pep_status === 'true',
        }),
      });
    } catch (err) {
      console.error('Trigger failed:', err);
    } finally {
      setSubmitting(false);
    }
  };

  const displayedEvent = inspectedEvent ?? activeEvent;
  const riskClass = displayedEvent ? getRiskClass(displayedEvent.fastPathResult.risk_score) : 'low';
  const actionClass = displayedEvent?.report ? getActionClass(displayedEvent.report.recommended_action) : '';

  return (
    <div className={`app-root ${historySidebarCollapsed ? 'history-collapsed' : ''}`}>
      {/* ── Header ───────────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-brand">
          <div className="header-logo"><Shield size={19} /></div>
          <div>
            <div className="header-title">FraudControl</div>
            <div className="header-subtitle">Autonomous Agentic Fraud Operations Console</div>
          </div>
        </div>
        <div className="header-right">
          <button
            className="icon-button"
            onClick={() => setTheme(current => current === 'dark' ? 'light' : 'dark')}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          {/* Pause Stream Control Button */}
          <button
            className={`btn-secondary ${isPaused ? 'btn-paused' : ''}`}
            style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem', padding: '0.4rem 0.8rem' }}
            onClick={() => setIsPaused(!isPaused)}
          >
            {isPaused ? <Play size={13} style={{ color: 'var(--success)' }} /> : <Pause size={13} style={{ color: 'var(--warning)' }} />}
            {isPaused ? 'Resume Stream' : 'Pause Stream'}
          </button>

          {queue.length > 0 && (
            <div className="queue-badge">
              <Clock size={12} /> {queue.length} buffered
            </div>
          )}
          <div className={`conn-status ${isConnected ? 'live' : 'offline'}`}>
            <div className="conn-dot" />
            {isConnected ? 'LIVE STREAM' : 'OFFLINE'}
          </div>
        </div>
      </header>

      {/* ── Main ─────────────────────────────────────────────────────── */}
      <main className={`main-content ${historySidebarCollapsed ? 'history-collapsed' : ''}`}>
        {/* Controls Bar */}
        <div className="controls-bar">
          {isConnected ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.78rem' }}>
              <Radio size={14} style={{ color: 'var(--success)', animation: 'live-pulse 1.5s ease infinite' }} />
              Kafka/RabbitMQ Stream Active
              {isPaused && (
                <span style={{ color: 'var(--warning)', background: 'var(--warning-dim)', border: '1px solid rgba(255,184,48,0.3)', padding: '0.1rem 0.5rem', borderRadius: '4px', fontSize: '0.72rem', fontWeight: 600 }}>
                  <Pause size={12} /> Stream paused
                </span>
              )}
            </div>
          ) : (
            <div style={{ color: 'var(--danger)', fontSize: '0.78rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Activity size={14} /> Connecting to backend...
            </div>
          )}
          <button id="btn-manual-trigger" className="btn-trigger" onClick={() => setShowModal(true)}>
            <Play size={14} /> Manual Investigation Trigger
          </button>
        </div>

        {/* ── Stats Bar ──────────────────────────────────────────────── */}
        <div className="metrics-panel">
          <button
            className="metrics-header-toggle"
            onClick={() => setMetricsExpanded(current => !current)}
            aria-expanded={metricsExpanded}
          >
            <span className="metrics-header-title"><Activity size={15} /> Metrics</span>
            <span className="metrics-header-summary">
              {stats.total_processed} processed · {stats.blocked} blocked · {stats.held} held · {stats.approved} approved
            </span>
            {metricsExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
          {metricsExpanded && <div className="stats-bar">
            <div className="stat-item">
              <span className="stat-label">Total Processed</span>
              <span className="stat-value total">{stats.total_processed}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Auto-Blocked</span>
              <span className="stat-value blocked">{stats.blocked}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Held for Review</span>
              <span className="stat-value held">{stats.held}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Approved</span>
              <span className="stat-value approved">{stats.approved}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Fraud Rate</span>
              <span className="stat-value rate">{stats.fraud_rate_pct}%</span>
            </div>
          </div>}
        </div>

        {/* Persistent transaction history sidebar */}
        <aside className="history-sidebar" aria-label="Transaction history">
          <div className="history-sidebar-header">
            {!historySidebarCollapsed && (
              <div>
                <div className="history-sidebar-title"><Clock size={14} /> Transactions</div>
                <div className="history-sidebar-count">{historyList.length} processed</div>
              </div>
            )}
            <button
              className="icon-button history-collapse-button"
              onClick={() => setHistorySidebarCollapsed(current => !current)}
              aria-label={historySidebarCollapsed ? 'Expand transaction history' : 'Collapse transaction history'}
              title={historySidebarCollapsed ? 'Expand transaction history' : 'Collapse transaction history'}
            >
              {historySidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
            </button>
          </div>

          {!historySidebarCollapsed && (
            <>
              {inspectedEvent && (
                <button className="return-live-button" onClick={() => setInspectedEvent(null)}>
                  <Radio size={13} /> Return to live transaction
                </button>
              )}
              <div className="history-list">
              {historyList.map((item, idx) => {
                const actionTaken = item.actionResult?.action_taken || item.report?.recommended_action || (item.fastPathResult.is_flagged ? 'FLAGGED' : 'APPROVED');
                const isSelected = displayedEvent?.transaction.customer_id === item.transaction.customer_id;
                const isBlock = actionTaken.includes('BLOCK');
                const isHold = actionTaken.includes('HELD') || actionTaken.includes('Hold');
                return (
                  <button
                    key={idx}
                    className={`history-item ${isSelected ? 'selected' : ''} ${isBlock ? 'blocked' : isHold ? 'held' : 'approved'}`}
                    onClick={() => {
                      setInspectedEvent(item);
                    }}
                  >
                    <span className="history-item-main">
                      <span className="history-item-id">{item.transaction.customer_id}</span>
                      <span className="history-item-type">{item.transaction.type}</span>
                    </span>
                    <span className="history-item-meta">
                      <span>{fmt(item.transaction.amount)}</span>
                      <span>{item.fastPathResult.risk_score.toFixed(0)}%</span>
                      {isBlock ? <ShieldX size={14} /> : isHold ? <Clock size={14} /> : <CheckCircle2 size={14} />}
                    </span>
                  </button>
                );
              })}
              {historyList.length === 0 && (
                <div className="history-empty"><Search size={20} /> No transactions yet</div>
              )}
              </div>
            </>
          )}
        </aside>

        {/* Idle State */}
        {!displayedEvent && (
          <div className="card">
            <div className="idle-state">
              <div className="idle-icon"><Search size={30} /></div>
              <p>
                No active transaction. Run <code>python producer.py</code> to start the live stream,
                or click <strong>Manual Investigation Trigger</strong> to demo the agent workflow instantly.
              </p>
            </div>
          </div>
        )}

        {/* Dashboard Grid */}
        {displayedEvent && (
          <div className="dashboard-grid slide-in">

            {/* ── Col 1: Fast Path ─────────────────────────────────── */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <div className="card">
                <div className="card-header">
                  <ArrowRightLeft size={14} className="icon" />
                  Incoming Transaction — Fast Path
                  {displayedEvent.manual && (
                    <span style={{ marginLeft: 'auto', fontSize: '0.65rem', color: 'var(--warning)', background: 'var(--warning-dim)', border: '1px solid rgba(255,184,48,0.3)', borderRadius: '4px', padding: '0.1rem 0.4rem' }}>
                      MANUAL TRIGGER
                    </span>
                  )}
                </div>
                <div className="card-body">
                  <div className="data-row">
                    <span className="data-label">Customer ID</span>
                    <span className="data-value mono">{displayedEvent.transaction.customer_id}</span>
                  </div>
                  <div className="data-row">
                    <span className="data-label">Type</span>
                    <span className="data-value">{displayedEvent.transaction.type}</span>
                  </div>
                  <div className="data-row">
                    <span className="data-label">Origin Balance</span>
                    <span className="data-value">{fmt(displayedEvent.transaction.oldbalanceOrg)}</span>
                  </div>
                  <div className="data-row">
                    <span className="data-label">Dest Balance</span>
                    <span className="data-value">{fmt(displayedEvent.transaction.oldbalanceDest)}</span>
                  </div>

                  <div className="metric-block">
                    <div>
                      <div className="metric-label">Transaction Amount</div>
                      <div className="metric-amount">{fmt(displayedEvent.transaction.amount)}</div>
                    </div>
                    <Zap size={22} style={{ color: 'var(--accent)', opacity: 0.6 }} />
                  </div>

                  <div className="risk-gauge-wrap">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem' }}>
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>XGBoost + TabNet Risk Score</span>
                      <span className={`risk-score-label ${riskClass}`}>{displayedEvent.fastPathResult.risk_score.toFixed(1)}%</span>
                    </div>
                    <div className="risk-gauge-bar">
                      <div
                        className={`risk-gauge-fill ${riskClass}`}
                        style={{ width: `${displayedEvent.fastPathResult.risk_score}%` }}
                      />
                    </div>
                  </div>

                  {displayedEvent.fastPathResult.is_flagged ? (
                    <div className="status-pill block"><ShieldAlert size={14} /> FLAGGED — Routed to Slow Path (≥70%)</div>
                  ) : (
                    <div className="status-pill approve"><ShieldCheck size={14} /> Approved — Below 70% Threshold</div>
                  )}

                  {/* Profile Expander */}
                  <div className="expander">
                    <button id="btn-expand-profile" className="expander-trigger" onClick={() => setProfileExpanded(!profileExpanded)}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}><Eye size={12} /> Customer Profile</span>
                      {profileExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                    </button>
                    {profileExpanded && (
                      <div className="expander-content">
                        <div className="data-row"><span className="data-label">KYC Status</span><span className="data-value">{displayedEvent.profile.kyc_status}</span></div>
                        <div className="data-row"><span className="data-label">PEP</span><span className="data-value">{displayedEvent.profile.pep_status ? 'Yes' : 'No'}</span></div>
                        <div className="data-row"><span className="data-label">Risk Tier</span><span className="data-value">{displayedEvent.profile.risk_tier}</span></div>
                        <div className="data-row"><span className="data-label">Account Age</span><span className="data-value">{displayedEvent.profile.account_age_days} days</span></div>
                        <div className="data-row"><span className="data-label">Counterparties (30d)</span><span className="data-value">{displayedEvent.profile.unique_counterparties_30d}</span></div>
                        <div className="data-row"><span className="data-label">Shared IPs</span><span className="data-value">{displayedEvent.profile.shared_ip_count}</span></div>
                        <div className="data-row"><span className="data-label">Failed Logins</span><span className="data-value">{displayedEvent.profile.recent_failed_logins}</span></div>
                        <div className="data-row"><span className="data-label">Session Velocity</span><span className="data-value">{displayedEvent.profile.session_velocity_seconds}s</span></div>
                        <div className="data-row"><span className="data-label">Avg TX Amount</span><span className="data-value">{fmt(displayedEvent.profile.historical_avg_tx_amount)}</span></div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* ── Col 2: Slow Path + Agent Timeline + HITL Controls ──── */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

              {/* Agent Timeline */}
              {displayedEvent.fastPathResult.is_flagged && (
                <div className="card">
                  <div className="card-header">
                    <Bot size={14} className="icon" />
                    Agent Execution Timeline
                  </div>
                  <div className="card-body">
                    <div className="agent-timeline">
                      {AGENT_STEPS.map((step) => {
                        const isDone    = inspectedEvent?.report ? true : completedSteps.has(step.key);
                        const isActive  = !inspectedEvent && activeStep === step.key && !isDone;
                        const Icon = step.icon;
                        return (
                          <div key={step.key} className={`timeline-step ${isDone ? 'completed' : ''}`}>
                            <div className={`step-icon-wrap ${isDone ? 'done' : isActive ? 'running' : 'pending'}`}>
                              <Icon size={13} />
                            </div>
                            <div className="step-body">
                              <div className="step-name">{step.name}</div>
                              <div className={`step-status ${isDone ? 'done' : isActive ? 'running' : ''}`}>
                                {isDone ? '✓ Complete' : isActive ? '● Running...' : step.desc}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Live Thinking Stream */}
                    {!inspectedEvent && (isInvestigating || thinkingLines.length > 0) && (
                      <div style={{ marginTop: '1rem' }}>
                        <div className="section-header">Live Agent Reasoning</div>
                        <div className="thinking-stream" ref={thinkingRef}>
                          {thinkingLines.length === 0 && (
                            <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem', fontFamily: 'JetBrains Mono' }}>
                              Waiting for agents...
                            </div>
                          )}
                          {thinkingLines.map((line, i) => (
                            <div key={i} className="thinking-line">
                              <span className="agent-tag">{line.agent.split(' ')[0]}</span>
                              <span className="msg-text">{line.message}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* SAR Report & Interactive Action Controls */}
              {displayedEvent.fastPathResult.is_flagged && (
                <div className="card">
                  <div className="card-header">
                    <FileText size={14} className="icon" />
                    Agentic Investigation Report & Actions
                    {!inspectedEvent && isInvestigating && !displayedEvent.report && (
                      <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--accent)', fontSize: '0.72rem', fontWeight: 500 }}>
                        <div className="spinner" style={{ width: 12, height: 12, borderWidth: 2 }} />
                        Agents investigating...
                      </span>
                    )}
                  </div>
                  <div className="card-body">
                    {!displayedEvent.report ? (
                      <div className="loader-wrap">
                        <div className="spinner" />
                        <p>4 specialist agents analyzing in parallel...</p>
                        <p style={{ fontSize: '0.7rem', opacity: 0.6 }}>Transaction · Customer · Graph (Neo4j) · Memory (SQLite)</p>
                      </div>
                    ) : (
                      <div>
                        {/* Decision */}
                        <div className={`status-pill ${actionClass}`} style={{ marginBottom: '1rem' }}>
                          {actionClass === 'block' ? <ShieldX size={14} /> : actionClass === 'approve' ? <ShieldCheck size={14} /> : <AlertTriangle size={14} />}
                          {displayedEvent.report.recommended_action}
                        </div>

                        {/* Confidence */}
                        <div className="confidence-bar">
                          <span className="conf-label">Confidence</span>
                          <div className="conf-track">
                            <div className="conf-fill" style={{ width: `${displayedEvent.report.confidence_score}%` }} />
                          </div>
                          <span className="conf-score">{displayedEvent.report.confidence_score.toFixed(0)}%</span>
                        </div>

                        {/* Self-reflection */}
                        {displayedEvent.report.reflection_triggered && (
                          <div className="reflection-badge">
                            <Brain size={13} /> Self-reflection triggered — agent autonomously re-queried RAG with refined query
                          </div>
                        )}

                        {/* SAR Explanation */}
                        <div style={{ marginTop: '1rem' }}>
                          <div className="section-header">SAR Explanation</div>
                          <p className="sar-explanation">{displayedEvent.report.sar_explanation}</p>
                        </div>

                        {/* Risk Flags */}
                        <div style={{ marginTop: '0.75rem' }}>
                          <div className="section-header">Risk Signals ({displayedEvent.report.reasons.length})</div>
                          <ul className="flag-list">
                            {displayedEvent.report.reasons.map((r, i) => (
                              <li key={i} className={`flag-item ${r.startsWith('CRITICAL') ? 'critical' : ''}`}>{r}</li>
                            ))}
                          </ul>
                        </div>

                        {/* Reasoning Trace */}
                        {displayedEvent.report.reasoning_trace?.length > 0 && (
                          <div className="expander">
                            <button className="expander-trigger" onClick={() => setPolicyExpanded(!policyExpanded)}>
                              <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}><Brain size={12} /> Reasoning Trace + Policy Citations</span>
                              {policyExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                            </button>
                            {policyExpanded && (
                              <div className="expander-content">
                                {displayedEvent.report.reasoning_trace.map((t, i) => (
                                  <div key={i} className="thinking-line" style={{ marginBottom: '0.4rem' }}>
                                    <span className="agent-tag">STEP {i + 1}</span>
                                    <span className="msg-text">{t}</span>
                                  </div>
                                ))}
                                <div style={{ marginTop: '0.75rem', fontFamily: 'JetBrains Mono', fontSize: '0.7rem', color: 'var(--text-muted)', whiteSpace: 'pre-wrap' }}>
                                  {displayedEvent.report.policy_citations}
                                </div>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Action Taken Banner */}
                        {displayedEvent.actionResult && (
                          <div className={`action-banner ${getActionTakenClass(displayedEvent.actionResult.action_taken)}`}>
                            <span className="action-banner-icon">
                              {displayedEvent.actionResult.action_taken.includes('BLOCK') ? <ShieldX size={18} /> :
                               displayedEvent.actionResult.action_taken.includes('HELD') ? <Clock size={18} /> : <CheckCircle2 size={18} />}
                            </span>
                            <div className="action-banner-text">
                              <div className="action-banner-title">
                                Action Agent Executed: {displayedEvent.actionResult.action_taken.replace(/_/g, ' ')}
                              </div>
                              {displayedEvent.actionResult.action_description}
                              {displayedEvent.actionResult.alert_id && (
                                <div style={{ marginTop: '0.3rem', fontSize: '0.68rem', opacity: 0.7, fontFamily: 'JetBrains Mono' }}>
                                  {displayedEvent.actionResult.alert_id}
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        {/* ── Interactive OTP Challenge & HITL Override Box ── */}
                        <div style={{ marginTop: '1.25rem', borderTop: '1px solid var(--border)', paddingTop: '1rem' }}>
                          <div className="section-header" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--warning)' }}>
                            <Key size={13} /> Interactive Agent Actions & Human-In-The-Loop (HITL)
                          </div>

                          {/* OTP Challenge Widget */}
                          {displayedEvent.actionResult?.otp_challenge && (
                            <div style={{ background: 'var(--warning-dim)', border: '1px solid rgba(255,184,48,0.3)', borderRadius: 'var(--radius-md)', padding: '0.85rem', marginBottom: '0.85rem' }}>
                              <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--warning)', display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.3rem' }}>
                                <Key size={14} /> Customer SMS OTP verification
                              </div>
                              <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginBottom: '0.6rem' }}>
                                Agent dispatched 6-digit OTP code <code style={{ color: 'var(--text-primary)', background: 'var(--bg-surface)', padding: '0.1rem 0.4rem', borderRadius: '4px', fontWeight: 700, fontFamily: 'JetBrains Mono' }}>{displayedEvent.actionResult.otp_challenge.code}</code> to customer device.
                              </p>
                              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                                <input
                                  className="form-input"
                                  style={{ width: '130px', padding: '0.35rem 0.6rem', fontSize: '0.8rem', fontFamily: 'JetBrains Mono' }}
                                  placeholder={displayedEvent.actionResult.otp_challenge.code}
                                  value={otpInput}
                                  onChange={e => setOtpInput(e.target.value)}
                                />
                                <button
                                  className="btn-primary"
                                  style={{ padding: '0.35rem 0.8rem', fontSize: '0.75rem' }}
                                  onClick={() => handleVerifyOTP()}
                                >
                                  Verify Customer OTP
                                </button>
                                <button
                                  className="btn-secondary"
                                  style={{ padding: '0.35rem 0.65rem', fontSize: '0.72rem' }}
                                  onClick={() => handleVerifyOTP(displayedEvent.actionResult?.otp_challenge?.code)}
                                >
                                  Simulate Auto-Pass
                                </button>
                              </div>
                              {otpStatusMsg && (
                                <div style={{ fontSize: '0.72rem', marginTop: '0.45rem', fontWeight: 600 }}>{otpStatusMsg}</div>
                              )}
                            </div>
                          )}

                          {/* HITL Override Controls */}
                          <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', padding: '0.85rem' }}>
                            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                              <UserCheck size={14} style={{ color: 'var(--accent)' }} /> Compliance Officer Manual Override
                            </div>
                            <input
                              className="form-input"
                              style={{ padding: '0.35rem 0.6rem', fontSize: '0.75rem', marginBottom: '0.6rem' }}
                              placeholder="Reason for manual override (optional notes)..."
                              value={hitlNotes}
                              onChange={e => setHitlNotes(e.target.value)}
                            />
                            <div style={{ display: 'flex', gap: '0.5rem' }}>
                              <button
                                className="btn-secondary"
                                disabled={hitlSubmitting}
                                style={{ padding: '0.4rem 0.8rem', fontSize: '0.75rem', color: 'var(--success)', borderColor: 'rgba(0,229,160,0.3)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}
                                onClick={() => handleHITLOverride('Approve')}
                              >
                                <UserCheck size={13} /> {hitlSubmitting ? 'Applying…' : 'Approve Override'}
                              </button>
                              <button
                                className="btn-secondary"
                                disabled={hitlSubmitting}
                                style={{ padding: '0.4rem 0.8rem', fontSize: '0.75rem', color: 'var(--danger)', borderColor: 'rgba(255,77,106,0.3)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}
                                onClick={() => handleHITLOverride('Block')}
                              >
                                <UserX size={13} /> {hitlSubmitting ? 'Applying…' : 'Force Block Account'}
                              </button>
                            </div>
                            {hitlStatusMsg && (
                              <div style={{ fontSize: '0.72rem', marginTop: '0.45rem', fontWeight: 600 }}>{hitlStatusMsg}</div>
                            )}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {!displayedEvent.fastPathResult.is_flagged && (
                <div className="card">
                  <div className="card-header"><Bot size={14} className="icon" />Agentic Slow Path</div>
                  <div className="card-body">
                    <div className="status-pill approve"><CheckCircle2 size={14} /> Transaction cleared by Fast Path — Slow Path skipped</div>
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.75rem', lineHeight: 1.6 }}>
                      Risk score {displayedEvent.fastPathResult.risk_score.toFixed(1)}% is below the 70% flagging threshold.
                      The transaction was instantly approved to minimize latency and LLM cost.
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* ── Col 3: Blocked Accounts Sidebar ──────────────────── */}
            {metricsExpanded && <div className="operations-sidebar">

              {/* Blocked Accounts */}
              <div className="card">
                <div className="card-header">
                  <Lock size={14} className="icon" />
                  Blocked Accounts
                  <span style={{ marginLeft: 'auto', background: 'var(--danger-dim)', color: 'var(--danger)', fontSize: '0.68rem', fontWeight: 700, padding: '0.1rem 0.45rem', borderRadius: '99px', border: '1px solid rgba(255,77,106,0.3)' }}>
                    {blockedAccounts.length}
                  </span>
                </div>
                <div className="card-body">
                  {blockedAccounts.length === 0 ? (
                    <div className="empty-state">No accounts blocked yet</div>
                  ) : (
                    <div className="blocked-list">
                      {blockedAccounts.slice(0, 10).map((acc, i) => (
                        <div key={i} className="blocked-item">
                          <div className="blocked-id">{acc.customer_id}</div>
                          <div className="blocked-meta">
                            {acc.blocked_at ? fmtDate(acc.blocked_at) : ''} · Risk: {acc.risk_score?.toFixed(1)}%
                          </div>
                          {acc.primary_reason && (
                            <div className="blocked-reason">{acc.primary_reason}</div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Alert Feed */}
              <div className="card">
                <div className="card-header">
                  <AlertTriangle size={14} className="icon" />
                  Live Alert Feed
                  <span style={{ marginLeft: 'auto', background: 'var(--warning-dim)', color: 'var(--warning)', fontSize: '0.68rem', fontWeight: 700, padding: '0.1rem 0.45rem', borderRadius: '99px', border: '1px solid rgba(255,184,48,0.3)' }}>
                    {alertItems.length}
                  </span>
                </div>
                <div className="card-body">
                  {alertItems.length === 0 ? (
                    <div className="empty-state">No alerts yet</div>
                  ) : (
                    <div className="alerts-list">
                      {alertItems.slice(0, 8).map((a, i) => (
                        <div key={i} className={`alert-item ${a.severity}`}>
                          <div className="alert-id">{a.id} · {a.severity}</div>
                          <div className="alert-msg">{a.message}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Active Agent Tools & Telemetry */}
              <div className="card">
                <div className="card-header"><Cpu size={14} className="icon" />Active Agent Tools & Environment</div>
                <div className="card-body" style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.3rem' }}>
                    <span className="tool-name"><Wrench size={14} /> <strong>Neo4j Cypher Tool</strong></span>
                    <span style={{ color: 'var(--success)', fontWeight: 600 }}>Active (Graph Agent)</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.3rem' }}>
                    <span className="tool-name"><HardDrive size={14} /> <strong>SQLite Memory Store</strong></span>
                    <span style={{ color: 'var(--success)', fontWeight: 600 }}>Active (Memory Agent)</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.3rem' }}>
                    <span className="tool-name"><BookOpen size={14} /> <strong>Chroma Vector RAG</strong></span>
                    <span style={{ color: 'var(--info)', fontWeight: 600 }}>Top-6 Diverse Chunks</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.4rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.3rem' }}>
                    <span className="tool-name"><Key size={14} /> <strong>SMS OTP Challenge</strong></span>
                    <span style={{ color: 'var(--warning)', fontWeight: 600 }}>Active (HITL / 2FA)</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span className="tool-name"><Siren size={14} /> <strong>Action Registry Executor</strong></span>
                    <span style={{ color: 'var(--accent)', fontWeight: 600 }}>Auto Block / Hold / Approve</span>
                  </div>
                </div>
              </div>
            </div>}

          </div>
        )}
      </main>

      {/* ── Manual Trigger Modal ─────────────────────────────────────── */}
      {showModal && (
        <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && setShowModal(false)}>
          <div className="modal-box">
            <div className="modal-title">
              <Play size={16} style={{ color: 'var(--accent)' }} />
              Manual Investigation Trigger
              <button onClick={() => setShowModal(false)} style={{ marginLeft: 'auto', background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
                <X size={16} />
              </button>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '1.25rem', lineHeight: 1.6 }}>
              Manually trigger the full 4-agent investigation workflow on any transaction. Pre-filled with a highly suspicious demo case.
            </p>

            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Customer ID</label>
                <input className="form-input" value={form.customer_id} onChange={e => setForm(f => ({...f, customer_id: e.target.value}))} />
              </div>
              <div className="form-group">
                <label className="form-label">Transaction Type</label>
                <input className="form-input" value={form.type} onChange={e => setForm(f => ({...f, type: e.target.value}))} />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Amount ($)</label>
                <input className="form-input" type="number" value={form.amount} onChange={e => setForm(f => ({...f, amount: e.target.value}))} />
              </div>
              <div className="form-group">
                <label className="form-label">Origin Balance</label>
                <input className="form-input" type="number" value={form.oldbalanceOrg} onChange={e => setForm(f => ({...f, oldbalanceOrg: e.target.value}))} />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label className="form-label">KYC Status</label>
                <input className="form-input" value={form.kyc_status} onChange={e => setForm(f => ({...f, kyc_status: e.target.value}))} />
              </div>
              <div className="form-group">
                <label className="form-label">Risk Tier</label>
                <input className="form-input" value={form.risk_tier} onChange={e => setForm(f => ({...f, risk_tier: e.target.value}))} />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Account Age (days)</label>
                <input className="form-input" type="number" value={form.account_age_days} onChange={e => setForm(f => ({...f, account_age_days: e.target.value}))} />
              </div>
              <div className="form-group">
                <label className="form-label">Shared IP Count</label>
                <input className="form-input" type="number" value={form.shared_ip_count} onChange={e => setForm(f => ({...f, shared_ip_count: e.target.value}))} />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Failed Logins</label>
                <input className="form-input" type="number" value={form.recent_failed_logins} onChange={e => setForm(f => ({...f, recent_failed_logins: e.target.value}))} />
              </div>
              <div className="form-group">
                <label className="form-label">Session Velocity (s)</label>
                <input className="form-input" type="number" value={form.session_velocity_seconds} onChange={e => setForm(f => ({...f, session_velocity_seconds: e.target.value}))} />
              </div>
            </div>

            <div className="modal-actions">
              <button className="btn-secondary" onClick={() => setShowModal(false)}>Cancel</button>
              <button id="btn-run-agents" className="btn-primary" onClick={handleTrigger} disabled={submitting}>
                {submitting ? 'Running...' : '🚀 Run Agents'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
