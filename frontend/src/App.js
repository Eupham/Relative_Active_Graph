import React, { useState, useEffect, useCallback, useRef } from 'react';
import Sidebar from './components/Sidebar';
import ChatPane from './components/ChatPane';
import TrainingModal from './components/TrainingModal';
import { Activity, ChevronDown, ChevronUp } from 'lucide-react';

// When React is built and served by FastAPI, both live on the same origin.
// Relative paths work automatically.
const API = '';

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

let sessionCounter = 1;
function newSession() {
  return {
    id: Date.now(),
    title: `Session ${sessionCounter++}`,
    createdAt: new Date(),
    messages: [],
  };
}

export default function App() {
  const [sessions, setSessions] = useState([newSession()]);
  const [activeId, setActiveId] = useState(sessions[0].id);
  const [trainingOpen, setTrainingOpen] = useState(false);
  const [trainingState, setTrainingState] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnRef = useRef(null);

  // ── WebSocket ──────────────────────────────────────────────────
  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${window.location.host}/api/ws`);
    ws.onopen  = () => { setWsConnected(true); };
    ws.onclose = () => {
      setWsConnected(false);
      reconnRef.current = setTimeout(connectWs, 3000);
    };
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'state') setTrainingState(msg.data);
      } catch {}
    };
    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connectWs();
    // Initial fetch
    fetch(`${API}/api/training/status`)
      .then(r => r.json())
      .then(setTrainingState)
      .catch(() => {});
    return () => {
      wsRef.current?.close();
      clearTimeout(reconnRef.current);
    };
  }, [connectWs]);

  // ── Session helpers ────────────────────────────────────────────
  const handleNewChat = () => {
    const s = newSession();
    setSessions(prev => [s, ...prev]);
    setActiveId(s.id);
  };

  const activeSession = sessions.find(s => s.id === activeId);

  const appendMessage = (msg) => {
    setSessions(prev => prev.map(s =>
      s.id === activeId
        ? { ...s, messages: [...s.messages, msg], title: s.messages.length === 0 ? (msg.text?.slice(0, 32) || s.title) : s.title }
        : s
    ));
  };

  const isTrainingRunning = trainingState?.running === true;

  return (
    <div className="app-shell">
      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-logo">
          <div className="header-logo-icon">⚛</div>
          RAG Engine
        </div>
        <div className="header-spacer" />
        <div className="header-status">
          <div className={`status-dot ${wsConnected ? 'active' : ''}`} />
          {wsConnected ? 'Live' : 'Offline'}
        </div>
        <button
          className={`btn-training ${trainingOpen ? 'open' : ''}`}
          onClick={() => setTrainingOpen(v => !v)}
        >
          <Activity size={14} />
          Training
          {isTrainingRunning && (
            <span className="training-active-badge">
              <span className="status-dot active" style={{width:5,height:5}} />
              Running
            </span>
          )}
          {trainingOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>
      </header>

      {/* ── Training Modal (drops from header) ─────────────────── */}
      {trainingOpen && (
        <TrainingModal
          apiUrl={API}
          trainingState={trainingState}
          onClose={() => setTrainingOpen(false)}
        />
      )}

      <div className="app-body">
        {/* ── Left Sidebar ─────────────────────────────────────── */}
        <Sidebar
          sessions={sessions}
          activeId={activeId}
          onSelect={setActiveId}
          onNewChat={handleNewChat}
        />

        {/* ── Main Chat ────────────────────────────────────────── */}
        <main className="main-content">
          <ChatPane
            session={activeSession}
            apiUrl={API}
            onMessage={appendMessage}
          />
        </main>
      </div>
    </div>
  );
}
