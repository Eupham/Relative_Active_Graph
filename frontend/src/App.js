import React, { useState, useEffect, useCallback, useRef } from 'react';
import Header from './components/Header';
import TrainingPanel from './components/TrainingPanel';
import MetricsPanel from './components/MetricsPanel';
import EventLog from './components/EventLog';
import InferencePanel from './components/InferencePanel';
import PoissonPanel from './components/PoissonPanel';
import EngineStats from './components/EngineStats';

const API = process.env.REACT_APP_BACKEND_URL || '';
const WS_URL = API.replace(/^http/, 'ws') + '/api/ws';

export default function App() {
  const [state, setState] = useState(null);
  const [tab, setTab] = useState('training');
  const [advancedView, setAdvancedView] = useState(false);
  const [events, setEvents] = useState([]);
  const wsRef = useRef(null);
  const reconnectRef = useRef(null);

  const connectWs = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;
    try {
      const ws = new WebSocket(WS_URL);
      ws.onopen = () => { console.log('WS connected'); };
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'state') setState(msg.data);
          if (msg.type === 'event') setEvents(prev => [...prev.slice(-299), msg.data]);
        } catch {}
      };
      ws.onclose = () => {
        reconnectRef.current = setTimeout(connectWs, 2000);
      };
      wsRef.current = ws;
    } catch {}
  }, []);

  useEffect(() => {
    connectWs();
    // Initial fetch
    fetch(`${API}/api/training/status`).then(r => r.json()).then(setState).catch(() => {});
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
    };
  }, [connectWs]);

  return (
    <div className="min-h-screen bg-void">
      <Header state={state} tab={tab} setTab={setTab} advancedView={advancedView} setAdvancedView={setAdvancedView} />
      <main className="max-w-[1800px] mx-auto px-4 pb-8 pt-4">
        {tab === 'training' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            <div className="lg:col-span-4">
              <TrainingPanel state={state} apiUrl={API} />
            </div>
            <div className="lg:col-span-8">
              <MetricsPanel state={state} advancedView={advancedView} />
            </div>
            {advancedView && (
              <>
                <div className="lg:col-span-6">
                  <PoissonPanel state={state} />
                </div>
                <div className="lg:col-span-6">
                  <EngineStats state={state} />
                </div>
              </>
            )}
            <div className="lg:col-span-12">
              <EventLog events={state?.events || events} advancedView={advancedView} />
            </div>
          </div>
        )}
        {tab === 'inference' && (
          <InferencePanel apiUrl={API} />
        )}
      </main>
    </div>
  );
}
