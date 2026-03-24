import React, { useState } from 'react';
import { X, Play, Square, Activity } from 'lucide-react';
import { QualityChart, GraphGrowthChart } from './TrainingCharts';

const LANGUAGES = ['en','de','fr','es','zh','ar','ja','pt','ru','it'];

function formatDuration(secs) {
  if (!secs) return '0s';
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function formatNum(n) {
  if (n === undefined || n === null) return '—';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
  return String(n);
}

export default function TrainingModal({ apiUrl, trainingState: ts, onClose }) {
  const [config, setConfig] = useState({
    language: 'en',
    epochs: 1,
    max_sentences: 1000,
    passage_chars: 2000,
  });
  const [busy, setBusy] = useState(false);

  const isRunning = ts?.running === true;

  // Safely access history arrays (ts may be null on first load)
  const qualityHistory = (ts?.quality_history) ?? [];

  // Approximate graph growth over passages from cumulative totals
  const totalNodes = ts?.global_nodes ?? 0;
  const totalEdges = ts?.global_edges ?? 0;
  const graphHistory = qualityHistory.map((pt, i) => ({
    passage: pt.passage,
    nodes: Math.round(((i + 1) / Math.max(qualityHistory.length, 1)) * totalNodes),
    edges: Math.round(((i + 1) / Math.max(qualityHistory.length, 1)) * totalEdges),
  }));

  const handleStart = async () => {
    setBusy(true);
    try {
      await fetch(`${apiUrl}/api/training/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          language: config.language,
          epochs: Number(config.epochs),
          max_sentences: Number(config.max_sentences),
          passage_chars: Number(config.passage_chars),
        }),
      });
    } catch (e) {
      console.error('Start error', e);
    } finally {
      setBusy(false);
    }
  };

  const handleStop = async () => {
    setBusy(true);
    try {
      await fetch(`${apiUrl}/api/training/stop`, { method: 'POST' });
    } catch (e) {
      console.error('Stop error', e);
    } finally {
      setBusy(false);
    }
  };

  const events = (ts?.events || []).slice().reverse().slice(0, 50);

  return (
    <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="training-modal">
        {/* Header */}
        <div className="modal-header">
          <div className="modal-title">
            <Activity size={15} />
            Training
            {isRunning && (
              <span className="training-active-badge">
                <span className="status-dot active" style={{width:5,height:5}} />
                Running
              </span>
            )}
          </div>
          <button className="btn-close" onClick={onClose}><X size={14} /></button>
        </div>

        <div className="modal-body">
          {/* Config */}
          <div>
            <div className="config-grid">
              <div className="form-field">
                <label className="form-label">Language</label>
                <select
                  className="form-input"
                  value={config.language}
                  onChange={e => setConfig(c => ({...c, language: e.target.value}))}
                  disabled={isRunning}
                >
                  {LANGUAGES.map(l => <option key={l} value={l}>{l.toUpperCase()}</option>)}
                </select>
              </div>
              <div className="form-field">
                <label className="form-label">Epochs</label>
                <input
                  className="form-input"
                  type="number" min={1} max={100}
                  value={config.epochs}
                  onChange={e => setConfig(c => ({...c, epochs: e.target.value}))}
                  disabled={isRunning}
                />
              </div>
              <div className="form-field">
                <label className="form-label">Max Sentences</label>
                <input
                  className="form-input"
                  type="number" min={10} step={100}
                  value={config.max_sentences}
                  onChange={e => setConfig(c => ({...c, max_sentences: e.target.value}))}
                  disabled={isRunning}
                />
              </div>
              <div className="form-field">
                <label className="form-label">Passage Chars</label>
                <input
                  className="form-input"
                  type="number" min={500} step={500}
                  value={config.passage_chars}
                  onChange={e => setConfig(c => ({...c, passage_chars: e.target.value}))}
                  disabled={isRunning}
                />
              </div>
            </div>

            <div className="config-actions" style={{marginTop: 12}}>
              {!isRunning ? (
                <button className="btn-primary" onClick={handleStart} disabled={busy}>
                  <Play size={13} />
                  Start Training
                </button>
              ) : (
                <button className="btn-danger" onClick={handleStop} disabled={busy}>
                  <Square size={13} />
                  Stop
                </button>
              )}
              {ts?.error && (
                <span style={{fontSize:12, color:'var(--danger)', marginLeft:8}}>
                  ✗ {ts.error}
                </span>
              )}
            </div>
          </div>

          {/* Live Stats */}
          <div className="stats-bar">
            <div className="stat-card">
              <div className="stat-label">Passages</div>
              <div className="stat-value">{formatNum(ts?.passages)}</div>
              <div className="stat-sub">of {formatNum(ts?.max_sentences)} sentences</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Mean Quality</div>
              <div className="stat-value" style={{color:'var(--accent)'}}>
                {ts?.mean_quality !== undefined ? (ts.mean_quality * 100).toFixed(1) + '%' : '—'}
              </div>
              <div className="stat-sub">semantic signal</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">λ Noise</div>
              <div className="stat-value" style={{color:'var(--warning)'}}>
                {ts?.poisson?.lambda !== undefined ? ts.poisson.lambda.toFixed(3) : '—'}
              </div>
              <div className="stat-sub">EMA {ts?.poisson?.ema_success !== undefined ? (ts.poisson.ema_success * 100).toFixed(0) + '%' : '—'}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Graph Nodes</div>
              <div className="stat-value" style={{color:'#8b5cf6'}}>{formatNum(ts?.global_nodes)}</div>
              <div className="stat-sub">{formatNum(ts?.global_edges)} edges</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Rules</div>
              <div className="stat-value">{formatNum(ts?.rules_induced)}</div>
              <div className="stat-sub">induced</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Elapsed</div>
              <div className="stat-value">{formatDuration(ts?.elapsed_sec)}</div>
              <div className="stat-sub">epoch {ts?.epoch || 0}/{ts?.total_epochs || 1}</div>
            </div>
          </div>

          {/* Charts */}
          <div className="charts-grid">
            <QualityChart data={qualityHistory} />
            <GraphGrowthChart history={graphHistory} />
          </div>

          {/* Event Log */}
          {events.length > 0 && (
            <div>
              <div className="form-label" style={{marginBottom:6}}>Engine Log</div>
              <div className="event-log">
                {events.map((ev, i) => (
                  <div key={i} className="event-line">
                    <span className="event-time">
                      {new Date(ev.ts * 1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'})}
                    </span>
                    <span className={`event-level-${ev.level}`}>[{ev.level}]</span>
                    <span className="event-msg">{ev.message}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
