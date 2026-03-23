import React, { useState } from 'react';
import { Play, Square, Settings } from 'lucide-react';

export default function TrainingPanel({ state, apiUrl }) {
  const [language, setLanguage] = useState('en');
  const [epochs, setEpochs] = useState(1);
  const [maxSentences, setMaxSentences] = useState(1000);
  const [passageChars, setPassageChars] = useState(2000);
  const running = state?.running || false;

  const startTraining = async () => {
    await fetch(`${apiUrl}/api/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ language, epochs, max_sentences: maxSentences, passage_chars: passageChars }),
    });
  };

  const stopTraining = async () => {
    await fetch(`${apiUrl}/api/training/stop`, { method: 'POST' });
  };

  const progress = state?.max_sentences > 0
    ? Math.min(100, ((state?.sentences || 0) / state.max_sentences) * 100)
    : 0;

  return (
    <div className="bg-panel border border-white/5 p-4 space-y-4" data-testid="training-panel">
      <div className="flex items-center justify-between">
        <h2 className="font-heading text-lg font-semibold tracking-wide uppercase text-neon-cyan flex items-center gap-2">
          <Settings className="w-4 h-4" />
          Training Config
        </h2>
      </div>

      <div className="space-y-3">
        <div>
          <label className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500 block mb-1">Language</label>
          <select
            data-testid="config-language"
            value={language}
            onChange={e => setLanguage(e.target.value)}
            disabled={running}
            className="w-full bg-black border border-white/10 text-white font-mono text-sm py-2 px-3 focus:border-neon-cyan focus:ring-1 focus:ring-neon-cyan/50 transition-all disabled:opacity-40"
          >
            {['en','de','fr','es','it','nl','pt','ru','zh','ja','ko','ar','hi','tr','pl','sv','fi'].map(l => (
              <option key={l} value={l}>{l.toUpperCase()}</option>
            ))}
          </select>
        </div>
        <div className="grid grid-cols-3 gap-2">
          <div>
            <label className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500 block mb-1">Epochs</label>
            <input
              data-testid="config-epochs"
              type="number" min={1} max={100} value={epochs}
              onChange={e => setEpochs(Number(e.target.value))}
              disabled={running}
              className="w-full bg-black border border-white/10 text-white font-mono text-sm py-2 px-3 text-right focus:border-neon-cyan disabled:opacity-40"
            />
          </div>
          <div>
            <label className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500 block mb-1">Max Sent</label>
            <input
              data-testid="config-max-sentences"
              type="number" min={100} step={100} value={maxSentences}
              onChange={e => setMaxSentences(Number(e.target.value))}
              disabled={running}
              className="w-full bg-black border border-white/10 text-white font-mono text-sm py-2 px-3 text-right focus:border-neon-cyan disabled:opacity-40"
            />
          </div>
          <div>
            <label className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500 block mb-1">Passage</label>
            <input
              data-testid="config-passage-chars"
              type="number" min={500} step={500} value={passageChars}
              onChange={e => setPassageChars(Number(e.target.value))}
              disabled={running}
              className="w-full bg-black border border-white/10 text-white font-mono text-sm py-2 px-3 text-right focus:border-neon-cyan disabled:opacity-40"
            />
          </div>
        </div>
      </div>

      <div className="flex gap-2">
        {!running ? (
          <button
            data-testid="start-training-btn"
            onClick={startTraining}
            className="flex-1 flex items-center justify-center gap-2 bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/50
              hover:bg-neon-cyan/20 hover:shadow-[0_0_15px_rgba(0,240,255,0.3)] transition-all
              uppercase tracking-widest text-xs font-bold py-2.5 px-4"
          >
            <Play className="w-4 h-4" /> Start Training
          </button>
        ) : (
          <button
            data-testid="stop-training-btn"
            onClick={stopTraining}
            className="flex-1 flex items-center justify-center gap-2 bg-neon-red/10 text-neon-red border border-neon-red/50
              hover:bg-neon-red/20 transition-all uppercase tracking-widest text-xs font-bold py-2.5 px-4"
          >
            <Square className="w-4 h-4" /> Stop
          </button>
        )}
      </div>

      {/* Progress bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-[10px] font-mono text-slate-500">
          <span>PROGRESS</span>
          <span>{progress.toFixed(1)}%</span>
        </div>
        <div className="h-1.5 bg-black border border-white/5 overflow-hidden">
          <div
            className="h-full bg-neon-cyan transition-all duration-500 ease-out"
            style={{ width: `${progress}%`, boxShadow: '0 0 8px rgba(0,240,255,0.5)' }}
          />
        </div>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-2 gap-2">
        {[
          { label: 'Epoch', value: `${state?.epoch || 0}/${state?.total_epochs || 0}` },
          { label: 'Passages', value: state?.passages || 0 },
          { label: 'Sentences', value: state?.sentences || 0 },
          { label: 'Steps', value: state?.steps || 0 },
          { label: 'Mean Quality', value: (state?.mean_quality || 0).toFixed(4), color: 'text-neon-green' },
          { label: 'Elapsed', value: formatTime(state?.elapsed_sec || 0) },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-black/40 border border-white/5 p-2">
            <div className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-600">{label}</div>
            <div className={`font-mono text-sm ${color || 'text-slate-200'}`} data-testid={`stat-${label.toLowerCase().replace(/\s/g, '-')}`}>
              {value}
            </div>
          </div>
        ))}
      </div>

      {state?.error && (
        <div className="bg-neon-red/10 border border-neon-red/30 p-3 text-neon-red text-xs font-mono" data-testid="training-error">
          {state.error}
        </div>
      )}
    </div>
  );
}

function formatTime(sec) {
  if (!sec) return '0s';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}
