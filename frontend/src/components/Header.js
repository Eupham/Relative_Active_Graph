import React from 'react';
import { Activity, Zap, Eye, EyeOff } from 'lucide-react';

export default function Header({ state, tab, setTab, advancedView, setAdvancedView }) {
  const running = state?.running || false;
  return (
    <header className="border-b border-white/5 bg-panel/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-[1800px] mx-auto px-4 h-14 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-neon-cyan" />
            <span className="font-heading text-xl font-bold tracking-wide uppercase text-neon-cyan">
              CSRRE
            </span>
            <span className="font-heading text-xl font-light tracking-wide uppercase text-slate-500">
              Mission Control
            </span>
          </div>
          <div className={`w-2 h-2 rounded-full ml-2 ${running ? 'bg-neon-green animate-pulse-glow' : 'bg-slate-600'}`} />
          <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">
            {running ? 'TRAINING ACTIVE' : 'STANDBY'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <nav className="flex gap-1 mr-4" data-testid="nav-tabs">
            {['training', 'inference'].map(t => (
              <button
                key={t}
                data-testid={`tab-${t}`}
                onClick={() => setTab(t)}
                className={`px-4 py-1.5 text-[10px] font-bold uppercase tracking-[0.2em] transition-all
                  ${tab === t
                    ? 'bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/30'
                    : 'text-slate-500 hover:text-slate-300 border border-transparent'
                  }`}
              >
                {t}
              </button>
            ))}
          </nav>
          {tab === 'training' && (
            <button
              data-testid="toggle-advanced"
              onClick={() => setAdvancedView(!advancedView)}
              className="flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-bold uppercase tracking-[0.15em] border transition-all
                bg-white/5 text-slate-400 border-white/10 hover:bg-white/10"
            >
              {advancedView ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
              {advancedView ? 'Simple' : 'Advanced'}
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
