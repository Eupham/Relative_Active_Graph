import React, { useRef, useEffect } from 'react';
import { Terminal } from 'lucide-react';

export default function EventLog({ events, advancedView }) {
  const endRef = useRef(null);
  const displayed = advancedView ? events : events.filter(e => e.level !== 'metric');

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [displayed.length]);

  const levelColor = {
    info: 'text-neon-cyan',
    metric: 'text-neon-green',
    error: 'text-neon-red',
    warning: 'text-neon-amber',
  };

  return (
    <div className="bg-panel border border-white/5 p-4" data-testid="event-log">
      <div className="flex items-center justify-between mb-2">
        <h2 className="font-heading text-lg font-semibold tracking-wide uppercase text-neon-cyan flex items-center gap-2">
          <Terminal className="w-4 h-4" />
          System Log
        </h2>
        <span className="text-[10px] font-mono text-slate-600">
          {displayed.length} entries
        </span>
      </div>
      <div className="font-mono text-xs bg-black p-3 border border-white/5 h-48 overflow-y-auto space-y-0.5">
        {displayed.length === 0 && (
          <div className="text-slate-600">Waiting for events...</div>
        )}
        {displayed.map((evt, i) => (
          <div key={i} className="flex gap-2">
            <span className="text-slate-600 shrink-0">
              {evt.ts ? new Date(evt.ts * 1000).toLocaleTimeString() : '--:--:--'}
            </span>
            <span className={`shrink-0 uppercase w-12 ${levelColor[evt.level] || 'text-slate-400'}`}>
              [{evt.level}]
            </span>
            <span className="text-green-400/80">{evt.message}</span>
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}
