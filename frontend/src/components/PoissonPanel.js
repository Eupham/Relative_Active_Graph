import React from 'react';
import { Gauge } from 'lucide-react';

export default function PoissonPanel({ state }) {
  const poisson = state?.poisson || {};
  const history = state?.quality_history || [];
  const last20 = history.slice(-20);

  return (
    <div className="bg-panel border border-white/5 p-4 space-y-4" data-testid="poisson-panel">
      <h2 className="font-heading text-lg font-semibold tracking-wide uppercase text-neon-amber flex items-center gap-2">
        <Gauge className="w-4 h-4" />
        Adaptive Noise Controller
      </h2>

      <div className="grid grid-cols-3 gap-2">
        <StatBox label="Current Lambda" value={(poisson.lambda || 0).toFixed(4)} color="text-neon-amber" />
        <StatBox label="EMA Success" value={(poisson.ema_success || 0).toFixed(4)} color="text-neon-green" />
        <StatBox label="History Size" value={poisson.history_len || 0} />
      </div>

      <div className="space-y-1">
        <div className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500">
          Lambda Gauge [{poisson.min_lam || 0.05} - {poisson.max_lam || 5.0}]
        </div>
        <div className="h-3 bg-black border border-white/5 relative overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-neon-green via-neon-amber to-neon-red transition-all duration-500"
            style={{ width: `${Math.min(100, ((poisson.lambda || 0) / (poisson.max_lam || 5)) * 100)}%` }}
          />
          <div className="absolute inset-0 flex items-center justify-center text-[9px] font-mono text-white/50">
            {(poisson.lambda || 0).toFixed(3)}
          </div>
        </div>
      </div>

      <div className="text-[10px] font-mono text-slate-500 space-y-0.5">
        <div>Strategy: EMA-guided proportional control</div>
        <div>
          {(poisson.ema_success || 0) > 0.75 ? (
            <span className="text-neon-amber">INCREASING noise — system too comfortable</span>
          ) : (poisson.ema_success || 0) < 0.35 ? (
            <span className="text-neon-green">DECREASING noise — easing pressure</span>
          ) : (
            <span className="text-neon-cyan">GOLDILOCKS zone — fine-tuning</span>
          )}
        </div>
      </div>

      {/* Spark line of last 20 quality samples */}
      {last20.length > 0 && (
        <div className="bg-black/40 border border-white/5 p-2">
          <div className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500 mb-1">Last 20 passages</div>
          <div className="flex items-end gap-0.5 h-12">
            {last20.map((d, i) => (
              <div
                key={i}
                className="flex-1 rounded-t-sm transition-all"
                style={{
                  height: `${Math.max(4, d.quality * 100)}%`,
                  backgroundColor: d.quality > 0.5 ? '#39FF14' : d.quality > 0.3 ? '#FFB800' : '#FF2A6D',
                  opacity: 0.6 + (i / last20.length) * 0.4,
                }}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function StatBox({ label, value, color = 'text-slate-200' }) {
  return (
    <div className="bg-black/40 border border-white/5 p-2">
      <div className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-600">{label}</div>
      <div className={`font-mono text-sm ${color}`}>{value}</div>
    </div>
  );
}
