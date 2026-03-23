import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { TrendingUp } from 'lucide-react';

export default function MetricsPanel({ state, advancedView }) {
  const history = state?.quality_history || [];

  return (
    <div className="bg-panel border border-white/5 p-4 space-y-4" data-testid="metrics-panel">
      <div className="flex items-center justify-between">
        <h2 className="font-heading text-lg font-semibold tracking-wide uppercase text-neon-cyan flex items-center gap-2">
          <TrendingUp className="w-4 h-4" />
          Quality Metrics
        </h2>
        <span className="text-[10px] font-mono text-slate-600">
          {history.length} data points
        </span>
      </div>

      {/* Quality over time chart */}
      <div className="h-64 bg-black/40 border border-white/5 p-2" data-testid="quality-chart">
        {history.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={history}>
              <defs>
                <linearGradient id="qualityGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00F0FF" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#00F0FF" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#1F1F1F" strokeDasharray="3 3" />
              <XAxis dataKey="passage" stroke="#475569" tick={{ fontSize: 10, fontFamily: 'JetBrains Mono' }} />
              <YAxis stroke="#475569" tick={{ fontSize: 10, fontFamily: 'JetBrains Mono' }} domain={[0, 1]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0A0A0A', borderColor: '#333', color: '#fff', fontFamily: 'JetBrains Mono', fontSize: 11 }}
                itemStyle={{ color: '#ccc' }}
              />
              <Area type="monotone" dataKey="quality" stroke="#00F0FF" fill="url(#qualityGrad)" strokeWidth={2} dot={false} name="Quality" />
              {advancedView && (
                <Line type="monotone" dataKey="ema_success" stroke="#39FF14" strokeWidth={1.5} dot={false} strokeDasharray="4 2" name="EMA Success" />
              )}
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center text-slate-600 font-mono text-sm">
            Awaiting training data...
          </div>
        )}
      </div>

      {advancedView && history.length > 0 && (
        <div className="h-48 bg-black/40 border border-white/5 p-2" data-testid="lambda-chart">
          <div className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500 mb-1 px-2">
            Adaptive Poisson Lambda
          </div>
          <ResponsiveContainer width="100%" height="85%">
            <LineChart data={history}>
              <CartesianGrid stroke="#1F1F1F" strokeDasharray="3 3" />
              <XAxis dataKey="passage" stroke="#475569" tick={{ fontSize: 10, fontFamily: 'JetBrains Mono' }} />
              <YAxis stroke="#475569" tick={{ fontSize: 10, fontFamily: 'JetBrains Mono' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0A0A0A', borderColor: '#333', color: '#fff', fontFamily: 'JetBrains Mono', fontSize: 11 }}
              />
              <Line type="monotone" dataKey="lambda" stroke="#FFB800" strokeWidth={2} dot={false} name="Lambda" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
