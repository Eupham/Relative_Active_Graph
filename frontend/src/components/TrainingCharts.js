import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

const QUALITY_COLOR  = '#10a37f';
const NOISE_COLOR    = '#f59e0b';
const SUCCESS_COLOR  = '#3b82f6';
const NODES_COLOR    = '#8b5cf6';
const EDGES_COLOR    = '#ec4899';

function ChartTooltipStyle() {
  return {
    contentStyle: {
      background: '#1a1a1a',
      border: '1px solid #2a2a2a',
      borderRadius: 8,
      fontSize: 11,
      color: '#ececec',
    },
    labelStyle: { color: '#8e8ea0' },
  };
}

// ── Quality & Noise Chart ─────────────────────────────────────────
export function QualityChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div className="chart-card">
        <div className="chart-title">Semantic Quality & Noise Pressure</div>
        <div className="chart-legend">
          <div className="legend-item"><div className="legend-dot" style={{background:QUALITY_COLOR}} />Quality</div>
          <div className="legend-item"><div className="legend-dot" style={{background:NOISE_COLOR}} />λ Noise</div>
          <div className="legend-item"><div className="legend-dot" style={{background:SUCCESS_COLOR}} />EMA Success</div>
        </div>
        <div className="chart-empty">Waiting for training data…</div>
      </div>
    );
  }

  const { contentStyle, labelStyle } = ChartTooltipStyle();

  return (
    <div className="chart-card">
      <div className="chart-title">Semantic Quality & Noise Pressure</div>
      <div className="chart-legend">
        <div className="legend-item"><div className="legend-dot" style={{background:QUALITY_COLOR}} />Quality</div>
        <div className="legend-item"><div className="legend-dot" style={{background:NOISE_COLOR}} />λ Noise</div>
        <div className="legend-item"><div className="legend-dot" style={{background:SUCCESS_COLOR}} />EMA Success</div>
      </div>
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
          <XAxis
            dataKey="passage"
            tick={{ fill: '#565869', fontSize: 10 }}
            tickLine={false}
            label={{ value: 'Passage', position: 'insideBottomRight', offset: -4, fill: '#565869', fontSize: 10 }}
          />
          <YAxis tick={{ fill: '#565869', fontSize: 10 }} tickLine={false} domain={[0, 'auto']} />
          <Tooltip contentStyle={contentStyle} labelStyle={labelStyle} formatter={(v) => v.toFixed(4)} />
          <Line
            type="monotone"
            dataKey="quality"
            stroke={QUALITY_COLOR}
            dot={false}
            strokeWidth={2}
            name="Quality"
          />
          <Line
            type="monotone"
            dataKey="lambda"
            stroke={NOISE_COLOR}
            dot={false}
            strokeWidth={1.5}
            strokeDasharray="4 2"
            name="λ Noise"
          />
          <Line
            type="monotone"
            dataKey="ema_success"
            stroke={SUCCESS_COLOR}
            dot={false}
            strokeWidth={1.5}
            strokeDasharray="2 3"
            name="EMA Success"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Graph Growth Chart ────────────────────────────────────────────
export function GraphGrowthChart({ history }) {
  if (!history || history.length === 0) {
    return (
      <div className="chart-card">
        <div className="chart-title">ARG Graph Growth</div>
        <div className="chart-legend">
          <div className="legend-item"><div className="legend-dot" style={{background:NODES_COLOR}} />Nodes</div>
          <div className="legend-item"><div className="legend-dot" style={{background:EDGES_COLOR}} />Edges</div>
        </div>
        <div className="chart-empty">Waiting for training data…</div>
      </div>
    );
  }

  const { contentStyle, labelStyle } = ChartTooltipStyle();

  return (
    <div className="chart-card">
      <div className="chart-title">ARG Graph Growth</div>
      <div className="chart-legend">
        <div className="legend-item"><div className="legend-dot" style={{background:NODES_COLOR}} />Nodes</div>
        <div className="legend-item"><div className="legend-dot" style={{background:EDGES_COLOR}} />Edges</div>
      </div>
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={history} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
          <XAxis
            dataKey="passage"
            tick={{ fill: '#565869', fontSize: 10 }}
            tickLine={false}
          />
          <YAxis tick={{ fill: '#565869', fontSize: 10 }} tickLine={false} />
          <Tooltip contentStyle={contentStyle} labelStyle={labelStyle} />
          <Line
            type="monotone"
            dataKey="nodes"
            stroke={NODES_COLOR}
            dot={false}
            strokeWidth={2}
            name="Nodes"
          />
          <Line
            type="monotone"
            dataKey="edges"
            stroke={EDGES_COLOR}
            dot={false}
            strokeWidth={2}
            name="Edges"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
