import React from 'react';
import { Database, GitBranch, Box } from 'lucide-react';

export default function EngineStats({ state }) {
  return (
    <div className="bg-panel border border-white/5 p-4 space-y-4" data-testid="engine-stats">
      <h2 className="font-heading text-lg font-semibold tracking-wide uppercase text-neon-green flex items-center gap-2">
        <Database className="w-4 h-4" />
        Engine State
      </h2>

      <div className="grid grid-cols-3 gap-2">
        <StatCard
          icon={<Box className="w-3.5 h-3.5 text-neon-cyan" />}
          label="Global Nodes"
          value={state?.global_nodes || 0}
        />
        <StatCard
          icon={<GitBranch className="w-3.5 h-3.5 text-neon-green" />}
          label="Global Edges"
          value={state?.global_edges || 0}
        />
        <StatCard
          icon={<Database className="w-3.5 h-3.5 text-neon-amber" />}
          label="Rules Induced"
          value={state?.rules_induced || 0}
        />
      </div>

      <div className="bg-black/40 border border-white/5 p-3 space-y-2">
        <div className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500">Architecture Pipeline</div>
        <div className="font-mono text-[11px] text-slate-400 space-y-1">
          <PipelineStep label="C4 Stream" desc="HuggingFace mC4 corpus" active />
          <PipelineStep label="Token Extraction" desc="FNV-1a hashing, feature extraction" active />
          <PipelineStep label="Teacher Forcing" desc="Target-Constrained Derivation" active={state?.running} />
          <PipelineStep label="ATMS Attribution" desc="Counterfactual edge scoring" active={state?.running} />
          <PipelineStep label="MetaGrammar" desc="Rule induction on derivation failure" active={state?.running} />
          <PipelineStep label="Poisson Noise" desc={`Adaptive lambda=${(state?.poisson?.lambda || 0).toFixed(3)}`} active={state?.running} />
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon, label, value }) {
  return (
    <div className="bg-black/40 border border-white/5 p-2">
      <div className="flex items-center gap-1.5 mb-1">
        {icon}
        <span className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-600">{label}</span>
      </div>
      <div className="font-mono text-lg text-slate-200">{typeof value === 'number' ? value.toLocaleString() : value}</div>
    </div>
  );
}

function PipelineStep({ label, desc, active }) {
  return (
    <div className="flex items-center gap-2">
      <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${active ? 'bg-neon-green' : 'bg-slate-700'}`} />
      <span className="text-slate-300">{label}</span>
      <span className="text-slate-600 text-[10px]">{desc}</span>
    </div>
  );
}
