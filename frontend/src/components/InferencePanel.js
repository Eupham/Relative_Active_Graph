import React, { useState } from 'react';
import { Sparkles, Send, Search } from 'lucide-react';

export default function InferencePanel({ apiUrl }) {
  const [seedText, setSeedText] = useState('');
  const [maxTokens, setMaxTokens] = useState(64);
  const [language, setLanguage] = useState('en');
  const [output, setOutput] = useState(null);
  const [loading, setLoading] = useState(false);
  const [synonymWord, setSynonymWord] = useState('');
  const [synonymResult, setSynonymResult] = useState(null);

  const generate = async () => {
    if (!seedText.trim()) return;
    setLoading(true);
    setOutput(null);
    try {
      const res = await fetch(`${apiUrl}/api/inference/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed_text: seedText, max_tokens: maxTokens, language }),
      });
      const data = await res.json();
      setOutput(data);
    } catch (e) {
      setOutput({ error: e.message });
    }
    setLoading(false);
  };

  const querySynonyms = async () => {
    if (!synonymWord.trim()) return;
    setSynonymResult(null);
    try {
      const res = await fetch(`${apiUrl}/api/inference/synonym`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word: synonymWord, n: 5 }),
      });
      setSynonymResult(await res.json());
    } catch (e) {
      setSynonymResult({ error: e.message });
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4" data-testid="inference-panel">
      {/* Generation */}
      <div className="bg-panel border border-white/5 p-4 space-y-4">
        <h2 className="font-heading text-lg font-semibold tracking-wide uppercase text-neon-cyan flex items-center gap-2">
          <Sparkles className="w-4 h-4" />
          Free Generation
        </h2>

        <div className="space-y-3">
          <div>
            <label className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500 block mb-1">Seed Text</label>
            <textarea
              data-testid="inference-seed-input"
              value={seedText}
              onChange={e => setSeedText(e.target.value)}
              placeholder="Enter seed text for generation..."
              className="w-full bg-black border border-white/10 text-white font-mono text-sm py-2 px-3 h-24 resize-none
                focus:border-neon-cyan focus:ring-1 focus:ring-neon-cyan/50 transition-all placeholder-slate-600"
            />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500 block mb-1">Max Tokens</label>
              <input
                data-testid="inference-max-tokens"
                type="number" min={1} max={512} value={maxTokens}
                onChange={e => setMaxTokens(Number(e.target.value))}
                className="w-full bg-black border border-white/10 text-white font-mono text-sm py-2 px-3 text-right focus:border-neon-cyan"
              />
            </div>
            <div>
              <label className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500 block mb-1">Language</label>
              <select
                data-testid="inference-language"
                value={language}
                onChange={e => setLanguage(e.target.value)}
                className="w-full bg-black border border-white/10 text-white font-mono text-sm py-2 px-3 focus:border-neon-cyan"
              >
                {['en','de','fr','es','it','nl','pt','ru','zh','ja'].map(l => (
                  <option key={l} value={l}>{l.toUpperCase()}</option>
                ))}
              </select>
            </div>
          </div>
          <button
            data-testid="generate-btn"
            onClick={generate}
            disabled={loading || !seedText.trim()}
            className="w-full flex items-center justify-center gap-2 bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/50
              hover:bg-neon-cyan/20 hover:shadow-[0_0_15px_rgba(0,240,255,0.3)] transition-all
              uppercase tracking-widest text-xs font-bold py-2.5 px-4 disabled:opacity-40"
          >
            <Send className="w-4 h-4" />
            {loading ? 'Generating...' : 'Generate'}
          </button>
        </div>

        {output && (
          <div className="bg-black/40 border border-white/5 p-3 space-y-2" data-testid="inference-output">
            {output.error ? (
              <div className="text-neon-red font-mono text-xs">{output.error}</div>
            ) : (
              <>
                <div className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500">Output</div>
                <div className="font-mono text-sm text-neon-green leading-relaxed">
                  {output.output || '(empty output)'}
                </div>
                <div className="flex gap-4 text-[10px] font-mono text-slate-500">
                  <span>Quality: {output.quality}</span>
                  <span>Depth: {output.depth_used}</span>
                  <span>Satisfied: {output.satisfied ? 'Yes' : 'No'}</span>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Synonym Query */}
      <div className="bg-panel border border-white/5 p-4 space-y-4">
        <h2 className="font-heading text-lg font-semibold tracking-wide uppercase text-neon-green flex items-center gap-2">
          <Search className="w-4 h-4" />
          Synonym Query
        </h2>
        <div className="space-y-3">
          <div>
            <label className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500 block mb-1">Word</label>
            <input
              data-testid="synonym-input"
              type="text"
              value={synonymWord}
              onChange={e => setSynonymWord(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && querySynonyms()}
              placeholder="Enter a word..."
              className="w-full bg-black border border-white/10 text-white font-mono text-sm py-2 px-3
                focus:border-neon-green focus:ring-1 focus:ring-neon-green/50 transition-all placeholder-slate-600"
            />
          </div>
          <button
            data-testid="synonym-query-btn"
            onClick={querySynonyms}
            disabled={!synonymWord.trim()}
            className="w-full flex items-center justify-center gap-2 bg-neon-green/10 text-neon-green border border-neon-green/50
              hover:bg-neon-green/20 transition-all uppercase tracking-widest text-xs font-bold py-2.5 px-4 disabled:opacity-40"
          >
            <Search className="w-4 h-4" /> Query
          </button>
        </div>

        {synonymResult && (
          <div className="bg-black/40 border border-white/5 p-3 space-y-2" data-testid="synonym-output">
            {synonymResult.error ? (
              <div className="text-neon-red font-mono text-xs">{synonymResult.error}</div>
            ) : (
              <div className="font-mono text-xs text-slate-300">
                <pre className="whitespace-pre-wrap">{JSON.stringify(synonymResult, null, 2)}</pre>
              </div>
            )}
          </div>
        )}

        {/* Info about the teacher forcing pipeline */}
        <div className="bg-black/40 border border-white/5 p-3">
          <div className="text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500 mb-2">How It Works</div>
          <div className="font-mono text-[10px] text-slate-500 space-y-1.5 leading-relaxed">
            <p><span className="text-neon-cyan">Teacher Forcing</span> = Target-Constrained Derivation</p>
            <p><span className="text-neon-green">Attribution</span> = ATMS + SCM counterfactual scoring</p>
            <p><span className="text-neon-amber">Poisson Noise</span> = Adaptive rule-dropping for robustness</p>
            <p><span className="text-neon-red">Rule Induction</span> = MetaGrammar hypothesizes on failure</p>
            <p className="text-slate-600 mt-2">No backprop. No gradients. Structural survival of the fittest.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
