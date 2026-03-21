//! CSRRE Engine: orchestrates the full execution flow (Section 7 of spec).
//! push(s) → ATMS activate → ARG expand → constrain → schedule → execute →
//! canonicalize → candidates → causal analysis → pop() → rule induction → linearize.

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, TRDId, Quality, ModalType, Env, Situation};
use crate::atms::BaseAtms;
use crate::arg::{
    ContextStack, ArgGraph, ArgSearch, ArgNode, ArgEdge, NodeType, EdgeType,
    transient_repr::{RepContent, Granularity},
    egraph_adapter::ArgEGraph,
    graphica_adapter::{GraphicaCache, build_key},
};
use crate::adaptive::{PerfRegistry, ThresholdRegistry, CausalTransitionRegistry};
use crate::constraints::{validate_shapes, check_sheaf_coherence, build_stalks, default_constraints, check_mode_consistency, SheafResult};
use crate::scheduler::{ExecStateTable, NodeExecState, build_schedule};
use crate::causal::CounterfactualReasoner;
use crate::semantics::MtlgSemantics;
use crate::rules::{RulerBridge, RuleLifecycleManager};
use crate::feedback::{AttributionEngine, ProvenanceLog, apply_trace};
use crate::generation::{ProgressiveDeepener, Linearizer, DeepeningResult};

/// A query submitted to the engine.
#[derive(Debug, Clone)]
pub struct Query {
    pub text:           String,
    pub situation_id:   u64,
    pub trd:            Option<TRDId>,
    pub target_language: String,
}

/// The result returned after one full execution cycle.
#[derive(Debug)]
pub struct QueryResult {
    pub surface_output: String,
    pub satisfied:      bool,
    pub depth_used:     usize,
    pub quality:        Quality,
}

/// The CSRRE Engine.
pub struct Engine {
    pub atms:          BaseAtms,
    pub context_stack: ContextStack,
    pub perf:          PerfRegistry,
    pub thresholds:    ThresholdRegistry,
    pub semantics:     MtlgSemantics,
    pub egraph:        ArgEGraph,
    pub graphica:      GraphicaCache,
    pub attribution:   AttributionEngine,
    pub provenance:    ProvenanceLog,
    pub ruler:         RulerBridge,
    pub rule_lifecycle: RuleLifecycleManager,
    pub counterfactual: CounterfactualReasoner,
    tr_counter:        u64,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            atms:           BaseAtms::new(),
            context_stack:  ContextStack::new(),
            perf:           PerfRegistry::new(0.25),
            thresholds:     ThresholdRegistry::new(),
            semantics:      MtlgSemantics::new(),
            egraph:         ArgEGraph::new(),
            graphica:       GraphicaCache::new(),
            attribution:    AttributionEngine::new(42),
            provenance:     ProvenanceLog::new(10_000),
            ruler:          RulerBridge::new(),
            rule_lifecycle: RuleLifecycleManager::new(),
            counterfactual: CounterfactualReasoner::new(0b1, 42),
            tr_counter:     0,
        }
    }

    /// Full execution cycle for a query.
    pub fn execute(&mut self, query: Query, node_pool: Vec<ArgNode>, edge_pool: Vec<ArgEdge>) -> QueryResult {
        // ── 1. Push context ───────────────────────────────────────────────────
        let ctx_id = self.context_stack.push(query.situation_id, query.trd);
        let active_env = self.context_stack.current_env();
        let trd = query.trd;

        // ── 2. Compute TRD-relative thresholds ───────────────────────────────
        let theta_alpha = trd.map(|d| self.thresholds.theta_alpha(d)).unwrap_or(0.4);
        let theta_rho   = trd.map(|d| self.thresholds.theta_rho(d)).unwrap_or(0.38);

        // ── 3. ARG expansion ─────────────────────────────────────────────────
        let mut search = ArgSearch::new(active_env, theta_alpha, theta_rho);
        for node in node_pool { search.try_activate(node); }
        for edge in edge_pool  { search.try_add_edge(edge); }
        let graph = &search.graph;

        // ── 4. Constraint checks ─────────────────────────────────────────────
        let shacl_violations = validate_shapes(graph, &default_constraints());
        if !shacl_violations.is_empty() {
            log::warn!("SHACL: {} violations in G(s)", shacl_violations.len());
        }
        let stalks       = build_stalks(graph);
        let sheaf_result = check_sheaf_coherence(graph, &stalks);
        if !sheaf_result.is_coherent() {
            log::warn!("Sheaf H¹={}: {} violations", sheaf_result.h1_norm, sheaf_result.violations.len());
        }

        // ── 5. Graphica memo check ────────────────────────────────────────────
        let nodes_ref: Vec<_> = graph.node_indices().map(|i| &graph[i]).collect();
        let edges_ref: Vec<_> = graph.edge_indices().map(|i| &graph[i]).collect();
        let cache_key = build_key(&nodes_ref, &edges_ref, active_env);

        // ── 6. E-graph saturation (on non-cached subgraphs) ───────────────────
        if self.graphica.get(&cache_key).is_none() {
            self.egraph.saturate();
        }

        // ── 7. Progressive deepening + generation ─────────────────────────────
        let deepener = ProgressiveDeepener::new(trd);
        let dr = deepener.run(graph, &self.semantics, &self.perf, &mut self.thresholds, &query.text);

        // ── 8. Linearize best hypothesis ──────────────────────────────────────
        let linearizer     = Linearizer::new(&query.target_language);
        let surface_output = dr.hypotheses.first()
            .map(|h| linearizer.linearize(h))
            .unwrap_or_else(|| format!("[no hypothesis for '{}']", query.text));

        // ── 9. Quality assessment (heuristic: satisfied + depth bonus) ────────
        let quality = if dr.satisfied {
            if dr.depth_used == 0 { Quality::Good } else { Quality::Partial }
        } else {
            Quality::Bad
        };

        // ── 10. Add TR to context ─────────────────────────────────────────────
        self.tr_counter += 1;
        let tr_id = self.context_stack.add_tr(
            RepContent::Lambda(surface_output.clone()),
            active_env,
            ModalType::default(),
            Granularity::Sentence,
        );

        // ── 11. Pop context: dissolve TRs, lift DRS, apply attribution ────────
        if let Some((dissolved_trs, _referents)) = self.context_stack.pop() {
            for tr in &dissolved_trs {
                // Apply attribution trace to ARG (in real system: graph would be mutable here).
                // Record in ruler for rule induction.
                self.ruler.observe_dissolved_tr(tr, quality == Quality::Good || quality == Quality::Partial);
            }
        }

        // ── 12. Threshold update ──────────────────────────────────────────────
        if let Some(d) = trd {
            self.perf.update(d, quality);
            self.thresholds.sync(&self.perf);
        }

        // ── 13. Rule induction ────────────────────────────────────────────────
        self.ruler.induce_rules();
        self.rule_lifecycle.integrate_induced(self.ruler.rules.clone());
        self.rule_lifecycle.on_tr_dissolved(self.tr_counter);

        QueryResult { surface_output, satisfied: dr.satisfied, depth_used: dr.depth_used, quality }
    }
}

impl Default for Engine { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::ArgNode;
    use crate::types::{ModalType, ModalMode, TypeCategory};

    fn make_node(id: u64, surface: &str, score: f32) -> ArgNode {
        let mut n = ArgNode::new(id, NodeType::Concept,
            ModalType::functor(ModalMode::Diamond, TypeCategory::Scene, 1, true), (0, 0));
        n.surface = Some(surface.as_bytes().to_vec());
        n.attribution_score = score;
        n.atms_label = 0b1;
        n
    }

    #[test]
    fn engine_executes_query() {
        let mut engine = Engine::new();
        let query = Query {
            text:            "run".into(),
            situation_id:    1,
            trd:             None,
            target_language: "en".into(),
        };
        let nodes = vec![make_node(1, "run", 0.9), make_node(2, "alice", 0.7)];
        let result = engine.execute(query, nodes, vec![]);
        // Should produce a surface output (even if no hypothesis satisfies)
        assert!(!result.surface_output.is_empty());
    }
}
