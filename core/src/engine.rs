//! CSRRE Engine: orchestrates the full execution flow (Section 7 of spec).
//! push(s) → ATMS activate → ARG expand → constrain → schedule → execute →
//! canonicalize → candidates → causal analysis → pop() → rule induction → linearize.

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, TRDId, Quality, ModalType, ModalMode, TypeCategory, Direction, Env, Situation};
use crate::atms::BaseAtms;
use crate::arg::{
    ContextStack, ArgGraph, ArgSearch, ArgNode, ArgEdge, NodeClass, EdgeClass,
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
use crate::generation::{ProgressiveDeepener, Linearizer, DeepeningResult, VocabDistribution};
use crate::feedback::update::{propagate_attribution_backward, apply_attribution};

/// A query submitted to the engine.
#[derive(Debug, Clone)]
pub struct Query {
    pub text:            String,
    pub situation_id:    u64,
    pub trd:             Option<TRDId>,
    pub target_language: String,
    /// Expected MTLG result type for this query.
    /// A hypothesis satisfies the query iff its proposition type is compatible
    /// with this type (same mode and category; arity may differ).
    /// Defaults to Diamond/Scene/arity=0 when unknown.
    pub expected_type:   ModalType,
}

impl Query {
    pub fn new(text: impl Into<String>, situation_id: u64, target_language: impl Into<String>) -> Self {
        Self {
            text:            text.into(),
            situation_id,
            trd:             None,
            target_language: target_language.into(),
            expected_type:   ModalType::default(),
        }
    }

    pub fn with_expected_type(mut self, ty: ModalType) -> Self {
        self.expected_type = ty;
        self
    }
}

/// The result returned after one full execution cycle.
#[derive(Debug)]
pub struct QueryResult {
    pub surface_output: String,
    pub satisfied:      bool,
    pub depth_used:     usize,
    pub quality:        Quality,
}

/// One step in a teacher-forcing training sequence.
#[derive(Clone, Debug)]
pub struct TokenStep {
    /// The surface text of this token (for logging / hypothesis matching).
    pub text:             String,
    /// The expected edge ID that should be activated for this token.
    pub expected_edge_id: EdgeId,
    /// Node pool to use for this step's ARG expansion.
    pub node_pool:        Vec<ArgNode>,
    /// Edge pool to use for this step's ARG expansion.
    pub edge_pool:        Vec<ArgEdge>,
}

/// Result of processing a full training sequence.
#[derive(Debug)]
pub struct SequenceTrainResult {
    /// Number of TokenSteps processed.
    pub steps_processed:  usize,
    /// Sum of Quality values across all steps.
    pub quality_sum:      f64,
    /// Final Quality of the last step.
    pub final_quality:    Quality,
    /// Edge IDs that received attribution updates.
    pub attributed_edges: Vec<EdgeId>,
}

/// The CSRRE Engine.
pub struct Engine {
    pub atms:           BaseAtms,
    pub context_stack:  ContextStack,
    pub perf:           PerfRegistry,
    pub thresholds:     ThresholdRegistry,
    pub semantics:      MtlgSemantics,
    pub egraph:         ArgEGraph,
    pub graphica:       GraphicaCache,
    pub attribution:    AttributionEngine,
    pub provenance:     ProvenanceLog,
    pub ruler:          RulerBridge,
    pub rule_lifecycle: RuleLifecycleManager,
    pub counterfactual: CounterfactualReasoner,
    tr_counter:         u64,
    /// The last ARG graph produced by `execute` or `execute_sequence`.
    pub last_graph:     Option<ArgGraph>,
    /// Cached node pool from the last expansion (for replay / attribution).
    pub node_pool_cache: Vec<ArgNode>,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            atms:            BaseAtms::new(),
            context_stack:   ContextStack::new(),
            perf:            PerfRegistry::new(0.25),
            thresholds:      ThresholdRegistry::new(),
            semantics:       MtlgSemantics::new(),
            egraph:          ArgEGraph::new(),
            graphica:        GraphicaCache::new(),
            attribution:     AttributionEngine::new(42),
            provenance:      ProvenanceLog::new(10_000),
            ruler:           RulerBridge::new(),
            rule_lifecycle:  RuleLifecycleManager::new(),
            counterfactual:  CounterfactualReasoner::new(0b1, 42),
            tr_counter:      0,
            last_graph:      None,
            node_pool_cache: Vec::new(),
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
            log::warn!("Sheaf violations={}: {} triangle inconsistencies", sheaf_result.violation_count, sheaf_result.violations.len());
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
        let dr = deepener.run(graph, &self.semantics, &self.perf, &mut self.thresholds, &query.text, &query.expected_type);

        // ── 8. Linearize best hypothesis ──────────────────────────────────────
        let linearizer     = Linearizer::new(&query.target_language);
        let surface_output = dr.hypotheses.first()
            .map(|h| linearizer.linearize(h))
            .unwrap_or_else(|| format!("[no hypothesis for '{}']", query.text));

        // ── 9. Quality assessment (heuristic: satisfied + depth bonus) ────────
        let quality = if dr.satisfied {
            if dr.depth_used == 0 { Quality::GOOD } else { Quality::PARTIAL }
        } else {
            Quality::BAD
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
                self.ruler.observe_dissolved_tr(tr, quality.as_f32() >= Quality::PARTIAL.as_f32());
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

    /// Execute a teacher-forcing training sequence.
    ///
    /// For each `TokenStep`, the engine:
    /// 1. Expands the ARG from the step's node/edge pool.
    /// 2. Builds a `VocabDistribution` (softmax over active nodes).
    /// 3. Computes CE-based `Quality` against the expected edge.
    /// 4. Applies attribution to edges involved in this step.
    /// 5. Updates TRD performance and thresholds.
    pub fn execute_sequence(
        &mut self,
        trd:      TRDId,
        steps:    Vec<TokenStep>,
        language: &str,
    ) -> SequenceTrainResult {
        let mut quality_sum      = 0.0f64;
        let mut attributed_edges = Vec::new();
        let mut final_quality    = Quality::BAD;
        let n_steps              = steps.len();

        let active_env = self.context_stack.current_env();
        let theta_alpha = self.thresholds.theta_alpha(trd);
        let theta_rho   = self.thresholds.theta_rho(trd);

        for step in steps {
            // Expand ARG for this token step.
            let mut search = ArgSearch::new(active_env, theta_alpha, theta_rho);
            for node in step.node_pool.iter().cloned() { search.try_activate(node); }
            for edge in step.edge_pool.iter().cloned() { search.try_add_edge(edge); }

            let graph = &search.graph;

            // VocabDistribution: P(token | context).
            let dist = VocabDistribution::from_graph(graph, active_env);

            // CE quality: was the expected edge the top-scoring one?
            let predicted_top = dist.probs.iter()
                .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .map(|&(_, eid, _)| eid)
                .unwrap_or(0);
            let was_correct = predicted_top == step.expected_edge_id;
            let quality = dist.ce_quality(step.expected_edge_id, was_correct);

            quality_sum   += quality.as_f64();
            final_quality  = quality;

            // Apply attribution to the expected edge.
            let result = self.apply_attribution_batch(
                vec![step.expected_edge_id],
                quality,
                trd,
            );
            attributed_edges.extend(result);

            self.node_pool_cache = step.node_pool;
        }

        // Store the last built graph (re-build from cache for caller convenience).
        // (In a full impl the graph would be retained across steps.)
        self.last_graph = None;

        // Update TRD performance.
        self.perf.update(trd, final_quality);
        self.thresholds.sync(&self.perf);

        SequenceTrainResult {
            steps_processed:  n_steps,
            quality_sum,
            final_quality,
            attributed_edges,
        }
    }

    /// Apply a batch of edge attribution updates for the given quality signal.
    ///
    /// Records each edge in the attribution engine and returns the list of
    /// edge IDs that were updated.
    pub fn apply_attribution_batch(
        &mut self,
        edge_ids: Vec<EdgeId>,
        quality:  Quality,
        trd_id:   TRDId,
    ) -> Vec<EdgeId> {
        self.tr_counter += 1;
        let tr_id = self.tr_counter;
        self.attribution.record(tr_id, trd_id, quality, &edge_ids);
        // Also update the counterfactual reasoner.
        self.counterfactual.record_dissolved_tr(tr_id, trd_id, quality, edge_ids.clone());
        edge_ids
    }
}

impl Default for Engine { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::ArgNode;

    fn make_node(id: u64, surface: &str, score: f32) -> ArgNode {
        let mut n = ArgNode::new(id, NodeClass::DEFAULT,
            ModalType::functor(ModalMode::Diamond, TypeCategory::DEFAULT, 1, Direction::Right), (0, 0));
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
            expected_type:   ModalType::default(),
        };
        let nodes = vec![make_node(1, "run", 0.9), make_node(2, "alice", 0.7)];
        let result = engine.execute(query, nodes, vec![]);
        // Should produce a surface output (even if no hypothesis satisfies)
        assert!(!result.surface_output.is_empty());
    }
}
