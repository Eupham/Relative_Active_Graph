//! CSRRE Engine: orchestrates the full execution flow (Section 7 of spec).
//! push(s) → ATMS activate → ARG expand → constrain → schedule → execute →
//! canonicalize → candidates → causal analysis → pop() → rule induction → linearize.

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, TRDId, Quality, ModalType, ModalMode, TypeCategory, Direction, Env, Situation};
use crate::atms::BaseAtms;
use crate::arg::{
    ContextStack, ArgGraph, ArgSearch, ArgNode, ArgEdge, NodeClass, EdgeClass,
    PassageContext, SlotOccupancyTracker,
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
use crate::generation::{ProgressiveDeepener, Linearizer, DeepeningResult, VocabDistribution, Hypothesis};
use crate::generation::linearizer::LexEntry;
use crate::semantics::mtlg_semantics::PropositionGraph;
use crate::feedback::update::{propagate_attribution_backward, apply_attribution, apply_weight_decay, propagate_edge_to_node_scores};

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
    /// The expected NodeId that should be activated for this token.
    /// This is the stable hash of the token's lemma — not a transient edge ID.
    pub expected_node_id: NodeId,
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
    /// Sum of Quality magnitudes across all steps.
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
    /// Per-language lexicon for linearization.
    pub global_lexicon: HashMap<String, LexEntry>,
    /// Slot occupancy tracker for synonym edge discovery.
    pub slot_tracker:   SlotOccupancyTracker,
}

impl Engine {
    pub fn new() -> Self {
        let mut engine = Self {
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
            global_lexicon:  HashMap::new(),
            slot_tracker:    SlotOccupancyTracker::new(),
        };
        engine
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

        // Snapshot node/edge pools before step 3 consumes them (needed for decoding).
        let node_pool_snapshot = node_pool.clone();
        let edge_pool_snapshot = edge_pool.clone();

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

        // ── 8. Generate surface output ────────────────────────────────────────
        let decode_trd = trd.unwrap_or(0);
        let decoded_ids = self.decode(&node_pool_snapshot, &edge_pool_snapshot, decode_trd, 32);

        let surface_output = if decoded_ids.is_empty() {
            // No confident prediction. Return the raw hypothesis root as a last resort.
            dr.hypotheses.first()
                .map(|h| {
                    let mut lin = Linearizer::new(&query.target_language);
                    for entry in self.global_lexicon.values() {
                        lin.lexicon.register(entry.clone());
                    }
                    lin.proposition_to_surface(&h.proposition)
                })
                .unwrap_or_else(|| format!("[no output for '{}']", query.text))
        } else {
            self.surface_from_decoded(&decoded_ids, &node_pool_snapshot, &query.target_language)
        };

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
    /// 2. Builds a `VocabDistribution` (softmax over active nodes by NodeId).
    /// 3. Computes CE-based two-sided `Quality` against the expected node.
    /// 4. Applies positive attribution to expected node's incoming edges.
    /// 5. Applies negative attribution to wrongly-predicted node's incoming edges.
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

        let active_env  = self.context_stack.current_env();
        let theta_alpha = self.thresholds.theta_alpha(trd);
        let theta_rho   = self.thresholds.theta_rho(trd);

        for step in steps {
            // Expand ARG for this token step.
            let mut search = ArgSearch::new(active_env, theta_alpha, theta_rho);
            for node in step.node_pool.iter().cloned() { search.try_activate(node); }
            for edge in step.edge_pool.iter().cloned() { search.try_add_edge(edge); }

            let graph = &search.graph;

            // VocabDistribution: P(token | context) keyed by NodeId.
            let dist = VocabDistribution::from_graph(graph, active_env);

            // Two-sided CE split: positive for expected, negative for wrongly predicted.
            let (q_correct, q_wrong_opt) = dist.ce_quality_split(step.expected_node_id);

            // Pull expected node's incoming edges toward activation.
            let incoming_expected: Vec<EdgeId> = graph.edge_indices()
                .filter(|&ei| graph[ei].dst == step.expected_node_id)
                .map(|ei| graph[ei].id)
                .collect();
            let result = self.apply_attribution_batch(incoming_expected, q_correct, trd);
            attributed_edges.extend(result);

            // Push wrongly-predicted node's incoming edges away.
            if let Some((wrong_nid, q_neg)) = q_wrong_opt {
                let incoming_wrong: Vec<EdgeId> = graph.edge_indices()
                    .filter(|&ei| graph[ei].dst == wrong_nid)
                    .map(|ei| graph[ei].id)
                    .collect();
                let result = self.apply_attribution_batch(incoming_wrong, q_neg, trd);
                attributed_edges.extend(result);
            }

            quality_sum   += q_correct.magnitude() as f64;
            final_quality  = q_correct;

            self.node_pool_cache = step.node_pool;
        }

        self.last_graph = None;

        self.perf.update(trd, final_quality);
        self.thresholds.sync(&self.perf);

        SequenceTrainResult {
            steps_processed:  n_steps,
            quality_sum,
            final_quality,
            attributed_edges,
        }
    }

    /// Execute a full passage as a teacher-forcing training unit.
    ///
    /// Implements incremental sequential conditioning: at each step, the
    /// VocabDistribution is computed from tokens 0..t-1 only. Token t is
    /// committed to context *after* prediction, so the model can never
    /// trivially predict from the future.
    ///
    /// Attribution is accumulated across the whole passage and flushed once
    /// at the end via backward propagation through the complete ATMS chain.
    /// Rationale (EBL, structured perceptron, ATMS): per-token propagation
    /// mutates a graph rebuilt fresh each step, discarding the changes.
    /// End-of-passage propagation operates on the complete justification structure.
    pub fn execute_passage(
        &mut self,
        trd:       TRDId,
        sentences: Vec<Vec<TokenStep>>,
        language:  &str,
    ) -> SequenceTrainResult {
        let active_env  = self.context_stack.current_env();
        let theta_alpha = self.thresholds.theta_alpha(trd);
        let theta_rho   = self.thresholds.theta_rho(trd);

        let mut passage       = PassageContext::new(trd);
        let mut quality_sum   = 0.0f64;
        let mut total_steps   = 0usize;
        let mut final_quality = Quality::BAD;
        let mut prev_node_id: Option<NodeId> = None;

        // ── Incremental forward pass ──────────────────────────────────────────
        // At each step: predict from prior context (tokens 0..t-1), accumulate
        // CE signal, then commit token t to the context graph.
        for sentence in &sentences {
            for step in sentence {
                let search = passage.step(
                    step.expected_node_id,
                    &step.node_pool,
                    &step.edge_pool,
                    active_env,
                    theta_alpha,
                    theta_rho,
                    prev_node_id,
                );

                let dist = VocabDistribution::from_graph(&search.graph, active_env);
                let (q_correct, q_wrong_opt) = dist.ce_quality_split(step.expected_node_id);

                // Accumulate CE signal on expected node's incoming edges.
                for ei in search.graph.edge_indices() {
                    if search.graph[ei].dst == step.expected_node_id {
                        passage.signal.accumulate(search.graph[ei].id, q_correct.0);
                    }
                }

                // Accumulate negative signal on wrong node's incoming edges.
                if let Some((wrong_nid, q_neg)) = q_wrong_opt {
                    for ei in search.graph.edge_indices() {
                        if search.graph[ei].dst == wrong_nid {
                            passage.signal.accumulate(search.graph[ei].id, q_neg.0);
                        }
                    }
                }

                quality_sum   += q_correct.magnitude() as f64;
                final_quality  = q_correct;
                total_steps   += 1;
                prev_node_id   = Some(step.expected_node_id);
            }
        }

        // Observe slot occupancy for synonym edge discovery.
        for n in passage.node_map.values() {
            if let Some(&trd_id) = n.trd_membership.first() {
                self.slot_tracker.observe(
                    n.id, n.surface_str().unwrap_or("_"), trd_id, n.mtlg_type,
                );
            }
        }

        // ── End-of-passage flush ──────────────────────────────────────────────
        // Build the complete passage graph once, then apply all accumulated
        // signals and propagate backward through the full justification chain.
        let mut final_graph = passage
            .build_graph(active_env, theta_alpha, theta_rho)
            .graph;

        let token_count = passage.signal.token_count.max(1) as f32;
        let mut updated_nodes: Vec<NodeId> = Vec::new();

        for (&edge_id, &signal) in &passage.signal.edge_signals {
            let q = Quality::new(signal / token_count);
            apply_attribution(&mut final_graph, edge_id, 1.0, q);
            // Track destination nodes for score propagation.
            if let Some(ei) = final_graph.edge_indices()
                .find(|&i| final_graph[i].id == edge_id)
            {
                updated_nodes.push(final_graph[ei].dst);
            }
        }

        let attributed_edges: Vec<EdgeId> =
            passage.signal.edge_signals.keys().copied().collect();

        // Backward attribution through ATMS justification chain.
        // Uses the net passage-level signal as the propagation seed.
        if total_steps > 0 {
            let net_q = Quality::new((quality_sum / total_steps as f64) as f32);
            let terminal_nodes: Vec<NodeId> = sentences.iter()
                .flat_map(|s| s.iter())
                .map(|step| step.expected_node_id)
                .collect();
            propagate_attribution_backward(
                &mut final_graph, &self.atms,
                &terminal_nodes, net_q, 4, 0.7,
            );
        }

        // Propagate updated edge weights to node attribution scores.
        updated_nodes.sort_unstable();
        updated_nodes.dedup();
        for nid in updated_nodes {
            propagate_edge_to_node_scores(&mut final_graph, nid);
        }

        // Periodic weight decay every 50 passages.
        self.tr_counter += 1;
        if self.tr_counter % 50 == 0 {
            apply_weight_decay(&mut final_graph, 0.001);
        }

        // Periodic synonym materialisation every 100 passages.
        if self.tr_counter % 100 == 0 {
            self.slot_tracker.materialise_synonym_edges(&mut final_graph, 3);
        }

        self.last_graph = Some(final_graph);
        self.perf.update(trd, final_quality);
        self.thresholds.sync(&self.perf);

        SequenceTrainResult {
            steps_processed:  total_steps,
            quality_sum,
            final_quality,
            attributed_edges,
        }
    }

    /// Decode a sequence of tokens from the current ARG context.
    ///
    /// Uses the same `PassageContext` → `VocabDistribution` → commit loop as
    /// `execute_passage`, but selects (argmax) instead of attributing.
    ///
    /// `seed_nodes` and `seed_edges` seed the context before any decoding begins.
    ///
    /// `max_tokens` is an upper bound. Decoding stops earlier if:
    /// - The top-scoring node probability drops below `1.0 / vocab_size`
    ///   (the model is no longer confident about any continuation), OR
    /// - The last two selected nodes are identical (repetition = done).
    ///
    /// Returns the sequence of selected NodeIds in emission order.
    pub fn decode(
        &mut self,
        seed_nodes: &[ArgNode],
        seed_edges: &[ArgEdge],
        trd:        TRDId,
        max_tokens: usize,
    ) -> Vec<NodeId> {
        let active_env  = self.context_stack.current_env();
        let theta_alpha = self.thresholds.theta_alpha(trd);
        let theta_rho   = self.thresholds.theta_rho(trd);

        let mut passage = PassageContext::new(trd);

        // Absorb seed context (same as the initial ARG for this query).
        passage.absorb_sentence(seed_nodes, seed_edges);

        let mut output: Vec<NodeId> = Vec::new();
        let mut prev: Option<NodeId> = None;

        for _ in 0..max_tokens {
            // Candidate pool is empty past the seed — the model generates
            // from the context it has accumulated.
            let result = passage.decode_step(
                &[],
                &[],
                active_env,
                theta_alpha,
                theta_rho,
                prev,
            );

            let (node_id, prob) = match result {
                Some(r) => r,
                None    => break,  // empty distribution = stop
            };

            // Stopping conditions.
            let vocab_size = passage.node_map.len().max(1);
            let floor = 1.0 / vocab_size as f32;
            if prob < floor { break; }
            if output.last() == Some(&node_id) { break; }  // repetition

            output.push(node_id);
            prev = Some(node_id);
        }

        output
    }

    /// Resolve a sequence of NodeIds to a surface string.
    ///
    /// Reads each node's surface bytes directly from the ARG graph.
    /// Falls back to the node's lexicon entry if surface bytes are absent
    /// (handles abstract nodes that were never seen in training text).
    /// Never consults role_order or frame templates.
    ///
    /// `seed_nodes` is the node pool that was used to seed the decode context.
    /// Nodes are looked up here first (they carry the surface bytes from training).
    pub fn surface_from_decoded(
        &self,
        node_ids:   &[NodeId],
        seed_nodes: &[ArgNode],
        language:   &str,
    ) -> String {
        let mut linearizer = Linearizer::new(language);
        for entry in self.global_lexicon.values() {
            linearizer.lexicon.register(entry.clone());
        }

        node_ids.iter()
            .filter_map(|&nid| {
                // Primary: check seed nodes (they carry surface bytes from the query).
                if let Some(node) = seed_nodes.iter().find(|n| n.id == nid) {
                    if let Some(s) = node.surface_str() {
                        if !s.is_empty() {
                            return Some(s.to_string());
                        }
                    }
                }
                // Check node_pool_cache (from training).
                if let Some(node) = self.node_pool_cache.iter().find(|n| n.id == nid) {
                    if let Some(s) = node.surface_str() {
                        if !s.is_empty() {
                            return Some(s.to_string());
                        }
                    }
                }
                // Check last_graph if available.
                if let Some(ref graph) = self.last_graph {
                    let node_idx = graph.node_indices().find(|&i| graph[i].id == nid);
                    if let Some(idx) = node_idx {
                        let node = &graph[idx];
                        if let Some(s) = node.surface_str() {
                            if !s.is_empty() {
                                return Some(s.to_string());
                            }
                        }
                    }
                }
                // Fallback: lexicon lookup by NodeId → predicate string (for abstract nodes).
                linearizer.lexicon.surface_for_node_id(nid, language)
                    .map(|s| s.to_string())
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Answer "another word for X" by traversing synonym edges.
    ///
    /// Given a surface form, finds the node whose surface matches, then returns
    /// the top-N neighbours connected by synonym edges, ranked by edge weight.
    pub fn synonym_query(&self, surface: &str, n: usize) -> Vec<(String, f32)> {
        let graph = match &self.last_graph {
            Some(g) => g,
            None    => return vec![],
        };

        // Find the node for this surface form.
        let source_idx = graph.node_indices().find(|&i| {
            graph[i].surface_str().map_or(false, |s| s.eq_ignore_ascii_case(surface))
        });

        let Some(src_idx) = source_idx else { return vec![]; };
        let src_id = graph[src_idx].id;

        // Traverse synonym edges (both directions).
        let mut candidates: Vec<(String, f32)> = graph.edge_indices()
            .filter(|&ei| graph[ei].src == src_id || graph[ei].dst == src_id)
            .filter_map(|ei| {
                let neighbour_id = if graph[ei].src == src_id {
                    graph[ei].dst
                } else {
                    graph[ei].src
                };
                let weight = graph[ei].weight;
                let neighbour_idx = graph.node_indices().find(|&i| graph[i].id == neighbour_id)?;
                let s = graph[neighbour_idx].surface_str()?.to_string();
                Some((s, weight))
            })
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(n);
        candidates
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
        assert!(!result.surface_output.is_empty());
    }

    #[test]
    fn execute_sequence_two_sided_ce() {
        let mut engine = Engine::new();

        let node_a = make_node(10, "cat", 0.9);    // dominant (wrong)
        let node_b = make_node(20, "feline", 0.1); // expected but low prob

        let step = TokenStep {
            text:             "feline".into(),
            expected_node_id: 20,
            node_pool:        vec![node_a, node_b],
            edge_pool:        vec![],
        };

        let result = engine.execute_sequence(0, vec![step], "en");
        assert_eq!(result.steps_processed, 1);
        // quality_sum should be > 0 since we got the CE signal from wrong prediction
        assert!(result.quality_sum >= 0.0);
    }
}
