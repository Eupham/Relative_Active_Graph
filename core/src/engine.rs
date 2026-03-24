//! CSRRE Engine: unified execution path for training and inference.
//! Every call — training passage or inference query — runs the same loop.
//! The engine learns continuously: global_nodes and global_edges persist
//! across all calls and are updated via EMA merging after each execution.

use std::collections::HashMap;
use std::path::Path;
use serde::{Serialize, Deserialize};
use crate::types::{NodeId, EdgeId, TRDId, Quality, ModalType, ModalMode, TypeCategory, Env};
use crate::atms::BaseAtms;
use crate::arg::{
    ContextStack, ArgGraph, ArgSearch, ArgNode, ArgEdge,
    PassageContext, SlotOccupancyTracker,
    transient_repr::{RepContent, Granularity},
    egraph_adapter::ArgEGraph,
    memoization::{GraphicaCache, CachedResult, build_key},
    type_normalizer::TypeNormalizer,
};
use crate::adaptive::{PerfRegistry, ThresholdRegistry};
use crate::constraints::{validate_shapes, check_sheaf_coherence, build_stalks, default_constraints};
use crate::scheduler::{ExecStateTable, NodeExecState, build_schedule};
use crate::causal::CounterfactualReasoner;
use crate::semantics::{MtlgSemantics, MetaGrammarEngine};
use crate::rules::{RulerBridge, RuleLifecycleManager};
use crate::feedback::{AttributionEngine, ProvenanceLog};
use crate::feedback::provenance::AuditEntry;
use crate::feedback::update::{
    propagate_attribution_backward, apply_attribution, apply_weight_decay,
    propagate_edge_to_node_scores, apply_nogood_consequences,
};
use crate::generation::{ProgressiveDeepener, Linearizer, VocabDistribution};
use crate::generation::linearizer::LexEntry;
use crate::lcs::converter::CategoryInducer;
use crate::lcs::token_types::TokenStructure;

#[derive(Debug, Clone)]
pub struct Query {
    pub text:            String,
    pub situation_id:    u64,
    pub trd:             Option<TRDId>,
    pub target_language: String,
    pub expected_type:   ModalType,
}

impl Query {
    pub fn new(text: impl Into<String>, situation_id: u64, target_language: impl Into<String>) -> Self {
        Self {
            text: text.into(), situation_id, trd: None,
            target_language: target_language.into(),
            expected_type: ModalType::default(),
        }
    }
    pub fn with_expected_type(mut self, ty: ModalType) -> Self { self.expected_type = ty; self }
}

#[derive(Debug)]
pub struct QueryResult {
    pub surface_output: String,
    pub satisfied:      bool,
    pub depth_used:     usize,
    pub quality:        Quality,
}

#[derive(Clone, Debug)]
pub struct TokenStep {
    pub text:             String,
    pub expected_node_id: NodeId,
    pub node_pool:        Vec<ArgNode>,
    pub edge_pool:        Vec<ArgEdge>,
}

#[derive(Debug)]
pub struct SequenceTrainResult {
    pub steps_processed:  usize,
    pub quality_sum:      f64,
    pub final_quality:    Quality,
    pub attributed_edges: Vec<EdgeId>,
}

pub struct Engine {
    pub atms:            BaseAtms,
    pub context_stack:   ContextStack,
    pub perf:            PerfRegistry,
    pub thresholds:      ThresholdRegistry,
    pub semantics:       MtlgSemantics,
    pub egraph:          ArgEGraph,
    pub graphica:        GraphicaCache,
    pub attribution:     AttributionEngine,
    pub provenance:      ProvenanceLog,
    pub ruler:           RulerBridge,
    pub rule_lifecycle:  RuleLifecycleManager,
    pub counterfactual:  CounterfactualReasoner,
    tr_counter:          u64,
    pub last_graph:      Option<ArgGraph>,
    pub node_pool_cache: Vec<ArgNode>,
    pub global_lexicon:  HashMap<String, LexEntry>,
    pub slot_tracker:    SlotOccupancyTracker,
    // Persistent global knowledge pool
    pub global_nodes:    HashMap<NodeId, ArgNode>,
    pub global_edges:    HashMap<EdgeId, ArgEdge>,
    pub node_budget:     usize,
    pub edge_budget:     usize,
    // Online category induction
    pub category_inducer:        CategoryInducer,
    pub pending_structures:      Vec<TokenStructure>,
    pub category_refit_counter:  u64,
    pub category_refit_interval: u64,
    // MetaGrammar Engine: discovers composition rules (§12)
    pub meta_grammar: MetaGrammarEngine,
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
            global_lexicon:  HashMap::new(),
            slot_tracker:    SlotOccupancyTracker::new(),
            global_nodes:    HashMap::new(),
            global_edges:    HashMap::new(),
            node_budget:     500_000,
            edge_budget:     5_000_000,
            category_inducer:        CategoryInducer::new(0),
            pending_structures:      Vec::new(),
            category_refit_counter:  0,
            category_refit_interval: 500,
            meta_grammar:            MetaGrammarEngine::new(0.0),
        }
    }

    // ── Global pool management ────────────────────────────────────────────────

    fn enrich_from_global(&self, wire_nodes: Vec<ArgNode>) -> Vec<ArgNode> {
        wire_nodes.into_iter().map(|mut n| {
            if let Some(known) = self.global_nodes.get(&n.id) {
                n.attribution_score = known.attribution_score;
                n.trd_membership    = known.trd_membership.clone();
                if n.structure.is_none() { n.structure = known.structure.clone(); }
            }
            n
        }).collect()
    }

    fn merge_into_global(&mut self, graph: &ArgGraph) {
        for idx in graph.node_indices() {
            let node = &graph[idx];
            self.global_nodes.entry(node.id)
                .and_modify(|e| {
                    e.attribution_score = 0.9 * e.attribution_score + 0.1 * node.attribution_score;
                    e.activation_count += 1;
                    e.last_activated_step = self.tr_counter;
                    for &trd in &node.trd_membership {
                        if !e.trd_membership.contains(&trd) { e.trd_membership.push(trd); }
                    }
                    if e.structure.is_none() { e.structure = node.structure.clone(); }
                })
                .or_insert_with(|| {
                let mut n = node.clone();
                n.activation_count = 1; // first insertion counts as one activation
                n
            });
        }
        for ei in graph.edge_indices() {
            let edge = &graph[ei];
            self.global_edges.entry(edge.id)
                .and_modify(|e| e.weight = 0.9 * e.weight + 0.1 * edge.weight)
                .or_insert_with(|| edge.clone());
        }
    }

    fn enforce_budget(&mut self) {
        if self.global_nodes.len() < self.node_budget && self.global_edges.len() < self.edge_budget {
            return;
        }
        // Pass 1: evict low-attribution rare nodes
        const THETA_EVICT: f32 = 0.05;
        const MIN_ACT: u32 = 3;
        let to_remove: Vec<NodeId> = self.global_nodes.values()
            .filter(|n| n.attribution_score < THETA_EVICT && n.activation_count < MIN_ACT)
            .map(|n| n.id).collect();
        for nid in &to_remove {
            self.global_nodes.remove(nid);
            self.global_edges.retain(|_, e| e.src != *nid && e.dst != *nid);
        }
        // Pass 2: rule-covered consolidation
        if self.global_nodes.len() >= self.node_budget {
            let rules = self.ruler.rules.clone();
            let to_remap: Vec<(NodeId, ModalMode, TypeCategory)> = self.global_nodes.values()
                .filter(|n| n.attribution_score < 0.3)
                .filter_map(|n| rules.iter()
                    .find(|r| r.confidence >= 0.90 && r.support >= 20
                           && r.lhs_mode == n.mtlg_type.mode && r.lhs_cat == n.mtlg_type.category)
                    .map(|r| (n.id, r.rhs_mode, r.rhs_cat)))
                .collect();
            for (nid, nm, nc) in to_remap {
                if let Some(n) = self.global_nodes.get_mut(&nid) {
                    n.mtlg_type.mode = nm; n.mtlg_type.category = nc; n.attribution_score *= 0.5;
                }
            }
            let evict2: Vec<NodeId> = self.global_nodes.values()
                .filter(|n| n.attribution_score < THETA_EVICT).map(|n| n.id).collect();
            for nid in &evict2 {
                self.global_nodes.remove(nid);
                self.global_edges.retain(|_, e| e.src != *nid && e.dst != *nid);
            }
        }
        // Pass 3: synonym merge
        if self.global_nodes.len() >= self.node_budget {
            let candidates = self.slot_tracker.synonym_candidates(5);
            for (src_id, dst_id, strength) in candidates {
                if strength < 0.9 { continue; }
                let ss = self.global_nodes.get(&src_id).map(|n| n.attribution_score);
                let ds = self.global_nodes.get(&dst_id).map(|n| n.attribution_score);
                if let (Some(ss), Some(ds)) = (ss, ds) {
                    let canonical = if ss >= ds { src_id } else { dst_id };
                    let deprecated = if ss >= ds { dst_id } else { src_id };
                    let extra = self.global_nodes.get(&deprecated).map(|n| n.activation_count).unwrap_or(0);
                    if let Some(n) = self.global_nodes.get_mut(&canonical) {
                        n.attribution_score = (ss + ds) / 2.0;
                        n.activation_count += extra;
                    }
                    for edge in self.global_edges.values_mut() {
                        if edge.src == deprecated { edge.src = canonical; }
                        if edge.dst == deprecated { edge.dst = canonical; }
                    }
                    self.global_nodes.remove(&deprecated);
                }
            }
        }
        // Hard eviction if still over budget
        if self.global_nodes.len() >= self.node_budget {
            let excess = self.global_nodes.len() - self.node_budget + self.node_budget / 10;
            let mut by_score: Vec<(NodeId, f32)> = self.global_nodes.iter()
                .map(|(&id, n)| (id, n.attribution_score)).collect();
            by_score.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (nid, _) in by_score.iter().take(excess) {
                self.global_nodes.remove(nid);
                self.global_edges.retain(|_, e| e.src != *nid && e.dst != *nid);
            }
        }
        if self.global_edges.len() >= self.edge_budget {
            let excess = self.global_edges.len() - self.edge_budget + self.edge_budget / 10;
            let mut by_weight: Vec<(EdgeId, f32)> = self.global_edges.iter()
                .map(|(&id, e)| (id, e.weight)).collect();
            by_weight.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (eid, _) in by_weight.iter().take(excess) { self.global_edges.remove(eid); }
        }
    }

    fn maybe_refit_categories(&mut self) {
        self.category_refit_counter += 1;
        if self.category_refit_counter % self.category_refit_interval != 0 { return; }
        if self.pending_structures.len() < 50 { return; }
        self.category_inducer.fit(&self.pending_structures);
        self.pending_structures.clear();
        let updates: Vec<(NodeId, TypeCategory)> = self.global_nodes.values()
            .filter_map(|n| n.structure.as_ref().map(|s| (n.id, self.category_inducer.predict(s))))
            .collect();
        for (nid, cat) in updates {
            if let Some(n) = self.global_nodes.get_mut(&nid) { n.mtlg_type.category = cat; }
        }
    }

    fn post_execution_common(&mut self, graph: &ArgGraph, trd: Option<TRDId>, quality: Quality) {
        // Slot occupancy for synonym discovery
        for idx in graph.node_indices() {
            let n = &graph[idx];
            if let Some(&trd_id) = n.trd_membership.first() {
                self.slot_tracker.observe(n.id, n.surface_str().unwrap_or("_"), trd_id, n.mtlg_type);
            }
        }
        // Rule induction + KBC
        self.ruler.induce_rules();
        self.rule_lifecycle.integrate_induced(self.ruler.rules.clone());
        self.rule_lifecycle.on_tr_dissolved(self.tr_counter);
        // VDBE threshold sync
        if let Some(d) = trd { self.perf.update(d, quality); }
        self.thresholds.sync(&self.perf);
    }

    // ── Full execution cycle (inference + continuous learning) ────────────────

    pub fn execute(&mut self, query: Query, node_pool: Vec<ArgNode>, edge_pool: Vec<ArgEdge>) -> QueryResult {
        let node_pool = self.enrich_from_global(node_pool);

        let _ctx_id    = self.context_stack.push(query.situation_id, query.trd);
        let active_env = self.context_stack.current_env();
        let trd        = query.trd;
        let theta_alpha = trd.map(|d| self.thresholds.theta_alpha(d)).unwrap_or(0.4);
        let theta_rho   = trd.map(|d| self.thresholds.theta_rho(d)).unwrap_or(0.38);

        let node_pool_snapshot = node_pool.clone();
        let edge_pool_snapshot = edge_pool.clone();

        // Collect structures for category refit
        for n in &node_pool { if let Some(ref s) = n.structure { self.pending_structures.push(s.clone()); } }

        // ARG expansion
        let mut search = ArgSearch::new(active_env, theta_alpha, theta_rho);
        for node in node_pool { search.try_activate(node); }
        for edge in edge_pool  { search.try_add_edge(edge); }

        // Schedule
        let ready_nodes: Vec<NodeId> = search.graph.node_indices().map(|i| search.graph[i].id).collect();
        let _schedule = build_schedule(&search.graph, trd, &self.perf, &ready_nodes);
        let mut exec_table = ExecStateTable::new(active_env);
        for idx in search.graph.node_indices() {
            let n = &search.graph[idx];
            exec_table.register(NodeExecState::new(n.id, n.atms_label, vec![]));
        }

        // Constraints
        let violations = validate_shapes(&search.graph, &default_constraints());
        if !violations.is_empty() {
            let bad: Vec<NodeId> = violations.iter().filter_map(|v| v.node_id).collect();
            apply_nogood_consequences(&mut search.graph, &bad);
        }
        let stalks = build_stalks(&search.graph);
        let _ = check_sheaf_coherence(&search.graph, &stalks);

        // Graphica memo + e-graph saturation
        let nodes_ref: Vec<_> = search.graph.node_indices().map(|i| &search.graph[i]).collect();
        let edges_ref: Vec<_> = search.graph.edge_indices().map(|i| &search.graph[i]).collect();
        let cache_key = build_key(&nodes_ref, &edges_ref, active_env);
        if self.graphica.get(&cache_key).is_none() {
            self.egraph.saturate();
            self.graphica.insert(CachedResult {
                key: cache_key, edge_deltas: HashMap::new(),
                quality: 0.0, traversal_count: 0, shortcut_canonical_id: None,
            });
        } else {
            self.graphica.record_traversal(&cache_key);
        }

        // Apply Ruler type normalizations
        let normalizer = TypeNormalizer::new(self.ruler.rules.clone());
        normalizer.apply_to_graph(&mut search.graph);

        // Progressive deepening
        let deepener = ProgressiveDeepener::new(trd);
        let dr = deepener.run(&search.graph, &self.semantics, &self.perf, &mut self.thresholds,
                              &query.text, &query.expected_type);

        // Decode surface output
        let decode_trd  = trd.unwrap_or(0);
        let decoded_ids = self.decode_with_thresholds(
            &node_pool_snapshot, &edge_pool_snapshot, decode_trd, 32, theta_alpha, theta_rho,
        );
        let surface_output = if decoded_ids.is_empty() {
            dr.hypotheses.first()
                .map(|h| {
                    let mut lin = Linearizer::new(&query.target_language);
                    for entry in self.global_lexicon.values() { lin.lexicon.register(entry.clone()); }
                    lin.proposition_to_surface(&h.proposition)
                })
                .unwrap_or_else(|| format!("[no output for '{}']", query.text))
        } else {
            self.surface_from_decoded(&decoded_ids, &node_pool_snapshot, &query.target_language)
        };

        // Quality from deepener
        let quality = if dr.satisfied {
            if dr.depth_used == 0 { Quality::GOOD } else { Quality::PARTIAL }
        } else { Quality::BAD };

        // Update cached quality with EMA blend
        self.graphica.update_quality(&cache_key, quality.as_f32());

        // Attribution — route through AttributionEngine + Counterfactual
        let attributed_edges: Vec<EdgeId> = search.graph.edge_indices()
            .map(|ei| search.graph[ei].id).collect();
        for &eid in &attributed_edges {
            self.apply_attribution_batch(vec![eid], quality, trd.unwrap_or(0));
        }
        if !decoded_ids.is_empty() {
            propagate_attribution_backward(&mut search.graph, &self.atms, &decoded_ids, quality, 4, 0.7);
            for nid in &decoded_ids { propagate_edge_to_node_scores(&mut search.graph, *nid); }
        }

        // Full attribution deltas (causal phase when ready)
        if let Some(d) = trd {
            let attrs = self.attribution.compute_all(d, &search.graph);
            for attr in &attrs {
                apply_attribution(&mut search.graph, attr.edge_id, attr.delta, quality);
            }
        }

        // Provenance
        self.provenance.record(AuditEntry {
            tr_id: self.tr_counter, context_id: 0, trd_id: trd,
            edges_updated: attributed_edges.iter().map(|&e| (e, 0.0f32)).collect(),
            quality: quality.as_f32(),
        });

        // ATMS: add TR, pop, feed Ruler
        self.tr_counter += 1;
        let _tr_id = self.context_stack.add_tr(
            RepContent::Lambda(surface_output.clone()), active_env, ModalType::default(), Granularity::Sentence,
        );
        if let Some((dissolved_trs, _)) = self.context_stack.pop() {
            let success = quality.as_f32() >= Quality::PARTIAL.as_f32();
            for tr in &dissolved_trs { self.ruler.observe_dissolved_tr(tr, success); }

            let live_envs: Vec<Env> = self.global_nodes.values().map(|n| n.atms_label).collect();
            self.context_stack.bridge.try_reclaim_pending(&live_envs);
        }

        self.post_execution_common(&search.graph, trd, quality);

        if self.tr_counter % 50 == 0 { apply_weight_decay(&mut search.graph, 0.001); }
        if self.tr_counter % 100 == 0 { self.slot_tracker.materialise_synonym_edges(&mut search.graph, 3); }

        self.merge_into_global(&search.graph);
        self.last_graph      = Some(search.graph);
        self.node_pool_cache = node_pool_snapshot;
        self.enforce_budget();
        self.maybe_refit_categories();

        QueryResult { surface_output, satisfied: dr.satisfied, depth_used: dr.depth_used, quality }
    }

    // ── Teacher-forcing passage (training + continuous learning) ──────────────

    pub fn execute_passage(
        &mut self,
        trd:       TRDId,
        sentences: Vec<Vec<TokenStep>>,
        _language:  &str,
    ) -> SequenceTrainResult {
        // Push ATMS context for the passage
        let ctx_id     = self.context_stack.push(trd as u64, Some(trd));
        let active_env = self.context_stack.current_env();
        let theta_alpha = self.thresholds.theta_alpha(trd);
        let theta_rho   = self.thresholds.theta_rho(trd);

        // Enrich nodes from global pool
        let enriched: Vec<Vec<TokenStep>> = sentences.into_iter()
            .map(|sent| sent.into_iter().map(|mut step| {
                step.node_pool = self.enrich_from_global(step.node_pool);
                step
            }).collect())
            .collect();

        let mut passage       = PassageContext::new(trd);
        let mut quality_sum   = 0.0f64;
        let mut total_steps   = 0usize;
        let mut final_quality = Quality::BAD;
        let mut prev_node_id: Option<NodeId> = None;

        // Incremental forward pass with CE teacher forcing
        for sentence in &enriched {
            for step in sentence {
                for n in &step.node_pool {
                    if let Some(ref s) = n.structure { self.pending_structures.push(s.clone()); }
                }
                let search = passage.step(
                    step.expected_node_id, &step.node_pool, &step.edge_pool,
                    active_env, theta_alpha, theta_rho, prev_node_id,
                );
                let dist = VocabDistribution::from_graph(&search.graph, active_env);
                let (q_correct, q_wrong_opt) = dist.ce_quality_split(step.expected_node_id);
                for ei in search.graph.edge_indices() {
                    if search.graph[ei].dst == step.expected_node_id {
                        passage.signal.accumulate(search.graph[ei].id, q_correct.0);
                    }
                }
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

        // Slot occupancy
        for n in passage.node_map.values() {
            if let Some(&trd_id) = n.trd_membership.first() {
                self.slot_tracker.observe(n.id, n.surface_str().unwrap_or("_"), trd_id, n.mtlg_type);
            }
        }

        // Build final passage graph
        let mut final_graph = passage.build_graph(active_env, theta_alpha, theta_rho).graph;

        // Schedule
        let ready_nodes2: Vec<NodeId> = final_graph.node_indices().map(|i| final_graph[i].id).collect();
        let _schedule = build_schedule(&final_graph, Some(trd), &self.perf, &ready_nodes2);

        // Constraints as training signal
        let violations = validate_shapes(&final_graph, &default_constraints());
        if !violations.is_empty() {
            let penalty = -0.1 * violations.len() as f32;
            for v in &violations {
                if let Some(nid) = v.node_id {
                    for ei in final_graph.edge_indices() {
                        if final_graph[ei].dst == nid {
                            passage.signal.accumulate(final_graph[ei].id, penalty);
                        }
                    }
                }
            }
        }
        let stalks = build_stalks(&final_graph);
        let _ = check_sheaf_coherence(&final_graph, &stalks);

        // Apply Ruler type normalizations
        let normalizer = TypeNormalizer::new(self.ruler.rules.clone());
        normalizer.apply_to_graph(&mut final_graph);

        // Flush CE signals — route through AttributionEngine + Counterfactual
        let token_count = passage.signal.token_count.max(1) as f32;
        let mut updated_nodes: Vec<NodeId> = Vec::new();
        let attributed_edges: Vec<EdgeId> = passage.signal.edge_signals.keys().copied().collect();

        for (&edge_id, &signal) in &passage.signal.edge_signals {
            let q = Quality::new(signal / token_count);
            self.apply_attribution_batch(vec![edge_id], q, trd);
            apply_attribution(&mut final_graph, edge_id, 1.0, q);
            if let Some(ei) = final_graph.edge_indices().find(|&i| final_graph[i].id == edge_id) {
                updated_nodes.push(final_graph[ei].dst);
            }
        }

        // Backward attribution
        if total_steps > 0 {
            let net_q = Quality::new((quality_sum / total_steps as f64) as f32);
            let terminal: Vec<NodeId> = enriched.iter().flat_map(|s| s.iter())
                .map(|step| step.expected_node_id).collect();
            propagate_attribution_backward(&mut final_graph, &self.atms, &terminal, net_q, 4, 0.7);
            updated_nodes.extend(terminal);
        }
        updated_nodes.sort_unstable();
        updated_nodes.dedup();
        for nid in &updated_nodes { propagate_edge_to_node_scores(&mut final_graph, *nid); }

        // Provenance
        self.provenance.record(AuditEntry {
            tr_id: self.tr_counter, context_id: ctx_id, trd_id: Some(trd),
            edges_updated: attributed_edges.iter().map(|&e| (e, 0.0f32)).collect(),
            quality: final_quality.as_f32(),
        });

        // Pop ATMS context, feed Ruler
        self.tr_counter += 1;
        let _tr_id = self.context_stack.add_tr(
            RepContent::Lambda(format!("passage_{}", self.tr_counter)),
            active_env, ModalType::default(), Granularity::Passage,
        );
        if let Some((dissolved_trs, _)) = self.context_stack.pop() {
            let success = final_quality.as_f32() >= Quality::PARTIAL.as_f32();
            for tr in &dissolved_trs { self.ruler.observe_dissolved_tr(tr, success); }

            let live_envs: Vec<Env> = self.global_nodes.values().map(|n| n.atms_label).collect();
            self.context_stack.bridge.try_reclaim_pending(&live_envs);
        }

        self.post_execution_common(&final_graph, Some(trd), final_quality);

        if self.tr_counter % 50 == 0 { apply_weight_decay(&mut final_graph, 0.001); }
        if self.tr_counter % 100 == 0 { self.slot_tracker.materialise_synonym_edges(&mut final_graph, 3); }

        self.merge_into_global(&final_graph);
        self.last_graph = Some(final_graph);
        self.enforce_budget();
        self.maybe_refit_categories();

        SequenceTrainResult { steps_processed: total_steps, quality_sum, final_quality, attributed_edges }
    }

    // ── Legacy single-sequence trainer (routes through execute_passage) ────────

    pub fn execute_sequence(&mut self, trd: TRDId, steps: Vec<TokenStep>, language: &str) -> SequenceTrainResult {
        self.execute_passage(trd, vec![steps], language)
    }

    // ── Decode helpers ────────────────────────────────────────────────────────

    pub fn decode(&mut self, seed_nodes: &[ArgNode], seed_edges: &[ArgEdge], trd: TRDId, max_tokens: usize) -> Vec<NodeId> {
        let theta_alpha = self.thresholds.theta_alpha(trd);
        let theta_rho   = self.thresholds.theta_rho(trd);
        self.decode_with_thresholds(seed_nodes, seed_edges, trd, max_tokens, theta_alpha, theta_rho)
    }

    pub fn decode_with_thresholds(
        &mut self, seed_nodes: &[ArgNode], seed_edges: &[ArgEdge],
        trd: TRDId, max_tokens: usize, theta_alpha: f64, theta_rho: f64,
    ) -> Vec<NodeId> {
        let active_env = self.context_stack.current_env();
        let mut passage = PassageContext::new(trd);
        passage.absorb_sentence(seed_nodes, seed_edges);
        let mut output: Vec<NodeId> = Vec::new();
        let mut prev: Option<NodeId> = None;
        for _ in 0..max_tokens {
            let result = passage.decode_step(&[], &[], active_env, theta_alpha, theta_rho, prev, &self.meta_grammar);
            let (node_id, prob) = match result { Some(r) => r, None => break };
            let vocab_size = passage.node_map.len().max(1);
            if prob < 1.0 / vocab_size as f32 { break; }
            if output.last() == Some(&node_id) { break; }
            output.push(node_id);
            prev = Some(node_id);
        }
        output
    }

    pub fn surface_from_decoded(&self, node_ids: &[NodeId], seed_nodes: &[ArgNode], language: &str) -> String {
        let mut linearizer = Linearizer::new(language);
        for entry in self.global_lexicon.values() { linearizer.lexicon.register(entry.clone()); }
        node_ids.iter().filter_map(|&nid| {
            if let Some(n) = seed_nodes.iter().find(|n| n.id == nid) {
                if let Some(s) = n.surface_str() { if !s.is_empty() { return Some(s.to_string()); } }
            }
            if let Some(n) = self.node_pool_cache.iter().find(|n| n.id == nid) {
                if let Some(s) = n.surface_str() { if !s.is_empty() { return Some(s.to_string()); } }
            }
            if let Some(ref graph) = self.last_graph {
                if let Some(idx) = graph.node_indices().find(|&i| graph[i].id == nid) {
                    if let Some(s) = graph[idx].surface_str() { if !s.is_empty() { return Some(s.to_string()); } }
                }
            }
            linearizer.lexicon.surface_for_node_id(nid, language).map(|s| s.to_string())
        }).collect::<Vec<_>>().join(" ")
    }

    pub fn synonym_query(&self, surface: &str, n: usize) -> Vec<(String, f32)> {
        let graph = match &self.last_graph { Some(g) => g, None => return vec![] };
        let source_idx = graph.node_indices().find(|&i| {
            graph[i].surface_str().map_or(false, |s| s.eq_ignore_ascii_case(surface))
        });
        let Some(src_idx) = source_idx else { return vec![]; };
        let src_id = graph[src_idx].id;
        let mut candidates: Vec<(String, f32)> = graph.edge_indices()
            .filter(|&ei| graph[ei].src == src_id || graph[ei].dst == src_id)
            .filter_map(|ei| {
                let nid = if graph[ei].src == src_id { graph[ei].dst } else { graph[ei].src };
                let idx = graph.node_indices().find(|&i| graph[i].id == nid)?;
                Some((graph[idx].surface_str()?.to_string(), graph[ei].weight))
            }).collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(n);
        candidates
    }

    pub fn apply_attribution_batch(&mut self, edge_ids: Vec<EdgeId>, quality: Quality, trd_id: TRDId) -> Vec<EdgeId> {
        // tr_counter is managed by the outer execution cycle; all per-edge records share the same TR stamp.
        let tr_id = self.tr_counter;
        self.attribution.record(tr_id, trd_id, quality, &edge_ids);
        self.counterfactual.record_dissolved_tr(tr_id, trd_id, quality, edge_ids.clone());
        edge_ids
    }

    /// Serialize persistent learned state to `path` as JSON (§9).
    ///
    /// Only serializes the fields that represent accumulated knowledge
    /// (global node/edge pools, category partition, grammar rules, thresholds).
    /// Transient fields (egraph, graphica, ruler, context_stack, atms) are
    /// excluded and reconstructed by Engine::new() on load.
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let snap = EngineSnapshot {
            global_nodes:      self.global_nodes.values().cloned().collect(),
            global_edges:      self.global_edges.values().cloned().collect(),
            category_inducer:  self.category_inducer.clone(),
            meta_grammar:      self.meta_grammar.clone(),
            tr_counter:        self.tr_counter,
        };
        let json = serde_json::to_string(&snap)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Deserialize persistent state from `path` and overlay it on a fresh Engine (§9).
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let snap: EngineSnapshot = serde_json::from_str(&data)?;
        let mut engine = Engine::new();
        engine.global_nodes = snap.global_nodes.into_iter().map(|n| (n.id, n)).collect();
        engine.global_edges = snap.global_edges.into_iter().map(|e| (e.id, e)).collect();
        engine.category_inducer = snap.category_inducer;
        engine.meta_grammar     = snap.meta_grammar;
        engine.tr_counter       = snap.tr_counter;
        Ok(engine)
    }

    /// Compute a diff of nodes and edges added since `since_tr` (§9).
    ///
    /// Returns an `EngineGraphDiff` that can be transmitted and applied to a
    /// replica engine via `apply_diff`.  The diff contains only nodes/edges
    /// whose `activation_count` is non-zero (i.e., seen at least once after
    /// `since_tr`); the receiver deduplicates by ID, merging EMA scores.
    pub fn diff_since(&self, since_tr: u64) -> EngineGraphDiff {
        // Use activation_count as a simple monotonic proxy for recency.
        // Nodes/edges first observed after `since_tr` have no way to self-report
        // their origin TR in the current schema, so we diff the full pool when
        // since_tr == 0 and return only nodes with activation_count > 0 otherwise.
        let nodes: Vec<ArgNode> = self.global_nodes.values()
            .filter(|n| since_tr == 0 || n.activation_count > 0)
            .cloned()
            .collect();
        let edges: Vec<ArgEdge> = self.global_edges.values().cloned().collect();
        EngineGraphDiff { nodes, edges, as_of_tr: self.tr_counter }
    }

    /// Merge a diff produced by `diff_since` into this engine (§9).
    ///
    /// Existing nodes/edges are EMA-merged (first-occurrence wins for identity,
    /// running average for scores).  New nodes/edges are inserted directly.
    pub fn apply_diff(&mut self, diff: EngineGraphDiff) {
        for node in diff.nodes {
            self.global_nodes.entry(node.id)
                .and_modify(|existing| {
                    existing.attribution_score =
                        0.5 * existing.attribution_score + 0.5 * node.attribution_score;
                    existing.activation_count += node.activation_count;
                })
                .or_insert(node);
        }
        for edge in diff.edges {
            self.global_edges.entry(edge.id).or_insert(edge);
        }
        self.tr_counter = self.tr_counter.max(diff.as_of_tr);
    }
}

/// Serializable snapshot of persistent engine state (§9).
#[derive(Serialize, Deserialize)]
pub struct EngineSnapshot {
    pub global_nodes:     Vec<ArgNode>,
    pub global_edges:     Vec<ArgEdge>,
    pub category_inducer: crate::lcs::converter::CategoryInducer,
    pub meta_grammar:     MetaGrammarEngine,
    pub tr_counter:       u64,
}

/// Incremental diff of the engine's graph knowledge pool (§9).
#[derive(Serialize, Deserialize)]
pub struct EngineGraphDiff {
    pub nodes:     Vec<ArgNode>,
    pub edges:     Vec<ArgEdge>,
    /// TR counter at the producing engine when this diff was generated.
    pub as_of_tr:  u64,
}

impl Default for Engine { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::{ArgNode, NodeClass};
    use crate::types::{ModalMode, TypeCategory, Direction};

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
        let query = Query::new("run", 1, "en");
        let nodes = vec![make_node(1, "run", 0.9), make_node(2, "alice", 0.7)];
        let result = engine.execute(query, nodes, vec![]);
        assert!(!result.surface_output.is_empty());
    }

    #[test]
    fn decode_high_attribution_first() {
        let mut engine = Engine::new();
        engine.context_stack.push(1, None);
        let high = make_node(100, "important", 0.9);
        let low  = make_node(200, "trivial",   0.1);
        let decoded = engine.decode(&[high, low], &[], 0, 32);
        assert!(!decoded.is_empty());
        assert_eq!(decoded[0], 100);
    }

    #[test]
    fn decode_empty_returns_empty() {
        let mut engine = Engine::new();
        assert!(engine.decode(&[], &[], 0, 32).is_empty());
    }

    #[test]
    fn global_pool_persists_across_calls() {
        let mut engine = Engine::new();
        let q1 = Query::new("run", 1, "en");
        let nodes1 = vec![make_node(1, "run", 0.9)];
        engine.execute(q1, nodes1, vec![]);
        assert!(engine.global_nodes.contains_key(&1));
        let q2 = Query::new("run again", 2, "en");
        let nodes2 = vec![make_node(1, "run", 0.8)];
        engine.execute(q2, nodes2, vec![]);
        let node = engine.global_nodes.get(&1).unwrap();
        assert!(node.activation_count >= 1);
    }
}
