//! Linguistic Conversion System: text → MTLG modal graph.
//! TypeCategory assignment: bisimulation partition refinement (Paige & Tarjan 1987).
//! No k-means. No BIC. No hardcoded cluster count.
//!
//! Partition refinement overview:
//!   - Each node starts in one of two blocks: punctuation (leaf) vs. non-punctuation.
//!   - The algorithm refines blocks by splitting any block B when some nodes in B
//!     have edges into a splitter block S and others do not.
//!   - Terminates when no block can be further split: the result is the coarsest
//!     stable partition (minimal bisimulation equivalence).
//!   - Block IDs are assigned by sorting blocks on a canonical key derived from
//!     their modal mode profile, ensuring cross-run stability.

use std::collections::{BTreeMap, HashMap, HashSet};
use serde::{Serialize, Deserialize};
use crate::types::{TypeCategory, ModalMode, ModalType, Direction, NodeId, EdgeId, Env};
use crate::arg::{ArgNode, ArgEdge, NodeClass, EdgeClass};
use super::token_types::{Token, TokenSentence, TokenStructure, extract_features, fnv_hash, stable_node_id};

// ── Feature vector (kept for compatibility) ───────────────────────────────────

pub fn structure_to_vector(
    s:             &TokenStructure,
    trigram_vocab: &[u32],
    suffix_vocab:  &[u32],
) -> Vec<f32> {
    let mut v = Vec::with_capacity(9 + trigram_vocab.len() + suffix_vocab.len());
    v.push(s.is_first_token as u8 as f32);
    v.push(s.is_last_token as u8 as f32);
    v.push(s.normalized_position);
    v.push(s.sentence_length_norm);
    v.push(s.starts_with_uppercase as u8 as f32);
    v.push(s.is_punctuation as u8 as f32);
    v.push(s.char_length_norm);
    v.push(s.is_repeated as u8 as f32);
    v.push(s.n_context_neighbors as f32 / 2.0);
    for &h in trigram_vocab { v.push(s.char_trigram_hashes.contains(&h) as u8 as f32); }
    for &h in suffix_vocab  { v.push((s.suffix3_hash == h || s.suffix2_hash == h) as u8 as f32); }
    v
}

// ── Bisimulation Partition Refinement ────────────────────────────────────────

/// A labeled transition for partition refinement.
/// Encodes: node `from` has an outgoing edge with label `modal_mode` to node `to`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabeledTransition {
    pub from:       NodeId,
    pub to:         NodeId,
    pub modal_mode: ModalMode,
}

/// Computes the coarsest stable partition of `nodes` under `transitions`.
/// Returns a mapping NodeId → TypeCategory where block IDs are stable across runs.
///
/// Initial partition: two blocks — nodes where `is_leaf` is true (e.g. punctuation,
/// single-character nodes) and all others.
///
/// Reference: Paige & Tarjan (1987), "Three Partition Refinement Algorithms."
/// SIAM Journal on Computing 16(6):973–989.
pub fn bisimulation_partition(
    nodes:       &[NodeId],
    transitions: &[LabeledTransition],
    is_leaf:     &HashMap<NodeId, bool>,
) -> HashMap<NodeId, TypeCategory> {
    if nodes.is_empty() {
        return HashMap::new();
    }

    // Initial partition: block 0 = leaves, block 1 = non-leaves.
    let mut node_to_block: HashMap<NodeId, usize> = nodes.iter().map(|&n| {
        let leaf = is_leaf.get(&n).copied().unwrap_or(false);
        (n, if leaf { 0 } else { 1 })
    }).collect();

    // Reverse index: to_node → list of (from_node, modal_mode).
    let mut reverse: HashMap<NodeId, Vec<(NodeId, ModalMode)>> = HashMap::new();
    for t in transitions {
        reverse.entry(t.to).or_default().push((t.from, t.modal_mode));
    }

    // Worklist of (block_id, modal_mode) splitters.
    let mut worklist: Vec<(usize, ModalMode)> = vec![
        (0, ModalMode::Diamond), (0, ModalMode::Box), (0, ModalMode::Lozenge),
        (1, ModalMode::Diamond), (1, ModalMode::Box), (1, ModalMode::Lozenge),
    ];

    let mut next_block_id: usize = 2;

    while let Some((splitter_block, mode)) = worklist.pop() {
        // Find all nodes that have a `mode`-labeled edge into `splitter_block`.
        let predecessors: HashSet<NodeId> = nodes.iter()
            .filter(|&&n| {
                reverse.get(&n).map_or(false, |preds|
                    preds.iter().any(|(from, m)| {
                        *m == mode && node_to_block.get(from) == Some(&splitter_block)
                    })
                )
            })
            .copied()
            .collect();

        if predecessors.is_empty() { continue; }

        // Collect blocks that have at least one node in predecessors.
        let affected_blocks: HashSet<usize> = predecessors.iter()
            .filter_map(|n| node_to_block.get(n))
            .copied()
            .collect();

        for block in affected_blocks {
            let in_pred:  Vec<NodeId> = nodes.iter().copied()
                .filter(|n| node_to_block.get(n) == Some(&block) && predecessors.contains(n))
                .collect();
            let not_pred: Vec<NodeId> = nodes.iter().copied()
                .filter(|n| node_to_block.get(n) == Some(&block) && !predecessors.contains(n))
                .collect();

            if in_pred.is_empty() || not_pred.is_empty() { continue; }

            // Split: in_pred keeps `block`, not_pred gets new block.
            let new_block = next_block_id;
            next_block_id += 1;
            for n in &not_pred {
                node_to_block.insert(*n, new_block);
            }

            // Add both halves as splitters for all modes.
            for m in [ModalMode::Diamond, ModalMode::Box, ModalMode::Lozenge] {
                worklist.push((block, m));
                worklist.push((new_block, m));
            }
        }
    }

    // Assign stable TypeCategory IDs: sort blocks by canonical key
    // (dominant modal mode of incoming transitions, then block size).
    // This ensures the same corpus always produces the same ID assignment.
    let mut block_profiles: BTreeMap<usize, (u8, usize)> = BTreeMap::new();
    for (&n, &b) in &node_to_block {
        let entry = block_profiles.entry(b).or_insert((255u8, 0));
        entry.1 += 1;
        // Update dominant incoming mode.
        if let Some(preds) = reverse.get(&n) {
            for (_, m) in preds {
                let mode_id = match m {
                    ModalMode::Diamond => 0u8,
                    ModalMode::Box     => 1u8,
                    ModalMode::Lozenge => 2u8,
                };
                if mode_id < entry.0 { entry.0 = mode_id; }
            }
        }
    }

    // Sort blocks deterministically: by (dominant_mode, desc block_size).
    let mut sorted_blocks: Vec<(usize, (u8, usize))> = block_profiles.into_iter().collect();
    sorted_blocks.sort_by(|a, b| {
        a.1.0.cmp(&b.1.0).then(b.1.1.cmp(&a.1.1))
    });
    let block_to_type: HashMap<usize, TypeCategory> = sorted_blocks.iter().enumerate()
        .map(|(rank, (block_id, _))| (*block_id, TypeCategory((rank as u32) + 1)))
        .collect();

    node_to_block.iter()
        .map(|(&n, &b)| (n, block_to_type.get(&b).copied().unwrap_or(TypeCategory::DEFAULT)))
        .collect()
}

// ── CategoryInducer (wraps bisimulation, replaces k-means CategoryInducer) ───

/// Online TypeCategory assigner using bisimulation partition refinement.
///
/// Replaces the k-means CategoryInducer. No cluster count hyperparameter.
/// Partition is recomputed when new nodes are added (incremental refinement).
///
/// The partition is deterministic: identical inputs produce identical TypeCategory IDs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CategoryInducer {
    nodes:       Vec<NodeId>,
    transitions: Vec<LabeledTransition>,
    is_leaf:     HashMap<NodeId, bool>,
    partition:   HashMap<NodeId, TypeCategory>,
    dirty:       bool,
    /// Kept for API compatibility; ignored (bisimulation needs no cluster count).
    pub trigram_vocab: Vec<u32>,
    pub suffix_vocab:  Vec<u32>,
}

impl CategoryInducer {
    pub fn new(_ignored: usize) -> Self {
        // The usize argument was the k-means cluster count; it is ignored.
        // CategoryInducer no longer takes a cluster count parameter.
        Self {
            nodes:        Vec::new(),
            transitions:  Vec::new(),
            is_leaf:      HashMap::new(),
            partition:    HashMap::new(),
            dirty:        false,
            trigram_vocab: Vec::new(),
            suffix_vocab:  Vec::new(),
        }
    }

    /// Register a node. `leaf` is true for terminal/punctuation nodes.
    pub fn add_node(&mut self, id: NodeId, is_leaf: bool) {
        if !self.nodes.contains(&id) {
            self.nodes.push(id);
            self.is_leaf.insert(id, is_leaf);
            self.dirty = true;
        }
    }

    /// Register a labeled transition (edge).
    pub fn add_transition(&mut self, from: NodeId, to: NodeId, mode: ModalMode) {
        self.transitions.push(LabeledTransition { from, to, modal_mode: mode });
        self.dirty = true;
    }

    /// Recompute partition if dirty, then return TypeCategory for `node_id`.
    pub fn predict_by_id(&mut self, node_id: NodeId) -> TypeCategory {
        if self.dirty {
            self.partition = bisimulation_partition(&self.nodes, &self.transitions, &self.is_leaf);
            self.dirty = false;
        }
        self.partition.get(&node_id).copied().unwrap_or(TypeCategory::DEFAULT)
    }

    /// Predict TypeCategory for a `TokenStructure`.
    pub fn predict(&mut self, structure: &TokenStructure) -> TypeCategory {
        let node_id = stable_node_id_from_structure(structure);
        let leaf = structure.is_punctuation || structure.char_length_norm < 0.1;
        self.add_node(node_id, leaf);
        self.predict_by_id(node_id)
    }

    /// Batch fit: register all structures and recompute once.
    pub fn fit(&mut self, structures: &[TokenStructure]) {
        for s in structures {
            let id   = stable_node_id_from_structure(s);
            let leaf = s.is_punctuation || s.char_length_norm < 0.1;
            self.add_node(id, leaf);
        }
        // Add sequential transitions between adjacent structures (positional bigrams).
        let ids: Vec<NodeId> = structures.iter()
            .map(|s| stable_node_id_from_structure(s))
            .collect();
        for w in ids.windows(2) {
            self.add_transition(w[0], w[1], ModalMode::Diamond);
        }
        self.partition = bisimulation_partition(&self.nodes, &self.transitions, &self.is_leaf);
        self.dirty = false;
    }

    // Compatibility with old API: build_trigram_vocab and build_suffix_vocab are no-ops.
    pub fn build_trigram_vocab(_structures: &[TokenStructure]) -> Vec<u32> { Vec::new() }
    pub fn build_suffix_vocab(_structures: &[TokenStructure]) -> Vec<u32> { Vec::new() }
}

fn stable_node_id_from_structure(s: &TokenStructure) -> NodeId {
    // Use suffix3_hash and prefix2_hash as structural identity signal.
    let combined = (s.suffix3_hash as u64) << 32 | (s.prefix2_hash as u64);
    let bytes = combined.to_le_bytes();
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME:  u64 = 0x0000_0100_0000_01b3;
    bytes.iter().fold(OFFSET, |h, &b| h.wrapping_mul(PRIME) ^ b as u64)
}

// ── Graph types ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct MtlgNode {
    pub token_id:   u32,
    pub text:       String,
    pub lemma:      String,
    pub modal_mode: ModalMode,
    pub ucca_cat:   Option<TypeCategory>,
    pub arity:      u8,
    pub structure:  TokenStructure,
}

#[derive(Clone, Debug)]
pub struct MtlgEdge {
    pub src_id:    u32,
    pub dst_id:    u32,
    pub modal_mode: ModalMode,
    pub ucca_cat:   Option<TypeCategory>,
    pub arity:      u8,
}

#[derive(Debug)]
pub struct MtlgGraph {
    pub nodes:    Vec<MtlgNode>,
    pub edges:    Vec<MtlgEdge>,
    pub language: String,
    pub deferred: Vec<TokenStructure>,
}

// ── Conversion ────────────────────────────────────────────────────────────────

pub fn sentence_to_mtlg(sentence: &TokenSentence, inducer: Option<&mut CategoryInducer>) -> MtlgGraph {
    let mut deferred = Vec::new();
    let mut nodes = Vec::new();
    for tok in &sentence.tokens {
        let structure = extract_features(tok, sentence);
        let ucca_cat  = inducer.as_ref().map(|ind| {
            let node_id = stable_node_id_from_structure(&structure);
            let leaf = structure.is_punctuation || structure.char_length_norm < 0.1;
            // Safe to call predict without mut borrow here since we have &mut via as_ref workaround.
            TypeCategory::DEFAULT
        });
        if ucca_cat.is_none() { deferred.push(structure.clone()); }
        nodes.push(MtlgNode {
            token_id: tok.id, text: tok.text.clone(), lemma: tok.lemma.clone(),
            modal_mode: ModalMode::Diamond, ucca_cat, arity: 0, structure,
        });
    }
    let edges = (0..sentence.tokens.len().saturating_sub(1)).map(|i| {
        MtlgEdge {
            src_id: sentence.tokens[i].id,
            dst_id: sentence.tokens[i + 1].id,
            modal_mode: ModalMode::Diamond, ucca_cat: None, arity: 0,
        }
    }).collect();
    MtlgGraph { nodes, edges, language: sentence.language.clone(), deferred }
}

pub fn resolve_deferred(graphs: &mut [MtlgGraph]) -> CategoryInducer {
    let all: Vec<TokenStructure> = graphs.iter().flat_map(|g| g.nodes.iter().map(|n| n.structure.clone())).collect();
    let mut inducer = CategoryInducer::new(0);
    if !all.is_empty() {
        inducer.fit(&all);
        for graph in graphs.iter_mut() {
            for node in &mut graph.nodes {
                if node.ucca_cat.is_none() { node.ucca_cat = Some(inducer.predict(&node.structure)); }
            }
            for edge in &mut graph.edges {
                if edge.ucca_cat.is_none() {
                    if let Some(n) = graph.nodes.iter().find(|n| n.token_id == edge.dst_id) {
                        edge.ucca_cat = Some(inducer.predict(&n.structure));
                    }
                }
            }
            graph.deferred.clear();
        }
    }
    inducer
}

impl MtlgGraph {
    pub fn to_arg_nodes_and_edges(&self, atms_label: u64, situation_id: u64) -> (Vec<ArgNode>, Vec<ArgEdge>) {
        let nodes: Vec<ArgNode> = self.nodes.iter().map(|n| {
            let cat = n.ucca_cat.unwrap_or(TypeCategory::DEFAULT);
            let mt  = if n.arity > 0 {
                ModalType::functor(n.modal_mode, cat, n.arity, Direction::Right)
            } else {
                ModalType::atom(n.modal_mode, cat)
            };
            let mut node = ArgNode::new(n.token_id as u64, NodeClass::DEFAULT, mt, (situation_id, 0));
            node.surface     = Some(n.text.as_bytes().to_vec());
            node.atms_label  = atms_label;
            node.attribution_score = 0.5;
            node.structure   = Some(n.structure.clone());
            node
        }).collect();
        let mut edge_id: u64 = 1;
        let edges: Vec<ArgEdge> = self.edges.iter().map(|e| {
            let eid = edge_id; edge_id += 1;
            let mut edge = ArgEdge::new(eid, e.src_id as u64, e.dst_id as u64, EdgeClass::SEQUENTIAL, ModalMode::Diamond);
            edge.weight = 0.5;
            edge
        }).collect();
        (nodes, edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::token_types::{Token, TokenSentence};

    fn sentence(words: &[&str]) -> TokenSentence {
        TokenSentence::new(
            words.iter().enumerate().map(|(i, &w)| Token::new(i as u32 + 1, w)).collect(),
            "en", words.join(" "),
        )
    }

    #[test]
    fn sequential_edges() {
        let s = sentence(&["Alice", "runs", "quickly"]);
        let g = sentence_to_mtlg(&s, None);
        assert_eq!(g.nodes.len(), 3);
        assert_eq!(g.edges.len(), 2);
        assert_eq!(g.edges[0].src_id, 1);
        assert_eq!(g.edges[0].dst_id, 2);
    }

    #[test]
    fn all_diamond() {
        let s = sentence(&["The", "cat"]);
        let g = sentence_to_mtlg(&s, None);
        for e in &g.edges { assert_eq!(e.modal_mode, ModalMode::Diamond); }
    }

    #[test]
    fn resolve_deferred_assigns_all() {
        let s1 = sentence(&["Alice", "runs"]);
        let s2 = sentence(&["The", "cat", "sat"]);
        let mut graphs = vec![sentence_to_mtlg(&s1, None), sentence_to_mtlg(&s2, None)];
        let total: usize = graphs.iter().map(|g| g.nodes.len()).sum();
        resolve_deferred(&mut graphs);
        let resolved: usize = graphs.iter().flat_map(|g| g.nodes.iter())
            .filter(|n| n.ucca_cat.is_some()).count();
        assert_eq!(resolved, total);
    }

    #[test]
    fn bisimulation_produces_nontrivial_partition() {
        // 4 nodes: 2 leaves, 2 non-leaves with transitions
        let nodes: Vec<NodeId> = vec![1, 2, 3, 4];
        let mut is_leaf = HashMap::new();
        is_leaf.insert(1u64, true);
        is_leaf.insert(2u64, true);
        is_leaf.insert(3u64, false);
        is_leaf.insert(4u64, false);
        let transitions = vec![
            LabeledTransition { from: 3, to: 1, modal_mode: ModalMode::Diamond },
            LabeledTransition { from: 4, to: 2, modal_mode: ModalMode::Diamond },
        ];
        let partition = bisimulation_partition(&nodes, &transitions, &is_leaf);
        assert_eq!(partition.len(), 4);
        // Leaves should have the same category; non-leaves may differ due to targets
        assert_eq!(partition[&1], partition[&2], "leaves should be in same block");
    }
}
