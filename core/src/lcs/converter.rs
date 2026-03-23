//! Text-to-MTLG converter. No UD dependency.
//! Categories: k-means over surface features. All initial edges are Diamond (◇).

use std::collections::{BTreeSet, HashMap};
use rand::prelude::*;

use crate::types::{TypeCategory, ModalMode, ModalType, Direction, NodeId, EdgeId, Env};
use crate::arg::{ArgNode, ArgEdge, NodeClass, EdgeClass};
use super::token_types::{Token, TokenSentence, TokenStructure, extract_features, fnv_hash};

// ── Feature vector ────────────────────────────────────────────────────────────

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

// ── k-means ───────────────────────────────────────────────────────────────────

fn squared_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum()
}

fn run_kmeans(vectors: &[Vec<f32>], k: usize, rng: &mut StdRng) -> (Vec<Vec<f32>>, Vec<usize>, f64) {
    let n = vectors.len();
    if n == 0 || k == 0 { return (vec![], vec![], 0.0); }
    let k = k.min(n);

    let mut center_idx = vec![rng.gen_range(0..n)];
    while center_idx.len() < k {
        let dists: Vec<f64> = (0..n).map(|i|
            center_idx.iter().map(|&c| squared_dist(&vectors[i], &vectors[c])).fold(f64::INFINITY, f64::min)
        ).collect();
        let total: f64 = dists.iter().sum();
        if total == 0.0 { break; }
        let mut pick = rng.gen::<f64>() * total;
        for (i, &d) in dists.iter().enumerate() {
            pick -= d;
            if pick <= 0.0 { center_idx.push(i); break; }
        }
    }

    let mut centroids: Vec<Vec<f32>> = center_idx.iter().map(|&i| vectors[i].clone()).collect();
    let mut assignments = vec![0usize; n];
    for _ in 0..50 {
        let mut changed = false;
        for i in 0..n {
            let best = centroids.iter().enumerate()
                .min_by(|(_, a), (_, b)| squared_dist(&vectors[i], a).partial_cmp(&squared_dist(&vectors[i], b)).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(ci, _)| ci).unwrap_or(0);
            if assignments[i] != best { changed = true; assignments[i] = best; }
        }
        if !changed { break; }
        let dim = vectors[0].len();
        let mut sums = vec![vec![0.0f64; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, &c) in assignments.iter().enumerate() {
            for d in 0..dim { sums[c][d] += vectors[i][d] as f64; }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                centroids[c] = sums[c].iter().map(|&s| (s / counts[c] as f64) as f32).collect();
            }
        }
    }
    let inertia: f64 = (0..n).map(|i| squared_dist(&vectors[i], &centroids[assignments[i]])).sum();
    (centroids, assignments, inertia)
}

fn select_k_bic(vectors: &[Vec<f32>], k_min: usize, k_max: usize) -> usize {
    if vectors.is_empty() { return k_min; }
    let n = vectors.len();
    let d = vectors[0].len();
    if d == 0 || n < k_min { return k_min; }
    let mut best_k = k_min;
    let mut best_bic = f64::INFINITY;
    let mut rng = StdRng::seed_from_u64(42);
    for k in k_min..=k_max.min(n) {
        let (_, _, inertia) = run_kmeans(vectors, k, &mut rng);
        let var = inertia / (n as f64 * d as f64) + 1e-9;
        let bic = n as f64 * d as f64 * var.ln() + k as f64 * d as f64 * (n as f64).ln();
        if bic < best_bic { best_bic = bic; best_k = k; }
    }
    best_k
}

// ── CategoryInducer ───────────────────────────────────────────────────────────

pub struct CategoryInducer {
    n_clusters:      usize,
    pub trigram_vocab: Vec<u32>,
    pub suffix_vocab:  Vec<u32>,
    centroids:         Vec<Vec<f32>>,
    cluster_labels:    HashMap<usize, TypeCategory>,
}

impl CategoryInducer {
    pub fn new(n_clusters: usize) -> Self {
        Self { n_clusters, trigram_vocab: Vec::new(), suffix_vocab: Vec::new(),
               centroids: Vec::new(), cluster_labels: HashMap::new() }
    }

    pub fn build_trigram_vocab(structures: &[TokenStructure]) -> Vec<u32> {
        let mut v: BTreeSet<u32> = BTreeSet::new();
        for s in structures { v.extend(s.char_trigram_hashes.iter().copied()); }
        v.into_iter().collect()
    }

    pub fn build_suffix_vocab(structures: &[TokenStructure]) -> Vec<u32> {
        let mut v: BTreeSet<u32> = BTreeSet::new();
        for s in structures { v.insert(s.suffix3_hash); v.insert(s.suffix2_hash); }
        v.into_iter().collect()
    }

    pub fn fit(&mut self, tokens: &[TokenStructure]) {
        if tokens.is_empty() { return; }
        if self.trigram_vocab.is_empty() { self.trigram_vocab = Self::build_trigram_vocab(tokens); }
        if self.suffix_vocab.is_empty()  { self.suffix_vocab  = Self::build_suffix_vocab(tokens); }
        let vectors: Vec<Vec<f32>> = tokens.iter()
            .map(|s| structure_to_vector(s, &self.trigram_vocab, &self.suffix_vocab))
            .collect();
        let n = vectors.len();
        let k = if self.n_clusters == 0 { select_k_bic(&vectors, 2, 20) } else { self.n_clusters.min(n) };
        let mut rng = StdRng::seed_from_u64(42);
        let (centroids, _, _) = run_kmeans(&vectors, k, &mut rng);
        self.cluster_labels = (0..centroids.len()).map(|c| (c, TypeCategory(c as u32 + 1))).collect();
        self.centroids = centroids;
    }

    pub fn predict(&self, s: &TokenStructure) -> TypeCategory {
        if self.centroids.is_empty() { return TypeCategory::DEFAULT; }
        let vec = structure_to_vector(s, &self.trigram_vocab, &self.suffix_vocab);
        let best = self.centroids.iter().enumerate()
            .min_by(|(_, a), (_, b)| squared_dist(&vec, a).partial_cmp(&squared_dist(&vec, b))
                .unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
        self.cluster_labels.get(&best).copied().unwrap_or(TypeCategory::DEFAULT)
    }
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

pub fn sentence_to_mtlg(sentence: &TokenSentence, inducer: Option<&CategoryInducer>) -> MtlgGraph {
    let mut deferred = Vec::new();
    let mut nodes = Vec::new();
    for tok in &sentence.tokens {
        let structure = extract_features(tok, sentence);
        let ucca_cat  = inducer.map(|ind| ind.predict(&structure));
        if ucca_cat.is_none() { deferred.push(structure.clone()); }
        nodes.push(MtlgNode {
            token_id: tok.id, text: tok.text.clone(), lemma: tok.lemma.clone(),
            modal_mode: ModalMode::Diamond, ucca_cat, arity: 0, structure,
        });
    }
    let edges = (0..sentence.tokens.len().saturating_sub(1)).map(|i| {
        let dst_structure = extract_features(&sentence.tokens[i + 1], sentence);
        let ucca_cat = inducer.map(|ind| ind.predict(&dst_structure));
        MtlgEdge {
            src_id: sentence.tokens[i].id,
            dst_id: sentence.tokens[i + 1].id,
            modal_mode: ModalMode::Diamond, ucca_cat, arity: 0,
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
}
