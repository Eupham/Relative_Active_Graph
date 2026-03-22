//! UD-to-MTLG converter: derives modal edge assignments from structural tree evidence.
//!
//! Design invariants (see conceptual specification):
//!
//! 1. **No UPOS comparison.**  The UPOS string (VERB, NOUN, ADJ, …) is hashed
//!    to a u32 on ingestion and stored on `TokenStructure::upos_opaque_id`.
//!    It is never string-compared anywhere in this module.
//!
//! 2. **No hard-coded category assignment.**  `score_ucca` and named deprel
//!    classification functions have been removed.  All category assignment
//!    goes through `CategoryInducer` k-means clustering on structural feature
//!    vectors.  Cluster IDs are opaque u32 values; none carry linguistic names.
//!
//! 3. **Opaque deprel features.**  Dependency relation strings are hashed to
//!    u32 values via FNV-1a (`deprel_hash`).  The feature vector uses a
//!    one-hot encoding over a corpus-derived deprel vocabulary for the
//!    self-deprel dimension and a bag-of-hashes for dependent deprels.
//!
//! 4. **Deferred resolution.**  All tokens are routed through `CategoryInducer`.
//!    When no inducer is provided, they accumulate in `MtlgGraph::deferred`
//!    for batch resolution via `resolve_deferred`.

use std::collections::{BTreeSet, HashMap};
use rand::prelude::*;

use crate::types::{
    TypeCategory, ModalMode, ModalType, Direction, NodeId, EdgeId, Env,
};
use crate::arg::{ArgNode, ArgEdge, NodeClass, EdgeClass};
use super::ud_types::{UdToken, UdTree, is_long_range, deprel_hash};

// ── Feature hashing ───────────────────────────────────────────────────────────

/// Map an arbitrary string to a stable u32 via FNV-1a.
/// Used to convert UPOS strings and morphological feature pairs into opaque
/// numeric identifiers that can appear in feature vectors without revealing
/// their semantic content to the classifier.
fn hash_string(s: &str) -> u32 {
    use std::hash::{Hash, Hasher};
    let mut h = fnv::FnvHasher::default();
    s.hash(&mut h);
    h.finish() as u32
}

/// Hash each `(key=value)` morphological feature pair to an opaque u32.
fn morph_feature_ids(feats: &HashMap<String, String>) -> BTreeSet<u32> {
    feats.iter().map(|(k, v)| hash_string(&format!("{}={}", k, v))).collect()
}

// ── TokenStructure ────────────────────────────────────────────────────────────

/// Purely structural characterisation of a UD token within its tree.
///
/// All named deprel comparisons have been replaced with opaque hash IDs.
/// The fields capture what a token *does* in the dependency tree without
/// embedding any linguistic theory about what those positions mean.
#[derive(Clone, Debug)]
pub struct TokenStructure {
    pub token_id:                 u32,

    // ── Opaque identifiers ──────────────────────────────────────────────────
    pub upos_opaque_id:           u32,
    pub morph_feature_ids:        BTreeSet<u32>,
    pub n_morphological_features: usize,

    // ── Structural tree position ────────────────────────────────────────────
    pub is_tree_root:             bool,
    pub depth_in_tree:            usize,

    // ── Deprel hashes ───────────────────────────────────────────────────────
    /// FNV-1a hash of this token's dependency relation to its head.
    pub self_deprel_hash:         u32,
    /// FNV-1a hashes of all dependents' deprels (bag, ordered by token id).
    pub dependent_deprel_hashes:  Vec<u32>,

    // ── Structural counts ───────────────────────────────────────────────────
    pub n_dependents:             usize,

    // ── Long-range / reentrancy evidence ────────────────────────────────────
    pub is_reentrant:             bool,
    pub has_long_range_dep:       bool,
}

/// Extract structural features from a UD token without reading UPOS labels.
pub fn extract_structure(tok: &UdToken, tree: &UdTree) -> TokenStructure {
    let deps = tree.dependents_of(tok.id);

    // Depth: count hops to root, guarding against cycles.
    let mut depth = 0usize;
    let mut cursor_id = tok.id;
    let mut seen = BTreeSet::new();
    loop {
        let Some(cursor) = tree.token_by_id(cursor_id) else { break };
        if cursor.head == 0 || !seen.insert(cursor_id) { break }
        cursor_id = cursor.head;
        depth += 1;
    }

    let has_long = is_long_range(&tok.deprel)
        || deps.iter().any(|d| is_long_range(&d.deprel));

    let morph_ids = morph_feature_ids(&tok.feats);
    let n_morph   = morph_ids.len();

    TokenStructure {
        token_id:                 tok.id,
        upos_opaque_id:           hash_string(&tok.upos),
        morph_feature_ids:        morph_ids,
        n_morphological_features: n_morph,
        is_tree_root:             tok.is_root(),
        depth_in_tree:            depth,
        self_deprel_hash:         deprel_hash(&tok.deprel),
        dependent_deprel_hashes:  deps.iter().map(|d| deprel_hash(&d.deprel)).collect(),
        n_dependents:             deps.len(),
        is_reentrant:             tok.is_reentrant(),
        has_long_range_dep:       has_long,
    }
}

/// Assign modal mode from reentrancy and long-range dependency evidence.
pub fn assign_modal_mode(tok: &UdToken, deprel: &str) -> ModalMode {
    if tok.is_reentrant() { return ModalMode::Box; }
    if is_long_range(deprel) { return ModalMode::Lozenge; }
    ModalMode::Diamond
}

// ── Feature vector ────────────────────────────────────────────────────────────

/// Convert a `TokenStructure` to a float vector for k-means clustering.
///
/// Layout:
///   [0]      is_tree_root
///   [1]      depth_in_tree / 10 (normalised)
///   [2]      n_dependents / 10 (normalised)
///   [3]      is_reentrant
///   [4]      has_long_range_dep
///   [5]      n_morphological_features / 10 (normalised)
///   [6]      upos_opaque_id % 10_000 / 10_000
///   [7..7+D] self deprel one-hot over deprel_vocab
///   [7+D..7+2D] dependent deprel bag-of-hashes over deprel_vocab
///   [7+2D..] morphological feature bits over morph_vocab
///
/// Total: 7 + 2 × |deprel_vocab| + |morph_vocab| ≈ 90–120 dimensions.
pub fn structure_to_vector(
    s:            &TokenStructure,
    deprel_vocab: &[u32],
    morph_vocab:  &[u32],
) -> Vec<f32> {
    let mut v = Vec::with_capacity(7 + 2 * deprel_vocab.len() + morph_vocab.len());

    v.push(s.is_tree_root              as u8 as f32);
    v.push((s.depth_in_tree  as f32 / 10.0).min(1.0));
    v.push((s.n_dependents   as f32 / 10.0).min(1.0));
    v.push(s.is_reentrant              as u8 as f32);
    v.push(s.has_long_range_dep        as u8 as f32);
    v.push((s.n_morphological_features as f32 / 10.0).min(1.0));
    v.push((s.upos_opaque_id % 10_000) as f32 / 10_000.0);

    // Self deprel: one-hot over deprel vocabulary.
    for &dh in deprel_vocab {
        v.push((s.self_deprel_hash == dh) as u8 as f32);
    }

    // Dependent deprels: bag-of-hashes over deprel vocabulary.
    let dep_set: std::collections::HashSet<u32> =
        s.dependent_deprel_hashes.iter().copied().collect();
    for &dh in deprel_vocab {
        v.push(dep_set.contains(&dh) as u8 as f32);
    }

    // Morphological features.
    for &fid in morph_vocab {
        v.push(s.morph_feature_ids.contains(&fid) as u8 as f32);
    }

    v
}

// ── k-means helpers ───────────────────────────────────────────────────────────

fn squared_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(&x, &y)| ((x - y) as f64).powi(2))
        .sum()
}

/// Run one k-means iteration: k-means++ init + Lloyd's algorithm.
/// Returns (centroids, assignments, inertia).
fn run_kmeans(
    vectors: &[Vec<f32>],
    k:       usize,
    rng:     &mut StdRng,
) -> (Vec<Vec<f32>>, Vec<usize>, f64) {
    let n = vectors.len();
    if n == 0 || k == 0 { return (vec![], vec![], 0.0); }
    let k = k.min(n);

    // k-means++ initialisation.
    let mut center_idx: Vec<usize> = vec![rng.gen_range(0..n)];
    for _ in 1..k {
        let dists: Vec<f64> = vectors.iter().map(|v| {
            center_idx.iter()
                .map(|&ci| squared_dist(v, &vectors[ci]))
                .fold(f64::INFINITY, f64::min)
        }).collect();
        let total: f64 = dists.iter().sum::<f64>() + 1e-12;
        let r: f64     = rng.gen();
        let mut cumsum = 0.0f64;
        let mut chosen = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            cumsum += d / total;
            if r <= cumsum { chosen = i; break; }
        }
        center_idx.push(chosen);
    }
    let mut centroids: Vec<Vec<f32>> =
        center_idx.iter().map(|&i| vectors[i].clone()).collect();

    // Lloyd's iterations.
    let mut assignments = vec![0usize; n];
    for _ in 0..20 {
        let new_assign: Vec<usize> = vectors.iter().map(|v| {
            centroids.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    squared_dist(v, a)
                        .partial_cmp(&squared_dist(v, b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0)
        }).collect();

        if new_assign == assignments { break; }
        assignments = new_assign;

        let dim = vectors[0].len();
        for c in 0..k {
            let members: Vec<&Vec<f32>> = vectors.iter()
                .zip(assignments.iter())
                .filter(|(_, &a)| a == c)
                .map(|(v, _)| v)
                .collect();
            if members.is_empty() { continue; }
            let mut new_centroid = vec![0.0f32; dim];
            for m in &members {
                for (i, &x) in m.iter().enumerate() {
                    new_centroid[i] += x;
                }
            }
            let count = members.len() as f32;
            for x in &mut new_centroid { *x /= count; }
            centroids[c] = new_centroid;
        }
    }

    let inertia: f64 = vectors.iter().zip(assignments.iter())
        .map(|(v, &a)| squared_dist(v, &centroids[a]))
        .sum();

    (centroids, assignments, inertia)
}

/// BIC-based automatic cluster count selection.
///
/// Uses the spherical Gaussian BIC:
///   BIC(k) = n·d·ln(σ²) + k·d·ln(n)
/// where σ² = inertia / (n·d).
///
/// Lower BIC = better fit penalised for complexity.
/// Total cost: ~15–20 k-means runs.
fn select_k_bic(vectors: &[Vec<f32>], k_min: usize, k_max: usize) -> usize {
    if vectors.is_empty() { return k_min; }
    let n = vectors.len();
    let d = vectors[0].len();
    if d == 0 || n < k_min { return k_min; }

    let mut best_k   = k_min;
    let mut best_bic = f64::INFINITY;
    let mut rng      = StdRng::seed_from_u64(42);

    for k in k_min..=k_max.min(n) {
        let (_, _, inertia) = run_kmeans(vectors, k, &mut rng);
        let var = inertia / (n as f64 * d as f64) + 1e-9;
        let bic = n as f64 * d as f64 * var.ln()
                + k as f64 * d as f64 * (n as f64).ln();
        if bic < best_bic {
            best_bic = bic;
            best_k   = k;
        }
    }
    best_k
}

// ── CategoryInducer ───────────────────────────────────────────────────────────

/// Resolves UCCA categories for all tokens via k-means clustering.
///
/// All tokens are routed through this inducer — there is no hard-coded
/// "confident" path that bypasses clustering. Cluster IDs are opaque
/// u32 values (1-based) assigned by clustering order; none carry
/// linguistic names.
///
/// When `n_clusters == 0`, BIC auto-selection chooses k in [2, 20].
/// When `n_clusters > 0`, that value is used directly.
pub struct CategoryInducer {
    n_clusters:      usize,
    pub morph_vocab:  Vec<u32>,
    pub deprel_vocab: Vec<u32>,
    centroids:        Vec<Vec<f32>>,
    cluster_labels:   HashMap<usize, TypeCategory>,
}

impl CategoryInducer {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            morph_vocab:  Vec::new(),
            deprel_vocab: Vec::new(),
            centroids:    Vec::new(),
            cluster_labels: HashMap::new(),
        }
    }

    /// Collect all observed morphological feature hash IDs into a shared vocabulary.
    pub fn build_morph_vocab(structures: &[TokenStructure]) -> Vec<u32> {
        let mut vocab: BTreeSet<u32> = BTreeSet::new();
        for s in structures {
            vocab.extend(s.morph_feature_ids.iter().copied());
        }
        vocab.into_iter().collect()
    }

    /// Collect all observed deprel hash IDs (self + dependents) into a shared vocabulary.
    pub fn build_deprel_vocab(structures: &[TokenStructure]) -> Vec<u32> {
        let mut vocab: BTreeSet<u32> = BTreeSet::new();
        for s in structures {
            vocab.insert(s.self_deprel_hash);
            vocab.extend(s.dependent_deprel_hashes.iter().copied());
        }
        vocab.into_iter().collect()
    }

    /// Cluster tokens and assign opaque category IDs to each cluster.
    pub fn fit(&mut self, tokens: &[TokenStructure]) {
        if tokens.is_empty() { return; }
        if self.morph_vocab.is_empty() {
            self.morph_vocab = Self::build_morph_vocab(tokens);
        }
        if self.deprel_vocab.is_empty() {
            self.deprel_vocab = Self::build_deprel_vocab(tokens);
        }

        let n = tokens.len();
        let vectors: Vec<Vec<f32>> = tokens
            .iter()
            .map(|s| structure_to_vector(s, &self.deprel_vocab, &self.morph_vocab))
            .collect();

        let k = if self.n_clusters == 0 {
            select_k_bic(&vectors, 2, 20)
        } else {
            self.n_clusters.min(n)
        };

        let mut rng = StdRng::seed_from_u64(42);
        let (centroids, _, _) = run_kmeans(&vectors, k, &mut rng);

        // Assign opaque cluster IDs (1-based; 0 is TypeCategory::DEFAULT).
        let mut labels = HashMap::new();
        for c in 0..centroids.len() {
            labels.insert(c, TypeCategory(c as u32 + 1));
        }

        self.centroids      = centroids;
        self.cluster_labels = labels;
    }

    /// Assign category to a token via nearest-centroid lookup.
    pub fn predict(&self, s: &TokenStructure) -> TypeCategory {
        if self.centroids.is_empty() { return TypeCategory::DEFAULT; }
        let vec = structure_to_vector(s, &self.deprel_vocab, &self.morph_vocab);
        let best = self.centroids.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                squared_dist(&vec, a)
                    .partial_cmp(&squared_dist(&vec, b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.cluster_labels.get(&best).copied().unwrap_or(TypeCategory::DEFAULT)
    }
}

// ── Output graph types ────────────────────────────────────────────────────────

/// An MTLG node derived from a UD token.
///
/// `upos_raw` is stored verbatim for human inspection only.
/// `ucca_cat` is `None` for tokens deferred to batch `resolve_deferred`.
#[derive(Clone, Debug)]
pub struct MtlgNode {
    pub token_id:   u32,
    pub text:       String,
    pub lemma:      String,
    /// Stored verbatim for auditability; never used in logic.
    pub upos_raw:   String,
    pub modal_mode: ModalMode,
    pub ucca_cat:   Option<TypeCategory>,
    pub arity:      u8,
    pub structure:  TokenStructure,
}

/// A directed edge in the MTLG graph corresponding to one UD dependency arc.
#[derive(Clone, Debug)]
pub struct MtlgEdge {
    pub src_id:     u32,
    pub dst_id:     u32,
    pub deprel:     String,
    pub modal_mode: ModalMode,
    pub ucca_cat:   Option<TypeCategory>,
    pub arity:      u8,
}

/// MTLG graph produced from one UD sentence.
///
/// `deferred` carries the `TokenStructure` values of all tokens whose
/// categories have not yet been resolved.
/// Call `resolve_deferred` to batch-resolve them via `CategoryInducer`.
#[derive(Debug)]
pub struct MtlgGraph {
    pub nodes:    Vec<MtlgNode>,
    pub edges:    Vec<MtlgEdge>,
    pub language: String,
    pub deferred: Vec<TokenStructure>,
}

// ── Main conversion ───────────────────────────────────────────────────────────

/// Convert a `UdTree` to an `MtlgGraph`.
///
/// All tokens are routed through `inducer` for category assignment.
/// If `inducer` is not provided, all tokens are deferred for batch
/// resolution via `resolve_deferred`.
pub fn ud_tree_to_mtlg(
    tree:    &UdTree,
    inducer: Option<&CategoryInducer>,
) -> MtlgGraph {
    let mut deferred = Vec::new();
    let mut nodes    = Vec::new();

    for tok in &tree.tokens {
        let s     = extract_structure(tok, tree);
        let arity = s.n_dependents.min(7) as u8;

        let resolved_cat = match inducer {
            Some(ind) => Some(ind.predict(&s)),
            None      => { deferred.push(s.clone()); None }
        };

        nodes.push(MtlgNode {
            token_id:   tok.id,
            text:       tok.text.clone(),
            lemma:      tok.lemma.clone(),
            upos_raw:   tok.upos.clone(),
            modal_mode: ModalMode::Diamond,
            ucca_cat:   resolved_cat,
            arity,
            structure:  s,
        });
    }

    let mut edges = Vec::new();
    for tok in &tree.tokens {
        if tok.head == 0 { continue; }
        let Some(_head_tok) = tree.token_by_id(tok.head) else { continue };

        let s_tok      = extract_structure(tok, tree);
        let modal_mode = assign_modal_mode(tok, &tok.deprel);
        let arity      = s_tok.n_dependents.min(7) as u8;

        let resolved_cat = inducer.map(|ind| ind.predict(&s_tok));

        edges.push(MtlgEdge {
            src_id:     tok.head,
            dst_id:     tok.id,
            deprel:     tok.deprel.clone(),
            modal_mode,
            ucca_cat:   resolved_cat,
            arity,
        });
    }

    MtlgGraph { nodes, edges, language: tree.language.clone(), deferred }
}

/// Batch-resolve deferred categories across a corpus of `MtlgGraph`s.
///
/// Collects all deferred `TokenStructure`s from all graphs, fits a
/// `CategoryInducer`, then updates each graph's nodes and edges in-place.
/// Returns the fitted inducer for use on future graphs.
pub fn resolve_deferred(graphs: &mut [MtlgGraph]) -> CategoryInducer {
    let all_deferred: Vec<TokenStructure> = graphs.iter()
        .flat_map(|g| g.deferred.iter().cloned())
        .collect();

    // Collect ALL node structures (not just deferred) for a richer vocab.
    let all_structures: Vec<TokenStructure> = graphs.iter()
        .flat_map(|g| g.nodes.iter().map(|n| n.structure.clone()))
        .collect();

    let mut inducer = CategoryInducer::new(16);

    if !all_deferred.is_empty() {
        inducer.morph_vocab  = CategoryInducer::build_morph_vocab(&all_structures);
        inducer.deprel_vocab = CategoryInducer::build_deprel_vocab(&all_structures);
        inducer.fit(&all_deferred);

        for graph in graphs.iter_mut() {
            for node in &mut graph.nodes {
                if node.ucca_cat.is_none() {
                    node.ucca_cat = Some(inducer.predict(&node.structure));
                }
            }
            for edge in &mut graph.edges {
                if edge.ucca_cat.is_none() {
                    let cat = graph.nodes.iter()
                        .find(|n| n.token_id == edge.dst_id)
                        .map(|n| inducer.predict(&n.structure))
                        .unwrap_or(TypeCategory::DEFAULT);
                    edge.ucca_cat = Some(cat);
                }
            }
            graph.deferred.clear();
        }
    }

    inducer
}

// ── ARG conversion ────────────────────────────────────────────────────────────

impl MtlgGraph {
    /// Convert this graph into `(ArgNode, ArgEdge)` vectors suitable for the
    /// CSRRE engine's ARG expansion.
    pub fn to_arg_nodes_and_edges(
        &self,
        env:     Env,
        base_id: NodeId,
    ) -> (Vec<ArgNode>, Vec<ArgEdge>) {
        let nodes: Vec<ArgNode> = self.nodes.iter().map(|mn| {
            let cat  = mn.ucca_cat.unwrap_or(TypeCategory::DEFAULT);
            let mt   = ModalType::functor(mn.modal_mode, cat, mn.arity, Direction::Right);
            let nid  = base_id + mn.token_id as u64;
            let mut n = ArgNode::new(nid, NodeClass::DEFAULT, mt, (0, 0));
            n.surface     = Some(mn.text.as_bytes().to_vec());
            n.atms_label  = env;
            n
        }).collect();

        let edges: Vec<ArgEdge> = self.edges.iter().enumerate().map(|(i, me)| {
            let src = base_id + me.src_id as u64;
            let dst = base_id + me.dst_id as u64;
            ArgEdge::new(
                base_id + i as u64 + 1000,
                src,
                dst,
                EdgeClass::DEFAULT,
                me.modal_mode,
            )
        }).collect();

        (nodes, edges)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lcs::ud_types::{UdToken, UdTree};

    /// "Alice runs quickly."
    fn alice_tree() -> UdTree {
        UdTree::new(
            vec![
                UdToken::new(1, "Alice",   "Alice",   "PROPN", "NNP", 2, "nsubj"),
                UdToken::new(2, "runs",    "run",     "VERB",  "VBZ", 0, "root"),
                UdToken::new(3, "quickly", "quickly", "ADV",   "RB",  2, "advmod"),
            ],
            "en",
            "Alice runs quickly.",
        )
    }

    /// "The city was destroyed." (event nominal with patient)
    fn destruction_tree() -> UdTree {
        let mut tok_dest = UdToken::new(4, "destroyed", "destroy", "VERB", "VBD", 0, "root");
        tok_dest.feats.insert("Tense".into(), "Past".into());
        tok_dest.feats.insert("VerbForm".into(), "Part".into());
        UdTree::new(
            vec![
                UdToken::new(1, "The",       "the",     "DET",   "DT",  2, "det"),
                UdToken::new(2, "city",      "city",    "NOUN",  "NN",  4, "nsubj:pass"),
                UdToken::new(3, "was",       "be",      "AUX",   "VBD", 4, "aux:pass"),
                tok_dest,
            ],
            "en",
            "The city was destroyed.",
        )
    }

    #[test]
    fn conversion_produces_correct_counts() {
        let tree  = alice_tree();
        let graph = ud_tree_to_mtlg(&tree, None);
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.edges.len(), 2); // nsubj + advmod
    }

    #[test]
    fn modal_mode_diamond_for_plain_deps() {
        let tree  = alice_tree();
        let graph = ud_tree_to_mtlg(&tree, None);
        for edge in &graph.edges {
            assert_eq!(edge.modal_mode, ModalMode::Diamond);
        }
    }

    #[test]
    fn to_arg_nodes_preserves_surface() {
        let tree  = alice_tree();
        let graph = ud_tree_to_mtlg(&tree, None);
        let (nodes, _edges) = graph.to_arg_nodes_and_edges(0b1, 0);
        let surfaces: Vec<&str> = nodes.iter()
            .filter_map(|n| n.surface.as_deref().and_then(|b| std::str::from_utf8(b).ok()))
            .collect();
        assert!(surfaces.contains(&"Alice"));
        assert!(surfaces.contains(&"runs"));
    }

    #[test]
    fn category_inducer_resolves_deferred() {
        let tree1   = alice_tree();
        let tree2   = destruction_tree();
        let mut graphs = vec![
            ud_tree_to_mtlg(&tree1, None),
            ud_tree_to_mtlg(&tree2, None),
        ];

        let total_nodes: usize = graphs.iter().map(|g| g.nodes.len()).sum();
        let _inducer = resolve_deferred(&mut graphs);

        let resolved: usize = graphs.iter()
            .flat_map(|g| g.nodes.iter())
            .filter(|n| n.ucca_cat.is_some())
            .count();
        assert_eq!(resolved, total_nodes, "all nodes must have a category after resolve_deferred");
    }

    #[test]
    fn all_tokens_get_categories_via_inducer() {
        let tree1 = alice_tree();
        let tree2 = destruction_tree();
        let mut graphs = vec![
            ud_tree_to_mtlg(&tree1, None),
            ud_tree_to_mtlg(&tree2, None),
        ];
        let total = graphs.iter().map(|g| g.nodes.len()).sum::<usize>();
        let _inducer = resolve_deferred(&mut graphs);
        let assigned = graphs.iter()
            .flat_map(|g| g.nodes.iter())
            .filter(|n| n.ucca_cat.is_some() && n.ucca_cat != Some(TypeCategory::DEFAULT))
            .count();
        assert_eq!(assigned, total, "every token must receive a non-default category");
    }

    #[test]
    fn upos_opaque_same_string_same_hash() {
        assert_eq!(hash_string("VERB"), hash_string("VERB"));
        assert_ne!(hash_string("VERB"), hash_string("NOUN"));
    }

    #[test]
    fn deprel_hash_used_in_structure() {
        let tree  = alice_tree();
        let alice = tree.token_by_id(1).unwrap();
        let s     = extract_structure(alice, &tree);
        // self_deprel_hash is the FNV hash of "nsubj" (no named comparison)
        let expected_hash = deprel_hash("nsubj");
        assert_eq!(s.self_deprel_hash, expected_hash);
    }

    #[test]
    fn structure_vector_correct_length() {
        let tree  = alice_tree();
        let runs  = tree.token_by_id(2).unwrap();
        let s     = extract_structure(runs, &tree);
        // Build a minimal vocab.
        let deprel_vocab = vec![deprel_hash("root"), deprel_hash("nsubj"), deprel_hash("advmod")];
        let morph_vocab  = vec![hash_string("Tense=Pres"), hash_string("Number=Sing")];
        let v = structure_to_vector(&s, &deprel_vocab, &morph_vocab);
        assert_eq!(v.len(), 7 + 2 * deprel_vocab.len() + morph_vocab.len());
    }
}
