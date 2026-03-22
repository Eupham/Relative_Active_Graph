//! UD-to-MTLG converter: derives modal edge assignments from structural tree evidence.
//!
//! Design invariants (see conceptual specification):
//!
//! 1. **No UPOS comparison.**  The UPOS string (VERB, NOUN, ADJ, …) is hashed
//!    to a u32 on ingestion and stored on `TokenStructure::upos_opaque_id`.
//!    It is never string-compared anywhere in this module.
//!
//! 2. **Structural evidence only.**  Category assignment is driven by:
//!    - How many core arguments / clausal dependents does this token govern?
//!    - What relational role does this token fill toward its own head?
//!    - Is the token the syntactic root?
//!    - What morphological features does it carry (as opaque hash IDs)?
//!    Dependency relation strings describe *relational structure*; they are
//!    not the same as the UPOS category of the token itself.
//!
//! 3. **Deferred resolution.**  When structural evidence is insufficient the
//!    token is deferred to `CategoryInducer`, a k-means++ clusterer that
//!    discovers categories from structural feature vectors.

use std::collections::{BTreeSet, HashMap};
use rand::prelude::*;

use crate::types::{
    TypeCategory, ModalMode, ModalType, Direction, NodeId, EdgeId, Env,
};
use crate::arg::{ArgNode, ArgEdge, NodeClass, EdgeClass};
use super::ud_types::{
    UdToken, UdTree,
    is_core_arg, is_clausal, is_adverbial, is_predicative,
    is_connector, is_discourse, is_functional, is_long_range,
};

// ── Confidence constants ──────────────────────────────────────────────────────

const CONFIDENCE_HIGH:   f64 = 0.85;
const CONFIDENCE_MEDIUM: f64 = 0.70;
const CONFIDENCE_LOW:    f64 = 0.45;
const DEFER_THRESHOLD:   f64 = 0.50;

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
/// No UPOS category name appears in this struct or the function that builds it.
/// UPOS is stored as an opaque u32 hash for feature vector construction only.
/// The fields capture what a token *does* in the dependency tree.
#[derive(Clone, Debug)]
pub struct TokenStructure {
    pub token_id:                 u32,

    // ── Opaque identifiers ──────────────────────────────────────────────────
    // Stored for vector construction and auditability; never used in
    // named comparisons.
    pub upos_opaque_id:           u32,
    pub morph_feature_ids:        BTreeSet<u32>,
    pub n_morphological_features: usize,

    // ── Structural tree position ────────────────────────────────────────────
    pub is_tree_root:             bool,
    pub depth_in_tree:            usize,

    // ── Argument projection: what this token GOVERNS ────────────────────────
    pub n_core_arg_dependents:    usize,
    pub n_clausal_dependents:     usize,
    pub n_adverbial_dependents:   usize,
    pub n_predicative_dependents: usize,
    pub n_functional_dependents:  usize,
    pub n_total_dependents:       usize,

    // ── This token's relation to its own head ───────────────────────────────
    pub self_is_core_arg:         bool,
    pub self_is_clausal:          bool,
    pub self_is_adverbial:        bool,
    pub self_is_predicative:      bool,
    pub self_is_connector:        bool,
    pub self_is_discourse:        bool,
    pub self_is_functional:       bool,

    // ── Long-range / reentrancy evidence ────────────────────────────────────
    pub is_reentrant:             bool,
    pub has_long_range_dep:       bool,
}

/// Extract structural features from a UD token without reading UPOS labels.
pub fn extract_structure(tok: &UdToken, tree: &UdTree) -> TokenStructure {
    let deps = tree.dependents_of(tok.id);
    let deprel = tok.deprel.to_lowercase();

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

    let n_core    = deps.iter().filter(|d| is_core_arg(&d.deprel)).count();
    let n_clausal = deps.iter().filter(|d| is_clausal(&d.deprel)).count();
    let n_adverb  = deps.iter().filter(|d| is_adverbial(&d.deprel)).count();
    let n_pred    = deps.iter().filter(|d| is_predicative(&d.deprel)).count();
    let n_func    = deps.iter().filter(|d| is_functional(&d.deprel)).count();

    let has_long = is_long_range(&deprel)
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
        n_core_arg_dependents:    n_core,
        n_clausal_dependents:     n_clausal,
        n_adverbial_dependents:   n_adverb,
        n_predicative_dependents: n_pred,
        n_functional_dependents:  n_func,
        n_total_dependents:       deps.len(),
        self_is_core_arg:         is_core_arg(&deprel),
        self_is_clausal:          is_clausal(&deprel),
        self_is_adverbial:        is_adverbial(&deprel),
        self_is_predicative:      is_predicative(&deprel),
        self_is_connector:        is_connector(&deprel),
        self_is_discourse:        is_discourse(&deprel),
        self_is_functional:       is_functional(&deprel),
        is_reentrant:             tok.is_reentrant(),
        has_long_range_dep:       has_long,
    }
}

// ── Structural UCCA scoring ───────────────────────────────────────────────────

/// Assign UCCA category from structural evidence alone.
///
/// Returns `(category, confidence)`. When `confidence < DEFER_THRESHOLD` the
/// category is `None` — the token is passed to `CategoryInducer` for
/// cluster-based resolution.
///
/// Evidence hierarchy (see spec §"Structural UCCA scoring"):
/// 1. Governs core arguments or clausal dependents, or is tree root → Process.
/// 2. Fills connector or discourse role → Connector / Ground.
/// 3. Fills adverbial modification role → Adverbial.
/// 4. Fills predicative (stative) modification role → State.
/// 5. Fills core argument role, morphologically simple, no dependents → Participant.
/// 6. Heads a subordinate clause (as dependent) → Scene.
/// 7. Functional / auxiliary element → Scene.
/// 8. Insufficient evidence → deferred (None).
pub fn score_ucca(s: &TokenStructure) -> (Option<TypeCategory>, f64) {
    // ── Strongly eventive ────────────────────────────────────────────────────
    if s.n_core_arg_dependents >= 1 || s.n_clausal_dependents >= 1 || s.is_tree_root {
        return (Some(TypeCategory(1)), CONFIDENCE_HIGH);
    }

    // ── Structural role: connector or discourse ──────────────────────────────
    if s.self_is_connector  { return (Some(TypeCategory(2)), CONFIDENCE_HIGH); }
    if s.self_is_discourse  { return (Some(TypeCategory(3)), CONFIDENCE_HIGH); }

    // ── Structural role: adverbial modification ──────────────────────────────
    if s.self_is_adverbial  { return (Some(TypeCategory(4)), CONFIDENCE_MEDIUM); }

    // ── Structural role: predicative (stative) modification ──────────────────
    if s.self_is_predicative { return (Some(TypeCategory(5)), CONFIDENCE_MEDIUM); }

    // ── Structural role: core argument ───────────────────────────────────────
    if s.self_is_core_arg {
        if s.n_total_dependents == 0 && s.n_morphological_features <= 3 {
            // Morphologically simple leaf in core arg position: Participant.
            return (Some(TypeCategory(6)), CONFIDENCE_MEDIUM);
        }
        // Morphologically complex or has dependents — could be event nominal.
        // Defer to CategoryInducer: evidence is ambiguous.
        return (None, CONFIDENCE_LOW);
    }

    // ── Structural role: clausal (as dependent, not head) ────────────────────
    if s.self_is_clausal    { return (Some(TypeCategory::DEFAULT), CONFIDENCE_MEDIUM); }

    // ── Functional element ────────────────────────────────────────────────────
    if s.self_is_functional { return (Some(TypeCategory::DEFAULT), 0.60); }

    // ── Insufficient structural evidence ─────────────────────────────────────
    (None, 0.30)
}

/// Estimate functor arity from structural evidence.
/// Arity = number of argument slots remaining to be saturated.
pub fn compute_arity(s: &TokenStructure) -> u8 {
    if s.n_core_arg_dependents > 0 {
        return s.n_core_arg_dependents.min(7) as u8;
    }
    // Functional or relational head governing exactly one dependent: arity 1.
    if s.n_total_dependents == 1 && s.n_core_arg_dependents == 0 {
        return 1;
    }
    0
}

/// Assign modal mode from structural relation evidence.
pub fn assign_modal_mode(tok: &UdToken, deprel: &str) -> ModalMode {
    if tok.is_reentrant() { return ModalMode::Box; }
    if is_long_range(deprel) { return ModalMode::Lozenge; }
    ModalMode::Diamond
}

// ── Feature vector ────────────────────────────────────────────────────────────

const N_STRUCTURAL_FEATURES: usize = 19;

/// Convert a `TokenStructure` to a fixed-size float vector suitable for k-means.
///
/// The first 19 dimensions are normalised structural features.
/// The remaining dimensions are presence bits over the shared morphological vocabulary.
pub fn structure_to_vector(s: &TokenStructure, morph_vocab: &[u32]) -> Vec<f32> {
    let mut v = Vec::with_capacity(N_STRUCTURAL_FEATURES + morph_vocab.len());

    v.push(s.is_tree_root as u8 as f32);
    v.push((s.depth_in_tree as f32 / 10.0).min(1.0));
    v.push((s.n_core_arg_dependents as f32 / 5.0).min(1.0));
    v.push((s.n_clausal_dependents as f32 / 3.0).min(1.0));
    v.push((s.n_adverbial_dependents as f32 / 3.0).min(1.0));
    v.push((s.n_predicative_dependents as f32 / 3.0).min(1.0));
    v.push((s.n_functional_dependents as f32 / 5.0).min(1.0));
    v.push((s.n_total_dependents as f32 / 10.0).min(1.0));
    v.push(s.self_is_core_arg    as u8 as f32);
    v.push(s.self_is_clausal     as u8 as f32);
    v.push(s.self_is_adverbial   as u8 as f32);
    v.push(s.self_is_predicative as u8 as f32);
    v.push(s.self_is_connector   as u8 as f32);
    v.push(s.self_is_discourse   as u8 as f32);
    v.push(s.self_is_functional  as u8 as f32);
    v.push(s.is_reentrant        as u8 as f32);
    v.push(s.has_long_range_dep  as u8 as f32);
    v.push((s.n_morphological_features as f32 / 10.0).min(1.0));
    // Opaque UPOS as a normalised integer.
    v.push((s.upos_opaque_id % 10_000) as f32 / 10_000.0);

    for &fid in morph_vocab {
        v.push(if s.morph_feature_ids.contains(&fid) { 1.0 } else { 0.0 });
    }
    v
}

// ── CategoryInducer ───────────────────────────────────────────────────────────

/// Resolves UCCA categories for tokens where structural evidence was insufficient.
///
/// Uses k-means++ on structural feature vectors to cluster deferred tokens.
/// Each cluster is labelled by re-running `score_ucca` on a synthetic
/// `TokenStructure` reconstructed from the cluster centroid.
///
/// This is where the system discovers that morphological combination X clusters
/// with Process behaviour — without ever being told "X means verb".
pub struct CategoryInducer {
    n_clusters:     usize,
    pub morph_vocab: Vec<u32>,
    centroids:       Vec<Vec<f32>>,
    cluster_labels:  HashMap<usize, TypeCategory>,
}

impl CategoryInducer {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            morph_vocab:    Vec::new(),
            centroids:       Vec::new(),
            cluster_labels:  HashMap::new(),
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

    /// Cluster deferred tokens and assign UCCA labels to each cluster.
    pub fn fit(&mut self, deferred: &[TokenStructure]) {
        if deferred.is_empty() { return; }
        if self.morph_vocab.is_empty() {
            self.morph_vocab = Self::build_morph_vocab(deferred);
        }

        let n = deferred.len();
        let k = self.n_clusters.min(n);
        let vectors: Vec<Vec<f32>> = deferred
            .iter()
            .map(|s| structure_to_vector(s, &self.morph_vocab))
            .collect();

        // ── k-means++ initialisation ─────────────────────────────────────────
        let mut rng = StdRng::seed_from_u64(42);
        let mut center_idx: Vec<usize> = vec![rng.gen_range(0..n)];
        for _ in 1..k {
            let dists: Vec<f64> = vectors.iter().map(|v| {
                center_idx.iter()
                    .map(|&ci| squared_dist(v, &vectors[ci]))
                    .fold(f64::INFINITY, f64::min)
            }).collect();
            let total: f64 = dists.iter().sum::<f64>() + 1e-12;
            let r: f64 = rng.gen();
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

        // ── Lloyd's iterations ───────────────────────────────────────────────
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

            // Update centroids.
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

        // ── Label each cluster by re-scoring its centroid ────────────────────
        let mut labels = HashMap::new();
        for (c, centroid) in centroids.iter().enumerate() {
            let synthetic = centroid_to_structure(centroid);
            let (cat, conf) = score_ucca(&synthetic);
            labels.insert(c, if cat.is_some() && conf >= 0.50 {
                cat.unwrap()
            } else {
                TypeCategory::DEFAULT
            });
        }

        self.centroids      = centroids;
        self.cluster_labels = labels;
    }

    /// Assign UCCA category to a deferred token via nearest-centroid lookup.
    pub fn predict(&self, s: &TokenStructure) -> TypeCategory {
        if self.centroids.is_empty() { return TypeCategory::DEFAULT; }
        let vec = structure_to_vector(s, &self.morph_vocab);
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

fn squared_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(&x, &y)| ((x - y) as f64).powi(2))
        .sum()
}

/// Reconstruct a synthetic `TokenStructure` from a centroid vector's
/// structural slice (indices 0–17 defined in `structure_to_vector`).
fn centroid_to_structure(c: &[f32]) -> TokenStructure {
    let b = |i: usize| -> bool { c.get(i).copied().unwrap_or(0.0) > 0.5 };
    let n = |i: usize, scale: usize| -> usize {
        (c.get(i).copied().unwrap_or(0.0) * scale as f32).round() as usize
    };
    TokenStructure {
        token_id:                 u32::MAX,
        upos_opaque_id:           0,
        morph_feature_ids:        BTreeSet::new(),
        n_morphological_features: n(17, 10),
        is_tree_root:             b(0),
        depth_in_tree:            n(1, 10),
        n_core_arg_dependents:    n(2, 5),
        n_clausal_dependents:     n(3, 3),
        n_adverbial_dependents:   n(4, 3),
        n_predicative_dependents: n(5, 3),
        n_functional_dependents:  n(6, 5),
        n_total_dependents:       n(7, 10),
        self_is_core_arg:         b(8),
        self_is_clausal:          b(9),
        self_is_adverbial:        b(10),
        self_is_predicative:      b(11),
        self_is_connector:        b(12),
        self_is_discourse:        b(13),
        self_is_functional:       b(14),
        is_reentrant:             b(15),
        has_long_range_dep:       b(16),
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
/// categories could not be resolved from structural evidence alone.
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
/// If `inducer` is provided, deferred tokens are resolved immediately.
/// Otherwise they accumulate in `MtlgGraph::deferred` for batch resolution.
pub fn ud_tree_to_mtlg(
    tree:    &UdTree,
    inducer: Option<&CategoryInducer>,
) -> MtlgGraph {
    let mut deferred = Vec::new();
    let mut nodes    = Vec::new();

    for tok in &tree.tokens {
        let s          = extract_structure(tok, tree);
        let (cat, _)   = score_ucca(&s);
        let arity      = compute_arity(&s);

        let resolved_cat = match cat {
            Some(c) => Some(c),
            None    => match inducer {
                Some(ind) => Some(ind.predict(&s)),
                None      => { deferred.push(s.clone()); None }
            },
        };

        nodes.push(MtlgNode {
            token_id:   tok.id,
            text:       tok.text.clone(),
            lemma:      tok.lemma.clone(),
            upos_raw:   tok.upos.clone(),
            modal_mode: ModalMode::Diamond, // nodes default; edges carry the actual mode
            ucca_cat:   resolved_cat,
            arity,
            structure:  s,
        });
    }

    let mut edges = Vec::new();
    for tok in &tree.tokens {
        if tok.head == 0 { continue; }
        let Some(head_tok) = tree.token_by_id(tok.head) else { continue };

        let s_tok          = extract_structure(tok, tree);
        let (cat, _)       = score_ucca(&s_tok);
        let s_head         = extract_structure(head_tok, tree);
        let modal_mode     = assign_modal_mode(tok, &tok.deprel);

        let resolved_cat = match cat {
            Some(c) => Some(c),
            None    => inducer.map(|ind| ind.predict(&s_tok)),
        };

        edges.push(MtlgEdge {
            src_id:     tok.head,
            dst_id:     tok.id,
            deprel:     tok.deprel.clone(),
            modal_mode,
            ucca_cat:   resolved_cat,
            arity:      compute_arity(&s_head),
        });
    }

    MtlgGraph { nodes, edges, language: tree.language.clone(), deferred }
}

/// Batch-resolve deferred categories across a corpus of `MtlgGraph`s.
///
/// Collects all deferred `TokenStructure`s, fits a `CategoryInducer`,
/// then updates each graph's nodes and edges in-place.
/// Returns the fitted inducer for use on future graphs.
pub fn resolve_deferred(graphs: &mut [MtlgGraph]) -> CategoryInducer {
    let all_deferred: Vec<TokenStructure> = graphs.iter()
        .flat_map(|g| g.deferred.iter().cloned())
        .collect();

    let mut inducer = CategoryInducer::new(16);

    if !all_deferred.is_empty() {
        inducer.fit(&all_deferred);

        for graph in graphs.iter_mut() {
            for node in &mut graph.nodes {
                if node.ucca_cat.is_none() {
                    node.ucca_cat = Some(inducer.predict(&node.structure));
                }
            }
            for edge in &mut graph.edges {
                if edge.ucca_cat.is_none() {
                    // Find the destination node's structure for re-prediction.
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
    ///
    /// `env` is the ATMS environment to stamp on every node.
    /// `base_id` is the starting NodeId; node IDs are `base_id + token_id`.
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
    fn root_token_scores_process() {
        let tree = alice_tree();
        let runs = tree.token_by_id(2).unwrap();
        let s    = extract_structure(runs, &tree);
        assert!(s.is_tree_root);
        let (cat, conf) = score_ucca(&s);
        assert_eq!(cat, Some(TypeCategory(1)));
        assert!(conf >= CONFIDENCE_HIGH - 1e-6);
    }

    #[test]
    fn nsubj_leaf_scores_participant() {
        let tree = alice_tree();
        let alice = tree.token_by_id(1).unwrap();
        let s     = extract_structure(alice, &tree);
        assert!(s.self_is_core_arg);
        assert_eq!(s.n_total_dependents, 0);
        let (cat, _) = score_ucca(&s);
        assert_eq!(cat, Some(TypeCategory(6)));
    }

    #[test]
    fn advmod_leaf_scores_adverbial() {
        let tree    = alice_tree();
        let quickly = tree.token_by_id(3).unwrap();
        let s       = extract_structure(quickly, &tree);
        assert!(s.self_is_adverbial);
        let (cat, _) = score_ucca(&s);
        assert_eq!(cat, Some(TypeCategory(4)));
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
        // Build a small corpus with some deferred tokens.
        let tree1   = alice_tree();
        let tree2   = destruction_tree();
        let mut graphs = vec![
            ud_tree_to_mtlg(&tree1, None),
            ud_tree_to_mtlg(&tree2, None),
        ];

        // Some nodes may be deferred; collect all.
        let total_nodes: usize = graphs.iter().map(|g| g.nodes.len()).sum();
        let _inducer = resolve_deferred(&mut graphs);

        // After resolution, no node should have None category.
        let resolved: usize = graphs.iter()
            .flat_map(|g| g.nodes.iter())
            .filter(|n| n.ucca_cat.is_some())
            .count();
        assert_eq!(resolved, total_nodes, "all nodes must have a category after resolve_deferred");
    }

    #[test]
    fn arity_nonzero_for_head_with_core_args() {
        let tree = alice_tree();
        let runs = tree.token_by_id(2).unwrap();
        let s    = extract_structure(runs, &tree);
        // runs governs one core arg (nsubj)
        assert!(compute_arity(&s) >= 1);
    }

    #[test]
    fn upos_opaque_same_string_same_hash() {
        assert_eq!(hash_string("VERB"), hash_string("VERB"));
        assert_ne!(hash_string("VERB"), hash_string("NOUN"));
    }

    #[test]
    fn no_upos_comparison_in_scoring() {
        // Construct a token with UPOS deliberately set to a nonsense string.
        // Score must still produce a category driven by structural evidence.
        let tree = alice_tree();
        let mut fake_tok = tree.token_by_id(2).unwrap().clone();
        fake_tok.upos = "XYZZY_NOT_A_REAL_UPOS".into();
        // The tree still has Alice (nsubj) and quickly (advmod) under node 2.
        // Because it is the root and governs a core arg, score_ucca should
        // return Process regardless of the UPOS string.
        let s = extract_structure(&fake_tok, &tree);
        let (cat, _) = score_ucca(&s);
        assert_eq!(cat, Some(TypeCategory(1)),
            "category must come from structural evidence, not the UPOS string");
    }
}
