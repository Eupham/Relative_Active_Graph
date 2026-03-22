//! Character-to-MTLG converter.
//!
//! Input is a `CharSequence` of Unicode code points. Feature vectors are
//! derived entirely from each character's intrinsic Unicode properties.
//!
//! Design invariants:
//! 1. No string comparison on character content. All properties come from
//!    char::is_alphabetic() etc. and codepoint arithmetic.
//! 2. No hard-coded category assignment. All assignment goes through
//!    CategoryInducer k-means over the 8-dimensional feature vector.
//! 3. Deferred resolution. Tokens accumulate in MtlgGraph::deferred when
//!    no inducer is provided; resolve_deferred batch-resolves them.

use rand::prelude::*;

use crate::types::{
    TypeCategory, ModalMode, ModalType, Direction, NodeId, EdgeId, Env,
};
use crate::arg::{ArgNode, ArgEdge, NodeClass, EdgeClass};
use super::ud_types::CharSequence;

// ── Character feature structure ───────────────────────────────────────────────

/// Intrinsic characterisation of a Unicode code point.
/// All fields are derivable from the character itself; no external model.
#[derive(Clone, Debug)]
pub struct CharTokenStructure {
    pub token_id:       u32,
    pub is_alphabetic:  bool,
    pub is_uppercase:   bool,
    pub is_lowercase:   bool,
    pub is_numeric:     bool,
    pub is_punctuation: bool,
    pub is_whitespace:  bool,
    /// 1–4. ASCII=1, Latin-ext/Greek/Cyrillic≈2, CJK/BMP≈3, SMP=4.
    pub utf8_byte_len:  u8,
    /// codepoint % 10_000: fine-grained identity without full Unicode table.
    pub codepoint_mod:  u32,
}

impl CharTokenStructure {
    pub fn from_char(id: u32, ch: char) -> Self {
        Self {
            token_id:       id,
            is_alphabetic:  ch.is_alphabetic(),
            is_uppercase:   ch.is_uppercase(),
            is_lowercase:   ch.is_lowercase(),
            is_numeric:     ch.is_numeric(),
            is_punctuation: ch.is_ascii_punctuation()
                || matches!(ch, '。'|'、'|'！'|'？'|'…'|'—'|'–'),
            is_whitespace:  ch.is_whitespace(),
            utf8_byte_len:  ch.to_string().len() as u8,
            codepoint_mod:  (ch as u32) % 10_000,
        }
    }
}

/// Convert a `CharTokenStructure` to an 8-dimensional float vector.
///
/// [0] is_alphabetic  [1] is_uppercase  [2] is_lowercase
/// [3] is_numeric     [4] is_punctuation [5] is_whitespace
/// [6] (utf8_byte_len−1)/3.0  (script family: 0=ASCII … 1.0=4-byte)
/// [7] codepoint_mod/10_000.0 (fine-grained identity)
pub fn structure_to_vector(s: &CharTokenStructure) -> Vec<f32> {
    vec![
        s.is_alphabetic  as u8 as f32,
        s.is_uppercase   as u8 as f32,
        s.is_lowercase   as u8 as f32,
        s.is_numeric     as u8 as f32,
        s.is_punctuation as u8 as f32,
        s.is_whitespace  as u8 as f32,
        s.utf8_byte_len.saturating_sub(1) as f32 / 3.0,
        s.codepoint_mod as f32 / 10_000.0,
    ]
}

// ── MTLG types ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct MtlgNode {
    pub token_id:   u32,
    pub text:       String,
    pub lemma:      String,   // == text for character tokens
    pub modal_mode: ModalMode,
    pub ucca_cat:   Option<TypeCategory>,
    pub arity:      u8,
    pub structure:  CharTokenStructure,
}

#[derive(Clone, Debug)]
pub struct MtlgEdge {
    pub src_id:     u32,
    pub dst_id:     u32,
    pub modal_mode: ModalMode,
    pub ucca_cat:   Option<TypeCategory>,
    pub arity:      u8,
}

#[derive(Debug)]
pub struct MtlgGraph {
    pub nodes:    Vec<MtlgNode>,
    pub edges:    Vec<MtlgEdge>,
    pub language: String,
    pub deferred: Vec<CharTokenStructure>,
}

impl MtlgGraph {
    pub fn to_arg_nodes_and_edges(
        &self,
        env:     Env,
        base_id: NodeId,
    ) -> (Vec<ArgNode>, Vec<ArgEdge>) {
        let nodes: Vec<ArgNode> = self.nodes.iter().map(|mn| {
            let cat = mn.ucca_cat.unwrap_or(TypeCategory::DEFAULT);
            let mt  = ModalType::functor(mn.modal_mode, cat, mn.arity, Direction::Right);
            // Character identity matches Python's _stable_node_id.
            let nid = stable_node_id_for(&mn.text);
            let mut n = ArgNode::new(nid, NodeClass::DEFAULT, mt, (0, 0));
            n.surface    = Some(mn.text.as_bytes().to_vec());
            n.atms_label = env;
            n
        }).collect();

        let edges: Vec<ArgEdge> = self.edges.iter().enumerate().map(|(i, me)| {
            let src_text = self.nodes.iter()
                .find(|n| n.token_id == me.src_id)
                .map(|n| n.text.as_str()).unwrap_or("");
            let dst_text = self.nodes.iter()
                .find(|n| n.token_id == me.dst_id)
                .map(|n| n.text.as_str()).unwrap_or("");
            ArgEdge::new(
                base_id + i as u64 + 1000,
                stable_node_id_for(src_text),
                stable_node_id_for(dst_text),
                EdgeClass::SEQUENTIAL,
                me.modal_mode,
            )
        }).collect();

        (nodes, edges)
    }
}

/// Stable 48-bit hash of a character string matching Python's `_stable_node_id`.
fn stable_node_id_for(text: &str) -> NodeId {
    use sha2::{Sha256, Digest};
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    let result = h.finalize();
    u64::from_be_bytes(result[..8].try_into().unwrap_or([0u8; 8])) & 0x0000_FFFF_FFFF_FFFF
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert a `CharSequence` to an `MtlgGraph`.
///
/// Sequential edges connect each character to the next. All modal modes
/// start as Diamond; CategoryInducer assigns structural roles from
/// distributional context during training.
pub fn char_sequence_to_mtlg(
    seq:     &CharSequence,
    inducer: Option<&CategoryInducer>,
) -> MtlgGraph {
    let mut deferred = Vec::new();
    let mut nodes    = Vec::new();

    for tok in &seq.tokens {
        let s = CharTokenStructure::from_char(tok.id, tok.char());
        let resolved_cat = match inducer {
            Some(ind) => Some(ind.predict(&s)),
            None      => { deferred.push(s.clone()); None }
        };
        nodes.push(MtlgNode {
            token_id:   tok.id,
            text:       tok.text.clone(),
            lemma:      tok.text.clone(),
            modal_mode: ModalMode::Diamond,
            ucca_cat:   resolved_cat,
            arity:      0,
            structure:  s,
        });
    }

    let edges: Vec<MtlgEdge> = seq.tokens.windows(2).map(|w| MtlgEdge {
        src_id:     w[0].id,
        dst_id:     w[1].id,
        modal_mode: ModalMode::Diamond,
        ucca_cat:   None,
        arity:      0,
    }).collect();

    MtlgGraph { nodes, edges, language: seq.language.clone(), deferred }
}

// ── CategoryInducer ───────────────────────────────────────────────────────────

pub struct CategoryInducer {
    pub centroids:  Vec<Vec<f32>>,
    pub n_clusters: usize,
}

impl CategoryInducer {
    pub fn new(n_clusters: usize) -> Self {
        Self { centroids: Vec::new(), n_clusters }
    }

    pub fn predict(&self, s: &CharTokenStructure) -> TypeCategory {
        if self.centroids.is_empty() { return TypeCategory::DEFAULT; }
        let v = structure_to_vector(s);
        let best = self.centroids.iter().enumerate()
            .map(|(i, c)| (i, squared_dist(&v, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
        TypeCategory((best + 1) as u32)
    }

    pub fn fit(&mut self, structures: &[CharTokenStructure], rng: &mut StdRng) {
        let vectors: Vec<Vec<f32>> = structures.iter().map(structure_to_vector).collect();
        if vectors.is_empty() { return; }
        let (centroids, _, _) = run_kmeans(&vectors, self.n_clusters, rng);
        self.centroids = centroids;
    }
}

/// Batch-resolve deferred categories. Fits CategoryInducer over all structures
/// in the corpus and updates every node in-place.
pub fn resolve_deferred(graphs: &mut [MtlgGraph]) -> CategoryInducer {
    let all_structures: Vec<CharTokenStructure> = graphs.iter()
        .flat_map(|g| g.nodes.iter().map(|n| n.structure.clone()))
        .collect();

    let k = 16.min(all_structures.len().max(1));
    let mut inducer = CategoryInducer::new(k);
    if !all_structures.is_empty() {
        let mut rng = StdRng::seed_from_u64(42);
        inducer.fit(&all_structures, &mut rng);
    }

    for graph in graphs.iter_mut() {
        for node in &mut graph.nodes {
            if node.ucca_cat.is_none() {
                node.ucca_cat = Some(inducer.predict(&node.structure));
            }
        }
        graph.deferred.clear();
    }
    inducer
}

// ── K-means helpers ───────────────────────────────────────────────────────────

fn squared_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum()
}

fn run_kmeans(
    vectors: &[Vec<f32>],
    k:       usize,
    rng:     &mut StdRng,
) -> (Vec<Vec<f32>>, Vec<usize>, f64) {
    let n = vectors.len();
    if n == 0 || k == 0 { return (vec![], vec![], 0.0); }
    let k = k.min(n);
    let dim = vectors[0].len();

    // k-means++ init
    let mut center_idx: Vec<usize> = vec![rng.gen_range(0..n)];
    while center_idx.len() < k {
        let dists: Vec<f64> = vectors.iter()
            .map(|v| center_idx.iter()
                .map(|&ci| squared_dist(v, &vectors[ci]))
                .fold(f64::MAX, f64::min))
            .collect();
        let total: f64 = dists.iter().sum();
        if total == 0.0 { break; }
        let mut r = rng.gen::<f64>() * total;
        let next = dists.iter().position(|&d| { r -= d; r <= 0.0 }).unwrap_or(n - 1);
        center_idx.push(next);
    }

    let mut centroids: Vec<Vec<f32>> = center_idx.iter().map(|&i| vectors[i].clone()).collect();
    let mut assignments = vec![0usize; n];

    for _ in 0..20 {
        let mut changed = false;
        for (i, v) in vectors.iter().enumerate() {
            let best = centroids.iter().enumerate()
                .map(|(ci, c)| (ci, squared_dist(v, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(ci, _)| ci).unwrap_or(0);
            if assignments[i] != best { changed = true; assignments[i] = best; }
        }
        if !changed { break; }
        let mut sums   = vec![vec![0.0f64; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, &c) in assignments.iter().enumerate() {
            for (j, &x) in vectors[i].iter().enumerate() { sums[c][j] += x as f64; }
            counts[c] += 1;
        }
        for (c, sum) in sums.iter().enumerate() {
            if counts[c] > 0 {
                centroids[c] = sum.iter().map(|&s| (s / counts[c] as f64) as f32).collect();
            }
        }
    }

    let inertia: f64 = vectors.iter().enumerate()
        .map(|(i, v)| squared_dist(v, &centroids[assignments[i]])).sum();
    (centroids, assignments, inertia)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lcs::ud_types::CharSequence;

    #[test]
    fn conversion_node_and_edge_counts() {
        let seq   = CharSequence::from_str("hello", "en");
        let graph = char_sequence_to_mtlg(&seq, None);
        assert_eq!(graph.nodes.len(), 5);
        assert_eq!(graph.edges.len(), 4); // h→e e→l l→l l→o
    }

    #[test]
    fn all_edges_diamond_mode() {
        let seq   = CharSequence::from_str("hi", "en");
        let graph = char_sequence_to_mtlg(&seq, None);
        for e in &graph.edges { assert_eq!(e.modal_mode, ModalMode::Diamond); }
    }

    #[test]
    fn deferred_resolved_by_inducer() {
        let mut graphs = vec![
            char_sequence_to_mtlg(&CharSequence::from_str("hello world", "en"), None),
            char_sequence_to_mtlg(&CharSequence::from_str("foo bar baz", "en"), None),
        ];
        let total: usize = graphs.iter().map(|g| g.nodes.len()).sum();
        resolve_deferred(&mut graphs);
        let resolved: usize = graphs.iter()
            .flat_map(|g| g.nodes.iter())
            .filter(|n| n.ucca_cat.is_some()).count();
        assert_eq!(resolved, total);
        assert!(graphs.iter().all(|g| g.deferred.is_empty()));
    }

    #[test]
    fn feature_vector_is_8_dimensional() {
        let s = CharTokenStructure::from_char(1, 'A');
        assert_eq!(structure_to_vector(&s).len(), 8);
    }

    #[test]
    fn ascii_script_family_is_zero() {
        let s = CharTokenStructure::from_char(1, 'z');
        let v = structure_to_vector(&s);
        assert_eq!(v[6], 0.0); // (1-1)/3 = 0
    }

    #[test]
    fn cjk_script_family_near_two_thirds() {
        let s = CharTokenStructure::from_char(1, '中');
        let v = structure_to_vector(&s);
        assert!((v[6] - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn stable_node_id_is_48_bit() {
        let id = stable_node_id_for("a");
        assert!(id <= 0x0000_FFFF_FFFF_FFFF);
    }

    #[test]
    fn to_arg_nodes_preserves_surface() {
        let seq   = CharSequence::from_str("hi", "en");
        let graph = char_sequence_to_mtlg(&seq, None);
        let (nodes, _) = graph.to_arg_nodes_and_edges(0b1, 0);
        let surfaces: Vec<&str> = nodes.iter()
            .filter_map(|n| n.surface.as_deref()
                .and_then(|b| std::str::from_utf8(b).ok()))
            .collect();
        assert!(surfaces.contains(&"h"));
        assert!(surfaces.contains(&"i"));
    }
}
