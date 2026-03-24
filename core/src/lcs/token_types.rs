//! Simple token and sentence types — no Universal Dependencies dependency.
//! All structural meaning is learned through teacher forcing.

use std::collections::BTreeSet;
use crate::types::ModalMode;

pub fn fnv_hash(s: &str) -> u32 {
    const OFFSET: u32 = 0x811c_9dc5;
    const PRIME:  u32 = 0x0100_0193;
    s.bytes().fold(OFFSET, |h, b| (h ^ b as u32).wrapping_mul(PRIME))
}

/// General-purpose 64-bit FNV-1a over raw bytes.
/// All 64-bit hashing in the engine routes through this function.
pub fn fnv1a_64_bytes(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME:  u64 = 0x0000_0100_0000_01b3;
    bytes.iter().fold(OFFSET, |h, &b| (h ^ b as u64).wrapping_mul(PRIME))
}

pub fn stable_node_id(lemma: &str) -> u64 {
    fnv1a_64_bytes(lemma.as_bytes())
}

/// Context-free identity for a token structure.
/// This is the base e-class-like identity (morphological fingerprint only).
pub fn base_node_id_from_structure(s: &TokenStructure) -> u64 {
    let mut b = [0u8; 8];
    b[0..4].copy_from_slice(&s.suffix3_hash.to_le_bytes());
    b[4..8].copy_from_slice(&s.prefix2_hash.to_le_bytes());
    fnv1a_64_bytes(&b)
}

/// Map ModalMode to a single byte for hashing.
pub fn mode_to_u8(mode: ModalMode) -> u8 {
    match mode {
        ModalMode::Diamond  => 0,
        ModalMode::Box      => 1,
        ModalMode::Lozenge  => 2,
    }
}

/// Compose a contextual NodeId from a context-free base identity and context mask.
///
/// Layout hashed (all little-endian, 17 bytes total):
///   [base_node_id u64] [env u64] [mode_byte u8]
pub fn contextual_node_id_from_base(base_node_id: u64, env: u64, mode: ModalMode) -> u64 {
    let mut b = [0u8; 17];
    b[0..8].copy_from_slice(&base_node_id.to_le_bytes());
    b[8..16].copy_from_slice(&env.to_le_bytes());
    b[16] = mode_to_u8(mode);
    fnv1a_64_bytes(&b)
}

/// True when `node_env` contains every bit in `query_mask`.
#[inline]
pub fn context_mask_matches(node_env: u64, query_mask: u64) -> bool {
    (node_env & query_mask) == query_mask
}

/// Context-aware 64-bit node identity spanning the external character graph
/// and the internal ARG hypergraph.
///
/// Layout hashed (all little-endian, 17 bytes total):
///   [suffix3_hash u32] [prefix2_hash u32] [env u64] [mode_byte u8]
///
/// Two tokens with the same morphological shape (suffix3+prefix2)
/// are treated as the same external-graph class, but separate internal
/// states (different Env bitmasks or ModalMode roles) produce distinct
/// NodeIds — bridge between external context and internal hypergraph.
///
/// Python parity: `contextual_node_id` in `lcs/induction/tokenizer.py`
/// must produce identical results from the same byte sequence.
pub fn contextual_node_id(s: &TokenStructure, env: u64, mode: ModalMode) -> u64 {
    let mut b = [0u8; 17];
    b[0..4].copy_from_slice(&s.suffix3_hash.to_le_bytes());
    b[4..8].copy_from_slice(&s.prefix2_hash.to_le_bytes());
    b[8..16].copy_from_slice(&env.to_le_bytes());
    b[16] = mode_to_u8(mode);
    fnv1a_64_bytes(&b)
}

/// Infer ModalMode from structural features of the external character graph.
///
/// - `Box` (□): repeated tokens — structural contraction, shared sub-graph node
/// - `Lozenge` (◊): mid-to-late non-boundary position — potential displacement
/// - `Diamond` (◇): boundary or early position — primary linear composition
pub fn infer_modal_mode(s: &TokenStructure) -> ModalMode {
    if s.is_repeated {
        ModalMode::Box        // shared/contraction: same surface recurs
    } else if s.normalized_position > 0.5 && !s.is_last_token {
        ModalMode::Lozenge    // mid-to-late non-boundary: potential displacement
    } else {
        ModalMode::Diamond    // boundary or early: primary composition
    }
}

#[derive(Clone, Debug)]
pub struct Token {
    pub id:    u32,
    pub text:  String,
    pub lemma: String,
}

impl Token {
    pub fn new(id: u32, text: impl Into<String>) -> Self {
        let text  = text.into();
        let lemma = text.to_lowercase();
        Self { id, text, lemma }
    }
    pub fn node_id(&self) -> u64 { stable_node_id(&self.lemma) }
}

#[derive(Clone, Debug)]
pub struct TokenSentence {
    pub tokens:   Vec<Token>,
    pub language: String,
    pub text:     String,
}

impl TokenSentence {
    pub fn new(tokens: Vec<Token>, language: impl Into<String>, text: impl Into<String>) -> Self {
        Self { tokens, language: language.into(), text: text.into() }
    }
    pub fn token_by_id(&self, id: u32) -> Option<&Token> {
        self.tokens.iter().find(|t| t.id == id)
    }
    pub fn is_repeated(&self, lemma: &str) -> bool {
        self.tokens.iter().filter(|t| t.lemma == lemma).count() > 1
    }
}

/// Structural feature record derived from surface properties only.
/// No POS tags, no dependency labels.
#[derive(Clone, Debug)]
pub struct TokenStructure {
    pub token_id:             u32,
    pub is_first_token:       bool,
    pub is_last_token:        bool,
    pub normalized_position:  f32,
    pub sentence_length_norm: f32,
    pub starts_with_uppercase: bool,
    pub is_punctuation:       bool,
    pub is_repeated:          bool,
    pub char_length_norm:     f32,
    pub prefix2_hash:         u32,
    pub suffix3_hash:         u32,
    pub suffix2_hash:         u32,
    pub prev_lemma_hash:      u32,
    pub next_lemma_hash:      u32,
    pub n_context_neighbors:  usize,
    pub char_trigram_hashes:  BTreeSet<u32>,
}

pub fn extract_features(tok: &Token, sentence: &TokenSentence) -> TokenStructure {
    let n   = sentence.tokens.len().max(1);
    let pos = tok.id as usize - 1;
    let prev = sentence.tokens.get(pos.wrapping_sub(1));
    let next = sentence.tokens.get(pos + 1);
    let chars: Vec<char> = tok.text.chars().collect();
    let len = chars.len().max(1);

    let suffix3: String = chars[len.saturating_sub(3)..].iter().collect();
    let suffix2: String = chars[len.saturating_sub(2)..].iter().collect();
    let prefix2: String = chars[..len.min(2)].iter().collect();

    let mut trigrams = BTreeSet::new();
    let padded = format!("_{}_", tok.text.to_lowercase());
    let pchars: Vec<char> = padded.chars().collect();
    for w in pchars.windows(3) {
        trigrams.insert(fnv_hash(&w.iter().collect::<String>()));
    }

    TokenStructure {
        token_id:              tok.id,
        is_first_token:        tok.id == 1,
        is_last_token:         tok.id as usize == n,
        normalized_position:   if n > 1 { pos as f32 / (n - 1) as f32 } else { 0.0 },
        sentence_length_norm:  (n as f32 / 20.0).min(1.0),
        starts_with_uppercase: chars.first().map(|c| c.is_uppercase()).unwrap_or(false),
        is_punctuation:        tok.text.chars().all(|c| c.is_ascii_punctuation()),
        is_repeated:           sentence.is_repeated(&tok.lemma),
        char_length_norm:      (len as f32 / 15.0).min(1.0),
        prefix2_hash:          fnv_hash(&prefix2),
        suffix3_hash:          fnv_hash(&suffix3),
        suffix2_hash:          fnv_hash(&suffix2),
        prev_lemma_hash:       prev.map(|t| fnv_hash(&t.lemma)).unwrap_or(0),
        next_lemma_hash:       next.map(|t| fnv_hash(&t.lemma)).unwrap_or(0),
        n_context_neighbors:   prev.is_some() as usize + next.is_some() as usize,
        char_trigram_hashes:   trigrams,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sentence(words: &[&str]) -> TokenSentence {
        TokenSentence::new(
            words.iter().enumerate().map(|(i, &w)| Token::new(i as u32 + 1, w)).collect(),
            "en", words.join(" "),
        )
    }

    #[test]
    fn first_last_flags() {
        let s = sentence(&["Alice", "runs", "quickly"]);
        let f0 = extract_features(&s.tokens[0], &s);
        let f2 = extract_features(&s.tokens[2], &s);
        assert!(f0.is_first_token && !f0.is_last_token);
        assert!(f2.is_last_token && !f2.is_first_token);
    }

    #[test]
    fn repeated_detection() {
        let s = sentence(&["the", "cat", "sat", "on", "the", "mat"]);
        assert!(extract_features(&s.tokens[0], &s).is_repeated);
        assert!(!extract_features(&s.tokens[1], &s).is_repeated);
    }

    #[test]
    fn stable_node_id_deterministic() {
        assert_eq!(stable_node_id("alice"), stable_node_id("alice"));
        assert_ne!(stable_node_id("alice"), stable_node_id("runs"));
    }

    fn make_structure(suffix3: u32, prefix2: u32) -> TokenStructure {
        TokenStructure {
            token_id: 1, is_first_token: false, is_last_token: false,
            normalized_position: 0.3, sentence_length_norm: 0.5,
            starts_with_uppercase: false, is_punctuation: false,
            is_repeated: false, char_length_norm: 0.2,
            prefix2_hash: prefix2, suffix3_hash: suffix3,
            suffix2_hash: 0, prev_lemma_hash: 0, next_lemma_hash: 0,
            n_context_neighbors: 0, char_trigram_hashes: BTreeSet::new(),
        }
    }

    #[test]
    fn contextual_node_id_deterministic() {
        let s = make_structure(100, 200);
        let a = contextual_node_id(&s, 1, ModalMode::Diamond);
        let b = contextual_node_id(&s, 1, ModalMode::Diamond);
        assert_eq!(a, b, "same inputs must produce same id");
    }

    #[test]
    fn contextual_node_id_env_distinguishes() {
        let s = make_structure(100, 200);
        let a = contextual_node_id(&s, 1, ModalMode::Diamond);
        let b = contextual_node_id(&s, 2, ModalMode::Diamond);
        assert_ne!(a, b, "different env must produce different id");
    }

    #[test]
    fn contextual_node_id_mode_distinguishes() {
        let s = make_structure(100, 200);
        let a = contextual_node_id(&s, 1, ModalMode::Diamond);
        let b = contextual_node_id(&s, 1, ModalMode::Box);
        assert_ne!(a, b, "different mode must produce different id");
    }

    #[test]
    fn contextual_node_id_structure_change() {
        let a = contextual_node_id(&make_structure(100, 200), 0, ModalMode::Diamond);
        let b = contextual_node_id(&make_structure(999, 200), 0, ModalMode::Diamond);
        assert_ne!(a, b, "different suffix3 must produce different id");
    }

    #[test]
    fn context_mask_matches_subset() {
        assert!(context_mask_matches(0b10110, 0b00110));
        assert!(!context_mask_matches(0b10110, 0b11000));
    }

    #[test]
    fn infer_modal_mode_repeated_is_box() {
        let mut s = make_structure(0, 0);
        s.is_repeated = true;
        assert_eq!(infer_modal_mode(&s), ModalMode::Box);
    }

    #[test]
    fn infer_modal_mode_late_position_is_lozenge() {
        let mut s = make_structure(0, 0);
        s.normalized_position = 0.7;
        s.is_last_token = false;
        assert_eq!(infer_modal_mode(&s), ModalMode::Lozenge);
    }

    #[test]
    fn infer_modal_mode_boundary_is_diamond() {
        let s = make_structure(0, 0);
        assert_eq!(infer_modal_mode(&s), ModalMode::Diamond);
    }
}
