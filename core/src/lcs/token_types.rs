//! Simple token and sentence types — no Universal Dependencies dependency.
//! All structural meaning is learned through teacher forcing.

use std::collections::BTreeSet;

pub fn fnv_hash(s: &str) -> u32 {
    const OFFSET: u32 = 0x811c_9dc5;
    const PRIME:  u32 = 0x0100_0193;
    s.bytes().fold(OFFSET, |h, b| h.wrapping_mul(PRIME) ^ b as u32)
}

pub fn stable_node_id(lemma: &str) -> u64 {
    // FNV-1a 64-bit hash. Replaced SHA-256 (§7): faster, no crypto dep needed.
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME:  u64 = 0x0000_0100_0000_01b3;
    lemma.bytes().fold(OFFSET, |h, b| h.wrapping_mul(PRIME) ^ b as u64)
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
}
