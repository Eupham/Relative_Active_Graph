//! Character-sequence token and sequence types for the LCS pipeline.
//!
//! Replaces the UD CoNLL-U types. The vocabulary is now Unicode code points.
//! Structural roles are discovered by the CategoryInducer, not assigned by
//! a parser.

// ── Hash utilities ────────────────────────────────────────────────────────────

/// FNV-1a hash of an arbitrary string slice to a stable u32.
/// Retained for bootstrap artefact compatibility.
pub fn deprel_hash(s: &str) -> u32 {
    const OFFSET: u32 = 0x811c_9dc5;
    const PRIME:  u32 = 0x0100_0193;
    s.bytes().fold(OFFSET, |h, b| h.wrapping_mul(PRIME) ^ b as u32)
}

/// Returns true for UD relation strings that indicate long-range extraction.
/// Retained for bootstrap artefact compatibility; not used in the
/// character-level training path.
pub fn is_long_range(deprel: &str) -> bool {
    matches!(deprel, "acl:relcl" | "nsubj:outer" | "obj:outer")
}

// ── Character token ───────────────────────────────────────────────────────────

/// A single Unicode code point in a character sequence.
#[derive(Clone, Debug)]
pub struct CharToken {
    /// Sequential position (1-indexed).
    pub id:        u32,
    /// The character as a UTF-8 string.
    pub text:      String,
    /// Unicode scalar value.
    pub codepoint: u32,
}

impl CharToken {
    pub fn new(id: u32, ch: char) -> Self {
        Self { id, text: ch.to_string(), codepoint: ch as u32 }
    }

    pub fn char(&self) -> char {
        char::from_u32(self.codepoint).unwrap_or('\u{FFFD}')
    }
}

// ── Character sequence ────────────────────────────────────────────────────────

/// A sentence decomposed into Unicode code points.
/// Replaces UdTree as the input to the LCS converter.
#[derive(Clone, Debug)]
pub struct CharSequence {
    pub tokens:   Vec<CharToken>,
    pub language: String,
    pub text:     String,
}

impl CharSequence {
    pub fn from_str(text: &str, language: impl Into<String>) -> Self {
        let tokens = text.chars().enumerate()
            .map(|(i, ch)| CharToken::new(i as u32 + 1, ch))
            .collect();
        Self { tokens, language: language.into(), text: text.to_string() }
    }

    pub fn token_by_id(&self, id: u32) -> Option<&CharToken> {
        self.tokens.iter().find(|t| t.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_sequence_from_str() {
        let seq = CharSequence::from_str("abc", "en");
        assert_eq!(seq.tokens.len(), 3);
        assert_eq!(seq.tokens[0].text, "a");
        assert_eq!(seq.tokens[2].codepoint, 'c' as u32);
    }

    #[test]
    fn char_token_roundtrip_cjk() {
        let tok = CharToken::new(1, '가');
        assert_eq!(tok.char(), '가');
        assert_eq!(tok.text, "가");
    }

    #[test]
    fn deprel_hash_stable() {
        assert_eq!(deprel_hash("nsubj"), deprel_hash("nsubj"));
        assert_ne!(deprel_hash("nsubj"), deprel_hash("obj"));
    }
}
