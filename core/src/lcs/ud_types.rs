//! Universal Dependencies token and tree types.
//!
//! These mirror the UD CoNLL-U format. The UPOS field is stored verbatim for
//! auditability but is never used in classification logic — it is hashed to an
//! opaque integer in the converter and treated as an anonymous feature there.

use std::collections::HashMap;

// ── Dependency relation utilities ─────────────────────────────────────────────

/// FNV-1a hash of a deprel string. Produces an opaque u32 feature ID.
///
/// Used to convert UD relation strings into numeric feature dimensions
/// without embedding any linguistic theory about what the relation means.
/// The inducer learns which deprel patterns cluster together from data.
pub fn deprel_hash(deprel: &str) -> u32 {
    const OFFSET: u32 = 0x811c_9dc5;
    const PRIME:  u32 = 0x0100_0193;
    deprel.bytes().fold(OFFSET, |h, b| h.wrapping_mul(PRIME) ^ b as u32)
}

/// Long-range / extracted dependencies → ◊ (Lozenge) modal mode.
///
/// Used for modal mode assignment (structural bookkeeping), not category scoring.
pub fn is_long_range(deprel: &str) -> bool {
    matches!(deprel, "acl:relcl" | "nsubj:outer" | "obj:outer")
}

/// Enhanced UD relations that indicate shared arguments → □ (Box) mode.
const REENTRANT_RELATIONS: &[&str] = &["nsubj:outer", "obj:outer", "nsubj:xsubj"];

// ── Token ─────────────────────────────────────────────────────────────────────

/// A single token from a UD CoNLL-U parse.
///
/// `upos` is stored verbatim for serialization and auditability.
/// It is hashed to an opaque integer in [`crate::lcs::converter`] and never
/// used in any named comparison within the classification pipeline.
#[derive(Clone, Debug)]
pub struct UdToken {
    pub id:     u32,
    pub text:   String,
    pub lemma:  String,
    /// Universal POS tag — stored but never string-compared in scoring.
    pub upos:   String,
    pub xpos:   String,
    /// Head token ID; 0 means this token is (or attached to) the root.
    pub head:   u32,
    pub deprel: String,
    /// Enhanced dependencies string (CoNLL-U `deps` column), used only to
    /// detect reentrancy.
    pub deps:   String,
    /// Morphological features as key → value pairs (e.g. `"Number" → "Sing"`).
    pub feats:  HashMap<String, String>,
}

impl UdToken {
    pub fn new(
        id:     u32,
        text:   impl Into<String>,
        lemma:  impl Into<String>,
        upos:   impl Into<String>,
        xpos:   impl Into<String>,
        head:   u32,
        deprel: impl Into<String>,
    ) -> Self {
        Self {
            id,
            text:   text.into(),
            lemma:  lemma.into(),
            upos:   upos.into(),
            xpos:   xpos.into(),
            head,
            deprel: deprel.into(),
            deps:   String::new(),
            feats:  HashMap::new(),
        }
    }

    /// True when this token is the syntactic root of its sentence.
    pub fn is_root(&self) -> bool {
        self.deprel == "root"
    }

    /// True when the token participates in an enhanced-UD shared argument
    /// relation → UD □ (Box) mode.
    pub fn is_reentrant(&self) -> bool {
        REENTRANT_RELATIONS.iter().any(|r| self.deps.contains(r))
    }
}

// ── Tree ──────────────────────────────────────────────────────────────────────

/// A sentence parsed into a UD dependency tree.
#[derive(Clone, Debug)]
pub struct UdTree {
    pub tokens:   Vec<UdToken>,
    pub language: String,
    pub text:     String,
}

impl UdTree {
    pub fn new(tokens: Vec<UdToken>, language: impl Into<String>, text: impl Into<String>) -> Self {
        Self { tokens, language: language.into(), text: text.into() }
    }

    pub fn token_by_id(&self, id: u32) -> Option<&UdToken> {
        self.tokens.iter().find(|t| t.id == id)
    }

    pub fn dependents_of(&self, head_id: u32) -> Vec<&UdToken> {
        self.tokens.iter().filter(|t| t.head == head_id).collect()
    }

    pub fn root_tokens(&self) -> Vec<&UdToken> {
        self.tokens.iter().filter(|t| t.is_root()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_tree() -> UdTree {
        UdTree::new(
            vec![
                UdToken::new(1, "Alice", "Alice", "PROPN", "NNP", 2, "nsubj"),
                UdToken::new(2, "runs",  "run",   "VERB",  "VBZ", 0, "root"),
                UdToken::new(3, "quickly", "quickly", "ADV", "RB", 2, "advmod"),
            ],
            "en",
            "Alice runs quickly.",
        )
    }

    #[test]
    fn token_by_id() {
        let tree = simple_tree();
        assert_eq!(tree.token_by_id(2).map(|t| t.text.as_str()), Some("runs"));
        assert!(tree.token_by_id(99).is_none());
    }

    #[test]
    fn dependents_of() {
        let tree = simple_tree();
        let deps = tree.dependents_of(2);
        assert_eq!(deps.len(), 2); // nsubj + advmod
    }

    #[test]
    fn root_detection() {
        let tree = simple_tree();
        let roots = tree.root_tokens();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].text, "runs");
    }

    #[test]
    fn deprel_hash_stable_and_distinct() {
        // Same input always produces same hash.
        assert_eq!(deprel_hash("nsubj"), deprel_hash("nsubj"));
        // Distinct inputs produce distinct hashes.
        assert_ne!(deprel_hash("nsubj"), deprel_hash("obj"));
        assert_ne!(deprel_hash("root"), deprel_hash("dep"));
    }

    #[test]
    fn long_range_detection() {
        assert!(is_long_range("acl:relcl"));
        assert!(is_long_range("nsubj:outer"));
        assert!(!is_long_range("nsubj"));
        assert!(!is_long_range("obj"));
    }
}
