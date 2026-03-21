//! Universal Dependencies token and tree types.
//!
//! These mirror the UD CoNLL-U format. The UPOS field is stored verbatim for
//! auditability but is never used in classification logic — it is hashed to an
//! opaque integer in the converter and treated as an anonymous feature there.

use std::collections::HashMap;

// ── Dependency relation sets ──────────────────────────────────────────────────
// These describe *structural roles* in the dependency tree, not categorical
// identity of the token.  Using them as structural evidence is correct;
// comparing tok.upos to "VERB" would not be.

/// Relations in which the governed token fills a thematic argument role.
pub fn is_core_arg(deprel: &str) -> bool {
    matches!(deprel, "nsubj" | "obj" | "iobj" | "csubj" | "ccomp" | "xcomp")
}

/// Relations in which the governed token heads a subordinate clause.
pub fn is_clausal(deprel: &str) -> bool {
    matches!(deprel, "ccomp" | "xcomp" | "advcl" | "csubj" | "acl" | "acl:relcl")
}

/// Adverbial modification relations (manner, time, place, degree).
pub fn is_adverbial(deprel: &str) -> bool {
    matches!(deprel, "advmod" | "obl" | "dislocated")
}

/// Predicative modification relations (descriptive, property-bearing).
pub fn is_predicative(deprel: &str) -> bool {
    matches!(deprel, "amod" | "cop")
}

/// Coordination and subordination connectors.
pub fn is_connector(deprel: &str) -> bool {
    matches!(deprel, "cc" | "mark" | "punct" | "conj")
}

/// Discourse and vocative grounding.
pub fn is_discourse(deprel: &str) -> bool {
    matches!(deprel, "discourse" | "vocative")
}

/// Functional / auxiliary elements: do not project independent argument structure.
pub fn is_functional(deprel: &str) -> bool {
    matches!(
        deprel,
        "aux" | "det" | "case" | "clf" | "fixed" | "flat" | "compound"
            | "goeswith" | "reparandum" | "orphan" | "expl"
    )
}

/// Long-range / extracted dependencies → ◊ (Lozenge) mode.
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
    fn deprel_sets_are_disjoint_for_typical_cases() {
        assert!(is_core_arg("nsubj"));
        assert!(!is_functional("nsubj"));
        assert!(is_functional("aux"));
        assert!(!is_core_arg("aux"));
        assert!(is_connector("cc"));
        assert!(is_discourse("discourse"));
    }
}
