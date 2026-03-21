//! ARG → λ-expression → MTLG derivation → surface form.
//! Per-language lexicon lookup: each predicate/concept has a surface realization
//! in the target language. Language selected by TRD / query context.

use std::collections::HashMap;
use crate::types::{NodeId, TRDId, ModalType, ModalMode, TypeCategory};
use crate::arg::{ArgGraph, ArgNode};
use crate::semantics::mtlg_semantics::{MtlgSemantics, LambdaTerm, PropositionGraph};
use crate::generation::hypothesis::Hypothesis;

/// Per-language lexicon entry.
#[derive(Clone, Debug)]
pub struct LexEntry {
    pub predicate:   String,
    pub language:    String,
    pub surface:     String,
    pub modal_type:  ModalType,
}

/// Per-language lexicon (induced from mC4 in the LCS pipeline).
pub struct PerLanguageLexicon {
    /// language_code → predicate → surface.
    entries: HashMap<String, HashMap<String, LexEntry>>,
    /// Fallback language (when target language has no entry).
    default_lang: String,
}

impl PerLanguageLexicon {
    pub fn new(default_lang: impl Into<String>) -> Self {
        Self {
            entries:      HashMap::new(),
            default_lang: default_lang.into(),
        }
    }

    pub fn register(&mut self, entry: LexEntry) {
        self.entries
            .entry(entry.language.clone())
            .or_default()
            .insert(entry.predicate.clone(), entry);
    }

    /// Look up surface form for `predicate` in `language`.
    pub fn surface_for(&self, predicate: &str, language: &str) -> Option<&str> {
        self.entries.get(language)
            .and_then(|lex| lex.get(predicate))
            .map(|e| e.surface.as_str())
            .or_else(|| {
                // Fallback to default language.
                self.entries.get(&self.default_lang)
                    .and_then(|lex| lex.get(predicate))
                    .map(|e| e.surface.as_str())
            })
    }
}

/// The linearizer: converts a Hypothesis to a surface string.
pub struct Linearizer {
    pub lexicon:  PerLanguageLexicon,
    pub language: String,
}

impl Linearizer {
    pub fn new(language: impl Into<String>) -> Self {
        let lang = language.into();
        Self {
            lexicon:  PerLanguageLexicon::new(lang.clone()),
            language: lang,
        }
    }

    /// Linearize a hypothesis into a surface string in the target language.
    pub fn linearize(&self, hyp: &Hypothesis) -> String {
        self.proposition_to_surface(&hyp.proposition)
    }

    /// Convert a PropositionGraph to surface form via lexicon lookup.
    pub fn proposition_to_surface(&self, prop: &PropositionGraph) -> String {
        let root_surface = self.lexicon
            .surface_for(&prop.root, &self.language)
            .unwrap_or(&prop.root)
            .to_string();

        if prop.roles.is_empty() {
            return root_surface;
        }

        // Simple SOV linearization (language-specific word order would be TRD-parameterized).
        let args: Vec<String> = prop.roles.iter()
            .map(|(role, arg)| {
                let arg_surface = self.lexicon
                    .surface_for(arg, &self.language)
                    .unwrap_or(arg)
                    .to_string();
                arg_surface
            })
            .collect();

        format!("{} {}", root_surface, args.join(" "))
    }

    /// Linearize a λ-term to surface: evaluate, then look up in lexicon.
    pub fn lambda_to_surface(&self, semantics: &MtlgSemantics, term: LambdaTerm) -> String {
        let prop = semantics.sentence_level(term);
        self.proposition_to_surface(&prop)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::hypothesis::Hypothesis;
    use crate::semantics::mtlg_semantics::PropositionGraph;

    #[test]
    fn linearizes_with_lexicon() {
        let mut lin = Linearizer::new("en");
        lin.lexicon.register(LexEntry {
            predicate: "run".into(),
            language:  "en".into(),
            surface:   "runs".into(),
            modal_type: ModalType::default(),
        });
        lin.lexicon.register(LexEntry {
            predicate: "alice".into(),
            language:  "en".into(),
            surface:   "Alice".into(),
            modal_type: ModalType::default(),
        });

        let prop = PropositionGraph {
            root:    "run".into(),
            roles:   vec![("ARG0".into(), "alice".into())],
            lambda_str: "run(alice)".into(),
        };
        let hyp = Hypothesis {
            id: 0,
            lambda_str:  "run(alice)".into(),
            proposition: prop,
            relevance:   0.9,
            root_node:   1,
        };
        let surface = lin.linearize(&hyp);
        assert_eq!(surface, "runs Alice");
    }

    #[test]
    fn falls_back_to_predicate_when_no_entry() {
        let lin = Linearizer::new("en");
        let prop = PropositionGraph {
            root:    "unknown_pred".into(),
            roles:   vec![],
            lambda_str: "unknown_pred".into(),
        };
        let hyp = Hypothesis { id:0, lambda_str: "unknown_pred".into(), proposition: prop, relevance: 0.5, root_node: 1 };
        assert_eq!(lin.linearize(&hyp), "unknown_pred");
    }
}
