//! ARG → λ-expression → MTLG derivation → surface form.
//! Per-language lexicon lookup: each predicate/concept has a surface realization
//! in the target language. Language selected by TRD / query context.

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, TRDId, ModalType, ModalMode, TypeCategory, Env, Direction};
use crate::arg::{ArgGraph, ArgNode};
use crate::semantics::mtlg_semantics::{MtlgSemantics, LambdaTerm, PropositionGraph};
use crate::generation::hypothesis::Hypothesis;

/// One step in a sequential linearization: which node/edge produced which surface token.
#[derive(Clone, Debug)]
pub struct LinearizationStep {
    pub step:     usize,
    pub surface:  String,
    pub node_id:  NodeId,
    pub edge_id:  EdgeId,
}

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

    /// Return the full LexEntry for `predicate` in `language`, with fallback to default_lang.
    pub fn entry_for(&self, predicate: &str, language: &str) -> Option<&LexEntry> {
        self.entries.get(language)
            .and_then(|lex| lex.get(predicate))
            .or_else(|| {
                self.entries.get(&self.default_lang)
                    .and_then(|lex| lex.get(predicate))
            })
    }

    /// Look up surface form for `predicate` in `language`.
    pub fn surface_for(&self, predicate: &str, language: &str) -> Option<&str> {
        self.entry_for(predicate, language).map(|e| e.surface.as_str())
    }

    /// Reverse lookup: find the surface form for a NodeId.
    ///
    /// NodeIds in the global lexicon are derived from stable_node_id(predicate),
    /// so we can re-hash each entry's predicate to find a match.
    /// This is O(|lexicon|) and intended only as a fallback for abstract nodes.
    pub fn surface_for_node_id(&self, node_id: NodeId, language: &str) -> Option<&str> {
        use crate::lcs::stable_node_id;
        let lex = self.entries.get(language)
            .or_else(|| self.entries.get(&self.default_lang))?;
        lex.values().find(|entry| {
            stable_node_id(&entry.predicate) == node_id
        }).map(|e| e.surface.as_str())
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
    /// Always delegates to proposition_to_surface; no mode dispatch.
    pub fn linearize(&self, hyp: &Hypothesis) -> String {
        self.proposition_to_surface(&hyp.proposition)
    }

    /// Convert a PropositionGraph to a syntactically valid surface string
    /// solely by interpreting strict Categorial Grammar directional functor-argument 
    /// mappings (Direction::Left and Direction::Right). This fully deprecates 
    /// language-model-based Next Token Prediction in favor of algebraic graph unparsing.
    pub fn proposition_to_surface(&self, prop: &PropositionGraph) -> String {
        let root_surface = self.lexicon
            .surface_for(&prop.root, &self.language)
            .unwrap_or(prop.root.as_str())
            .to_string();

        if prop.roles.is_empty() {
            return root_surface;
        }

        let root_type = self.lexicon.entry_for(&prop.root, &self.language)
            .map(|e| &e.modal_type);
            
        let dir = root_type.map_or(Direction::Right, |mt| mt.direction);
        let mut surface = root_surface;

        for (_, arg) in &prop.roles {
            let arg_surf = self.lexicon
                .surface_for(arg, &self.language)
                .unwrap_or(arg.as_str())
                .to_string();
                
            // Strictly obey Categorial / L-System unparsing orientations
            if dir == Direction::Left {
                surface = format!("{} {}", arg_surf, surface);
            } else {
                surface = format!("{} {}", surface, arg_surf);
            }
        }

        surface
    }

    /// Linearize a λ-term to surface: evaluate, then look up in lexicon.
    pub fn lambda_to_surface(&self, semantics: &MtlgSemantics, term: LambdaTerm) -> String {
        let prop = semantics.sentence_level(term);
        self.proposition_to_surface(&prop)
    }

    /// Produce a sequence of `LinearizationStep`s from an ordered list of
    /// `(node_id, edge_id, predicate)` triples.
    ///
    /// Used by the sequential trainer: each step corresponds to one teacher-forced token.
    pub fn linearize_sequence(
        &self,
        steps: &[(NodeId, EdgeId, &str)],
    ) -> Vec<LinearizationStep> {
        steps.iter().enumerate().map(|(i, &(nid, eid, pred))| {
            let surface = self.lexicon
                .surface_for(pred, &self.language)
                .unwrap_or(pred)
                .to_string();
            LinearizationStep { step: i, surface, node_id: nid, edge_id: eid }
        }).collect()
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
        // Default direction is Right, so 'runs Alice'
        assert_eq!(surface, "runs Alice");

        // Change to Left facing
        lin.lexicon.register(LexEntry {
            predicate: "run".into(),
            language:  "en".into(),
            surface:   "runs".into(),
            modal_type: ModalType::functor(ModalMode::Diamond, TypeCategory::DEFAULT, 1, Direction::Left),
        });
        let surface_left = lin.linearize(&hyp);
        assert_eq!(surface_left, "Alice runs");
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
