//! ARG → λ-expression → MTLG derivation → surface form.
//! Per-language lexicon lookup: each predicate/concept has a surface realization
//! in the target language. Language selected by TRD / query context.

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, TRDId, ModalType, ModalMode, TypeCategory, Env};
use crate::arg::{ArgGraph, ArgNode};
use crate::semantics::mtlg_semantics::{MtlgSemantics, LambdaTerm, PropositionGraph};
use crate::generation::hypothesis::Hypothesis;
use super::vocab_distribution::VocabDistribution;

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
    /// Ordered sequence of role labels that determines surface word order.
    /// e.g. for `synonym_of` in English: ["ARG0", "ROOT", "ARG1"]
    /// which produces: "{ARG0} {ROOT surface} {ARG1}"
    /// Empty means: ROOT first, then roles in definition order (existing behaviour).
    pub role_order:  Vec<String>,
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

    /// Convert a PropositionGraph to surface form via lexicon lookup.
    ///
    /// If the predicate's LexEntry has a non-empty `role_order`, that order
    /// determines slot position. The special token "ROOT" in role_order is
    /// replaced with the predicate's surface. All other tokens are role labels
    /// whose argument values are looked up in the lexicon.
    ///
    /// If role_order is empty, falls back to: ROOT surface first, then args in
    /// definition order (preserves existing behaviour for all non-relational entries).
    pub fn proposition_to_surface(&self, prop: &PropositionGraph) -> String {
        let entry = self.lexicon.entry_for(&prop.root, &self.language);
        let root_surface = entry
            .map(|e| e.surface.as_str())
            .unwrap_or(prop.root.as_str())
            .to_string();

        if prop.roles.is_empty() {
            return root_surface;
        }

        let role_order: &[String] = entry
            .map(|e| e.role_order.as_slice())
            .unwrap_or(&[]);

        if role_order.is_empty() {
            // Existing behaviour: root then args in definition order.
            let args: Vec<String> = prop.roles.iter()
                .map(|(_, arg)| self.resolve_arg(arg))
                .collect();
            return format!("{} {}", root_surface, args.join(" "));
        }

        // Frame-ordered realization.
        let role_map: HashMap<&str, &str> = prop.roles.iter()
            .map(|(role, arg)| (role.as_str(), arg.as_str()))
            .collect();

        let tokens: Vec<String> = role_order.iter()
            .filter_map(|slot| {
                if slot == "ROOT" {
                    Some(root_surface.clone())
                } else {
                    role_map.get(slot.as_str())
                        .map(|arg| self.resolve_arg(arg))
                }
            })
            .collect();

        tokens.join(" ")
    }

    /// Resolve an argument string to its surface form via lexicon,
    /// falling back to the raw string if no entry exists.
    fn resolve_arg(&self, arg: &str) -> String {
        self.lexicon
            .surface_for(arg, &self.language)
            .unwrap_or(arg)
            .to_string()
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

    /// Decode a PropositionGraph autoregressively using VocabDistribution scoring.
    ///
    /// At each step:
    ///   1. Collect the open role slots from the proposition (those not yet filled).
    ///   2. Score all ARG nodes in those slots via VocabDistribution.
    ///   3. Select the highest-scoring node whose predicate matches an open slot arg.
    ///   4. Resolve that node's surface form via lexicon.
    ///   5. Emit the surface token and mark the slot filled.
    ///
    /// The resulting token sequence is ordered by VocabDistribution rank subject
    /// to the frame order in LexEntry.role_order (same structural constraint as
    /// proposition_to_surface, but ranking comes from learned attribution scores,
    /// not definition order).
    ///
    /// `graph` must be the ARG graph from the last Engine::execute call or a
    /// freshly built ArgSearch::graph for this query's node/edge pool.
    /// `env` must be the active ATMS environment for this query.
    pub fn autoregressive_linearize(
        &self,
        prop:  &PropositionGraph,
        graph: &ArgGraph,
        env:   Env,
    ) -> String {
        let dist = VocabDistribution::from_graph(graph, env);

        let entry = self.lexicon.entry_for(&prop.root, &self.language);
        let root_surface = entry
            .map(|e| e.surface.as_str())
            .unwrap_or(prop.root.as_str())
            .to_string();

        if prop.roles.is_empty() {
            return root_surface;
        }

        // Build a mutable slot map: role_label → (predicate_string, filled: bool)
        let mut slots: Vec<(String, String, bool)> = prop.roles.iter()
            .map(|(role, arg)| (role.clone(), arg.clone(), false))
            .collect();

        let role_order: Vec<String> = entry
            .map(|e| e.role_order.clone())
            .unwrap_or_default();

        // Determine surface order for slots.
        // If role_order is set, emit ROOT surface at the ROOT position and fill
        // the other slots in role_order sequence.
        // If role_order is empty, emit ROOT first then slots in definition order.
        let mut output_tokens: Vec<String> = Vec::new();

        if role_order.is_empty() {
            // Fallback: ROOT + definition-ordered args, scored by dist for ranking.
            output_tokens.push(root_surface.clone());
            // Sort remaining slots by their node's VocabDistribution score (descending),
            // then resolve surface.
            let mut scored_slots: Vec<(f32, String)> = slots.iter()
                .map(|(_, arg, _)| {
                    let score = dist.score_for_predicate(arg);
                    let surface = self.resolve_arg(arg);
                    (score, surface)
                })
                .collect();
            scored_slots.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            output_tokens.extend(scored_slots.into_iter().map(|(_, s)| s));
        } else {
            // Frame-ordered: iterate role_order positions.
            for slot_label in &role_order {
                if slot_label == "ROOT" {
                    output_tokens.push(root_surface.clone());
                } else if let Some((_, arg, _)) = slots.iter_mut().find(|(r, _, _)| r == slot_label) {
                    output_tokens.push(self.resolve_arg(arg));
                }
            }
        }

        output_tokens.join(" ")
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
            role_order: vec![],
        });
        lin.lexicon.register(LexEntry {
            predicate: "alice".into(),
            language:  "en".into(),
            surface:   "Alice".into(),
            modal_type: ModalType::default(),
            role_order: vec![],
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
