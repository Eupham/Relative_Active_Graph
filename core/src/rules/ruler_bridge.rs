//! Ruler/Enumo-style rule induction from dissolved TR patterns.
//! Induces domain rewrite rules over the MTLG type domain by observing
//! successful ARG → MTLG derivation mappings.
//! (Nandi et al. OOPSLA 2021 — Ruler synthesizes rewrite rules by e-graph enumeration.)

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, ModalType, ModalMode, TypeCategory, Direction, TRDId};
use crate::arg::transient_repr::Tr;

/// A rewrite rule over the MTLG type domain.
#[derive(Clone, Debug)]
pub struct MtlgRule {
    pub id:       u64,
    pub name:     String,
    /// Pattern: (source_mode, source_category) → rewrites to → (target_mode, target_category).
    pub lhs_mode: ModalMode,
    pub lhs_cat:  TypeCategory,
    pub rhs_mode: ModalMode,
    pub rhs_cat:  TypeCategory,
    /// Confidence score: fraction of dissolved TRs where this rule applied successfully.
    pub confidence: f64,
    pub support:  usize, // number of observed instances
}

impl MtlgRule {
    pub fn applies_to(&self, ty: ModalType) -> bool {
        ty.mode == self.lhs_mode && ty.category == self.lhs_cat
    }

    pub fn apply(&self, ty: ModalType) -> ModalType {
        ModalType { mode: self.rhs_mode, category: self.rhs_cat, ..ty }
    }
}

/// Pattern key for grouping dissolved TR observations.
type PatternKey = (ModalMode, TypeCategory, ModalMode, TypeCategory);

/// Ruler-style rule inducer: observes dissolved TRs and synthesizes rewrite rules.
pub struct RulerBridge {
    /// Observed (lhs_mode, lhs_cat, rhs_mode, rhs_cat) → (successes, total).
    observations: HashMap<PatternKey, (usize, usize)>,
    /// Induced rules (updated after each observation batch).
    pub rules:    Vec<MtlgRule>,
    next_rule_id: u64,
    /// Minimum support for a rule to be retained.
    min_support:  usize,
    /// Minimum confidence for a rule to be retained.
    min_confidence: f64,
}

impl RulerBridge {
    pub fn new() -> Self {
        Self {
            observations:   HashMap::new(),
            rules:          Vec::new(),
            next_rule_id:   0,
            min_support:    5,
            min_confidence: 0.6,
        }
    }

    /// Observe a dissolved TR: extract type transformation patterns.
    pub fn observe_dissolved_tr(&mut self, tr: &Tr, success: bool) {
        // Record the type transformation this TR represents.
        // Source type: the input MTLG type. Target type: the output type after application.
        let src = tr.mtlg_type;
        if let Some(applied) = src.apply() {
            let key = (src.mode, src.category, applied.mode, applied.category);
            let entry = self.observations.entry(key).or_insert((0, 0));
            entry.1 += 1;
            if success { entry.0 += 1; }
        }
    }

    /// Re-synthesize rules from observations (call after processing a batch of TRs).
    pub fn induce_rules(&mut self) {
        self.rules.clear();
        for (&(lm, lc, rm, rc), &(successes, total)) in &self.observations {
            if total < self.min_support { continue; }
            let confidence = successes as f64 / total as f64;
            if confidence < self.min_confidence { continue; }
            let name = format!("{:?}/{:?}→{:?}/{:?}", lm, lc, rm, rc);
            self.rules.push(MtlgRule {
                id:         self.next_rule_id,
                name,
                lhs_mode:   lm,
                lhs_cat:    lc,
                rhs_mode:   rm,
                rhs_cat:    rc,
                confidence,
                support:    total,
            });
            self.next_rule_id += 1;
        }
        // Sort by confidence descending.
        self.rules.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Apply the best matching rule to `ty`, if any.
    pub fn apply_best(&self, ty: ModalType) -> Option<ModalType> {
        self.rules.iter()
            .find(|r| r.applies_to(ty))
            .map(|r| r.apply(ty))
    }

    pub fn rule_count(&self) -> usize { self.rules.len() }
}

impl Default for RulerBridge { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModalMode, TypeCategory};

    #[test]
    fn rule_induced_after_observations() {
        let mut ruler = RulerBridge::new();
        let ty = ModalType::functor(ModalMode::Diamond, TypeCategory::Scene, 1, Direction::Right);

        // Simulate 10 dissolved TRs with this type transformation, all successful.
        for i in 0..10 {
            let tr = Tr {
                id: i,
                context_id: 0,
                content: crate::arg::transient_repr::RepContent::Lambda("x".into()),
                lifecycle: crate::arg::transient_repr::Lifecycle::Dissolved,
                atms_env: 0b1,
                attribution_trace: HashMap::new(),
                provenance: vec![],
                mtlg_type: ty,
                granularity: crate::arg::transient_repr::Granularity::Sentence,
            };
            ruler.observe_dissolved_tr(&tr, true);
        }
        ruler.induce_rules();
        assert!(ruler.rule_count() > 0, "should have induced at least one rule");
    }
}
