//! Modal lifting rules LR(τ→τ'): transform MTLG modal types during context shifts.
//! Algebraic weight scaling is handled inline; numerica_adapter owns path normalization.

use std::collections::HashMap;
use crate::types::{ModalType, ModalMode, TypeCategory, Direction, ContextId};
use crate::arg::transient_repr::Tr;

/// A lifting rule: transforms a source modal type to a target modal type.
#[derive(Clone, Debug)]
pub struct ModalLiftingRule {
    pub id:          u64,
    pub name:        String,
    pub source_mode: ModalMode,
    pub source_cat:  TypeCategory,
    pub target_mode: ModalMode,
    pub target_cat:  TypeCategory,
    /// Optional algebraic transformation tag (resolved by numerica_adapter).
    pub algebraic_transform: Option<String>,
}

impl ModalLiftingRule {
    pub fn new(
        id: u64,
        name: impl Into<String>,
        source_mode: ModalMode,
        source_cat:  TypeCategory,
        target_mode: ModalMode,
        target_cat:  TypeCategory,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            source_mode,
            source_cat,
            target_mode,
            target_cat,
            algebraic_transform: None,
        }
    }

    /// Apply the lifting rule to a modal type if it matches.
    pub fn apply(&self, ty: ModalType) -> Option<ModalType> {
        if ty.mode == self.source_mode && ty.category == self.source_cat {
            Some(ModalType {
                mode:     self.target_mode,
                category: self.target_cat,
                arity:    ty.arity,
                direction: ty.direction,
            })
        } else {
            None
        }
    }

    pub fn apply_weight(&self, weight: f32) -> f32 {
        // `algebraic_transform` tag is reserved for future numerica extensions.
        weight
    }
}

/// Registry of lifting rules indexed by (source_context, target_context).
pub struct LiftingRuleRegistry {
    /// (from_ctx, to_ctx) → list of applicable rules.
    rules: HashMap<(ContextId, ContextId), Vec<ModalLiftingRule>>,
    /// Default rules applied in any shift.
    default_rules: Vec<ModalLiftingRule>,
}

impl LiftingRuleRegistry {
    pub fn new() -> Self {
        Self {
            rules:         HashMap::new(),
            default_rules: Vec::new(),
        }
    }

    pub fn register(&mut self, from_ctx: ContextId, to_ctx: ContextId, rule: ModalLiftingRule) {
        self.rules.entry((from_ctx, to_ctx)).or_default().push(rule);
    }

    pub fn register_default(&mut self, rule: ModalLiftingRule) {
        self.default_rules.push(rule);
    }

    /// Lift a TR from one context to another, applying all matching rules.
    /// Returns Some(lifted_tr) if at least one rule matched, None if the TR should be dissolved.
    pub fn lift_tr(&self, tr: &Tr, from_ctx: ContextId, to_ctx: ContextId) -> Option<ModalType> {
        let rules = self.rules.get(&(from_ctx, to_ctx))
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        let all_rules = rules.iter().chain(self.default_rules.iter());
        for rule in all_rules {
            if let Some(new_type) = rule.apply(tr.mtlg_type) {
                return Some(new_type);
            }
        }
        None // No matching rule → TR should be dissolved
    }

    /// Lift a TR and return the algebraic weight scale for the matched rule.
    /// Used by the engine when shortcut materialization is active.
    /// Returns Some((new_type, weight_scale)) on match, None if dissolved.
    pub fn lift_tr_with_weight(
        &self,
        tr:       &Tr,
        from_ctx: ContextId,
        to_ctx:   ContextId,
    ) -> Option<(ModalType, f32)> {
        let rules = self.rules.get(&(from_ctx, to_ctx))
            .map(|v| v.as_slice()).unwrap_or(&[]);
        for rule in rules.iter().chain(self.default_rules.iter()) {
            if let Some(new_type) = rule.apply(tr.mtlg_type) {
                return Some((new_type, rule.apply_weight(1.0)));
            }
        }
        None
    }
}

impl Default for LiftingRuleRegistry {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_applies_matching_type() {
        let rule = ModalLiftingRule::new(
            1, "cat7-to-cat1",
            ModalMode::Diamond, TypeCategory(7),
            ModalMode::Diamond, TypeCategory(1),
        );
        let ty = ModalType::atom(ModalMode::Diamond, TypeCategory(7));
        let result = rule.apply(ty).unwrap();
        assert_eq!(result.category, TypeCategory(1));
        assert_eq!(result.mode, ModalMode::Diamond);
    }

    #[test]
    fn rule_rejects_non_matching() {
        let rule = ModalLiftingRule::new(
            1, "cat7-to-cat1",
            ModalMode::Diamond, TypeCategory(7),
            ModalMode::Diamond, TypeCategory(1),
        );
        let ty = ModalType::atom(ModalMode::Box, TypeCategory(7));
        assert!(rule.apply(ty).is_none());
    }
}
