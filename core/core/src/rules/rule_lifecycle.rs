//! Rule lifecycle: induction → activation → deprecation.
//! Rules accumulate evidence from dissolved TRs; their confidence rises or falls.
//! Rules below the deprecation threshold are removed from the active set.

use std::collections::HashMap;
use crate::types::TRDId;
use super::ruler_bridge::MtlgRule;

const DEPRECATION_THRESHOLD: f64 = 0.3;
const MAX_AGE_WITHOUT_SUPPORT: u64 = 1000;

/// Lifecycle state for a rule.
#[derive(Clone, Debug)]
pub struct RuleState {
    pub rule:              MtlgRule,
    pub last_used_at_tr:   u64,  // TR dissolution count when last used
    pub application_count: u64,
    pub trd_affinities:    HashMap<TRDId, f64>, // per-TRD effectiveness
}

impl RuleState {
    pub fn new(rule: MtlgRule) -> Self {
        Self {
            rule,
            last_used_at_tr:   0,
            application_count: 0,
            trd_affinities:    HashMap::new(),
        }
    }

    pub fn record_use(&mut self, current_tr: u64, trd_id: TRDId, quality: f64) {
        self.last_used_at_tr   = current_tr;
        self.application_count += 1;
        let affinity = self.trd_affinities.entry(trd_id).or_insert(0.5);
        // EMA update of TRD affinity
        *affinity = 0.9 * *affinity + 0.1 * quality;
    }

    pub fn is_stale(&self, current_tr: u64) -> bool {
        self.application_count > 0
            && (current_tr - self.last_used_at_tr) > MAX_AGE_WITHOUT_SUPPORT
    }

    pub fn should_deprecate(&self, current_tr: u64) -> bool {
        self.rule.confidence < DEPRECATION_THRESHOLD || self.is_stale(current_tr)
    }

    pub fn trd_affinity(&self, trd_id: TRDId) -> f64 {
        self.trd_affinities.get(&trd_id).copied().unwrap_or(0.5)
    }
}

/// Manages the full lifecycle of all induced rules.
pub struct RuleLifecycleManager {
    rules:      HashMap<u64, RuleState>,
    current_tr: u64,
}

impl RuleLifecycleManager {
    pub fn new() -> Self {
        Self { rules: HashMap::new(), current_tr: 0 }
    }

    /// Add newly induced rules from RulerBridge.
    pub fn integrate_induced(&mut self, induced: Vec<MtlgRule>) {
        for rule in induced {
            self.rules.entry(rule.id).or_insert_with(|| RuleState::new(rule));
        }
    }

    /// Called on each TR dissolution.
    pub fn on_tr_dissolved(&mut self, tr_id: u64) {
        self.current_tr = tr_id;
        // Remove stale/low-confidence rules.
        self.rules.retain(|_, state| !state.should_deprecate(self.current_tr));
    }

    pub fn record_rule_use(&mut self, rule_id: u64, trd_id: TRDId, quality: f64) {
        if let Some(state) = self.rules.get_mut(&rule_id) {
            state.record_use(self.current_tr, trd_id, quality);
        }
    }

    pub fn active_rules(&self) -> Vec<&MtlgRule> {
        self.rules.values()
            .filter(|s| !s.should_deprecate(self.current_tr))
            .map(|s| &s.rule)
            .collect()
    }

    pub fn active_count(&self) -> usize { self.active_rules().len() }
}

impl Default for RuleLifecycleManager { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModalMode, TypeCategory, ModalType};

    fn make_rule(id: u64, confidence: f64) -> MtlgRule {
        MtlgRule {
            id,
            name:       "test".into(),
            lhs_mode:   ModalMode::Diamond,
            lhs_cat:    TypeCategory::DEFAULT,
            rhs_mode:   ModalMode::Diamond,
            rhs_cat:    TypeCategory(1),
            confidence,
            support:    10,
        }
    }

    #[test]
    fn low_confidence_rule_deprecated() {
        let mut mgr = RuleLifecycleManager::new();
        mgr.integrate_induced(vec![make_rule(1, 0.2)]);
        mgr.on_tr_dissolved(1);
        assert_eq!(mgr.active_count(), 0); // confidence < 0.3 → deprecated
    }

    #[test]
    fn high_confidence_rule_survives() {
        let mut mgr = RuleLifecycleManager::new();
        mgr.integrate_induced(vec![make_rule(1, 0.8)]);
        mgr.on_tr_dissolved(1);
        assert_eq!(mgr.active_count(), 1);
    }
}
