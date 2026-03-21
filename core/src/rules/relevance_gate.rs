//! TRD-relative relevance gate: Eff/Effort threshold for rule application.
//! A rule is applied only if its expected effectiveness > effort cost in the current TRD.

use crate::types::TRDId;
use crate::adaptive::PerfRegistry;
use super::{ruler_bridge::MtlgRule, rule_lifecycle::RuleState};

/// Compute rule effort: a function of how many nodes the rule touches (estimated by arity).
pub fn rule_effort(rule: &MtlgRule) -> f64 {
    // Simpler rules (fewer type transformations) have lower effort.
    if rule.lhs_mode == rule.rhs_mode { 0.3 } else { 0.7 }
}

/// Compute rule effectiveness in a TRD: confidence × TRD affinity.
pub fn rule_effectiveness(rule: &MtlgRule, state: &RuleState, trd_id: TRDId) -> f64 {
    rule.confidence * state.trd_affinity(trd_id)
}

/// Should this rule be applied in the current TRD context?
/// Gate condition: effectiveness > effort.
pub fn should_apply(rule: &MtlgRule, state: &RuleState, trd_id: TRDId, perf: &PerfRegistry) -> bool {
    let effectiveness = rule_effectiveness(rule, state, trd_id);
    let effort        = rule_effort(rule);
    // Also require the TRD to be performing reasonably (p_ema > 0.3).
    let trd_perf = perf.p_ema(trd_id);
    effectiveness > effort && trd_perf > 0.3
}

/// Rank rules for a TRD: highest (effectiveness − effort) × trd_performance first.
pub fn rank_rules<'a>(
    rules:   &[(&'a MtlgRule, &'a RuleState)],
    trd_id:  TRDId,
    perf:    &PerfRegistry,
) -> Vec<(&'a MtlgRule, f64)> {
    let trd_perf = perf.p_ema(trd_id);
    let mut ranked: Vec<_> = rules.iter()
        .map(|&(rule, state)| {
            let eff    = rule_effectiveness(rule, state, trd_id);
            let effort = rule_effort(rule);
            let score  = (eff - effort) * trd_perf;
            (rule, score)
        })
        .filter(|&(_, score)| score > 0.0)
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModalMode, TypeCategory, Quality};

    fn make_rule(confidence: f64, same_mode: bool) -> MtlgRule {
        MtlgRule {
            id:       1,
            name:     "test".into(),
            lhs_mode: ModalMode::Diamond,
            lhs_cat:  TypeCategory::Scene,
            rhs_mode: if same_mode { ModalMode::Diamond } else { ModalMode::Box },
            rhs_cat:  TypeCategory::Process,
            confidence,
            support:  10,
        }
    }

    fn make_state(rule: &MtlgRule) -> RuleState {
        let mut s = RuleState::new(rule.clone());
        s.trd_affinities.insert(0, 0.8);
        s
    }

    #[test]
    fn high_confidence_same_mode_passes_gate() {
        let rule  = make_rule(0.9, true);
        let state = make_state(&rule);
        let mut perf = PerfRegistry::new(0.25);
        for _ in 0..10 { perf.update(0, Quality::Good); }
        assert!(should_apply(&rule, &state, 0, &perf));
    }

    #[test]
    fn low_trd_performance_blocks_gate() {
        let rule  = make_rule(0.7, true);
        let state = make_state(&rule);
        let mut perf = PerfRegistry::new(0.25);
        for _ in 0..10 { perf.update(0, Quality::Bad); }
        assert!(!should_apply(&rule, &state, 0, &perf));
    }
}
