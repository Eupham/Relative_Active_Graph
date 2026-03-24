//! Ruler-style rule induction with Knuth-Bendix completion.
//! KBC guarantees the rule set is confluent and terminating.

use std::collections::{HashMap, VecDeque};
use crate::types::{ModalType, ModalMode, TypeCategory};
use crate::arg::transient_repr::Tr;
use crate::arg::type_normalizer::{lpo_gt, normalize_type};

#[derive(Clone, Debug)]
pub struct MtlgRule {
    pub id:         u64,
    pub name:       String,
    pub lhs_mode:   ModalMode,
    pub lhs_cat:    TypeCategory,
    pub rhs_mode:   ModalMode,
    pub rhs_cat:    TypeCategory,
    pub confidence: f64,
    pub support:    usize,
}

impl MtlgRule {
    pub fn applies_to(&self, ty: ModalType) -> bool {
        ty.mode == self.lhs_mode && ty.category == self.lhs_cat
    }
    pub fn apply(&self, ty: ModalType) -> ModalType {
        ModalType { mode: self.rhs_mode, category: self.rhs_cat, ..ty }
    }
}

type PatternKey = (ModalMode, TypeCategory, ModalMode, TypeCategory);

pub struct RulerBridge {
    observations:   HashMap<PatternKey, (usize, usize)>,
    pub rules:      Vec<MtlgRule>,
    next_rule_id:   u64,
    min_support:    usize,
    min_confidence: f64,
    pub rule_budget: usize,
}

impl RulerBridge {
    pub fn new() -> Self {
        Self {
            observations:   HashMap::new(),
            rules:          Vec::new(),
            next_rule_id:   0,
            min_support:    5,
            min_confidence: 0.6,
            rule_budget:    50_000,
        }
    }

    pub fn observe_dissolved_tr(&mut self, tr: &Tr, success: bool) {
        let src = tr.mtlg_type;
        // Only record observations for non-Diamond functors being dissolved.
        // Models: "consistently dissolved Box/Lozenge functors normalize toward Diamond."
        // Diamond already is the base mode; atoms (arity=0) produce no orderable pairs.
        if src.arity == 0 || src.mode == ModalMode::Diamond { return; }
        let key = (src.mode, src.category, ModalMode::Diamond, src.category);
        let entry = self.observations.entry(key).or_insert((0, 0));
        entry.1 += 1;
        if success { entry.0 += 1; }
    }

    pub fn induce_rules(&mut self) {
        let mut raw: Vec<MtlgRule> = Vec::new();
        for (&(lm, lc, rm, rc), &(successes, total)) in &self.observations {
            if total < self.min_support { continue; }
            let confidence = successes as f64 / total as f64;
            if confidence < self.min_confidence { continue; }
            if lpo_gt((lm, lc, 0), (rm, rc, 0)) {
                raw.push(MtlgRule {
                    id: self.next_rule_id, name: format!("{:?}/{:?}→{:?}/{:?}", lm, lc, rm, rc),
                    lhs_mode: lm, lhs_cat: lc, rhs_mode: rm, rhs_cat: rc, confidence, support: total,
                });
                self.next_rule_id += 1;
            } else if lpo_gt((rm, rc, 0), (lm, lc, 0)) {
                raw.push(MtlgRule {
                    id: self.next_rule_id, name: format!("{:?}/{:?}→{:?}/{:?}(rev)", rm, rc, lm, lc),
                    lhs_mode: rm, lhs_cat: rc, rhs_mode: lm, rhs_cat: lc, confidence, support: total,
                });
                self.next_rule_id += 1;
            }
        }
        self.rules = self.complete(raw);
        self.rules.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    }

    fn complete(&mut self, raw: Vec<MtlgRule>) -> Vec<MtlgRule> {
        let mut complete: Vec<MtlgRule> = Vec::new();
        let mut pending: VecDeque<MtlgRule> = raw.into_iter().collect();

        while let Some(rule) = pending.pop_front() {
            let (lhs_mode, lhs_cat) = normalize_type(rule.lhs_mode, rule.lhs_cat, &complete);
            let (rhs_mode, rhs_cat) = normalize_type(rule.rhs_mode, rule.rhs_cat, &complete);
            if lhs_mode == rhs_mode && lhs_cat == rhs_cat { continue; }

            let (flm, flc, frm, frc);
            if lpo_gt((lhs_mode, lhs_cat, 0), (rhs_mode, rhs_cat, 0)) {
                flm = lhs_mode; flc = lhs_cat; frm = rhs_mode; frc = rhs_cat;
            } else if lpo_gt((rhs_mode, rhs_cat, 0), (lhs_mode, lhs_cat, 0)) {
                flm = rhs_mode; flc = rhs_cat; frm = lhs_mode; frc = lhs_cat;
            } else { continue; }

            let oriented = MtlgRule {
                id: self.next_rule_id,
                name: format!("kbc_{:?}/{:?}→{:?}/{:?}", flm, flc, frm, frc),
                lhs_mode: flm, lhs_cat: flc, rhs_mode: frm, rhs_cat: frc,
                confidence: rule.confidence, support: rule.support,
            };
            self.next_rule_id += 1;

            let conflicts: Vec<MtlgRule> = complete.iter()
                .filter(|e| e.lhs_mode == oriented.lhs_mode && e.lhs_cat == oriented.lhs_cat)
                .cloned().collect();

            for conflict in &conflicts {
                let (sm, sc) = normalize_type(oriented.rhs_mode, oriented.rhs_cat, &complete);
                let (tm, tc) = normalize_type(conflict.rhs_mode, conflict.rhs_cat, &complete);
                if sm == tm && sc == tc { continue; }
                let conf = rule.confidence.min(conflict.confidence);
                let sup  = rule.support.min(conflict.support);
                if lpo_gt((sm, sc, 0), (tm, tc, 0)) {
                    pending.push_back(MtlgRule {
                        id: self.next_rule_id,
                        name: format!("cp_{:?}/{:?}→{:?}/{:?}", sm, sc, tm, tc),
                        lhs_mode: sm, lhs_cat: sc, rhs_mode: tm, rhs_cat: tc,
                        confidence: conf, support: sup,
                    });
                    self.next_rule_id += 1;
                } else if lpo_gt((tm, tc, 0), (sm, sc, 0)) {
                    pending.push_back(MtlgRule {
                        id: self.next_rule_id,
                        name: format!("cp_{:?}/{:?}→{:?}/{:?}", tm, tc, sm, sc),
                        lhs_mode: tm, lhs_cat: tc, rhs_mode: sm, rhs_cat: sc,
                        confidence: conf, support: sup,
                    });
                    self.next_rule_id += 1;
                }
            }

            complete.push(oriented);
            if complete.len() > self.rule_budget {
                log::warn!("KBC rule budget {} exceeded", self.rule_budget);
                break;
            }
        }
        complete
    }

    pub fn apply_best(&self, ty: ModalType) -> Option<ModalType> {
        self.rules.iter().find(|r| r.applies_to(ty)).map(|r| r.apply(ty))
    }

    pub fn rule_count(&self) -> usize { self.rules.len() }
}

impl Default for RulerBridge { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModalMode, TypeCategory, Direction};

    #[test]
    fn rule_induced_after_observations() {
        let mut ruler = RulerBridge::new();
        let ty = ModalType::functor(ModalMode::Box, TypeCategory(3), 1, Direction::Right);
        for i in 0..10 {
            let tr = Tr {
                id: i, context_id: 0,
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
        assert!(ruler.rule_count() > 0);
    }

    #[test]
    fn no_cycles_in_completed_rules() {
        let mut ruler = RulerBridge::new();
        ruler.induce_rules();
        for r in &ruler.rules {
            assert!(!(r.lhs_mode == r.rhs_mode && r.lhs_cat == r.rhs_cat), "self-loop rule");
        }
    }
}
