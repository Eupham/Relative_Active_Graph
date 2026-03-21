//! STO Situation: typed partial world. Carries infon support and TRD assignment.
//! "No nodes are globally asserted. Structure is situation-relative and emergent."

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::types::{TRDId, InfonId, NodeId, Infon, Situation};

/// A TRD (Transient Relative Domain): cluster of co-activating situation types,
/// infon patterns, and MTLG modal type profiles.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trd {
    pub id:           TRDId,
    pub label:        String,
    /// Characteristic infon patterns that activate this TRD.
    pub infon_patterns: Vec<String>,
    /// Modal mode distribution: mode → expected frequency in [0,1].
    pub modal_profile: HashMap<String, f32>,
    /// Initial variance estimate for threshold computation (from mC4 bootstrapping).
    pub var_max_prior: f64,
}

impl Trd {
    pub fn new(id: TRDId, label: impl Into<String>) -> Self {
        Self {
            id,
            label:          label.into(),
            infon_patterns: Vec::new(),
            modal_profile:  HashMap::new(),
            var_max_prior:  0.25, // default: σ²_max = 0.25 (maximum for Bernoulli p=0.5)
        }
    }

    /// Match score: how well does this situation's infon set match TRD patterns?
    pub fn match_score(&self, infons: &[Infon]) -> f32 {
        if self.infon_patterns.is_empty() { return 0.5; }
        let matched = infons.iter().filter(|infon| {
            self.infon_patterns.iter().any(|p| infon.relation.contains(p.as_str()))
        }).count();
        matched as f32 / self.infon_patterns.len() as f32
    }
}

/// The Situation Registry: manages situations, TRDs, and their infon support.
pub struct SituationRegistry {
    pub situations: HashMap<u64, Situation>,
    pub trds:       HashMap<TRDId, Trd>,
    /// Infon pool: InfonId → Infon.
    pub infons:     HashMap<InfonId, Infon>,
    next_trd_id:    TRDId,
}

impl SituationRegistry {
    pub fn new() -> Self {
        Self {
            situations:  HashMap::new(),
            trds:        HashMap::new(),
            infons:      HashMap::new(),
            next_trd_id: 0,
        }
    }

    pub fn register_situation(&mut self, sit: Situation) {
        self.situations.insert(sit.id, sit);
    }

    pub fn register_trd(&mut self, mut trd: Trd) -> TRDId {
        let id = self.next_trd_id;
        trd.id = id;
        self.trds.insert(id, trd);
        self.next_trd_id += 1;
        id
    }

    pub fn register_infon(&mut self, infon: Infon) {
        self.infons.insert(infon.id, infon);
    }

    /// Whether situation s ⊨ infon σ (Barwise & Perry 1983).
    ///
    /// Conditions:
    /// 1. Both situation and infon are registered.
    /// 2. The infon's polarity is positive (negative facts require a negation layer
    ///    not yet present in the base system).
    /// 3. Every NodeId in the infon's arg list is anchored in the situation's
    ///    active_nodes set — the argument roles are filled by entities in s.
    pub fn supports(&self, sit_id: u64, infon_id: InfonId) -> bool {
        let sit   = match self.situations.get(&sit_id)  { Some(s) => s, None => return false };
        let infon = match self.infons.get(&infon_id)    { Some(i) => i, None => return false };

        // Polarity: only positive facts are supported in the base layer.
        if !infon.polarity { return false; }

        // Argument anchoring: every arg NodeId must be in sit.active_nodes.
        infon.args.iter().all(|arg_id| sit.active_nodes.contains(arg_id))
    }

    /// Collect the infons that situation `sit_id` actually supports.
    fn supported_infons(&self, sit_id: u64) -> Vec<Infon> {
        let sit = match self.situations.get(&sit_id) { Some(s) => s, None => return vec![] };
        self.infons.values()
            .filter(|infon| {
                infon.polarity
                    && infon.args.iter().all(|arg_id| sit.active_nodes.contains(arg_id))
            })
            .cloned()
            .collect()
    }

    /// Look up the best-matching TRD for a situation's supported infon set.
    pub fn lookup_trd(&self, situation_id: u64) -> Option<TRDId> {
        let infons = self.supported_infons(situation_id);
        self.trds.values()
            .max_by(|a, b| {
                a.match_score(&infons)
                    .partial_cmp(&b.match_score(&infons))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|t| t.id)
    }
}

impl Default for SituationRegistry {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trd_match_score() {
        let mut trd = Trd::new(0, "code-gen");
        trd.infon_patterns = vec!["function".into(), "return".into()];
        let infons = vec![
            Infon { id: 0, relation: "function-call".into(), args: vec![], polarity: true },
        ];
        let score = trd.match_score(&infons);
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn supports_checks_polarity_and_anchoring() {
        let mut reg = SituationRegistry::new();

        let mut sit = Situation::new(1, "test", 0);
        sit.anchor(10);
        sit.anchor(20);
        reg.register_situation(sit);

        // Positive infon with anchored args — should be supported
        reg.register_infon(Infon { id: 100, relation: "rel".into(), args: vec![10, 20], polarity: true });
        assert!(reg.supports(1, 100));

        // Negative infon — should not be supported
        reg.register_infon(Infon { id: 101, relation: "rel".into(), args: vec![10], polarity: false });
        assert!(!reg.supports(1, 101));

        // Positive infon with unanchored arg — should not be supported
        reg.register_infon(Infon { id: 102, relation: "rel".into(), args: vec![99], polarity: true });
        assert!(!reg.supports(1, 102));
    }

    #[test]
    fn lookup_trd_uses_supported_infons() {
        let mut reg = SituationRegistry::new();

        let mut sit = Situation::new(1, "test", 0);
        sit.anchor(5);
        reg.register_situation(sit);

        // Infon anchored in situation
        reg.register_infon(Infon { id: 1, relation: "function-call".into(), args: vec![5], polarity: true });

        let mut trd = Trd::new(0, "code-gen");
        trd.infon_patterns = vec!["function".into()];
        reg.register_trd(trd);

        let result = reg.lookup_trd(1);
        assert!(result.is_some());
    }
}
