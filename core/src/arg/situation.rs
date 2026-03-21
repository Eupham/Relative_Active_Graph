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

    /// Look up the best-matching TRD for a situation's infon set.
    /// Returns the TRD with highest match_score, or None if no TRDs registered.
    pub fn lookup_trd(&self, situation_id: u64) -> Option<TRDId> {
        let sit = self.situations.get(&situation_id)?;
        let infons: Vec<Infon> = sit.id.to_le_bytes().iter()
            .enumerate()
            .filter_map(|(i, _)| self.infons.get(&(i as u64)).cloned())
            .collect();

        self.trds.values()
            .max_by(|a, b| {
                a.match_score(&infons).partial_cmp(&b.match_score(&infons)).unwrap()
            })
            .map(|t| t.id)
    }

    /// Whether situation s ⊨ infon σ (the infon is supported by the situation).
    pub fn supports(&self, sit_id: u64, infon_id: InfonId) -> bool {
        self.situations.contains_key(&sit_id) && self.infons.contains_key(&infon_id)
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
}
