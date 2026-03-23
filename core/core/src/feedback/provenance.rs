//! Audit trail and TRD crystallization.
//! Tracks which TRs contributed to which ARG modifications, for debugging and TRD evolution.

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, ContextId, TRDId};

/// One entry in the audit trail: records an ARG modification caused by a dissolved TR.
#[derive(Clone, Debug)]
pub struct AuditEntry {
    pub tr_id:       u64,
    pub context_id:  ContextId,
    pub trd_id:      Option<TRDId>,
    pub edges_updated: Vec<(EdgeId, f32)>,  // (edge_id, delta_applied)
    pub quality:     f32,
}

/// TRD crystallization: co-activating UCCA-type profiles cluster into TRD entries.
#[derive(Clone, Debug)]
pub struct TrdCrystallizationEntry {
    pub trd_id:         TRDId,
    /// Modal profile: {mode_name → frequency}.
    pub modal_profile:  HashMap<String, f32>,
    /// Count of dissolved TRs that contributed to this TRD's profile.
    pub support:        usize,
}

impl TrdCrystallizationEntry {
    pub fn new(trd_id: TRDId) -> Self {
        Self { trd_id, modal_profile: HashMap::new(), support: 0 }
    }

    /// Update modal profile from a dissolved TR.
    pub fn update_from_tr(&mut self, modal_mode: &str, weight: f32) {
        let entry = self.modal_profile.entry(modal_mode.to_string()).or_insert(0.0);
        // EMA update.
        *entry = 0.9 * *entry + 0.1 * weight;
        self.support += 1;
    }
}

/// The provenance audit log.
pub struct ProvenanceLog {
    entries:        Vec<AuditEntry>,
    crystallization: HashMap<TRDId, TrdCrystallizationEntry>,
    max_entries:    usize,
}

impl ProvenanceLog {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries:         Vec::new(),
            crystallization: HashMap::new(),
            max_entries,
        }
    }

    pub fn record(&mut self, entry: AuditEntry) {
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0); // sliding window
        }
        // Update TRD crystallization if present.
        if let Some(trd_id) = entry.trd_id {
            let cryst = self.crystallization.entry(trd_id)
                .or_insert_with(|| TrdCrystallizationEntry::new(trd_id));
            for &(_, delta) in &entry.edges_updated {
                cryst.update_from_tr("diamond", delta.abs());
            }
        }
        self.entries.push(entry);
    }

    /// Edges that contributed most across dissolved TRs (top-k by cumulative delta).
    pub fn top_contributing_edges(&self, k: usize) -> Vec<(EdgeId, f32)> {
        let mut totals: HashMap<EdgeId, f32> = HashMap::new();
        for entry in &self.entries {
            for &(eid, delta) in &entry.edges_updated {
                *totals.entry(eid).or_insert(0.0) += delta.abs();
            }
        }
        let mut sorted: Vec<_> = totals.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(k).collect()
    }

    /// TRDs with the highest support (most dissolved TRs crystallized into them).
    pub fn top_trds(&self, k: usize) -> Vec<TRDId> {
        let mut sorted: Vec<_> = self.crystallization.values().collect();
        sorted.sort_by(|a, b| b.support.cmp(&a.support));
        sorted.into_iter().take(k).map(|e| e.trd_id).collect()
    }

    pub fn entry_count(&self) -> usize { self.entries.len() }
}

impl Default for ProvenanceLog {
    fn default() -> Self { Self::new(10_000) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_retrieves_top_edges() {
        let mut log = ProvenanceLog::new(100);
        log.record(AuditEntry {
            tr_id: 1, context_id: 0, trd_id: Some(0),
            edges_updated: vec![(10, 0.3), (20, 0.1)],
            quality: 1.0,
        });
        log.record(AuditEntry {
            tr_id: 2, context_id: 0, trd_id: Some(0),
            edges_updated: vec![(10, 0.5)],
            quality: 0.5,
        });
        let top = log.top_contributing_edges(2);
        assert_eq!(top[0].0, 10); // edge 10 has total 0.8
    }

    #[test]
    fn crystallization_accumulates() {
        let mut log = ProvenanceLog::new(100);
        for i in 0..5 {
            log.record(AuditEntry {
                tr_id: i, context_id: 0, trd_id: Some(0),
                edges_updated: vec![(1, 0.5)], quality: 1.0,
            });
        }
        assert!(log.crystallization[&0].support > 0);
    }
}
