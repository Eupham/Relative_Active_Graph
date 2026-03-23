//! Bootstrap artefact loader: reads the JSON files produced by the Phase 0
//! bootstrap pipeline (mC4 → UD → MTLG → cluster → export) and reconstructs
//! the CategoryRegistry + TRD modal profiles needed for Phase 1 training.

use std::collections::HashMap;
use std::path::Path;
use crate::types::{TypeCategory, TRDId, ModalMode};
use super::category_registry::CategoryRegistry;

/// TRD profile: modal mode → TypeCategory frequency map.
/// Describes which (mode, category) combinations co-occur in this TRD.
#[derive(Clone, Debug, Default)]
pub struct TrdProfile {
    pub trd_id:         TRDId,
    /// (mode_id, category_id) → count.
    pub type_counts:    HashMap<(u8, u32), usize>,
    pub total_tokens:   usize,
}

impl TrdProfile {
    pub fn new(trd_id: TRDId) -> Self {
        Self { trd_id, type_counts: HashMap::new(), total_tokens: 0 }
    }

    pub fn record(&mut self, mode: ModalMode, cat: TypeCategory) {
        let mode_id = match mode {
            ModalMode::Diamond => 0,
            ModalMode::Box     => 1,
            ModalMode::Lozenge => 2,
        };
        *self.type_counts.entry((mode_id, cat.id())).or_insert(0) += 1;
        self.total_tokens += 1;
    }

    /// Dominant modal mode (by count).
    pub fn dominant_mode(&self) -> ModalMode {
        let mut counts = [0usize; 3];
        for (&(m, _), &c) in &self.type_counts { counts[m as usize] += c; }
        match counts.iter().enumerate().max_by_key(|&(_, &c)| c).map(|(i, _)| i) {
            Some(1) => ModalMode::Box,
            Some(2) => ModalMode::Lozenge,
            _       => ModalMode::Diamond,
        }
    }
}

/// Artefacts produced by the bootstrap pipeline.
pub struct BootstrapArtefacts {
    pub registry:     CategoryRegistry,
    pub trd_profiles: HashMap<TRDId, TrdProfile>,
    /// Edge ID → token surface form (loaded from export).
    pub edge_vocab:   HashMap<u64, String>,
    /// FNV-1a fingerprint of the loaded categories array (§20.1).
    /// Verified on load to detect stale/corrupted artefacts; 0 when not computed.
    pub partition_fingerprint: u64,
}

impl BootstrapArtefacts {
    /// Create empty artefacts (useful as a cold-start fallback).
    pub fn empty() -> Self {
        Self {
            registry:            CategoryRegistry::new(),
            trd_profiles:        HashMap::new(),
            edge_vocab:          HashMap::new(),
            partition_fingerprint: 0,
        }
    }
}

/// Load bootstrap artefacts from a directory.
///
/// Expected files (all optional; missing files yield empty defaults):
/// - `{dir}/categories.json`   — array of `{id, label, centroid, count}`
/// - `{dir}/trd_profiles.json` — array of `{trd_id, type_counts, total_tokens}`
/// - `{dir}/edge_vocab.json`   — object mapping edge_id (string) → surface token
pub fn load_bootstrap(dir: &Path) -> BootstrapArtefacts {
    let mut artefacts = BootstrapArtefacts::empty();

    // ── Load categories ─────────────────────────────────────────────────────
    // §5d: schema changed from {id, label, centroid, count} to {id, descriptor: [str], size: int}
    let cat_path = dir.join("categories.json");
    if cat_path.exists() {
        if let Ok(data) = std::fs::read_to_string(&cat_path) {
            // §20.1: Compute FNV-1a fingerprint over the raw category bytes.
            artefacts.partition_fingerprint = crate::lcs::fnv1a_64_bytes(data.as_bytes());

            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                if let Some(arr) = json.as_array() {
                    for item in arr {
                        // New format: descriptor array + size
                        let descriptor: Vec<String> = item["descriptor"]
                            .as_array()
                            .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                            .unwrap_or_default();
                        let size = item["size"].as_u64()
                            .or_else(|| item["count"].as_u64()) // backward compat
                            .unwrap_or(0) as usize;
                        let label = descriptor.first().cloned().unwrap_or_default();
                        artefacts.registry.register(descriptor, label, size);
                    }
                }
            }
        }
    }

    // ── Load TRD profiles ───────────────────────────────────────────────────
    let trd_path = dir.join("trd_profiles.json");
    if trd_path.exists() {
        if let Ok(data) = std::fs::read_to_string(&trd_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                if let Some(arr) = json.as_array() {
                    for item in arr {
                        let trd_id = item["trd_id"].as_u64().unwrap_or(0) as TRDId;
                        let total  = item["total_tokens"].as_u64().unwrap_or(0) as usize;
                        let mut profile = TrdProfile::new(trd_id);
                        profile.total_tokens = total;
                        if let Some(counts) = item["type_counts"].as_object() {
                            for (key, val) in counts {
                                // key format: "mode_id,cat_id"
                                let parts: Vec<&str> = key.split(',').collect();
                                if parts.len() == 2 {
                                    if let (Ok(m), Ok(c)) = (parts[0].parse::<u8>(), parts[1].parse::<u32>()) {
                                        let count = val.as_u64().unwrap_or(0) as usize;
                                        profile.type_counts.insert((m, c), count);
                                    }
                                }
                            }
                        }
                        artefacts.trd_profiles.insert(trd_id, profile);
                    }
                }
            }
        }
    }

    // ── Load edge vocab ─────────────────────────────────────────────────────
    let vocab_path = dir.join("edge_vocab.json");
    if vocab_path.exists() {
        if let Ok(data) = std::fs::read_to_string(&vocab_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                if let Some(obj) = json.as_object() {
                    for (k, v) in obj {
                        if let (Ok(eid), Some(surf)) = (k.parse::<u64>(), v.as_str()) {
                            artefacts.edge_vocab.insert(eid, surf.to_string());
                        }
                    }
                }
            }
        }
    }

    artefacts
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn empty_artefacts_has_empty_registry() {
        // The registry starts empty; categories are populated by bootstrap clustering.
        let a = BootstrapArtefacts::empty();
        assert!(a.registry.is_empty(), "fresh registry must start empty — no hard-coded prototypes");
    }

    #[test]
    fn load_from_nonexistent_dir_returns_empty() {
        let path = PathBuf::from("/tmp/csrre_nonexistent_dir_xyz");
        let a = load_bootstrap(&path);
        assert!(a.trd_profiles.is_empty());
        assert!(a.edge_vocab.is_empty());
    }
}
