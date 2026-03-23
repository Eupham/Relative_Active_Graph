//! CategoryRegistry: maps runtime-discovered TypeCategory IDs to descriptors and labels.
//! Populated by the bootstrap pipeline and persisted as JSON for the training loop.
//!
//! §5c: Replaced k-means centroid with community descriptor (top-5 surface forms).

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::types::TypeCategory;

/// One entry in the category registry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CategoryEntry {
    pub id:         TypeCategory,
    /// Representative surface forms (top 5 by frequency) for this structural equivalence class.
    /// Replaces k-means centroid. (§5c)
    pub descriptor: Vec<String>,
    /// Human-readable label (derived from top descriptor form).
    pub label:      String,
    /// Number of tokens assigned to this category during bootstrap.
    pub count:      usize,
}

/// Registry: ID → entry.
///
/// Starts empty. All category IDs are assigned by the bootstrap CategoryInducer.
/// ID 0 is reserved for TypeCategory::DEFAULT (unassigned). IDs 1+ are opaque
/// cluster IDs — none carry linguistic names.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CategoryRegistry {
    entries: HashMap<u32, CategoryEntry>,
    next_id: u32,
}

impl CategoryRegistry {
    pub fn new() -> Self {
        // Start from 1; 0 is reserved for TypeCategory::DEFAULT.
        Self { entries: HashMap::new(), next_id: 1 }
    }

    /// Register a discovered community and return its assigned TypeCategory.
    pub fn register(&mut self, descriptor: Vec<String>, label: String, count: usize) -> TypeCategory {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(id, CategoryEntry {
            id: TypeCategory(id),
            descriptor,
            label,
            count,
        });
        TypeCategory(id)
    }

    /// Update the descriptor for an existing category (§5c, renamed from update_centroid).
    pub fn update_descriptor(&mut self, cat: TypeCategory, descriptor: Vec<String>, count: usize) {
        if let Some(entry) = self.entries.get_mut(&cat.id()) {
            entry.descriptor = descriptor;
            entry.count      = count;
        }
    }

    /// Kept for backward compatibility; delegates to update_descriptor.
    pub fn update_centroid(&mut self, cat: TypeCategory, descriptor: Vec<String>, count: usize) {
        self.update_descriptor(cat, descriptor, count);
    }

    /// Nearest category by lexical overlap between query neighbors and descriptor surface forms (§5c).
    /// Replaces Euclidean distance over centroids.
    pub fn nearest_by_descriptor(&self, query_neighbors: &[&str]) -> TypeCategory {
        self.entries.values()
            .filter(|e| !e.descriptor.is_empty())
            .max_by_key(|e| {
                let desc_set: std::collections::HashSet<&str> =
                    e.descriptor.iter().map(|s| s.as_str()).collect();
                query_neighbors.iter().filter(|&&n| desc_set.contains(n)).count()
            })
            .map(|e| e.id)
            .unwrap_or(TypeCategory::DEFAULT)
    }

    /// Convenience alias for nearest_by_descriptor with an empty query.
    pub fn nearest(&self, _features: &[f32]) -> TypeCategory {
        // Legacy call site compatibility: centroid-based nearest is removed.
        // Returns DEFAULT; callers should migrate to nearest_by_descriptor.
        TypeCategory::DEFAULT
    }

    pub fn get(&self, cat: TypeCategory) -> Option<&CategoryEntry> {
        self.entries.get(&cat.id())
    }

    pub fn all_entries(&self) -> impl Iterator<Item = &CategoryEntry> {
        self.entries.values()
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}

impl Default for CategoryRegistry {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_starts_empty() {
        let reg = CategoryRegistry::new();
        assert!(reg.is_empty(), "registry must start with no entries");
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn register_returns_new_id() {
        let mut reg = CategoryRegistry::new();
        let cat = reg.register(vec!["run".into(), "running".into()], "cluster_1".into(), 10);
        assert!(cat.id() >= 1, "first cluster ID must be >= 1");
        assert!(reg.get(cat).is_some());
    }

    #[test]
    fn nearest_by_descriptor_finds_overlap() {
        let mut reg = CategoryRegistry::new();
        let c1 = reg.register(vec!["run".into(), "walk".into()], "motion".into(), 100);
        let c2 = reg.register(vec!["think".into(), "believe".into()], "cognition".into(), 50);
        let result = reg.nearest_by_descriptor(&["run", "sprint"]);
        assert_eq!(result, c1);
        let result2 = reg.nearest_by_descriptor(&["think", "ponder"]);
        assert_eq!(result2, c2);
    }
}
