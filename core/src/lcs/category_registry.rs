//! CategoryRegistry: maps runtime-discovered TypeCategory IDs to centroids and labels.
//! Populated by the bootstrap pipeline and persisted as JSON for the training loop.

use std::collections::HashMap;
use crate::types::TypeCategory;

/// One entry in the category registry.
#[derive(Clone, Debug)]
pub struct CategoryEntry {
    pub id:       TypeCategory,
    /// K-means centroid in structural feature space.
    pub centroid: Vec<f32>,
    /// Human-readable label (derived from dominant structural pattern).
    pub label:    String,
    /// Number of tokens assigned to this category during bootstrap.
    pub count:    usize,
}

/// Registry: ID → entry.
///
/// Starts empty. All category IDs are assigned by the bootstrap CategoryInducer.
/// ID 0 is reserved for TypeCategory::DEFAULT (unassigned). IDs 1+ are opaque
/// cluster IDs — none carry linguistic names.
pub struct CategoryRegistry {
    entries: HashMap<u32, CategoryEntry>,
    next_id: u32,
}

impl CategoryRegistry {
    pub fn new() -> Self {
        // Start from 1; 0 is reserved for TypeCategory::DEFAULT.
        Self { entries: HashMap::new(), next_id: 1 }
    }

    /// Register a discovered cluster centroid and return its assigned TypeCategory.
    pub fn register(&mut self, centroid: Vec<f32>, label: String, count: usize) -> TypeCategory {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(id, CategoryEntry {
            id: TypeCategory(id),
            centroid,
            label,
            count,
        });
        TypeCategory(id)
    }

    /// Update the centroid for an existing prototype category.
    pub fn update_centroid(&mut self, cat: TypeCategory, centroid: Vec<f32>, count: usize) {
        if let Some(entry) = self.entries.get_mut(&cat.id()) {
            entry.centroid = centroid;
            entry.count    = count;
        }
    }

    /// Nearest registered category by Euclidean distance in feature space.
    pub fn nearest(&self, features: &[f32]) -> TypeCategory {
        self.entries.values()
            .filter(|e| !e.centroid.is_empty())
            .min_by(|a, b| {
                let da = squared_dist(features, &a.centroid);
                let db = squared_dist(features, &b.centroid);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|e| e.id)
            .unwrap_or(TypeCategory::DEFAULT)
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

fn squared_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(&x, &y)| ((x - y) as f64).powi(2))
        .sum()
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
        let cat = reg.register(vec![0.1, 0.2], "cluster_1".into(), 10);
        assert!(cat.id() >= 1, "first cluster ID must be >= 1");
        assert!(reg.get(cat).is_some());
    }

    #[test]
    fn nearest_finds_closest_centroid() {
        let mut reg = CategoryRegistry::new();
        let c1 = reg.register(vec![1.0, 0.0], "cluster_1".into(), 100);
        let c2 = reg.register(vec![0.0, 1.0], "cluster_2".into(), 50);
        let result = reg.nearest(&[0.9, 0.1]);
        assert_eq!(result, c1);
        let result2 = reg.nearest(&[0.1, 0.9]);
        assert_eq!(result2, c2);
    }
}
