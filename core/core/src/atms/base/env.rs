//! Env = u64 bitvector over assumption space (up to 64 concurrent situation assumptions).
//! Context switch is O(1): just change which Env is "active."

use crate::types::Env;

/// E₁ ⊆ E₂ — E₁ subsumes E₂ (every assumption in E₁ is in E₂).
#[inline]
pub fn subsumes(e1: Env, e2: Env) -> bool {
    e1 & e2 == e1
}

/// Set union of two environments.
#[inline]
pub fn union(e1: Env, e2: Env) -> Env {
    e1 | e2
}

/// Set intersection of two environments.
#[inline]
pub fn intersect(e1: Env, e2: Env) -> Env {
    e1 & e2
}

/// The empty environment (no assumptions active).
#[inline]
pub fn empty() -> Env {
    0
}

/// Singleton environment: exactly bit `bit` set.
/// Panics if bit >= 64.
#[inline]
pub fn singleton(bit: u8) -> Env {
    assert!(bit < 64, "assumption bit out of range");
    1u64 << bit
}

#[inline]
pub fn is_empty(e: Env) -> bool {
    e == 0
}

/// Number of assumptions in this environment.
#[inline]
pub fn assumption_count(e: Env) -> u32 {
    e.count_ones()
}

/// Remove assumption `bit` from environment.
#[inline]
pub fn remove_bit(e: Env, bit: u8) -> Env {
    e & !(1u64 << bit)
}

/// Check whether assumption `bit` is active in `e`.
#[inline]
pub fn has_bit(e: Env, bit: u8) -> bool {
    e & (1u64 << bit) != 0
}

/// Iterate over assumption bit positions set in `e`.
pub fn active_bits(e: Env) -> impl Iterator<Item = u8> {
    (0u8..64).filter(move |&b| has_bit(e, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsumes_reflexive() {
        let e = singleton(3) | singleton(7);
        assert!(subsumes(e, e));
    }

    #[test]
    fn subsumes_proper_subset() {
        let e1 = singleton(3);
        let e2 = singleton(3) | singleton(7);
        assert!(subsumes(e1, e2));
        assert!(!subsumes(e2, e1));
    }

    #[test]
    fn empty_subsumes_everything() {
        assert!(subsumes(empty(), singleton(5) | singleton(10)));
    }

    #[test]
    fn active_bits_correct() {
        let e = singleton(0) | singleton(3) | singleton(63);
        let bits: Vec<u8> = active_bits(e).collect();
        assert_eq!(bits, vec![0, 3, 63]);
    }
}
