//! Core HLLSet implementation
//!
//! HLLSet uses a bitmap-based register model where each bucket stores the
//! maximum leading zero count observed. This enables efficient set operations
//! through bitwise operations on the register bitmaps.

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::io::Cursor;

/// Number of registers (buckets) - must be power of 2
/// M = 2^P where P is precision bits
pub const P: u32 = 10;
pub const M: u32 = 1 << P; // 1024 registers

/// Maximum value a register can hold (5 bits = 0-31)
pub const MAX_REGISTER_VALUE: u32 = 31;

/// Alpha constant for bias correction (for M=1024)
/// alpha_m = 0.7213 / (1 + 1.079/m) for m >= 128
pub const ALPHA_M: f64 = 0.7213 / (1.0 + 1.079 / (M as f64));

/// HLLSet - HyperLogLog with Set Algebra
///
/// The core data structure that enables probabilistic set operations.
/// Each register is represented as a bitmap position: register_index * 32 + value
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HLLSet {
    /// Bitmap storing register values
    /// Position = bucket_index * 32 + register_value
    registers: RoaringBitmap,
}

impl Default for HLLSet {
    fn default() -> Self {
        Self::new()
    }
}

impl HLLSet {
    /// Create a new empty HLLSet
    pub fn new() -> Self {
        Self {
            registers: RoaringBitmap::new(),
        }
    }

    /// Create HLLSet from tokens (strings)
    pub fn from_tokens<I, S>(tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[u8]>,
    {
        let mut hllset = Self::new();
        for token in tokens {
            hllset.add_token(token.as_ref());
        }
        hllset
    }

    /// Create HLLSet from pre-computed 64-bit hashes
    pub fn from_hashes<I>(hashes: I) -> Self
    where
        I: IntoIterator<Item = u64>,
    {
        let mut hllset = Self::new();
        for hash in hashes {
            hllset.add_hash(hash);
        }
        hllset
    }

    /// Add a token to the HLLSet
    pub fn add_token(&mut self, token: &[u8]) {
        let hash = murmur3_hash(token);
        self.add_hash(hash);
    }

    /// Add a pre-computed hash to the HLLSet
    ///
    /// Uses standard HLL algorithm with leading zeros:
    /// - Bottom P bits → bucket index  
    /// - Remaining bits → count leading zeros + 1 (rho value)
    /// - Store max rho per bucket
    ///
    /// Note: Cython/Julia use trailing zeros with bitmap registers.
    /// We use leading zeros with value encoding in RoaringBitmap.
    /// Both achieve equivalent HLL estimation.
    pub fn add_hash(&mut self, hash: u64) {
        // Extract bucket index from lower P bits
        let bucket = (hash as u32) & (M - 1);

        // Count leading zeros in the (64-P) remaining bits, then +1
        // Since remaining is stored in u64, leading_zeros() counts from bit 63.
        // We need to subtract P to get the count within our (64-P) bit window.
        let remaining = hash >> P;
        let leading_zeros = if remaining == 0 {
            // All bits are zero, so leading zeros = (64-P)
            64 - P
        } else {
            // leading_zeros() on u64 counts from bit 63, but our value
            // only uses bits 0..(64-P). Subtract P to correct.
            remaining.leading_zeros() - P
        };
        // rho = leading_zeros + 1, capped at MAX_REGISTER_VALUE
        let rho = (leading_zeros + 1).min(MAX_REGISTER_VALUE);

        // Update register if new value is larger
        self.set_register(bucket, rho);
    }

    /// Set a register value (only if larger than current)
    fn set_register(&mut self, bucket: u32, value: u32) {
        let current = self.get_register(bucket);
        if value > current {
            // Clear old value
            if current > 0 {
                self.registers.remove(bucket * 32 + current);
            }
            // Set new value
            self.registers.insert(bucket * 32 + value);
        }
    }

    /// Get current register value for a bucket
    pub fn get_register(&self, bucket: u32) -> u32 {
        let base = bucket * 32;
        // Find the highest set bit in this bucket's range
        for v in (1..=MAX_REGISTER_VALUE).rev() {
            if self.registers.contains(base + v) {
                return v;
            }
        }
        0
    }

    /// Estimate cardinality using HyperLogLog++ algorithm with bias correction
    ///
    /// Uses Google's HLL++ algorithm (Heule, Nunkesser, Hall, 2013) with
    /// empirical bias correction tables for precision 10.
    ///
    /// HLLSet is immutable after creation, so this is a pure function.
    pub fn cardinality(&self) -> f64 {
        // Calculate harmonic mean of 2^(-register)
        let mut sum = 0.0f64;
        let mut zeros = 0u32;

        for bucket in 0..M {
            let val = self.get_register(bucket);
            sum += 2.0f64.powi(-(val as i32));
            if val == 0 {
                zeros += 1;
            }
        }

        // Empty set fast-path
        if zeros == M {
            return 0.0;
        }

        // Raw estimate: alpha * m^2 / sum
        let raw_estimate = ALPHA_M * (M as f64) * (M as f64) / sum;

        // Apply bias correction from HLL++ empirical tables
        let bias = crate::bias::estimate_bias(P, raw_estimate);
        let corrected = (raw_estimate - bias - 1.0).max(0.0);

        // For very small cardinalities with empty registers, use linear counting
        // Linear counting threshold: ~2.5 * M
        if zeros > 0 && raw_estimate <= 2.5 * (M as f64) {
            let linear_count = (M as f64) * ((M as f64) / (zeros as f64)).ln();
            linear_count.round().max(0.0)
        } else {
            corrected.round()
        }
    }

    /// Union of two HLLSets (A ∪ B)
    ///
    /// For each register, take the maximum value from either set.
    /// This is achieved by OR-ing the bitmaps and then keeping only
    /// the maximum bit per bucket.
    pub fn union(&self, other: &HLLSet) -> HLLSet {
        let mut result = HLLSet::new();

        for bucket in 0..M {
            let val_a = self.get_register(bucket);
            let val_b = other.get_register(bucket);
            let max_val = val_a.max(val_b);
            if max_val > 0 {
                result.registers.insert(bucket * 32 + max_val);
            }
        }

        result
    }

    /// Intersection of two HLLSets (A ∩ B)
    ///
    /// For each register, take the minimum value from both sets.
    /// A zero in either set means that bucket contributes nothing to intersection.
    pub fn intersection(&self, other: &HLLSet) -> HLLSet {
        let mut result = HLLSet::new();

        for bucket in 0..M {
            let val_a = self.get_register(bucket);
            let val_b = other.get_register(bucket);
            let min_val = val_a.min(val_b);
            if min_val > 0 {
                result.registers.insert(bucket * 32 + min_val);
            }
        }

        result
    }

    /// Difference of two HLLSets (A \ B)
    ///
    /// For each register, if A has a higher value than B, keep A's excess.
    /// This represents elements that are "more likely" to be only in A.
    pub fn difference(&self, other: &HLLSet) -> HLLSet {
        let mut result = HLLSet::new();

        for bucket in 0..M {
            let val_a = self.get_register(bucket);
            let val_b = other.get_register(bucket);
            // Saturating subtraction - keep A's contribution if larger
            let diff_val = if val_a > val_b { val_a - val_b } else { 0 };
            if diff_val > 0 {
                result.registers.insert(bucket * 32 + diff_val);
            }
        }

        result
    }

    /// Symmetric difference (XOR) of two HLLSets (A ⊕ B)
    ///
    /// Elements in A or B but not both.
    /// |A ⊕ B| = |A ∪ B| - |A ∩ B|
    pub fn symmetric_difference(&self, other: &HLLSet) -> HLLSet {
        let mut result = HLLSet::new();

        for bucket in 0..M {
            let val_a = self.get_register(bucket);
            let val_b = other.get_register(bucket);
            // XOR-like behavior: absolute difference
            let xor_val = if val_a > val_b {
                val_a - val_b
            } else {
                val_b - val_a
            };
            if xor_val > 0 {
                result.registers.insert(bucket * 32 + xor_val);
            }
        }

        result
    }

    /// Jaccard similarity: |A ∩ B| / |A ∪ B|
    pub fn jaccard_similarity(&mut self, other: &mut HLLSet) -> f64 {
        let mut union_set = self.union(other);
        let mut inter_set = self.intersection(other);

        let union_card = union_set.cardinality();
        let inter_card = inter_set.cardinality();

        if union_card == 0.0 {
            return 1.0; // Both empty = identical
        }

        inter_card / union_card
    }

    /// Merge another HLLSet into this one (in-place union)
    pub fn merge(&mut self, other: &HLLSet) {
        for bucket in 0..M {
            let other_val = other.get_register(bucket);
            if other_val > 0 {
                self.set_register(bucket, other_val);
            }
        }
    }

    /// Generate content-addressable key from sorted tokens
    pub fn content_key<I, S>(tokens: I) -> String
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut sorted: Vec<String> = tokens.into_iter().map(|s| s.as_ref().to_string()).collect();
        sorted.sort();
        sorted.dedup();

        let joined = sorted.join("\0");

        let mut hasher = Sha1::new();
        hasher.update(joined.as_bytes());
        let hash = hasher.finalize();

        format!("hllset:{}", hex::encode(hash))
    }

    /// Get number of non-zero registers
    pub fn non_zero_registers(&self) -> u32 {
        let mut count = 0;
        for bucket in 0..M {
            if self.get_register(bucket) > 0 {
                count += 1;
            }
        }
        count
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        // RoaringBitmap serialized size + struct overhead
        self.registers.serialized_size() + std::mem::size_of::<Self>()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        bincode::deserialize(bytes).ok()
    }

    /// Get all register positions (for debugging)
    pub fn dump_positions(&self) -> Vec<(u32, u32)> {
        let mut positions = Vec::new();
        for bucket in 0..M {
            let val = self.get_register(bucket);
            if val > 0 {
                positions.push((bucket, val));
            }
        }
        positions
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.registers.is_empty()
    }
}

/// MurmurHash3 (64-bit) for consistent hashing
fn murmur3_hash(data: &[u8]) -> u64 {
    let mut cursor = Cursor::new(data);
    murmur3::murmur3_x64_128(&mut cursor, 0)
        .map(|h| h as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_hllset() {
        let mut hll = HLLSet::new();
        assert_eq!(hll.cardinality(), 0.0);
        assert!(hll.is_empty());
    }

    #[test]
    fn test_basic_add() {
        let mut hll = HLLSet::from_tokens(["apple", "banana", "cherry"]);
        let card = hll.cardinality();
        assert!(card >= 2.5 && card <= 3.5, "Expected ~3, got {}", card);
    }

    #[test]
    fn test_union() {
        let hll_a = HLLSet::from_tokens(["a", "b", "c"]);
        let hll_b = HLLSet::from_tokens(["b", "c", "d"]);

        let mut union = hll_a.union(&hll_b);
        let card = union.cardinality();
        assert!(card >= 3.5 && card <= 4.5, "Expected ~4, got {}", card);
    }

    #[test]
    fn test_intersection() {
        let hll_a = HLLSet::from_tokens(["a", "b", "c"]);
        let hll_b = HLLSet::from_tokens(["b", "c", "d"]);

        let mut inter = hll_a.intersection(&hll_b);
        let card = inter.cardinality();
        assert!(card >= 1.5 && card <= 2.5, "Expected ~2, got {}", card);
    }

    #[test]
    fn test_content_key() {
        let key1 = HLLSet::content_key(["a", "b", "c"]);
        let key2 = HLLSet::content_key(["c", "b", "a"]); // Different order
        let key3 = HLLSet::content_key(["a", "b", "d"]); // Different content

        assert_eq!(key1, key2, "Same tokens should produce same key");
        assert_ne!(key1, key3, "Different tokens should produce different key");
        assert!(key1.starts_with("hllset:"));
    }

    #[test]
    fn test_jaccard_similarity() {
        let mut hll_a = HLLSet::from_tokens(["a", "b", "c"]);
        let mut hll_b = HLLSet::from_tokens(["b", "c", "d"]);

        let sim = hll_a.jaccard_similarity(&mut hll_b);
        // |{b,c}| / |{a,b,c,d}| = 2/4 = 0.5
        assert!(sim >= 0.3 && sim <= 0.7, "Expected ~0.5, got {}", sim);
    }

    #[test]
    fn test_serialization() {
        let hll = HLLSet::from_tokens(["test", "serialization"]);
        let bytes = hll.to_bytes();
        let restored = HLLSet::from_bytes(&bytes).unwrap();

        assert_eq!(
            hll.dump_positions(),
            restored.dump_positions(),
            "Serialization should preserve data"
        );
    }

    #[test]
    fn test_large_set() {
        let tokens: Vec<String> = (0..10000).map(|i| format!("token_{}", i)).collect();
        let mut hll = HLLSet::from_tokens(&tokens);
        let card = hll.cardinality();

        // HLL should estimate within ~3% for large sets
        let error = (card - 10000.0).abs() / 10000.0;
        assert!(error < 0.05, "Error too high: {}%", error * 100.0);
    }
}
