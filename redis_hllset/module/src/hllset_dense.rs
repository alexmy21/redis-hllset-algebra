//! Core HLLSet implementation with dense bitmap registers
//!
//! Each register is a 32-bit bitmap where bit k is set if an element
//! hashed to this bucket had k trailing zeros. This enables true bitwise
//! set algebra (OR, AND, AND-NOT, XOR) matching Julia HllSets.jl.

use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::io::Cursor;

/// Number of registers (buckets) - must be power of 2
/// M = 2^P where P is precision bits
pub const P: u32 = 10;
pub const M: usize = 1 << P; // 1024 registers

/// Alpha constant for bias correction (for M=1024)
/// alpha_m = 0.7213 / (1 + 1.079/m) for m >= 128
pub const ALPHA_M: f64 = 0.7213 / (1.0 + 1.079 / (M as f64));

/// HLLSet - HyperLogLog with Set Algebra
///
/// Dense register format: M × 32-bit bitmaps.
/// Each register stores ALL observed trailing zero counts as set bits.
/// This enables true bitwise set operations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HLLSet {
    /// Dense array of M registers, each is a 32-bit bitmap
    /// Bit k is set if an element had k trailing zeros
    registers: Vec<u32>,
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
            registers: vec![0u32; M],
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
    /// Uses trailing zeros like Cython/Julia implementation:
    /// - Bottom P bits → bucket index
    /// - Remaining bits → count trailing zeros
    /// - Set bit at position tz in the register bitmap
    ///
    /// This preserves ALL observations, enabling true set algebra.
    pub fn add_hash(&mut self, hash: u64) {
        // Extract bucket index from lower P bits
        let bucket = (hash as usize) & (M - 1);

        // Remaining bits → count trailing zeros
        let remaining = hash >> P;
        let tz = if remaining == 0 {
            // All zeros: use max position (31)
            31
        } else {
            // Count trailing zeros, cap at 31 for 32-bit register
            remaining.trailing_zeros().min(31)
        };

        // Set bit at position tz (OR into the bitmap)
        self.registers[bucket] |= 1u32 << tz;
    }

    /// Get the highest set bit position in a register (1-indexed)
    /// Returns 0 if register is empty.
    ///
    /// Equivalent to Julia's maxidx(): 32 - leading_zeros(register)
    #[inline]
    fn highest_set_bit(value: u32) -> u32 {
        if value == 0 {
            0
        } else {
            32 - value.leading_zeros()
        }
    }

    /// Get register value for cardinality estimation
    /// Returns the highest set bit position (1-indexed), or 0 if empty.
    pub fn get_register(&self, bucket: usize) -> u32 {
        Self::highest_set_bit(self.registers[bucket])
    }

    /// Estimate cardinality using HyperLogLog++ algorithm with bias correction
    ///
    /// Uses Google's HLL++ algorithm (Heule, Nunkesser, Hall, 2013) with
    /// empirical bias correction tables for precision 10.
    ///
    /// For each register, uses highest_set_bit() to get the max observed
    /// trailing zero count, matching Julia's count() function.
    pub fn cardinality(&self) -> f64 {
        // Calculate harmonic mean of 2^(-highest_set_bit)
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
        if zeros == M as u32 {
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
    /// Bitwise OR of all registers. This preserves ALL observations
    /// from both sets.
    pub fn union(&self, other: &HLLSet) -> HLLSet {
        let mut result = HLLSet::new();
        for i in 0..M {
            result.registers[i] = self.registers[i] | other.registers[i];
        }
        result
    }

    /// Intersection of two HLLSets (A ∩ B)
    ///
    /// Bitwise AND of all registers. Only keeps observations
    /// present in BOTH sets.
    pub fn intersection(&self, other: &HLLSet) -> HLLSet {
        let mut result = HLLSet::new();
        for i in 0..M {
            result.registers[i] = self.registers[i] & other.registers[i];
        }
        result
    }

    /// Difference of two HLLSets (A \ B)
    ///
    /// Bitwise AND-NOT: keeps bits in A that are NOT in B.
    pub fn difference(&self, other: &HLLSet) -> HLLSet {
        let mut result = HLLSet::new();
        for i in 0..M {
            result.registers[i] = self.registers[i] & !other.registers[i];
        }
        result
    }

    /// Symmetric difference (XOR) of two HLLSets (A ⊕ B)
    ///
    /// Bitwise XOR: elements in A or B but not both.
    pub fn symmetric_difference(&self, other: &HLLSet) -> HLLSet {
        let mut result = HLLSet::new();
        for i in 0..M {
            result.registers[i] = self.registers[i] ^ other.registers[i];
        }
        result
    }

    /// Jaccard similarity: |A ∩ B| / |A ∪ B|
    pub fn jaccard_similarity(&self, other: &HLLSet) -> f64 {
        let union_set = self.union(other);
        let inter_set = self.intersection(other);

        let union_card = union_set.cardinality();
        let inter_card = inter_set.cardinality();

        if union_card == 0.0 {
            return 1.0; // Both empty = identical
        }

        inter_card / union_card
    }

    /// Merge another HLLSet into this one (in-place union)
    pub fn merge(&mut self, other: &HLLSet) {
        for i in 0..M {
            self.registers[i] |= other.registers[i];
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
        self.registers.iter().filter(|&&r| r != 0).count() as u32
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        // M registers * 4 bytes each + struct overhead
        M * 4 + std::mem::size_of::<Self>()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        bincode::deserialize(bytes).ok()
    }

    /// Get raw register bitmap for a bucket (for debugging)
    pub fn get_register_bitmap(&self, bucket: usize) -> u32 {
        self.registers[bucket]
    }

    /// Get all non-zero registers with their values (for debugging)
    pub fn dump_positions(&self) -> Vec<(u32, u32)> {
        self.registers
            .iter()
            .enumerate()
            .filter(|(_, &r)| r != 0)
            .map(|(i, &r)| (i as u32, Self::highest_set_bit(r)))
            .collect()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.registers.iter().all(|&r| r == 0)
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
        let hll = HLLSet::new();
        assert_eq!(hll.cardinality(), 0.0);
        assert!(hll.is_empty());
    }

    #[test]
    fn test_basic_cardinality() {
        let hll = HLLSet::from_tokens(vec!["a", "b", "c", "d", "e"]);
        let card = hll.cardinality();
        // Should be close to 5
        assert!(card >= 3.0 && card <= 7.0);
    }

    #[test]
    fn test_union() {
        let a = HLLSet::from_tokens(vec!["a", "b", "c"]);
        let b = HLLSet::from_tokens(vec!["c", "d", "e"]);
        let union = a.union(&b);
        
        // Union should have cardinality close to 5
        let card = union.cardinality();
        assert!(card >= 3.0 && card <= 7.0);
    }

    #[test]
    fn test_intersection() {
        let a = HLLSet::from_tokens(vec!["a", "b", "c"]);
        let b = HLLSet::from_tokens(vec!["c", "d", "e"]);
        let inter = a.intersection(&b);
        
        // Intersection has only "c" - cardinality should be small
        let card = inter.cardinality();
        assert!(card >= 0.0 && card <= 3.0);
    }

    #[test]
    fn test_bitwise_operations() {
        // Verify that bitwise operations work correctly
        let mut a = HLLSet::new();
        let mut b = HLLSet::new();
        
        // Manually set some bits
        a.registers[0] = 0b11110000;
        b.registers[0] = 0b00111100;
        
        let union = a.union(&b);
        assert_eq!(union.registers[0], 0b11111100);
        
        let inter = a.intersection(&b);
        assert_eq!(inter.registers[0], 0b00110000);
        
        let diff = a.difference(&b);
        assert_eq!(diff.registers[0], 0b11000000);
        
        let xor = a.symmetric_difference(&b);
        assert_eq!(xor.registers[0], 0b11001100);
    }

    #[test]
    fn test_highest_set_bit() {
        assert_eq!(HLLSet::highest_set_bit(0), 0);
        assert_eq!(HLLSet::highest_set_bit(1), 1);
        assert_eq!(HLLSet::highest_set_bit(2), 2);
        assert_eq!(HLLSet::highest_set_bit(0b1000), 4);
        assert_eq!(HLLSet::highest_set_bit(0b10101010), 8);
        assert_eq!(HLLSet::highest_set_bit(0x80000000), 32);
    }
}
