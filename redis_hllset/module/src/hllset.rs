//! HLLSet - HyperLogLog with Set Algebra
//!
//! Architecture:
//! - **Storage format**: Roaring bitmap (compressed) - each bit position encodes
//!   (register_index * 32 + bit_position) where bit_position represents a trailing
//!   zeros count that was observed.
//! - **Dense format**: Vec<u32> with M registers, each is a 32-bit bitmap.
//!   Used for cardinality estimation.
//!
//! Key insight: HLLSet is a 3D tensor - 2^P registers × 32 bits each.
//! The bit position IS the trailing zeros count. All bits are preserved,
//! enabling true set algebra (not just max).
//!
//! Set operations work directly on Roaring bitmap format for efficiency.
//! Cardinality estimation inflates to dense format to compute harmonic mean.

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::io::Cursor;

/// Number of precision bits (P)
pub const P: u32 = 10;

/// Number of registers (M = 2^P)
pub const M: usize = 1 << P; // 1024 registers

/// Bits per register (for 32-bit trailing zeros)
pub const BITS_PER_REG: u32 = 32;

/// Total bits in the tensor (M * 32)
pub const TOTAL_BITS: u32 = (M as u32) * BITS_PER_REG;

/// Alpha constant for HLL bias correction (for M=1024)
pub const ALPHA_M: f64 = 0.7213 / (1.0 + 1.079 / (M as f64));

/// HLLSet - HyperLogLog with Set Algebra
///
/// Stored as Roaring bitmap for compression.
/// Inflated to dense Vec<u32> for cardinality estimation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HLLSet {
    /// Compressed bitmap storage
    /// Bit at position (reg * 32 + tz) means register `reg` observed `tz` trailing zeros
    bitmap: RoaringBitmap,
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
            bitmap: RoaringBitmap::new(),
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

    /// Create HLLSet from dense registers
    pub fn from_dense(registers: &[u32]) -> Self {
        let mut hllset = Self::new();
        for (reg, &value) in registers.iter().enumerate().take(M) {
            for bit in 0..32u32 {
                if (value >> bit) & 1 == 1 {
                    let pos = (reg as u32) * BITS_PER_REG + bit;
                    hllset.bitmap.insert(pos);
                }
            }
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
    /// Algorithm (matching Julia/Cython):
    /// - Bottom P bits → register index
    /// - Remaining bits → count trailing zeros
    /// - Set bit at position tz in register's bitmap
    pub fn add_hash(&mut self, hash: u64) {
        // Extract register index from lower P bits
        let reg = (hash as u32) & ((M as u32) - 1);

        // Remaining bits → count trailing zeros
        let remaining = hash >> P;
        let tz = if remaining == 0 {
            // All zeros: max position (capped at 31 for 32-bit register)
            31
        } else {
            // Count trailing zeros, cap at 31
            remaining.trailing_zeros().min(31)
        };

        // Set bit at position (reg * 32 + tz)
        let pos = reg * BITS_PER_REG + tz;
        self.bitmap.insert(pos);
    }

    /// Inflate Roaring bitmap to dense Vec<u32> registers
    ///
    /// This is needed for cardinality estimation which requires
    /// the highest set bit in each register.
    pub fn to_dense(&self) -> Vec<u32> {
        let mut registers = vec![0u32; M];
        for pos in self.bitmap.iter() {
            let reg = (pos / BITS_PER_REG) as usize;
            let bit = pos % BITS_PER_REG;
            if reg < M {
                registers[reg] |= 1u32 << bit;
            }
        }
        registers
    }

    /// Get the highest set bit position in a register (1-indexed)
    /// Returns 0 if register is empty.
    ///
    /// Equivalent to Julia's maxidx(): total_bits - leading_zeros
    #[inline]
    fn highest_set_bit(value: u32) -> u32 {
        if value == 0 {
            0
        } else {
            32 - value.leading_zeros()
        }
    }

    /// Estimate cardinality using Horvitz-Thompson estimator
    ///
    /// This is the correct estimator for bitmap registers where we store
    /// the SET of observed states (trailing-zero counts), not just the maximum.
    ///
    /// For each state s (bit position), we count c_s = number of registers
    /// that have bit s set. Then we estimate frequency of state s as:
    ///
    ///   f̂_s = -n * ln(1 - c_s/n)
    ///
    /// Total cardinality = Σ f̂_s for all states s.
    ///
    /// This is unbiased under Poisson sampling model.
    /// Reference: Horvitz-Thompson estimator for presence/absence data.
    pub fn cardinality(&self) -> f64 {
        self.cardinality_ht()
    }

    /// Horvitz-Thompson cardinality estimator for bitmap HLLSet
    ///
    /// For bitmap registers storing the SET of all observed trailing-zero counts,
    /// the proper estimator is:
    ///
    ///   c_s = count of registers with bit s set
    ///   f̂_s = -n * ln(1 - c_s/n)  (estimated frequency for state s)
    ///   cardinality = Σ f̂_s
    ///
    /// For saturated states (c_s = n), we use geometric extrapolation from
    /// the highest non-saturated state, since each state s has expected
    /// frequency 2x of state s+1.
    pub fn cardinality_ht(&self) -> f64 {
        let n = M as f64;
        let m = M as u32;
        let mut total = 0.0f64;

        // Collect c_s values for all bit positions
        let mut c_values = [0u32; BITS_PER_REG as usize];
        for bit_pos in 0..BITS_PER_REG {
            c_values[bit_pos as usize] = self.count_registers_with_bit(bit_pos);
        }

        // Find first non-saturated state and its estimate
        let mut last_non_sat: i32 = -1;
        let mut last_f_hat = 0.0f64;
        
        for bit_pos in 0..BITS_PER_REG {
            let c_s = c_values[bit_pos as usize];
            if c_s > 0 && c_s < m {
                last_non_sat = bit_pos as i32;
                let ratio = c_s as f64 / n;
                last_f_hat = -n * (1.0 - ratio).ln();
                break;
            }
        }

        // Calculate estimates with saturation handling
        for bit_pos in 0..BITS_PER_REG {
            let c_s = c_values[bit_pos as usize];
            
            if c_s == 0 {
                continue;
            } else if c_s < m {
                // Normal HT estimate
                let ratio = c_s as f64 / n;
                total += -n * (1.0 - ratio).ln();
            } else {
                // Saturated: extrapolate using geometric series
                // State s has 2x the frequency of state s+1
                if last_non_sat > bit_pos as i32 {
                    // Extrapolate: each lower bit doubles the frequency
                    total += last_f_hat * 2.0f64.powi(last_non_sat - bit_pos as i32);
                } else {
                    // No non-saturated state found yet, use fallback
                    total += n * n.ln();
                }
            }
        }

        total.round().max(0.0)
    }

    /// Count how many registers have a specific bit position set
    fn count_registers_with_bit(&self, bit_pos: u32) -> u32 {
        let mut count = 0u32;
        
        // Iterate through sparse bitmap positions
        for pos in self.bitmap.iter() {
            let bit = pos % BITS_PER_REG;
            
            if bit == bit_pos {
                count += 1;
            }
        }
        
        count
    }

    /// Legacy HLL cardinality (for comparison)
    #[allow(dead_code)]
    pub fn cardinality_hll(&self) -> f64 {
        let registers = self.to_dense();

        // Calculate harmonic mean of 2^(-highest_set_bit)
        let mut sum = 0.0f64;
        let mut zeros = 0u32;

        for &reg_value in registers.iter() {
            let max_bit = Self::highest_set_bit(reg_value);
            sum += 2.0f64.powi(-(max_bit as i32));
            if max_bit == 0 {
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
        if zeros > 0 && raw_estimate <= 2.5 * (M as f64) {
            let linear_count = (M as f64) * ((M as f64) / (zeros as f64)).ln();
            linear_count.round().max(0.0)
        } else {
            corrected.round()
        }
    }

    /// Union of two HLLSets (A ∪ B)
    ///
    /// Roaring bitmap OR operation - keeps ALL bits from both sets.
    pub fn union(&self, other: &HLLSet) -> HLLSet {
        let mut result = self.bitmap.clone();
        result |= &other.bitmap;
        HLLSet { bitmap: result }
    }

    /// Intersection of two HLLSets (A ∩ B)
    ///
    /// Roaring bitmap AND operation - keeps only bits present in BOTH sets.
    pub fn intersection(&self, other: &HLLSet) -> HLLSet {
        let mut result = self.bitmap.clone();
        result &= &other.bitmap;
        HLLSet { bitmap: result }
    }

    /// Difference of two HLLSets (A \ B)
    ///
    /// Roaring bitmap AND-NOT: bits in A that are NOT in B.
    pub fn difference(&self, other: &HLLSet) -> HLLSet {
        let mut result = self.bitmap.clone();
        result -= &other.bitmap;
        HLLSet { bitmap: result }
    }

    /// Symmetric difference (XOR) of two HLLSets (A ⊕ B)
    ///
    /// Roaring bitmap XOR: bits in A or B but not both.
    pub fn symmetric_difference(&self, other: &HLLSet) -> HLLSet {
        let mut result = self.bitmap.clone();
        result ^= &other.bitmap;
        HLLSet { bitmap: result }
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
        self.bitmap |= &other.bitmap;
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

    /// Get SHA1 hash of the bitmap content (for content-addressing)
    pub fn content_hash(&self) -> String {
        let bytes = self.to_bytes();
        let mut hasher = Sha1::new();
        hasher.update(&bytes);
        let hash = hasher.finalize();
        hex::encode(hash)
    }

    /// Get number of set bits in the Roaring bitmap
    pub fn cardinality_bits(&self) -> u64 {
        self.bitmap.len()
    }

    /// Get number of non-zero registers
    pub fn non_zero_registers(&self) -> u32 {
        let registers = self.to_dense();
        registers.iter().filter(|&&r| r != 0).count() as u32
    }

    /// Get memory usage in bytes (serialized size)
    pub fn memory_usage(&self) -> usize {
        self.bitmap.serialized_size()
    }

    /// Serialize to bytes (Roaring bitmap format)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.bitmap.serialized_size());
        self.bitmap.serialize_into(&mut bytes).unwrap_or_default();
        bytes
    }

    /// Deserialize from bytes (Roaring bitmap format)
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        RoaringBitmap::deserialize_from(bytes)
            .ok()
            .map(|bitmap| HLLSet { bitmap })
    }

    /// Get raw register bitmap for a specific register (inflates to dense)
    pub fn get_register_bitmap(&self, reg: usize) -> u32 {
        let mut value = 0u32;
        let start = (reg as u32) * BITS_PER_REG;
        for bit in 0..BITS_PER_REG {
            if self.bitmap.contains(start + bit) {
                value |= 1u32 << bit;
            }
        }
        value
    }

    /// Get register value for HLL (highest set bit, 1-indexed)
    pub fn get_register(&self, reg: usize) -> u32 {
        Self::highest_set_bit(self.get_register_bitmap(reg))
    }

    /// Get all non-zero registers with their bitmap values (for debugging)
    pub fn dump_positions(&self) -> Vec<(u32, u32)> {
        let registers = self.to_dense();
        registers
            .iter()
            .enumerate()
            .filter(|(_, &r)| r != 0)
            .map(|(i, &r)| (i as u32, Self::highest_set_bit(r)))
            .collect()
    }

    /// Get all active (reg, zeros) positions where bits are set.
    /// 
    /// This is the 2D tensor view used for disambiguation:
    /// each active position represents a potential token inscription.
    /// 
    /// Returns: Vec of (register_index, zeros_count) tuples
    pub fn active_positions(&self) -> Vec<(u32, u32)> {
        let mut positions = Vec::new();
        
        // Iterate directly through sparse bitmap
        for pos in self.bitmap.iter() {
            let reg = pos / BITS_PER_REG;
            let zeros = pos % BITS_PER_REG;
            positions.push((reg, zeros));
        }
        
        positions
    }
    
    /// Get popcount (total number of set bits)
    pub fn popcount(&self) -> u64 {
        self.bitmap.len()
    }
    
    /// Get popcount per register
    pub fn register_popcounts(&self) -> Vec<u32> {
        let registers = self.to_dense();
        registers
            .iter()
            .map(|&r| r.count_ones())
            .collect()
    }
    
    /// Get c_s counts (number of registers with bit s set) for HT estimator
    pub fn bit_counts(&self) -> Vec<u32> {
        let mut counts = vec![0u32; BITS_PER_REG as usize];
        
        for pos in self.bitmap.iter() {
            let bit = (pos % BITS_PER_REG) as usize;
            counts[bit] += 1;
        }
        
        counts
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.bitmap.is_empty()
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
        assert!(card >= 3.0 && card <= 7.0, "card={}", card);
    }

    #[test]
    fn test_union() {
        let a = HLLSet::from_tokens(vec!["a", "b", "c"]);
        let b = HLLSet::from_tokens(vec!["c", "d", "e"]);
        let union = a.union(&b);

        // Union should have cardinality close to 5
        let card = union.cardinality();
        assert!(card >= 3.0 && card <= 7.0, "card={}", card);
    }

    #[test]
    fn test_roaring_vs_dense() {
        // Create HLLSet with some data
        let hll = HLLSet::from_tokens(vec!["test1", "test2", "test3"]);

        // Convert to dense and back
        let dense = hll.to_dense();
        let hll2 = HLLSet::from_dense(&dense);

        // Should have same cardinality
        assert_eq!(hll.cardinality(), hll2.cardinality());
    }

    #[test]
    fn test_serialization() {
        let hll = HLLSet::from_tokens(vec!["a", "b", "c"]);
        let bytes = hll.to_bytes();
        let hll2 = HLLSet::from_bytes(&bytes).unwrap();

        assert_eq!(hll.cardinality(), hll2.cardinality());
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

    #[test]
    fn test_bit_encoding() {
        let mut hll = HLLSet::new();

        // Manually add a hash that goes to register 5 with 3 trailing zeros
        // Hash = (reg << 0) | (remaining bits that have 3 trailing zeros)
        // remaining = 0b1000 has 3 trailing zeros
        let hash: u64 = 5 | (0b1000 << P);
        hll.add_hash(hash);

        // Check that bit 3 is set in register 5
        let reg5 = hll.get_register_bitmap(5);
        assert_eq!(reg5, 0b1000, "reg5={:b}", reg5);
    }

    #[test]
    fn test_roaring_operations() {
        // Test that Roaring bitmap operations work correctly
        let a = HLLSet::from_tokens(vec!["a", "b", "c"]);
        let b = HLLSet::from_tokens(vec!["b", "c", "d"]);

        // Check that operations don't panic
        let _union = a.union(&b);
        let _inter = a.intersection(&b);
        let _diff = a.difference(&b);
        let _xor = a.symmetric_difference(&b);
    }
}
