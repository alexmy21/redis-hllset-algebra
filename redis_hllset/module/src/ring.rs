//! XOR Ring Algebra for HLLSet decomposition
//!
//! This module implements server-side Gaussian elimination over GF(2)
//! for decomposing HLLSets into XOR of basis elements.
//!
//! # Key Concepts
//!
//! - **Ring**: A set of linearly independent HLLSets (basis)
//! - **Decomposition**: Express any HLLSet as XOR of basis elements
//! - **BitMatrix**: Row-reduced echelon form for efficient elimination
//!
//! # Algorithm
//!
//! For each new HLLSet H:
//! 1. Convert to bitvector (M*32 bits for p-bit precision)
//! 2. Gaussian eliminate against current basis
//! 3. If residual non-zero: H is independent → add to basis
//! 4. If residual zero: H = B₁ ⊕ B₂ ⊕ ... (dependent)

use std::collections::HashMap;

/// Number of bits per HLLSet register
pub const BITS_PER_REG: usize = 32;

/// A compressed bitvector using u64 words
#[derive(Clone, Debug)]
pub struct BitVec {
    words: Vec<u64>,
    num_bits: usize,
}

impl BitVec {
    /// Create a new zero bitvector with given number of bits
    pub fn new(num_bits: usize) -> Self {
        let num_words = (num_bits + 63) / 64;
        Self {
            words: vec![0u64; num_words],
            num_bits,
        }
    }

    /// Create from dense HLLSet registers (Vec<u32>)
    /// Each register contributes 32 bits
    pub fn from_registers(registers: &[u32]) -> Self {
        let num_bits = registers.len() * BITS_PER_REG;
        let mut bv = Self::new(num_bits);
        
        for (reg_idx, &value) in registers.iter().enumerate() {
            // Register i contributes bits [i*32, i*32+32)
            let base_bit = reg_idx * BITS_PER_REG;
            for bit in 0..32 {
                if (value >> bit) & 1 == 1 {
                    bv.set(base_bit + bit);
                }
            }
        }
        bv
    }

    /// Get bit at position
    #[inline]
    pub fn get(&self, pos: usize) -> bool {
        if pos >= self.num_bits {
            return false;
        }
        let word_idx = pos / 64;
        let bit_idx = pos % 64;
        (self.words[word_idx] >> bit_idx) & 1 == 1
    }

    /// Set bit at position
    #[inline]
    pub fn set(&mut self, pos: usize) {
        if pos >= self.num_bits {
            return;
        }
        let word_idx = pos / 64;
        let bit_idx = pos % 64;
        self.words[word_idx] |= 1u64 << bit_idx;
    }

    /// Clear bit at position
    #[inline]
    pub fn clear(&mut self, pos: usize) {
        if pos >= self.num_bits {
            return;
        }
        let word_idx = pos / 64;
        let bit_idx = pos % 64;
        self.words[word_idx] &= !(1u64 << bit_idx);
    }

    /// XOR with another bitvector (in-place)
    pub fn xor_assign(&mut self, other: &BitVec) {
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a ^= *b;
        }
    }

    /// Check if all bits are zero
    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Check if any bit is set
    pub fn any(&self) -> bool {
        self.words.iter().any(|&w| w != 0)
    }

    /// Find position of first set bit (None if all zero)
    pub fn first_one(&self) -> Option<usize> {
        for (word_idx, &word) in self.words.iter().enumerate() {
            if word != 0 {
                let bit = word.trailing_zeros() as usize;
                let pos = word_idx * 64 + bit;
                if pos < self.num_bits {
                    return Some(pos);
                }
            }
        }
        None
    }

    /// Count number of set bits (popcount)
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Clone this bitvector
    pub fn clone_vec(&self) -> Self {
        Self {
            words: self.words.clone(),
            num_bits: self.num_bits,
        }
    }
}

/// Result of decomposing an HLLSet against a ring
#[derive(Debug, Clone)]
pub struct DecomposeResult {
    /// SHA1 of the decomposed HLLSet
    pub sha1: String,
    
    /// True if this HLLSet is linearly independent (new base)
    pub is_new_base: bool,
    
    /// SHA1s of basis elements that XOR to this HLLSet
    /// If is_new_base=true, contains only this HLLSet's SHA1
    /// If is_new_base=false, contains SHA1s of bases to XOR
    pub expression: Vec<String>,
    
    /// Ring rank before this operation
    pub rank_before: usize,
    
    /// Ring rank after this operation
    pub rank_after: usize,
}

/// BitMatrix in row-reduced echelon form for Gaussian elimination
#[derive(Clone, Debug)]
pub struct BitMatrix {
    /// Rows of the matrix (each is a bitvector)
    rows: Vec<BitVec>,
    
    /// Pivot column for each row
    pivots: Vec<usize>,
    
    /// Number of columns (bits per HLLSet)
    num_cols: usize,
}

impl BitMatrix {
    /// Create empty matrix with given column count
    pub fn new(num_cols: usize) -> Self {
        Self {
            rows: Vec::new(),
            pivots: Vec::new(),
            num_cols,
        }
    }

    /// Get the rank (number of independent rows)
    pub fn rank(&self) -> usize {
        self.rows.len()
    }

    /// Reduce a vector against current basis.
    /// Returns (reduced_vector, row_indices_used)
    /// 
    /// The reduced_vector is the residual after eliminating with basis rows.
    /// row_indices_used indicates which basis rows were XORed.
    pub fn reduce(&self, vec: &BitVec) -> (BitVec, Vec<usize>) {
        let mut residual = vec.clone_vec();
        let mut used = Vec::new();
        
        for (row_idx, &pivot) in self.pivots.iter().enumerate() {
            if residual.get(pivot) {
                // XOR with this basis row to eliminate the pivot
                residual.xor_assign(&self.rows[row_idx]);
                used.push(row_idx);
            }
        }
        
        (residual, used)
    }

    /// Add a new row to the matrix, maintaining row-echelon form.
    /// The row should already be reduced (residual from reduce()).
    /// Returns the pivot column used, or None if row was zero.
    pub fn add_row(&mut self, row: BitVec) -> Option<usize> {
        if let Some(pivot) = row.first_one() {
            // Insert in sorted position by pivot
            let insert_pos = self.pivots.iter().position(|&p| p > pivot).unwrap_or(self.pivots.len());
            
            self.rows.insert(insert_pos, row);
            self.pivots.insert(insert_pos, pivot);
            
            // Re-reduce all rows above to maintain proper RREF
            // (This ensures pivot column has only one 1)
            for i in 0..insert_pos {
                if self.rows[i].get(pivot) {
                    let new_row = self.rows[insert_pos].clone_vec();
                    self.rows[i].xor_assign(&new_row);
                }
            }
            
            Some(pivot)
        } else {
            None // Zero row, don't add
        }
    }
}

/// Ring state for XOR decomposition
#[derive(Clone, Debug)]
pub struct RingState {
    /// Ring identifier
    pub ring_id: String,
    
    /// Precision bits (p)
    pub p_bits: u8,
    
    /// Number of registers (2^p_bits)
    pub num_registers: usize,
    
    /// Total bits per HLLSet (num_registers * 32)
    pub num_bits: usize,
    
    /// Basis matrix in row-reduced echelon form
    matrix: BitMatrix,
    
    /// SHA1s of basis elements (in same order as matrix rows)
    basis_sha1s: Vec<String>,
    
    /// Creation timestamp
    pub created_at: f64,
    
    /// Last update timestamp
    pub updated_at: f64,
}

impl RingState {
    /// Create a new ring state
    pub fn new(ring_id: String, p_bits: u8) -> Self {
        let num_registers = 1usize << p_bits;
        let num_bits = num_registers * BITS_PER_REG;
        
        Self {
            ring_id,
            p_bits,
            num_registers,
            num_bits,
            matrix: BitMatrix::new(num_bits),
            basis_sha1s: Vec::new(),
            created_at: 0.0,
            updated_at: 0.0,
        }
    }

    /// Get current rank (number of independent bases)
    pub fn rank(&self) -> usize {
        self.matrix.rank()
    }

    /// Get list of basis SHA1s
    pub fn basis(&self) -> &[String] {
        &self.basis_sha1s
    }

    /// Decompose an HLLSet (given as dense registers) into XOR of bases.
    /// 
    /// # Arguments
    /// * `registers` - Dense u32 registers from HLLSet
    /// * `sha1` - SHA1 of the HLLSet
    /// 
    /// # Returns
    /// DecomposeResult with expression and whether it's a new base
    pub fn decompose(&mut self, registers: &[u32], sha1: String) -> DecomposeResult {
        let rank_before = self.rank();
        
        // Convert registers to bitvector
        let bitvec = BitVec::from_registers(registers);
        
        // Reduce against current basis
        let (residual, used_indices) = self.matrix.reduce(&bitvec);
        
        if residual.any() {
            // New independent element - add to basis
            self.matrix.add_row(residual);
            self.basis_sha1s.push(sha1.clone());
            
            DecomposeResult {
                sha1,
                is_new_base: true,
                expression: vec![self.basis_sha1s.last().unwrap().clone()],
                rank_before,
                rank_after: self.rank(),
            }
        } else {
            // Dependent - express as XOR of existing bases
            let expression: Vec<String> = used_indices.iter()
                .map(|&i| self.basis_sha1s[i].clone())
                .collect();
            
            DecomposeResult {
                sha1,
                is_new_base: false,
                expression,
                rank_before,
                rank_after: self.rank(), // Unchanged
            }
        }
    }

    /// Express an HLLSet (by registers) as XOR of basis elements.
    /// Unlike decompose(), this doesn't modify the ring.
    pub fn express(&self, registers: &[u32]) -> Vec<String> {
        let bitvec = BitVec::from_registers(registers);
        let (_residual, used_indices) = self.matrix.reduce(&bitvec);
        
        used_indices.iter()
            .map(|&i| self.basis_sha1s[i].clone())
            .collect()
    }

    /// Serialize ring state to HashMap for Redis storage
    pub fn to_redis_hash(&self) -> HashMap<String, String> {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        
        let mut map = HashMap::new();
        map.insert("ring_id".into(), self.ring_id.clone());
        map.insert("p_bits".into(), self.p_bits.to_string());
        map.insert("rank".into(), self.rank().to_string());
        map.insert("basis_sha1s".into(), serde_json::to_string(&self.basis_sha1s).unwrap_or_default());
        map.insert("created_at".into(), self.created_at.to_string());
        map.insert("updated_at".into(), self.updated_at.to_string());
        
        // Serialize matrix (compact representation)
        let matrix_data = self.serialize_matrix();
        map.insert("matrix_data".into(), STANDARD.encode(&matrix_data));
        map.insert("pivots".into(), serde_json::to_string(&self.matrix.pivots).unwrap_or_default());
        
        map
    }

    /// Deserialize ring state from Redis hash
    pub fn from_redis_hash(map: &HashMap<String, String>) -> Option<Self> {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        
        let ring_id = map.get("ring_id")?.clone();
        let p_bits: u8 = map.get("p_bits")?.parse().ok()?;
        let num_registers = 1usize << p_bits;
        let num_bits = num_registers * BITS_PER_REG;
        
        let basis_sha1s: Vec<String> = serde_json::from_str(map.get("basis_sha1s")?).ok()?;
        let created_at: f64 = map.get("created_at")?.parse().ok()?;
        let updated_at: f64 = map.get("updated_at")?.parse().ok()?;
        
        // Deserialize matrix
        let matrix_data = STANDARD.decode(map.get("matrix_data")?).ok()?;
        let pivots: Vec<usize> = serde_json::from_str(map.get("pivots")?).ok()?;
        
        let matrix = Self::deserialize_matrix(&matrix_data, &pivots, num_bits);
        
        Some(Self {
            ring_id,
            p_bits,
            num_registers,
            num_bits,
            matrix,
            basis_sha1s,
            created_at,
            updated_at,
        })
    }

    /// Serialize matrix to bytes (compact format)
    fn serialize_matrix(&self) -> Vec<u8> {
        let mut data = Vec::new();
        
        // Number of rows
        let num_rows = self.matrix.rows.len() as u32;
        data.extend_from_slice(&num_rows.to_le_bytes());
        
        // Each row: just store the u64 words
        for row in &self.matrix.rows {
            for &word in &row.words {
                data.extend_from_slice(&word.to_le_bytes());
            }
        }
        
        data
    }

    /// Deserialize matrix from bytes
    fn deserialize_matrix(data: &[u8], pivots: &[usize], num_bits: usize) -> BitMatrix {
        let num_words = (num_bits + 63) / 64;
        let mut cursor = 0;
        
        // Read number of rows
        let num_rows = u32::from_le_bytes(data[cursor..cursor+4].try_into().unwrap_or([0;4])) as usize;
        cursor += 4;
        
        let mut rows = Vec::with_capacity(num_rows);
        
        for _ in 0..num_rows {
            let mut words = Vec::with_capacity(num_words);
            for _ in 0..num_words {
                let word = u64::from_le_bytes(data[cursor..cursor+8].try_into().unwrap_or([0;8]));
                words.push(word);
                cursor += 8;
            }
            rows.push(BitVec { words, num_bits });
        }
        
        BitMatrix {
            rows,
            pivots: pivots.to_vec(),
            num_cols: num_bits,
        }
    }
}

/// W lattice commit - snapshot of ring state at a time point
#[derive(Clone, Debug)]
pub struct WCommit {
    /// Time index
    pub time_index: u64,
    
    /// Ring ID
    pub ring_id: String,
    
    /// Basis SHA1s at this time
    pub basis_sha1s: Vec<String>,
    
    /// Rank at this time
    pub rank: usize,
    
    /// Timestamp
    pub timestamp: f64,
    
    /// Optional metadata (JSON)
    pub metadata: Option<String>,
}

impl WCommit {
    /// Create from current ring state
    pub fn from_ring(ring: &RingState, time_index: u64, metadata: Option<String>) -> Self {
        Self {
            time_index,
            ring_id: ring.ring_id.clone(),
            basis_sha1s: ring.basis_sha1s.clone(),
            rank: ring.rank(),
            timestamp: 0.0, // Set by caller
            metadata,
        }
    }

    /// Serialize to Redis hash
    pub fn to_redis_hash(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("time_index".into(), self.time_index.to_string());
        map.insert("ring_id".into(), self.ring_id.clone());
        map.insert("basis_sha1s".into(), serde_json::to_string(&self.basis_sha1s).unwrap_or_default());
        map.insert("rank".into(), self.rank.to_string());
        map.insert("timestamp".into(), self.timestamp.to_string());
        if let Some(ref meta) = self.metadata {
            map.insert("metadata".into(), meta.clone());
        }
        map
    }

    /// Deserialize from Redis hash
    pub fn from_redis_hash(map: &HashMap<String, String>) -> Option<Self> {
        Some(Self {
            time_index: map.get("time_index")?.parse().ok()?,
            ring_id: map.get("ring_id")?.clone(),
            basis_sha1s: serde_json::from_str(map.get("basis_sha1s")?).ok()?,
            rank: map.get("rank")?.parse().ok()?,
            timestamp: map.get("timestamp")?.parse().ok()?,
            metadata: map.get("metadata").cloned(),
        })
    }
}

/// Diff between two W commits
#[derive(Debug)]
pub struct WDiff {
    pub added_bases: Vec<String>,
    pub removed_bases: Vec<String>,
    pub shared_bases: Vec<String>,
    pub delta_rank: i64,
}

impl WDiff {
    /// Compute diff between two commits
    pub fn compute(w1: &WCommit, w2: &WCommit) -> Self {
        use std::collections::HashSet;
        
        let set1: HashSet<_> = w1.basis_sha1s.iter().cloned().collect();
        let set2: HashSet<_> = w2.basis_sha1s.iter().cloned().collect();
        
        let added: Vec<_> = set2.difference(&set1).cloned().collect();
        let removed: Vec<_> = set1.difference(&set2).cloned().collect();
        let shared: Vec<_> = set1.intersection(&set2).cloned().collect();
        
        Self {
            added_bases: added,
            removed_bases: removed,
            shared_bases: shared,
            delta_rank: w2.rank as i64 - w1.rank as i64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitvec_basic() {
        let mut bv = BitVec::new(100);
        assert!(bv.is_zero());
        
        bv.set(0);
        bv.set(63);
        bv.set(64);
        bv.set(99);
        
        assert!(bv.get(0));
        assert!(bv.get(63));
        assert!(bv.get(64));
        assert!(bv.get(99));
        assert!(!bv.get(50));
        assert!(!bv.is_zero());
        assert_eq!(bv.count_ones(), 4);
        assert_eq!(bv.first_one(), Some(0));
    }

    #[test]
    fn test_bitvec_xor() {
        let mut bv1 = BitVec::new(128);
        let mut bv2 = BitVec::new(128);
        
        bv1.set(0);
        bv1.set(10);
        bv1.set(64);
        
        bv2.set(10);
        bv2.set(64);
        bv2.set(100);
        
        bv1.xor_assign(&bv2);
        
        // After XOR: 0, 100 should be set (10, 64 cancel)
        assert!(bv1.get(0));
        assert!(!bv1.get(10));
        assert!(!bv1.get(64));
        assert!(bv1.get(100));
    }

    #[test]
    fn test_ring_decompose_independent() {
        let mut ring = RingState::new("test".into(), 4); // 16 registers
        
        // First HLLSet - should become a base
        let mut regs1 = vec![0u32; 16];
        regs1[0] = 0b0001; // Bit 0 set in register 0
        
        let result = ring.decompose(&regs1, "sha1_1".into());
        assert!(result.is_new_base);
        assert_eq!(result.expression.len(), 1);
        assert_eq!(ring.rank(), 1);
        
        // Second HLLSet (different) - should also become a base
        let mut regs2 = vec![0u32; 16];
        regs2[1] = 0b0010; // Bit 1 set in register 1
        
        let result = ring.decompose(&regs2, "sha1_2".into());
        assert!(result.is_new_base);
        assert_eq!(ring.rank(), 2);
    }

    #[test]
    fn test_ring_decompose_dependent() {
        let mut ring = RingState::new("test".into(), 4);
        
        // HLLSet A
        let mut regs_a = vec![0u32; 16];
        regs_a[0] = 0b0001;
        ring.decompose(&regs_a, "sha1_a".into());
        
        // HLLSet B
        let mut regs_b = vec![0u32; 16];
        regs_b[1] = 0b0010;
        ring.decompose(&regs_b, "sha1_b".into());
        
        // HLLSet C = A XOR B (dependent)
        let mut regs_c = vec![0u32; 16];
        regs_c[0] = 0b0001;
        regs_c[1] = 0b0010;
        
        let result = ring.decompose(&regs_c, "sha1_c".into());
        assert!(!result.is_new_base);
        assert_eq!(result.expression.len(), 2);
        assert!(result.expression.contains(&"sha1_a".to_string()));
        assert!(result.expression.contains(&"sha1_b".to_string()));
        assert_eq!(ring.rank(), 2); // Rank unchanged
    }
}
