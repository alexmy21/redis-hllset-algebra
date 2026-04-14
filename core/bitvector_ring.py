"""
BitVector Ring Algebra - Pure algebraic operations on fixed-size bit vectors.

This module provides the HLLSet-agnostic ring layer that operates purely on
bit vectors. It implements:

1. Boolean Ring Structure:
   - Addition: XOR (symmetric difference)
   - Multiplication: AND (intersection)
   - Additive identity: 0 (zero vector)
   - Multiplicative identity: 1 (all-ones vector)

2. Lattice Structure:
   - Join: OR (union)
   - Meet: AND (intersection)
   - Partial order: subset relation

3. Compression via Gaussian Elimination over F_2:
   - Basis computation
   - Coefficient vectors
   - Reconstruction

4. Error Detection:
   - X + X = 0 (characteristic 2)
   - X · X = X (idempotence)

Bridge Laws (connecting ring and lattice):
   - A ∪ B = (A △ B) △ (A ∩ B)
   - A ∖ B = A ∧ ¬B
   - ¬A = 1 ⊕ A

References:
- HLLSet_Ring_Lattice.md (theory)
- INTRODUCTION_TO_HLLSETS.md (HLLSet semantics)

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np


class BitVector:
    """
    Fixed-size bit vector with N bits.
    
    Implements the boolean ring (Z/2Z)^N with:
    - XOR as addition
    - AND as multiplication
    - Complement via XOR with all-ones
    
    Attributes:
        N: Number of bits
        value: Integer representation of the bit vector
        mask: Bit mask for N bits
    """
    
    def __init__(self, value: int = 0, N: int = 32):
        """
        Create a BitVector with N bits.
        
        Args:
            value: Integer representation (masked to N bits)
            N: Number of bits (default 32)
        """
        self.N = N
        self.mask = (1 << N) - 1
        self.value = value & self.mask
    
    @classmethod
    def zeros(cls, N: int = 32) -> 'BitVector':
        """Create zero vector (additive identity)."""
        return cls(0, N)
    
    @classmethod
    def ones(cls, N: int = 32) -> 'BitVector':
        """Create all-ones vector (multiplicative identity)."""
        return cls((1 << N) - 1, N)
    
    @classmethod
    def from_bits(cls, bits: List[int], N: int = 32) -> 'BitVector':
        """
        Create from list of bit positions (0-indexed from LSB).
        
        Args:
            bits: List of positions where bits should be 1
            N: Total number of bits
            
        Returns:
            BitVector with specified bits set
        """
        value = 0
        for b in bits:
            if 0 <= b < N:
                value |= (1 << b)
        return cls(value, N)
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'BitVector':
        """
        Create from numpy array of uint32 registers.
        
        This is the primary way to create a BitVector from HLLSet registers.
        
        Args:
            arr: numpy array of uint32 values (HLL registers)
            
        Returns:
            BitVector with total bits = len(arr) * 32
        """
        N = len(arr) * 32
        value = 0
        for i, reg in enumerate(arr):
            value |= (int(reg) << (i * 32))
        return cls(value, N)
    
    def to_numpy(self, dtype=np.uint32) -> np.ndarray:
        """
        Convert to numpy array of uint32 registers.
        
        Returns:
            numpy array with shape (N // 32,) of uint32
        """
        n_regs = (self.N + 31) // 32
        arr = np.zeros(n_regs, dtype=dtype)
        value = self.value
        for i in range(n_regs):
            arr[i] = value & 0xFFFFFFFF
            value >>= 32
        return arr
    
    # === Ring Operations ===
    
    def __xor__(self, other: 'BitVector') -> 'BitVector':
        """Ring addition: symmetric difference (XOR)."""
        assert self.N == other.N, "BitVectors must have same size"
        return BitVector(self.value ^ other.value, self.N)
    
    def __and__(self, other: 'BitVector') -> 'BitVector':
        """Ring multiplication: intersection (AND)."""
        assert self.N == other.N, "BitVectors must have same size"
        return BitVector(self.value & other.value, self.N)
    
    def __or__(self, other: 'BitVector') -> 'BitVector':
        """Lattice join: union (OR)."""
        assert self.N == other.N, "BitVectors must have same size"
        return BitVector(self.value | other.value, self.N)
    
    def __invert__(self) -> 'BitVector':
        """Complement (NOT)."""
        return BitVector(~self.value & self.mask, self.N)
    
    def diff(self, other: 'BitVector') -> 'BitVector':
        """Set difference: self ∖ other = self AND NOT other."""
        return self & ~other
    
    # === Comparison ===
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BitVector):
            return NotImplemented
        return self.value == other.value and self.N == other.N
    
    def __hash__(self) -> int:
        return hash((self.value, self.N))
    
    def is_subset(self, other: 'BitVector') -> bool:
        """Check if self ⊆ other (lattice order)."""
        return (self.value & ~other.value) == 0
    
    def is_zero(self) -> bool:
        """Check if zero vector."""
        return self.value == 0
    
    # === Metrics ===
    
    def popcount(self) -> int:
        """Number of set bits (Hamming weight)."""
        return self.value.bit_count()
    
    def hamming_distance(self, other: 'BitVector') -> int:
        """Hamming distance = |self XOR other|."""
        return (self ^ other).popcount()
    
    # === Serialization ===
    
    def to_bits(self) -> List[int]:
        """Return list of positions where bits are 1."""
        bits = []
        v = self.value
        pos = 0
        while v:
            if v & 1:
                bits.append(pos)
            v >>= 1
            pos += 1
        return bits
    
    def to_binary_str(self, width: Optional[int] = None) -> str:
        """Binary string (MSB first for visualization)."""
        w = width or self.N
        return f"{self.value:0{w}b}"
    
    def __repr__(self) -> str:
        if self.N <= 64:
            return f"BitVector(0x{self.value:0{(self.N + 3) // 4}x}, N={self.N})"
        return f"BitVector(popcount={self.popcount()}, N={self.N})"


@dataclass
class CompressionResult:
    """Result of compressing a BitVector."""
    vector_id: int
    coefficients: int
    original: BitVector
    basis_rank: int


class BitVectorRing:
    """
    Pure algebra on fixed-size bit vectors with Gaussian elimination compression.
    
    This is the core algebraic engine that is HLLSet-agnostic. It works on
    any fixed-size bit vectors and provides:
    
    1. Ring operations (XOR, AND)
    2. Lattice operations (OR, AND, complement)
    3. Basis computation via Gaussian elimination over F_2
    4. Compression to coefficient vectors
    5. Error detection via ring invariants
    
    For HLLSet, use N = 2^P * 32 (e.g., N = 32768 for P=10).
    """
    
    def __init__(self, N: int = 32):
        """
        Create a BitVectorRing for N-bit vectors.
        
        Args:
            N: Number of bits in each vector
        """
        self.N = N
        self.basis: List[BitVector] = []
        self.leading_bits: List[int] = []
        self.vectors: Dict[int, Tuple[BitVector, int]] = {}  # id -> (original, coeffs)
        self.next_id: int = 0
        self.vector_to_id: Dict[int, int] = {}  # value -> id (for deduplication)
    
    # === Basic Operations ===
    
    @staticmethod
    def xor(a: BitVector, b: BitVector) -> BitVector:
        """Ring addition: symmetric difference."""
        return a ^ b
    
    @staticmethod
    def and_(a: BitVector, b: BitVector) -> BitVector:
        """Ring multiplication: intersection."""
        return a & b
    
    @staticmethod
    def or_(a: BitVector, b: BitVector) -> BitVector:
        """Lattice join: union."""
        return a | b
    
    @staticmethod
    def diff(a: BitVector, b: BitVector) -> BitVector:
        """Set difference: A ∖ B."""
        return a.diff(b)
    
    @staticmethod
    def complement(a: BitVector) -> BitVector:
        """Complement."""
        return ~a
    
    # === Basis Management ===
    
    def _reduce_with_basis(self, vec: BitVector) -> BitVector:
        """Reduce vector by current basis using Gaussian elimination."""
        reduced = BitVector(vec.value, self.N)
        for i, b in enumerate(self.basis):
            if reduced.value & (1 << self.leading_bits[i]):
                reduced = reduced ^ b
        return reduced
    
    @staticmethod
    def _find_leading_bit(vec: BitVector) -> int:
        """Find highest set bit position (-1 if zero)."""
        if vec.value == 0:
            return -1
        return vec.value.bit_length() - 1
    
    def add_to_basis(self, vec: BitVector) -> bool:
        """
        Add vector to basis using Gaussian elimination over F_2.
        
        Args:
            vec: BitVector to potentially add to basis
            
        Returns:
            True if vector increased rank (was independent)
        """
        # Reduce by current basis
        reduced = self._reduce_with_basis(vec)
        
        if reduced.is_zero():
            return False
        
        # Find leading bit
        lead = self._find_leading_bit(reduced)
        
        # Insert maintaining descending order of leading bits
        insert_pos = 0
        while insert_pos < len(self.leading_bits) and self.leading_bits[insert_pos] > lead:
            insert_pos += 1
        
        self.basis.insert(insert_pos, reduced)
        self.leading_bits.insert(insert_pos, lead)
        
        # Canonicalize: eliminate this leading bit from other basis vectors
        for i in range(len(self.basis)):
            if i != insert_pos and (self.basis[i].value & (1 << lead)):
                self.basis[i] = self.basis[i] ^ reduced
                if i < insert_pos and self.leading_bits[i] == lead:
                    self.leading_bits[i] = self._find_leading_bit(self.basis[i])
        
        return True
    
    def rank(self) -> int:
        """Current rank of the basis."""
        return len(self.basis)
    
    def get_basis(self) -> List[BitVector]:
        """Return a copy of current basis vectors."""
        return [BitVector(b.value, self.N) for b in self.basis]
    
    # === Compression ===
    
    def _compute_coefficients(self, vec: BitVector) -> int:
        """
        Compute coefficients for a vector against current basis.
        
        Args:
            vec: BitVector to decompose
            
        Returns:
            Integer where bit i is set if basis[i] is used
        """
        coeffs = 0
        remaining = BitVector(vec.value, self.N)
        
        for i, b in enumerate(self.basis):
            lead = self.leading_bits[i]
            if remaining.value & (1 << lead):
                coeffs |= (1 << i)
                remaining = remaining ^ b
        
        # remaining should be zero if vec is in span of basis
        return coeffs
    
    def compress(self, vec: BitVector) -> int:
        """
        Compress a bit vector to coefficients against current basis.
        
        This adds the vector to the basis if it's independent, then
        computes the coefficient representation.
        
        Note: Coefficients are computed lazily and may be recomputed
        when check_consistency is called after more vectors are added.
        
        Args:
            vec: BitVector to compress
            
        Returns:
            Integer ID for retrieving the compressed vector
        """
        # Check for duplicate
        if vec.value in self.vector_to_id:
            return self.vector_to_id[vec.value]
        
        # Ensure vector is in the basis span
        self.add_to_basis(vec)
        
        # Compute coefficients against current basis
        coeffs = self._compute_coefficients(vec)
        
        # Store compressed representation
        vector_id = self.next_id
        self.next_id += 1
        self.vectors[vector_id] = (vec, coeffs)
        self.vector_to_id[vec.value] = vector_id
        
        return vector_id
    
    def finalize(self) -> None:
        """
        Finalize compression by recomputing all coefficients.
        
        Call this after adding all vectors to ensure consistency.
        The basis may change during insertions (via Gaussian elimination),
        so coefficients computed during compress() may become stale.
        """
        for vec_id, (original, _) in self.vectors.items():
            coeffs = self._compute_coefficients(original)
            self.vectors[vec_id] = (original, coeffs)
    
    def decompress(self, vec_id: int) -> BitVector:
        """Retrieve original bit vector from ID."""
        if vec_id not in self.vectors:
            raise ValueError(f"Vector ID {vec_id} not found")
        return BitVector(self.vectors[vec_id][0].value, self.N)
    
    def get_coefficients(self, vec_id: int) -> int:
        """Get coefficient vector for compressed representation."""
        if vec_id not in self.vectors:
            raise ValueError(f"Vector ID {vec_id} not found")
        return self.vectors[vec_id][1]
    
    def reconstruct_from_coeffs(self, coeffs: int) -> BitVector:
        """Reconstruct bit vector from coefficient vector using current basis."""
        result = BitVector.zeros(self.N)
        for i in range(len(self.basis)):
            if coeffs & (1 << i):
                result = result ^ self.basis[i]
        return result
    
    # === Error Detection ===
    
    def check_ring_invariants(self, vec: BitVector) -> Tuple[bool, str]:
        """
        Check that vector satisfies ring invariants.
        
        Returns:
            (passed, message) tuple
        """
        # X + X = 0
        x_plus_x = vec ^ vec
        if not x_plus_x.is_zero():
            return False, "Failed: X + X != 0"
        
        # X · X = X
        x_times_x = vec & vec
        if x_times_x != vec:
            return False, "Failed: X · X != X"
        
        return True, "All invariants passed"
    
    def check_consistency(self, vec_id: int) -> bool:
        """Check that compressed representation reconstructs correctly."""
        if vec_id not in self.vectors:
            return False
        original, coeffs = self.vectors[vec_id]
        reconstructed = self.reconstruct_from_coeffs(coeffs)
        return original == reconstructed
    
    # === Linear Algebra Utilities ===
    
    def span_dimension(self, vector_ids: List[int]) -> int:
        """Compute dimension of span of given vectors."""
        temp_basis = []
        temp_leads = []
        
        for vid in vector_ids:
            vec = self.vectors[vid][0]
            reduced = BitVector(vec.value, self.N)
            for i, b in enumerate(temp_basis):
                if reduced.value & (1 << temp_leads[i]):
                    reduced = reduced ^ b
            if not reduced.is_zero():
                lead = self._find_leading_bit(reduced)
                temp_basis.append(reduced)
                temp_leads.append(lead)
                # Sort by leading bit descending
                order = sorted(range(len(temp_basis)), key=lambda i: -temp_leads[i])
                temp_basis = [temp_basis[i] for i in order]
                temp_leads = [temp_leads[i] for i in order]
        
        return len(temp_basis)
    
    def to_matrix(self, vector_ids: List[int]) -> np.ndarray:
        """Convert vectors to binary matrix (for visualization)."""
        matrix = np.zeros((len(vector_ids), self.N), dtype=np.int8)
        for i, vid in enumerate(vector_ids):
            vec = self.vectors[vid][0]
            bits = vec.to_bits()
            for b in bits:
                matrix[i, b] = 1
        return matrix
    
    # === Statistics ===
    
    def stats(self) -> Dict:
        """Return compression statistics."""
        return {
            'N': self.N,
            'rank': self.rank(),
            'num_vectors': len(self.vectors),
            'compression_ratio': self.N / max(1, self.rank()),
            'theoretical_bits': self.N,
            'coefficient_bits': self.rank(),
        }
    
    # === Visualization ===
    
    def show_basis(self):
        """Display current basis vectors."""
        if not self.basis:
            print("Basis is empty")
            return
        
        print(f"Basis (rank = {len(self.basis)}):")
        for i, b in enumerate(self.basis):
            bits = b.to_bits()
            if len(bits) > 10:
                bits_str = f"[{bits[0]}, {bits[1]}, ..., {bits[-2]}, {bits[-1]}]"
            else:
                bits_str = f"[{', '.join(map(str, bits))}]"
            print(f"  b_{i}: leading={self.leading_bits[i]}, bits={bits_str}")


# Convenience functions for common operations

def verify_bridge_law(a: BitVector, b: BitVector) -> bool:
    """Verify: A ∪ B = (A △ B) △ (A ∩ B)."""
    union = a | b
    sym_diff = a ^ b
    intersect = a & b
    bridge = sym_diff ^ intersect
    return union == bridge


def batch_compress(ring: BitVectorRing, vectors: List[BitVector]) -> List[int]:
    """
    Compress multiple vectors efficiently.
    
    Adds all vectors first, then finalizes to ensure consistent coefficients.
    """
    ids = [ring.compress(v) for v in vectors]
    ring.finalize()  # Recompute coefficients after basis is stable
    return ids


def batch_decompress(ring: BitVectorRing, ids: List[int]) -> List[BitVector]:
    """Decompress multiple vectors."""
    return [ring.decompress(id) for id in ids]
