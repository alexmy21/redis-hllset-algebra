# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
BitVector Core — Fast Cython implementation of ring algebra on bit vectors.

This module provides the performance-critical foundation for the HLLSet
ring algebra layer. It implements:

1. BitVectorCore: Fixed-size bit vector with O(1) ring operations
2. GaussianEliminator: Basis compression over F_2
3. Bulk operations: Vectorized XOR, AND, OR for batches

All operations are designed to work with or without the GIL for
maximum performance in multi-threaded scenarios.

Matches the Python API in bitvector_ring.py but with Cython speedups.

Author: HLLSet Algebra Project
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint32_t, uint64_t, int64_t
from libc.string cimport memset, memcpy

cnp.import_array()


# =========================================================================
# BitVectorCore — Fast fixed-size bit vector
# =========================================================================

cdef class BitVectorCore:
    """
    Fixed-size bit vector with fast ring operations.
    
    Stores bits as array of uint32 words for efficient bulk operations.
    Supports vectors up to 2^20 bits (for HLLSet with P=16).
    
    Ring operations:
        XOR (^) — addition in F_2
        AND (&) — multiplication in F_2
        OR (|)  — lattice join (union)
        NOT (~) — complement
    """
    
    cdef public int N                    # Total number of bits
    cdef public int n_words              # Number of uint32 words
    cdef public cnp.ndarray words        # Storage array
    cdef uint32_t[::1] words_view        # Fast typed memoryview
    
    def __init__(self, int N):
        """
        Create a BitVectorCore with N bits (initialized to zero).
        
        Args:
            N: Number of bits (must be multiple of 32)
        """
        if N <= 0:
            raise ValueError("N must be positive")
        if N % 32 != 0:
            # Round up to multiple of 32
            N = ((N + 31) // 32) * 32
        
        self.N = N
        self.n_words = N // 32
        self.words = np.zeros(self.n_words, dtype=np.uint32)
        self.words_view = self.words
    
    @staticmethod
    def from_numpy(cnp.ndarray[cnp.uint32_t, ndim=1] arr):
        """Create from numpy array of uint32 words."""
        cdef int n_words = len(arr)
        cdef BitVectorCore bv = BitVectorCore(n_words * 32)
        bv.words[:] = arr
        return bv
    
    @staticmethod
    def from_bits(list bit_positions, int N):
        """Create with specific bits set."""
        cdef BitVectorCore bv = BitVectorCore(N)
        cdef int pos, word_idx, bit_idx
        
        for pos in bit_positions:
            if 0 <= pos < N:
                word_idx = pos // 32
                bit_idx = pos % 32
                bv.words_view[word_idx] |= (<uint32_t>1 << bit_idx)
        
        return bv
    
    # === Ring Operations (all return new BitVectorCore) ===
    
    def xor(self, BitVectorCore other):
        """Ring addition: XOR (symmetric difference)."""
        if self.N != other.N:
            raise ValueError("BitVectors must have same size")
        
        cdef BitVectorCore result = BitVectorCore(self.N)
        cdef int i
        
        with nogil:
            for i in range(self.n_words):
                result.words_view[i] = self.words_view[i] ^ other.words_view[i]
        
        return result
    
    def and_(self, BitVectorCore other):
        """Ring multiplication: AND (intersection)."""
        if self.N != other.N:
            raise ValueError("BitVectors must have same size")
        
        cdef BitVectorCore result = BitVectorCore(self.N)
        cdef int i
        
        with nogil:
            for i in range(self.n_words):
                result.words_view[i] = self.words_view[i] & other.words_view[i]
        
        return result
    
    def or_(self, BitVectorCore other):
        """Lattice join: OR (union)."""
        if self.N != other.N:
            raise ValueError("BitVectors must have same size")
        
        cdef BitVectorCore result = BitVectorCore(self.N)
        cdef int i
        
        with nogil:
            for i in range(self.n_words):
                result.words_view[i] = self.words_view[i] | other.words_view[i]
        
        return result
    
    def not_(self):
        """Complement: NOT."""
        cdef BitVectorCore result = BitVectorCore(self.N)
        cdef int i
        
        with nogil:
            for i in range(self.n_words):
                result.words_view[i] = ~self.words_view[i]
        
        return result
    
    def diff(self, BitVectorCore other):
        """Set difference: self AND NOT other."""
        if self.N != other.N:
            raise ValueError("BitVectors must have same size")
        
        cdef BitVectorCore result = BitVectorCore(self.N)
        cdef int i
        
        with nogil:
            for i in range(self.n_words):
                result.words_view[i] = self.words_view[i] & (~other.words_view[i])
        
        return result
    
    # === In-place operations (for performance) ===
    
    cdef void xor_inplace(self, BitVectorCore other) nogil:
        """In-place XOR."""
        cdef int i
        for i in range(self.n_words):
            self.words_view[i] ^= other.words_view[i]
    
    cdef void and_inplace(self, BitVectorCore other) nogil:
        """In-place AND."""
        cdef int i
        for i in range(self.n_words):
            self.words_view[i] &= other.words_view[i]
    
    cdef void or_inplace(self, BitVectorCore other) nogil:
        """In-place OR."""
        cdef int i
        for i in range(self.n_words):
            self.words_view[i] |= other.words_view[i]
    
    cdef void clear(self) nogil:
        """Set all bits to zero."""
        cdef int i
        for i in range(self.n_words):
            self.words_view[i] = 0
    
    # === Metrics ===
    
    def popcount(self):
        """Count number of set bits (Hamming weight)."""
        cdef int count = 0
        cdef int i
        cdef uint32_t val
        
        with nogil:
            for i in range(self.n_words):
                val = self.words_view[i]
                # Brian Kernighan's algorithm
                while val:
                    val &= val - 1
                    count += 1
        
        return count
    
    def is_zero(self):
        """Check if all bits are zero."""
        cdef int i
        
        for i in range(self.n_words):
            if self.words_view[i] != 0:
                return False
        
        return True
    
    def leading_bit(self):
        """
        Find position of highest set bit (0-indexed from LSB).
        Returns -1 if all zeros.
        """
        cdef int i, pos
        cdef uint32_t val
        
        # Scan from highest word down
        for i in range(self.n_words - 1, -1, -1):
            val = self.words_view[i]
            if val != 0:
                # Find highest bit in this word
                pos = 31
                while pos >= 0 and not (val & (<uint32_t>1 << pos)):
                    pos -= 1
                return i * 32 + pos
        
        return -1
    
    def get_bit(self, int pos):
        """Get bit at position."""
        if pos < 0 or pos >= self.N:
            return False
        cdef int word_idx = pos // 32
        cdef int bit_idx = pos % 32
        return bool(self.words_view[word_idx] & (<uint32_t>1 << bit_idx))
    
    def set_bit(self, int pos):
        """Set bit at position."""
        if 0 <= pos < self.N:
            self.words_view[pos // 32] |= (<uint32_t>1 << (pos % 32))
    
    def clear_bit(self, int pos):
        """Clear bit at position."""
        if 0 <= pos < self.N:
            self.words_view[pos // 32] &= ~(<uint32_t>1 << (pos % 32))
    
    def to_bits(self):
        """Return list of positions where bits are set."""
        result = []
        cdef int i, j
        cdef uint32_t val
        
        for i in range(self.n_words):
            val = self.words_view[i]
            if val != 0:
                for j in range(32):
                    if val & (<uint32_t>1 << j):
                        result.append(i * 32 + j)
        
        return result
    
    def to_numpy(self):
        """Return copy of underlying uint32 array."""
        return self.words.copy()
    
    def copy(self):
        """Create a deep copy."""
        cdef BitVectorCore result = BitVectorCore(self.N)
        result.words[:] = self.words
        return result
    
    def __eq__(self, other):
        if not isinstance(other, BitVectorCore):
            return NotImplemented
        cdef BitVectorCore o = <BitVectorCore>other
        if self.N != o.N:
            return False
        cdef int i
        for i in range(self.n_words):
            if self.words_view[i] != o.words_view[i]:
                return False
        return True
    
    def __repr__(self):
        cdef int pc = self.popcount()
        return f"BitVectorCore(N={self.N}, popcount={pc})"


# =========================================================================
# GaussianEliminatorCore — Fast basis compression over F_2
# =========================================================================

cdef class GaussianEliminatorCore:
    """
    Gaussian elimination over F_2 for basis compression.
    
    Maintains a reduced row echelon basis of bit vectors.
    Vectors are stored with descending leading-bit order.
    
    Uses BitVectorCore for underlying storage and operations.
    """
    
    cdef public int N                    # Bits per vector
    cdef public list basis               # List of BitVectorCore
    cdef public list leading_bits        # Leading bit position per basis vector
    cdef public int max_rank             # Maximum allowed rank (optional limit)
    
    def __init__(self, int N, int max_rank=0):
        """
        Create eliminator for N-bit vectors.
        
        Args:
            N: Number of bits per vector
            max_rank: Maximum rank (0 = unlimited)
        """
        self.N = N
        self.basis = []
        self.leading_bits = []
        self.max_rank = max_rank if max_rank > 0 else N
    
    def rank(self):
        """Current rank of basis."""
        return len(self.basis)
    
    def add_vector(self, BitVectorCore vec):
        """
        Add vector to basis using Gaussian elimination.
        
        Returns:
            True if vector increased rank (was independent)
            False if vector was in span of current basis
        """
        if vec.N != self.N:
            raise ValueError(f"Expected {self.N}-bit vector, got {vec.N}")
        
        if len(self.basis) >= self.max_rank:
            return False
        
        # Reduce by current basis
        cdef BitVectorCore reduced = vec.copy()
        cdef int i
        cdef int lead
        cdef BitVectorCore b
        
        for i in range(len(self.basis)):
            lead = self.leading_bits[i]
            if reduced.get_bit(lead):
                b = <BitVectorCore>self.basis[i]
                reduced.xor_inplace(b)
        
        if reduced.is_zero():
            return False
        
        # Find leading bit of reduced vector
        lead = reduced.leading_bit()
        if lead < 0:
            return False
        
        # Insert maintaining descending order
        cdef int insert_pos = 0
        while insert_pos < len(self.leading_bits):
            if self.leading_bits[insert_pos] < lead:
                break
            insert_pos += 1
        
        self.basis.insert(insert_pos, reduced)
        self.leading_bits.insert(insert_pos, lead)
        
        # Canonicalize: eliminate this bit from other basis vectors
        for i in range(len(self.basis)):
            if i != insert_pos:
                b = <BitVectorCore>self.basis[i]
                if b.get_bit(lead):
                    b.xor_inplace(reduced)
                    # Update leading bit if changed
                    if i < insert_pos:
                        self.leading_bits[i] = b.leading_bit()
        
        return True
    
    def compute_coefficients(self, BitVectorCore vec):
        """
        Compute coefficient vector for expressing vec in terms of basis.
        
        Returns:
            Integer where bit i is set if basis[i] is used
        """
        cdef int64_t coeffs = 0
        cdef BitVectorCore remaining = vec.copy()
        cdef int i, lead
        cdef BitVectorCore b
        
        for i in range(len(self.basis)):
            lead = self.leading_bits[i]
            if remaining.get_bit(lead):
                coeffs |= (<int64_t>1 << i)
                b = <BitVectorCore>self.basis[i]
                remaining.xor_inplace(b)
        
        return int(coeffs)
    
    def reconstruct(self, int64_t coeffs):
        """
        Reconstruct vector from coefficient representation.
        
        Args:
            coeffs: Integer coefficient vector
            
        Returns:
            BitVectorCore reconstructed from basis
        """
        cdef BitVectorCore result = BitVectorCore(self.N)
        cdef int i
        cdef BitVectorCore b
        
        for i in range(len(self.basis)):
            if coeffs & (<int64_t>1 << i):
                b = <BitVectorCore>self.basis[i]
                result.xor_inplace(b)
        
        return result
    
    def get_basis_vectors(self):
        """Return list of basis vectors (copies)."""
        return [(<BitVectorCore>b).copy() for b in self.basis]


# =========================================================================
# Bulk operations — Vectorized for batches
# =========================================================================

def bulk_xor(list vectors):
    """
    XOR all vectors together (bulk symmetric difference).
    
    Args:
        vectors: List of BitVectorCore with same N
        
    Returns:
        BitVectorCore with XOR of all inputs
    """
    if not vectors:
        raise ValueError("Empty vector list")
    
    cdef BitVectorCore first = <BitVectorCore>vectors[0]
    cdef BitVectorCore result = first.copy()
    cdef int i
    cdef BitVectorCore v
    
    for i in range(1, len(vectors)):
        v = <BitVectorCore>vectors[i]
        result.xor_inplace(v)
    
    return result


def bulk_or(list vectors):
    """
    OR all vectors together (bulk union).
    
    Args:
        vectors: List of BitVectorCore with same N
        
    Returns:
        BitVectorCore with OR of all inputs
    """
    if not vectors:
        raise ValueError("Empty vector list")
    
    cdef BitVectorCore first = <BitVectorCore>vectors[0]
    cdef BitVectorCore result = first.copy()
    cdef int i
    cdef BitVectorCore v
    
    for i in range(1, len(vectors)):
        v = <BitVectorCore>vectors[i]
        result.or_inplace(v)
    
    return result


def bulk_and(list vectors):
    """
    AND all vectors together (bulk intersection).
    
    Args:
        vectors: List of BitVectorCore with same N
        
    Returns:
        BitVectorCore with AND of all inputs
    """
    if not vectors:
        raise ValueError("Empty vector list")
    
    cdef BitVectorCore first = <BitVectorCore>vectors[0]
    cdef BitVectorCore result = first.copy()
    cdef int i
    cdef BitVectorCore v
    
    for i in range(1, len(vectors)):
        v = <BitVectorCore>vectors[i]
        result.and_inplace(v)
    
    return result


def pairwise_hamming(list vectors):
    """
    Compute pairwise Hamming distances.
    
    Args:
        vectors: List of BitVectorCore with same N
        
    Returns:
        2D numpy array of distances
    """
    cdef int n = len(vectors)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] result = np.zeros((n, n), dtype=np.int32)
    cdef int i, j
    cdef BitVectorCore vi, vj, diff
    
    for i in range(n):
        vi = <BitVectorCore>vectors[i]
        for j in range(i + 1, n):
            vj = <BitVectorCore>vectors[j]
            diff = vi.xor(vj)
            result[i, j] = diff.popcount()
            result[j, i] = result[i, j]
    
    return result
