# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
HLL Core Operations in Cython — Bitmap Register Model

Faithful port of the original HllSets.jl Julia implementation.

Each register is a uint32 bitmap where bit k is set when an element
with k trailing zeros is hashed to that bucket. This preserves the
full set of observations, enabling TRUE bitwise set operations:

  Union:        OR  (A | B)      — elements in either set
  Intersection: AND (A & B)      — elements in both sets
  Difference:   AND-NOT (A & ~B) — elements in A but not B
  XOR:          XOR (A ^ B)      — elements in exactly one set

All operations return new HLLCore instances (immutability).

Cardinality is computed via standard HLL algorithm using highest_set_bit
(equivalent to Julia's maxidx) per register → harmonic mean + corrections.

Serialization: Registers flattened to bit positions → Roaring bitmap.
  bit_position = register_index * 32 + bit_index
  Prefixed with b'HLL2' for format identification.
  Backward-compatible: reads old uint8 format (position*256 + value).
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.math cimport log, pow, sqrt
import hashlib

# Initialize numpy
cnp.import_array()


cdef extern from "Python.h":
    void PyEval_InitThreads()


# Alpha constants for bias correction
cdef double ALPHA_16 = 0.673
cdef double ALPHA_32 = 0.697
cdef double ALPHA_64 = 0.709
cdef double ALPHA_INF = 0.7213 / (1.0 + 1.079 / 65536.0)  # For m >= 128

# Serialization format marker
FORMAT_MARKER = b'HLL2'


# =========================================================================
# Inline C-level helper functions (all nogil, thread-safe)
# =========================================================================

cdef inline uint64_t murmur_hash64(const char* data, int length, uint64_t seed) nogil:
    """
    Fast MurmurHash64A implementation for token hashing.
    Thread-safe (no GIL required).
    """
    cdef uint64_t h = seed ^ (length * 0xc6a4a7935bd1e995ULL)
    cdef const uint64_t* data64 = <const uint64_t*>data
    cdef int nblocks = length // 8
    cdef uint64_t k
    cdef int i

    # Process 8-byte chunks
    for i in range(nblocks):
        k = data64[i]
        k *= 0xc6a4a7935bd1e995ULL
        k ^= k >> 47
        k *= 0xc6a4a7935bd1e995ULL
        h ^= k
        h *= 0xc6a4a7935bd1e995ULL

    # Process remaining bytes
    cdef const uint8_t* tail = <const uint8_t*>(data + nblocks * 8)
    cdef int remaining = length & 7

    if remaining >= 7:
        h ^= <uint64_t>tail[6] << 48
    if remaining >= 6:
        h ^= <uint64_t>tail[5] << 40
    if remaining >= 5:
        h ^= <uint64_t>tail[4] << 32
    if remaining >= 4:
        h ^= <uint64_t>tail[3] << 24
    if remaining >= 3:
        h ^= <uint64_t>tail[2] << 16
    if remaining >= 2:
        h ^= <uint64_t>tail[1] << 8
    if remaining >= 1:
        h ^= <uint64_t>tail[0]
        h *= 0xc6a4a7935bd1e995ULL

    # Finalize
    h ^= h >> 47
    h *= 0xc6a4a7935bd1e995ULL
    h ^= h >> 47

    return h


cdef inline int trailing_zeros_64(uint64_t value) nogil:
    """Count trailing zeros in 64-bit integer. Returns 64 if value == 0."""
    if value == 0:
        return 64
    cdef int count = 0
    while (value & 1) == 0:
        count += 1
        value >>= 1
    return count


cdef inline int highest_set_bit(uint32_t value) nogil:
    """
    Return 1-indexed position of highest set bit in uint32.
    Returns 0 if value == 0.

    Equivalent to Julia's maxidx():
        total_bits = sizeof(x) * 8
        leading_zeros_count = leading_zeros(x)
        return total_bits - leading_zeros_count
    """
    if value == 0:
        return 0
    cdef int pos = 0
    while value:
        value >>= 1
        pos += 1
    return pos


# =========================================================================
# HLLCore — Bitmap register HyperLogLog
# =========================================================================

cdef class HLLCore:
    """
    Core HyperLogLog with bitmap registers (uint32).

    Each register is a 32-bit bitmap storing observed trailing-zero counts.
    Bit k is set if at least one element hashed to this bucket had exactly
    k trailing zeros in its hash suffix.

    This enables true bitwise set algebra (OR, AND, AND-NOT, XOR),
    matching the original HllSets.jl Julia implementation.
    """

    cdef public int p_bits
    cdef public int m              # Number of registers = 2^p_bits
    cdef public cnp.ndarray registers
    cdef uint32_t[::1] registers_view  # Fast typed memoryview

    def __init__(self, int p_bits=12):
        """
        Create HLLCore.

        Args:
            p_bits: Precision (4–16). Default 12 = 4096 registers.
        """
        if p_bits < 4 or p_bits > 16:
            raise ValueError("p_bits must be between 4 and 16")

        self.p_bits = p_bits
        self.m = 1 << p_bits
        self.registers = np.zeros(self.m, dtype=np.uint32)
        self.registers_view = self.registers

    # =====================================================================
    # Token ingestion
    # =====================================================================

    def add_token(self, str token, uint64_t seed=0):
        """Add a single token to the HLLCore."""
        cdef bytes token_bytes = token.encode('utf-8')
        cdef const char* data = token_bytes
        cdef int length = len(token_bytes)
        self._add_token_c(data, length, seed)

    cdef void _add_token_c(self, const char* data, int length, uint64_t seed) nogil:
        """
        Internal: hash token, compute bucket and trailing zeros,
        set corresponding bit in the register bitmap.

        Equivalent to Julia's add!():
            h = u_hash(x; seed=seed)
            bin = getbin(hll, h)
            idx = getzeros(hll, h)
            if idx <= 32
                hll.counts[bin] |= (1 << (idx - 1))
            end

        We use 0-indexed bit positions: bit tz is set for tz trailing zeros.
        """
        cdef uint64_t hash_val = murmur_hash64(data, length, seed)

        # Bottom P bits → bucket index
        cdef uint32_t bucket = hash_val & ((1 << self.p_bits) - 1)

        # Remaining bits → count trailing zeros
        cdef uint64_t remaining = hash_val >> self.p_bits
        cdef int tz = trailing_zeros_64(remaining)

        # Set bit at position tz (cap at 31 for uint32)
        if tz < 32:
            self.registers_view[bucket] = self.registers_view[bucket] | (<uint32_t>1 << tz)

    def add_batch(self, list tokens, uint64_t seed=0):
        """
        Add batch of tokens efficiently.
        Encodes tokens first, then processes with GIL released.
        """
        cdef int i, n = len(tokens)
        cdef bytes token_bytes
        cdef const char* data
        cdef int length

        for i in range(n):
            token_bytes = tokens[i].encode('utf-8')
            data = token_bytes
            length = len(token_bytes)
            with nogil:
                self._add_token_c(data, length, seed)

    def compute_reg_zeros_batch(self, list tokens, uint64_t seed=0):
        """
        Compute (bucket, trailing_zeros) pairs WITHOUT adding to registers.

        Used for adjacency matrix construction — exposes the hash decomposition
        without modifying state.

        Args:
            tokens: List of token strings
            seed: Hash seed (must match seed used for add_batch)

        Returns:
            List of (bucket, trailing_zeros) tuples, one per token.
        """
        cdef int i, n = len(tokens)
        cdef bytes token_bytes
        cdef const char* data
        cdef int length
        cdef uint64_t hash_val
        cdef uint32_t bucket
        cdef uint64_t remaining
        cdef int tz

        result = []
        for i in range(n):
            token_bytes = tokens[i].encode('utf-8')
            data = token_bytes
            length = len(token_bytes)

            hash_val = murmur_hash64(data, length, seed)
            bucket = hash_val & ((1 << self.p_bits) - 1)
            remaining = hash_val >> self.p_bits
            tz = trailing_zeros_64(remaining)

            result.append((int(bucket), int(tz)))

        return result

    def add_from_hashes(self, list hashes, uint32_t p_bits):
        """
        Add batch of pre-computed hashes directly to registers.
        
        Args:
            hashes: List of 64-bit integer hashes
            p_bits: Precision bits (must match instance config)
        """
        cdef int i, n = len(hashes)
        cdef uint64_t hash_val
        cdef uint32_t bucket
        cdef uint64_t remaining
        cdef int tz
        
        # Determine bit mask for bucket index
        cdef uint32_t mask = (1 << p_bits) - 1

        for i in range(n):
            hash_val = <uint64_t>hashes[i]
            
            # Extract bucket index (bottom P bits)
            bucket = hash_val & mask
            
            # Remaining bits -> count trailing zeros
            remaining = hash_val >> p_bits
            tz = trailing_zeros_64(remaining)
            
            # Set bit in register
            if tz < 32:
                self.registers_view[bucket] = self.registers_view[bucket] | (<uint32_t>1 << tz)


    def cardinality(self):
        """
        Estimate cardinality using HLL++ with empirical bias correction.

        Uses highest_set_bit() per register (equivalent to Julia's maxidx),
        then harmonic mean with alpha correction, PLUS Google HLL++ empirical
        bias correction from Heule et al. 2013.

        Matches Julia HllSets.jl count() exactly:
            harmonic_mean = sizeof(x) / sum(1 / 1 << maxidx(i) for i in x.counts)
            biased_estimate = α(x) * sizeof(x) * harmonic_mean
            return max(0, round(biased_estimate - bias(x, biased_estimate) - 1))
        """
        from core.hll_constants import estimate_bias

        cdef double raw_sum = self._compute_raw_estimate()
        cdef int m = self.m
        cdef double alpha, biased_estimate, bias_correction
        cdef int zero_count

        # Empty set fast-path: if all registers are zero, cardinality is 0
        zero_count = self._count_empty_registers()
        if zero_count == m:
            return 0.0

        # Alpha bias correction constant
        if m == 16:
            alpha = ALPHA_16
        elif m == 32:
            alpha = ALPHA_32
        elif m == 64:
            alpha = ALPHA_64
        else:
            alpha = ALPHA_INF

        # Harmonic mean: m / raw_sum, then biased_estimate = alpha * m * harmonic_mean
        # This is equivalent to alpha * m^2 / raw_sum
        biased_estimate = alpha * m * m / raw_sum

        # HLL++ empirical bias correction (Google, Heule et al. 2013)
        # For precisions 4-18, use lookup tables with linear interpolation
        bias_correction = estimate_bias(self.p_bits, biased_estimate)

        # Julia formula: max(0, round(biased_estimate - bias - 1))
        return max(0.0, round(biased_estimate - bias_correction - 1))

    cdef double _compute_raw_estimate(self) nogil:
        """
        Compute raw HLL sum: Σ 2^(-highest_set_bit(register[i]))

        For each register, highest_set_bit gives the 1-indexed position
        of the maximum observed trailing-zero count. This is equivalent
        to the standard HLL register value M[j].
        """
        cdef double sum_val = 0.0
        cdef int i
        cdef int max_bit

        for i in range(self.m):
            max_bit = highest_set_bit(self.registers_view[i])
            sum_val += pow(2.0, -<double>max_bit)

        return sum_val

    cdef int _count_empty_registers(self) nogil:
        """Count registers with no bits set (empty buckets)."""
        cdef int count = 0
        cdef int i

        for i in range(self.m):
            if self.registers_view[i] == 0:
                count += 1

        return count

    # =====================================================================
    # Set operations — ALL bitwise on uint32 register bitmaps
    # =====================================================================

    def union(self, HLLCore other):
        """
        Union: bitwise OR of register bitmaps.

        Julia equivalent:
            z.counts[i] = x.counts[i] | y.counts[i]

        Returns new HLLCore (immutable).
        """
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot union HLLs with different p_bits")

        cdef HLLCore result = HLLCore(self.p_bits)
        cdef int i

        with nogil:
            for i in range(self.m):
                result.registers_view[i] = (
                    self.registers_view[i] | other.registers_view[i]
                )

        return result

    def intersect(self, HLLCore other):
        """
        Intersection: bitwise AND of register bitmaps.

        Julia equivalent:
            z.counts[i] = x.counts[i] & y.counts[i]

        Returns new HLLCore with only the observations present in BOTH sets.
        This is a TRUE intersection, not an approximation.
        """
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot intersect HLLs with different p_bits")

        cdef HLLCore result = HLLCore(self.p_bits)
        cdef int i

        with nogil:
            for i in range(self.m):
                result.registers_view[i] = (
                    self.registers_view[i] & other.registers_view[i]
                )

        return result

    def difference(self, HLLCore other):
        """
        Difference (A - B): bitwise AND-NOT of register bitmaps.

        Julia equivalent (set_comp):
            z.counts[i] = x.counts[i] & ~y.counts[i]

        Returns elements in self but NOT in other.
        """
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot diff HLLs with different p_bits")

        cdef HLLCore result = HLLCore(self.p_bits)
        cdef int i

        with nogil:
            for i in range(self.m):
                result.registers_view[i] = (
                    self.registers_view[i] & (~other.registers_view[i])
                )

        return result

    def symmetric_difference(self, HLLCore other):
        """
        Symmetric difference: bitwise XOR of register bitmaps.

        Julia equivalent (set_xor):
            z.counts[i] = xor(x.counts[i], y.counts[i])

        Returns elements in exactly one of the two sets.
        """
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot XOR HLLs with different p_bits")

        cdef HLLCore result = HLLCore(self.p_bits)
        cdef int i

        with nogil:
            for i in range(self.m):
                result.registers_view[i] = (
                    self.registers_view[i] ^ other.registers_view[i]
                )

        return result

    # =====================================================================
    # Similarity measures
    # =====================================================================

    def intersect_cardinality(self, HLLCore other):
        """
        Intersection cardinality via bitwise AND then HLL cardinality.

        More accurate than inclusion-exclusion for the bitmap model,
        since we have a true intersection sketch.
        """
        return self.intersect(other).cardinality()

    def jaccard_similarity(self, HLLCore other):
        """
        Jaccard similarity: |A ∩ B| / |A ∪ B|

        Uses true bitwise intersection and union.
        """
        cdef double card_union = self.union(other).cardinality()
        if card_union == 0:
            return 0.0

        cdef double card_intersect = self.intersect(other).cardinality()
        return card_intersect / card_union

    def cosine_similarity(self, HLLCore other):
        """
        Cosine similarity: |A ∩ B| / sqrt(|A| * |B|)

        Uses true bitwise intersection.
        """
        cdef double card_a = self.cardinality()
        cdef double card_b = other.cardinality()

        if card_a == 0 or card_b == 0:
            return 0.0

        cdef double card_intersect = self.intersect(other).cardinality()
        return card_intersect / sqrt(card_a * card_b)

    # =====================================================================
    # Register access
    # =====================================================================

    def get_registers(self):
        """Get registers as numpy uint32 array (copy for safety)."""
        return self.registers.copy()

    def set_registers(self, cnp.ndarray new_registers):
        """
        Set registers from numpy array.

        Accepts uint32 (native) or uint8 (legacy conversion).
        Legacy uint8 values are interpreted as max-zeros+1 and
        converted to bitmap with single highest bit set.
        """
        if len(new_registers) != self.m:
            raise ValueError(f"Expected {self.m} registers, got {len(new_registers)}")

        cdef int i
        cdef int val

        if new_registers.dtype == np.uint8:
            # Legacy: uint8 value was max(leading_zeros + 1)
            # Convert to bitmap: set bit at position (value - 1)
            for i in range(self.m):
                val = int(new_registers[i])
                if val > 0 and val <= 32:
                    self.registers_view[i] = <uint32_t>1 << (val - 1)
                else:
                    self.registers_view[i] = 0
        else:
            self.registers[:] = new_registers.astype(np.uint32)

    # =====================================================================
    # Serialization — Roaring bitmap of bit positions
    # =====================================================================

    def get_registers_roaring(self):
        """
        Serialize registers as Roaring bitmap of bit positions.

        Flattens the 2^P × 32 bitmap tensor to 1D bit positions:
            For register i, if bit j is set → position = i * 32 + j

        This matches Julia's to_binary_tensor() → flatten_tensor() approach,
        but compressed with Roaring bitmap.

        Prefixed with b'HLL2' for format identification.
        """
        try:
            from pyroaring import BitMap
        except ImportError:
            raise ImportError(
                "pyroaring is required for Roaring bitmap compression. "
                "Install with: pip install pyroaring"
            )

        rb = BitMap()
        cdef int i, j
        cdef uint32_t val

        for i in range(self.m):
            val = self.registers_view[i]
            if val != 0:
                for j in range(32):
                    if val & (<uint32_t>1 << j):
                        rb.add(i * 32 + j)

        return FORMAT_MARKER + rb.serialize()

    def set_registers_roaring(self, bytes data):
        """
        Deserialize registers from Roaring bitmap.

        Auto-detects format:
          - HLL2 (new): b'HLL2' prefix, bit-position encoding
          - Legacy (old): No prefix, position*256+value encoding

        Legacy data is converted to bitmap model by setting the
        highest bit corresponding to the old register value.
        Cardinality is preserved; intersection of legacy + new is
        NOT meaningful (different hash models).
        """
        try:
            from pyroaring import BitMap
        except ImportError:
            raise ImportError(
                "pyroaring is required for Roaring bitmap compression. "
                "Install with: pip install pyroaring"
            )

        cdef int encoded, reg_idx, bit_idx, old_val

        # Clear all registers
        self.registers[:] = 0

        if data[:4] == FORMAT_MARKER:
            # ---- New bitmap format (HLL2) ----
            rb = BitMap.deserialize(data[4:])
            for encoded in rb:
                reg_idx = encoded // 32
                bit_idx = encoded % 32
                if reg_idx < self.m:
                    self.registers_view[reg_idx] = (
                        self.registers_view[reg_idx] | (<uint32_t>1 << bit_idx)
                    )
        else:
            # ---- Legacy uint8 format ----
            # Old encoding: position * 256 + value
            # value was the uint8 register (max leading_zeros + 1)
            # Convert: set bit (value - 1) in bitmap to preserve cardinality
            import warnings
            warnings.warn(
                "Loading legacy HLL format (uint8 registers). "
                "Cardinality is preserved but set operations between "
                "legacy and new HLLSets are not meaningful. "
                "Re-ingest data for full bitmap accuracy.",
                UserWarning,
                stacklevel=3
            )
            rb = BitMap.deserialize(data)
            for encoded in rb:
                reg_idx = encoded // 256
                old_val = encoded % 256
                if reg_idx < self.m and old_val > 0 and old_val <= 32:
                    self.registers_view[reg_idx] = (
                        self.registers_view[reg_idx] | (<uint32_t>1 << (old_val - 1))
                    )

    def get_compression_stats(self):
        """
        Get compression statistics for current registers.

        Returns dict with sizes, ratios, and bit counts.
        """
        cdef int original_size = self.m * 4  # uint32 = 4 bytes per register
        compressed = self.get_registers_roaring()
        cdef int compressed_size = len(compressed)

        cdef int non_zero = 0
        cdef int total_bits = 0
        cdef int i, j
        cdef uint32_t val

        for i in range(self.m):
            val = self.registers_view[i]
            if val != 0:
                non_zero += 1
                for j in range(32):
                    if val & (<uint32_t>1 << j):
                        total_bits += 1

        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': <double>original_size / <double>compressed_size if compressed_size > 0 else 0.0,
            'non_zero_registers': non_zero,
            'total_bits_set': total_bits,
        }

    # =====================================================================
    # Copy and pickle support
    # =====================================================================

    def copy(self):
        """Create a deep copy."""
        cdef HLLCore result = HLLCore(self.p_bits)
        result.registers[:] = self.registers
        return result

    def __reduce__(self):
        """Support for pickling (needed for multiprocessing)."""
        return (
            HLLCore,
            (self.p_bits,),
            {'registers': self.registers}
        )

    def __setstate__(self, state):
        """
        Restore from pickle.
        Handles both uint32 (new) and uint8 (legacy) register arrays.
        """
        regs = state['registers']
        if regs.dtype == np.uint8:
            # Legacy pickle — convert uint8 to bitmap
            self.set_registers(regs)
        else:
            self.registers[:] = regs
