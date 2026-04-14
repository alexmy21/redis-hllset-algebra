"""
HLLSet - Immutable C/Cython Backend with Batch Processing

Design principles:
- HLLSet instances are fully immutable
- All operations return new instances
- Batch processing is the primary mode for token ingestion
- Multi-batch processing can be parallelized (thread-safe C backend)
- No in-place modifications

Batch Processing Pattern:
    # Single batch
    hll = HLLSet.from_batch(['token1', 'token2', ...])
    
    # Multi-batch with parallel processing
    batches = [batch1, batch2, batch3]
    hll_combined = HLLSet.from_batches(batches, parallel=True)
    
    # Accumulating pattern
    hll1 = HLLSet.from_batch(batch1)
    hll2 = HLLSet.from_batch(batch2)
    hll_combined = hll1.union(hll2)  # Immutable merge

Hash Configuration:
    HLLSet is the SINGLE SOURCE OF TRUTH for hash settings.
    All hash-dependent modules should import from here:
    
        from .hllset import HLLSet, DEFAULT_HASH_CONFIG
        
    Hash config includes:
    - hash_type: 'sha1', 'sha256', 'murmur3', etc.
    - p_bits: HLL precision bits (register count = 2^p_bits)
    - seed: Hash seed for reproducibility
    
    The hash function is available directly from HLLSet:
        h = HLLSet.hash("token")  # Returns 32-bit int
        reg, zeros = HLLSet.hash_to_reg_zeros("token")  # Returns (reg, zeros)
"""

from __future__ import annotations
from typing import Set, Tuple, Union, List, Optional, Iterable, Callable
from dataclasses import dataclass
from enum import Enum
import hashlib
import struct
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os


# =============================================================================
# REGISTER FORMAT SPECIFICATION
# =============================================================================
# Each HLL register is a uint32 BITMAP (not uint8 max value).
# Bit k is set when an element with k trailing zeros was observed.
#
# Operations:
#   - Union:        OR  (registers_a | registers_b)
#   - Intersection: AND (registers_a & registers_b)
#   - Difference:   AND-NOT (registers_a & ~registers_b)
#
# This matches Julia HllSets.jl exactly.
# =============================================================================

REGISTER_DTYPE = np.uint32  # All HLL registers use this dtype


# =============================================================================
# MURMURHASH64A - Pure Python implementation matching HLLCore
# =============================================================================

def murmur_hash64a(data: bytes, seed: int = 0) -> int:
    """
    MurmurHash64A implementation matching HLLCore's Cython version.
    
    This is the exact same algorithm used in hll_core.pyx, ensuring
    that HashConfig.hash() produces identical results to HLLCore.add_batch().
    
    Args:
        data: Bytes to hash
        seed: 64-bit seed value
        
    Returns:
        64-bit unsigned hash value
    """
    M = 0xc6a4a7935bd1e995
    R = 47
    MASK64 = 0xFFFFFFFFFFFFFFFF
    
    length = len(data)
    h = (seed ^ (length * M)) & MASK64
    
    # Process 8-byte chunks
    nblocks = length // 8
    for i in range(nblocks):
        k = struct.unpack_from('<Q', data, i * 8)[0]
        k = (k * M) & MASK64
        k ^= (k >> R)
        k = (k * M) & MASK64
        h ^= k
        h = (h * M) & MASK64
    
    # Process remaining bytes
    tail = data[nblocks * 8:]
    remaining = len(tail)
    
    if remaining >= 7:
        h ^= tail[6] << 48
    if remaining >= 6:
        h ^= tail[5] << 40
    if remaining >= 5:
        h ^= tail[4] << 32
    if remaining >= 4:
        h ^= tail[3] << 24
    if remaining >= 3:
        h ^= tail[2] << 16
    if remaining >= 2:
        h ^= tail[1] << 8
    if remaining >= 1:
        h ^= tail[0]
        h = (h * M) & MASK64
    
    # Finalize
    h ^= (h >> R)
    h = (h * M) & MASK64
    h ^= (h >> R)
    
    return h


# =============================================================================
# HASH CONFIGURATION - SINGLE SOURCE OF TRUTH
# =============================================================================

class HashType(Enum):
    """Supported hash algorithms."""
    MURMUR3 = "murmur3"  # Fast, matches HLLCore's internal MurmurHash64
    SHA1 = "sha1"        # 160-bit, standard for HLL (32-bit prefix used)
    SHA256 = "sha256"    # 256-bit, higher entropy but slower


@dataclass(frozen=True)
class HashConfig:
    """
    Centralized hash configuration for the entire system.
    
    This is immutable and serves as the single source of truth for
    all hash-related parameters used across hllset, mf_algebra, kernel, etc.
    
    IMPORTANT: Default is MURMUR3 to match HLLCore's internal hash.
    This ensures consistency between HashConfig.hash_to_reg_zeros() and
    HLLSet.from_batch() / HLLCore operations.
    
    Attributes:
        hash_type: Algorithm to use ('murmur3', 'sha1', 'sha256')
        p_bits: HLL precision bits (m = 2^p_bits registers)
        seed: Hash seed for reproducibility
        h_bits: Number of bits to extract from hash (default: 32)
    
    Example:
        >>> config = HashConfig(hash_type=HashType.MURMUR3, p_bits=10, seed=42)
        >>> h = config.hash("hello")
        >>> reg, zeros = config.hash_to_reg_zeros("hello")
    """
    hash_type: HashType = HashType.MURMUR3  # Match HLLCore's internal hash
    p_bits: int = 10       # 2^10 = 1024 registers
    seed: int = 42         # Default seed for reproducibility
    h_bits: int = 64       # Bits in hash (64 for MurmurHash64A)
    
    def hash(self, content: str) -> int:
        """
        Compute hash of content as integer.
        
        Returns:
            64-bit unsigned integer hash (for MURMUR3)
            32-bit for SHA variants
        """
        if self.hash_type == HashType.MURMUR3:
            # Use our MurmurHash64A implementation matching HLLCore
            h = murmur_hash64a(content.encode('utf-8'), self.seed)
        elif self.hash_type == HashType.SHA1:
            h = int(hashlib.sha1(content.encode()).hexdigest()[:8], 16)
        elif self.hash_type == HashType.SHA256:
            h = int(hashlib.sha256(content.encode()).hexdigest()[:8], 16)
        else:
            raise ValueError(f"Unsupported hash type: {self.hash_type}")
        return h
    
    def hash_with_seed(self, content: str) -> int:
        """
        Compute seeded hash of content.
        
        For MURMUR3, seed is already applied in hash().
        For SHA variants, combines content with seed string.
        """
        if self.hash_type == HashType.MURMUR3:
            # Seed already applied in murmur_hash64a()
            return self.hash(content)
        else:
            seeded = f"{self.seed}:{content}"
            return self.hash(seeded)
    
    def hash_to_reg_zeros(self, content: str, use_seed: bool = True) -> Tuple[int, int]:
        """
        Compute (register_index, trailing_zeros) from content.
        
        This is the fundamental operation for HLL insertion and
        identifier scheme indexing. Matches HLLCore's algorithm exactly:
        - reg = hash & ((1 << p_bits) - 1)
        - zeros = trailing zeros in (hash >> p_bits)
        
        Args:
            content: String to hash
            use_seed: Whether to apply seed (always True for MURMUR3)
            
        Returns:
            (reg, zeros) tuple where:
            - reg: Register index in [0, 2^p_bits)
            - zeros: Trailing zeros count in remaining bits
        """
        # For MURMUR3, seed is always used (built into hash())
        h = self.hash(content)
        
        # Bottom p_bits → register index (matches HLLCore)
        reg = h & ((1 << self.p_bits) - 1)
        
        # Remaining bits → count trailing zeros (matches HLLCore)
        remaining = h >> self.p_bits
        
        # Count trailing zeros
        if remaining == 0:
            zeros = self.h_bits - self.p_bits  # Max zeros (all bits were zero)
        else:
            # Find position of lowest set bit (trailing zeros count)
            zeros = (remaining & -remaining).bit_length() - 1
        
        return (reg, zeros)
    
    @property
    def num_registers(self) -> int:
        """Number of HLL registers (m = 2^p_bits)."""
        return 1 << self.p_bits
    
    @property  
    def max_zeros(self) -> int:
        """Maximum possible trailing zeros value."""
        return self.h_bits - self.p_bits


# Default configuration - THE SINGLE SOURCE OF TRUTH
# Uses MURMUR3 to match HLLCore's internal MurmurHash64A
DEFAULT_HASH_CONFIG = HashConfig(
    hash_type=HashType.MURMUR3,
    p_bits=10,
    seed=42,
    h_bits=64  # MurmurHash64A produces 64-bit hash
)

# Legacy constants for backward compatibility
# These are now derived from DEFAULT_HASH_CONFIG
P_BITS = DEFAULT_HASH_CONFIG.p_bits
SHARED_SEED = DEFAULT_HASH_CONFIG.seed

# Import C backend
try:
    from .hll_core import HLLCore
    C_BACKEND_AVAILABLE = True
except ImportError:
    C_BACKEND_AVAILABLE = False
    raise ImportError(
        "C backend (hll_core) not available. "
        "Please build the Cython extension with: python setup.py build_ext --inplace"
    )


def compute_sha1(data: Union[str, bytes, np.ndarray]) -> str:
    """Compute SHA1 hash of data."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    elif isinstance(data, np.ndarray):
        data = data.tobytes()
    return hashlib.sha1(data).hexdigest()


class HLLSet:
    """
    HLLSet with C/Cython backend.
    
    Treat instances as immutable. Methods return new instances.
    
    HLLSet is the SINGLE SOURCE OF TRUTH for hash configuration.
    All hash-related operations should use HLLSet's hash methods:
    
        # Class-level (uses default config):
        h = HLLSet.hash("token")
        reg, zeros = HLLSet.hash_to_reg_zeros("token")
        
        # Instance-level (uses instance config):
        hll = HLLSet.from_batch(tokens)
        h = hll.hash_token("token")
        reg, zeros = hll.token_to_reg_zeros("token")
    """
    
    # Class-level default configuration
    _default_config: HashConfig = DEFAULT_HASH_CONFIG
    
    def __init__(self, p_bits: int = P_BITS, seed: int = SHARED_SEED,
                 hash_type: HashType = HashType.SHA1,
                 _core: Optional[HLLCore] = None,
                 _config: Optional[HashConfig] = None):
        """
        Create HLLSet.
        
        Args:
            p_bits: Precision bits
            seed: Hash seed for reproducibility
            hash_type: Hash algorithm to use
            _core: Existing C HLLCore (internal use)
            _config: Explicit config (overrides other params, internal use)
        """
        # Use explicit config if provided, else build from params
        if _config is not None:
            self._config = _config
        else:
            self._config = HashConfig(
                hash_type=hash_type,
                p_bits=p_bits,
                seed=seed
            )
        
        self.p_bits = self._config.p_bits
        self.seed = self._config.seed
        self.hash_type = self._config.hash_type
        
        self._core = _core if _core is not None else HLLCore(self.p_bits)
        self._name: Optional[str] = None
        
        # Compute name from content
        self._compute_name()
    
    @property
    def config(self) -> HashConfig:
        """Get the hash configuration for this HLLSet instance."""
        return self._config
    
    def _compute_name(self):
        """Compute content-addressed name from registers."""
        registers = self.dump_numpy()
        self._name = compute_sha1(registers)
    
    # -------------------------------------------------------------------------
    # Hash Methods - SINGLE SOURCE OF TRUTH
    # -------------------------------------------------------------------------
    
    @classmethod
    def hash(cls, content: str, config: Optional[HashConfig] = None) -> int:
        """
        Compute hash of content using default or provided config.
        
        This is the PRIMARY hash method for the entire system.
        All modules should use HLLSet.hash() instead of direct hashlib calls.
        
        Args:
            content: String to hash
            config: Optional config (uses default if not provided)
            
        Returns:
            64-bit unsigned integer hash (MURMUR3) or 32-bit (SHA variants)
        """
        cfg = config or cls._default_config
        return cfg.hash(content)
    
    @classmethod 
    def hash_to_reg_zeros(cls, content: str, config: Optional[HashConfig] = None) -> Tuple[int, int]:
        """
        Compute (register_index, trailing_zeros) from content.
        
        This is the PRIMARY reg/zeros computation for the entire system.
        Matches HLLCore's internal algorithm exactly.
        
        Args:
            content: String to hash
            config: Optional config (uses default if not provided)
            
        Returns:
            (reg, zeros) tuple
        """
        cfg = config or cls._default_config
        return cfg.hash_to_reg_zeros(content)
    
    def hash_token(self, content: str) -> int:
        """
        Compute hash using this instance's configuration.
        
        Use when you need instance-specific hash settings.
        """
        return self._config.hash_with_seed(content)
    
    def token_to_reg_zeros(self, content: str) -> Tuple[int, int]:
        """
        Compute (reg, zeros) using this instance's configuration.
        
        Use when you need instance-specific hash settings.
        """
        return self._config.hash_to_reg_zeros(content)
    
    @classmethod
    def get_default_config(cls) -> HashConfig:
        """Get the default hash configuration."""
        return cls._default_config
    
    @classmethod
    def set_default_config(cls, config: HashConfig):
        """
        Set the default hash configuration.
        
        WARNING: This affects all new HLLSet instances and hash operations.
        Use sparingly, typically at application startup.
        """
        cls._default_config = config
        # Update legacy constants
        global P_BITS, SHARED_SEED
        P_BITS = config.p_bits
        SHARED_SEED = config.seed

    # -------------------------------------------------------------------------
    # Class Methods - Primary API (Immutable Batch Processing)
    # -------------------------------------------------------------------------
    
    @classmethod
    def from_batch(cls, tokens: Union[List[str], Set[str], Iterable[str]], 
                   p_bits: int = P_BITS, seed: int = SHARED_SEED,
                   config: Optional[HashConfig] = None) -> HLLSet:
        """
        Create HLLSet from a batch of tokens (PRIMARY FACTORY METHOD).
        
        This is the recommended way to create HLLSets. All tokens in the batch
        are processed together into a new immutable HLLSet instance.
        
        Args:
            tokens: Batch of tokens (list, set, or iterable)
            p_bits: Precision bits for HLL (used if config not provided)
            seed: Hash seed for consistency (used if config not provided)
            config: Optional HashConfig (overrides p_bits/seed if provided)
            
        Returns:
            New immutable HLLSet containing all tokens
            
        Example:
            >>> hll = HLLSet.from_batch(['token1', 'token2', 'token3'])
            >>> print(hll.cardinality())
            
            # With explicit config:
            >>> config = HashConfig(p_bits=12, seed=123)
            >>> hll = HLLSet.from_batch(tokens, config=config)
        """
        if not isinstance(tokens, list):
            tokens = list(tokens)
        
        # Use config if provided, else build from params
        if config is not None:
            hll = cls(_config=config)
            actual_seed = config.seed
        else:
            hll = cls(p_bits=p_bits, seed=seed)
            actual_seed = seed
        
        if tokens:
            hll._core.add_batch(tokens, actual_seed)
            hll._compute_name()
        
        return hll
    
    def absorb_and_track(self, tokens: Union[List[str], Set[str], Iterable[str]],
                         seed: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Absorb tokens into this HLLSet and return their (reg, zeros) identifiers.
        
        This allows building a LUT or Adjacency Matrix synchronously with
        HLLSet creation, guaranteeing that the hashes used for storage
        match the hashes used for indexing.
        
        Args:
            tokens: Batch of tokens
            seed: Hash seed (uses instance seed if not provided)
            
        Returns:
            List of (reg, zeros) tuples corresponding to input tokens
        """
        if not isinstance(tokens, list):
            tokens = list(tokens)
        
        actual_seed = seed if seed is not None else self.seed
            
        # 1. Compute identifiers (single source of truth)
        pairs = self.compute_reg_zeros_batch(tokens, self.p_bits, actual_seed)
        
        # 2. Update internal state using these identifiers
        # We use a specialized internal method to add precomputed pairs
        # to avoid re-hashing in the C backend.
        if hasattr(self._core, 'add_precomputed_batch'):
            self._core.add_precomputed_batch(pairs)
        else:
            # Fallback if C extension hasn't been updated yet:
            # We must use normal add_batch which re-hashes.
            # This is safe but less efficient.
            self._core.add_batch(tokens, actual_seed)
            
        # 3. Update name
        self._compute_name()
        
        return pairs

    def absorb_hashes(self, hashes: List[int]):
        """
        Absorb pre-computed 64-bit integer hashes.
        
        This allows external drivers (ManifoldOS) to manage hashing logic
        while HLLSet manages storage logic.
        
        Args:
            hashes: List of 64-bit integer hashes
        """
        if hasattr(self._core, 'add_from_hashes'):
            self._core.add_from_hashes(hashes, self.p_bits)
            self._compute_name()
        else:
            raise NotImplementedError("C backend 'add_from_hashes' not available. Check build.")

    @staticmethod
    def compute_reg_zeros_batch(tokens: Union[List[str], Set[str], Iterable[str]],
                                p_bits: int = P_BITS, seed: int = SHARED_SEED,
                                config: Optional[HashConfig] = None) -> List[Tuple[int, int]]:
        """
        Compute (reg, zeros) pairs for tokens WITHOUT creating HLLSet.
        
        This is a utility method for adjacency matrix construction to avoid
        duplicate hash calculations. When building AM, we need both:
        1. HLLSet for cardinality estimation (set operations)
        2. (reg, zeros) pairs for compact identifiers in AM
        
        Instead of:
            hll = HLLSet.from_batch(tokens)  # Calculates hashes
            pairs = [compute_reg_zeros(t) for t in tokens]  # RECALCULATES hashes!
        
        Use:
            pairs = HLLSet.compute_reg_zeros_batch(tokens)  # Calculate once
            hll = HLLSet.from_batch(tokens)  # Reuse cached calculation
        
        Args:
            tokens: Batch of tokens (list, set, or iterable)
            p_bits: Precision bits (used if config not provided)
            seed: Hash seed (used if config not provided)
            config: Optional HashConfig (overrides p_bits/seed)
        
        Returns:
            List of (reg, zeros) tuples, one per token
        
        Example:
            >>> tokens = ['hello', 'world']
            >>> pairs = HLLSet.compute_reg_zeros_batch(tokens)
            >>> print(pairs)  # [(512, 3), (789, 1)]
        """
        if not isinstance(tokens, list):
            tokens = list(tokens)
        
        if not tokens:
            return []
        
        # Use config values if provided
        if config is not None:
            p_bits = config.p_bits
            seed = config.seed
        
        # Use C backend to compute efficiently
        core = HLLCore(p_bits)
        return core.compute_reg_zeros_batch(tokens, seed)
    
    @classmethod
    def from_batches(cls, batches: List[Union[List[str], Set[str]]], 
                     p_bits: int = P_BITS, seed: int = SHARED_SEED,
                     parallel: bool = False, max_workers: Optional[int] = None,
                     config: Optional[HashConfig] = None) -> HLLSet:
        """
        Create HLLSet from multiple batches with optional parallel processing.
        
        Each batch is processed independently (can be parallelized), then all
        results are merged via union operation. This is efficient for large
        datasets that can be split into chunks.
        
        The C backend is thread-safe and supports true parallel processing!
        
        Args:
            batches: List of token batches
            p_bits: Precision bits for HLL (used if config not provided)
            seed: Hash seed (used if config not provided)
            parallel: If True, process batches in parallel
            max_workers: Number of parallel workers (None = CPU count)
            config: Optional HashConfig (overrides p_bits/seed)
            
        Returns:
            New immutable HLLSet containing union of all batches
            
        Example:
            >>> batches = [['a', 'b'], ['c', 'd'], ['e', 'f']]
            >>> hll = HLLSet.from_batches(batches, parallel=True)
        """
        if not batches:
            if config is not None:
                return cls(_config=config)
            return cls(p_bits=p_bits, seed=seed)
        
        if parallel:
            # TRUE parallel processing with C backend!
            max_workers = max_workers or os.cpu_count()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                if config is not None:
                    hlls = list(executor.map(
                        lambda b: cls.from_batch(b, config=config),
                        batches
                    ))
                else:
                    hlls = list(executor.map(
                        lambda b: cls.from_batch(b, p_bits=p_bits, seed=seed),
                        batches
                    ))
        else:
            # Sequential processing
            if config is not None:
                hlls = [cls.from_batch(b, config=config) for b in batches]
            else:
                hlls = [cls.from_batch(b, p_bits=p_bits, seed=seed) for b in batches]
        
        # Merge all HLLSets via union
        return cls.merge(hlls)
    
    @classmethod
    def merge(cls, hlls: List[HLLSet]) -> HLLSet:
        """
        Merge multiple HLLSets into one via union operation.
        
        This is the recommended way to combine multiple HLLSets. All input
        HLLSets must have the same p_bits.
        
        Args:
            hlls: List of HLLSet instances to merge
            
        Returns:
            New HLLSet containing union of all inputs
            
        Example:
            >>> hll1 = HLLSet.from_batch(['a', 'b'])
            >>> hll2 = HLLSet.from_batch(['c', 'd'])
            >>> merged = HLLSet.merge([hll1, hll2])
        """
        if not hlls:
            return cls()
        
        if len(hlls) == 1:
            return hlls[0]
        
        # Use bulk_union for efficiency
        return cls.bulk_union(hlls)
    
    @classmethod
    def bulk_union(cls, hlls: List[HLLSet]) -> HLLSet:
        """
        Bulk union using NumPy vectorized operations (SIMD-optimized).
        
        Stacks all register arrays and applies np.bitwise_or.reduce().
        This is MUCH faster than sequential union for combining many HLLSets.
        
        Performance: O(1) in number of HLLSets (vs O(n) for sequential)
        
        Args:
            hlls: List of HLLSet instances to merge
            
        Returns:
            New HLLSet containing union of all inputs
        """
        if not hlls:
            return cls()
        
        if len(hlls) == 1:
            return hlls[0]
        
        # Verify all have same p_bits
        p_bits = hlls[0].p_bits
        if not all(h.p_bits == p_bits for h in hlls):
            raise ValueError("All HLLSets must have the same p_bits")
        
        # Stack all register arrays: shape (n_hlls, m_registers)
        register_stack = np.stack([h._core.get_registers() for h in hlls], axis=0)
        
        # Vectorized bitwise OR across all HLLSets (SIMD-optimized!)
        merged_registers = np.bitwise_or.reduce(register_stack, axis=0)
        
        # Create result HLLSet
        result = cls(p_bits=p_bits)
        result._core.set_registers(merged_registers)
        result._compute_name()
        
        return result
    
    @classmethod
    def absorb(cls, tokens: Set[str], p_bits: int = P_BITS, seed: int = SHARED_SEED) -> HLLSet:
        """
        Create HLLSet from tokens (legacy method, use from_batch instead).
        
        Kept for backward compatibility. Prefer from_batch() for new code.
        """
        return cls.from_batch(tokens, p_bits=p_bits, seed=seed)
    
    @classmethod
    def add(cls, base: HLLSet, tokens: Union[str, List[str]], seed: int = SHARED_SEED) -> HLLSet:
        """
        Add tokens to an HLLSet, return new HLLSet (legacy method).
        
        Note: For better performance with large datasets, use from_batch()
        or from_batches() instead.
        
        Usage:
            h1 = HLLSet.from_batch(['a', 'b'])
            h2 = HLLSet.add(h1, 'c')  # h2 contains a,b,c; h1 unchanged
            h3 = HLLSet.add(h1, ['d', 'e'])  # batch add
        
        For accumulating batches, prefer:
            h1 = HLLSet.from_batch(batch1)
            h2 = HLLSet.from_batch(batch2)
            h_combined = h1.union(h2)
        """
        if isinstance(tokens, str):
            tokens = [tokens]
        
        if not tokens:
            return base  # No change needed
        
        # Create new HLLSet from tokens, then union with base
        tokens_hll = cls.from_batch(tokens, p_bits=base.p_bits, seed=seed)
        return base.union(tokens_hll)
    
    @classmethod
    def append(cls, base: HLLSet, tokens: Union[str, List[str]], seed: int = SHARED_SEED) -> HLLSet:
        """
        Append tokens to an HLLSet (alias for add).
        
        Same as add() - provided for API consistency.
        """
        return cls.add(base, tokens, seed)
    
    # -------------------------------------------------------------------------
    # Instance Methods - Return new instances
    # -------------------------------------------------------------------------
    
    def union(self, other: HLLSet) -> HLLSet:
        """Union with another HLLSet (returns new instance)."""
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot union HLLs with different p_bits")
        
        result_core = self._core.union(other._core)
        return HLLSet(p_bits=self.p_bits, _core=result_core)
    
    def intersect(self, other: HLLSet) -> HLLSet:
        """
        Intersection with another HLLSet (returns new composable instance).
        
        Uses bitwise AND on register bitmaps: result[i] = A[i] & B[i].
        Each uint32 register is a bitmap of observed trailing-zero counts;
        AND keeps only observations present in BOTH sets.
        
        This is a TRUE set intersection sketch, fully composable:
          - A ∩ A = A  (idempotent)
          - A ∩ B = B ∩ A  (commutative)
          - (A ∩ B) ∩ C = A ∩ (B ∩ C)  (associative)
          - A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)  (distributive)
        
        Matches Julia HllSets.jl: z.counts[i] = x.counts[i] & y.counts[i]
        """
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot intersect HLLs with different p_bits")
        
        result_core = self._core.intersect(other._core)
        return HLLSet(p_bits=self.p_bits, _core=result_core)
    
    def diff(self, other: HLLSet) -> HLLSet:
        """
        Difference A - B (returns new composable instance).
        
        Uses bitwise AND-NOT on register bitmaps: result[i] = A[i] & ~B[i].
        Keeps only the observations in A that are NOT also in B.
        
        Matches Julia HllSets.jl set_comp:
            z.counts[i] = x.counts[i] & ~y.counts[i]
        
        Fully composable — result is a valid HLLSet.
        """
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot diff HLLs with different p_bits")
        
        result_core = self._core.difference(other._core)
        return HLLSet(p_bits=self.p_bits, _core=result_core)
    
    def symmetric_difference(self, other: HLLSet) -> HLLSet:
        """
        Symmetric difference A △ B (returns new composable instance).
        
        Uses bitwise XOR on register bitmaps: result[i] = A[i] ^ B[i].
        Keeps only observations in exactly one of the two sets.
        
        Matches Julia HllSets.jl set_xor:
            z.counts[i] = xor(x.counts[i], y.counts[i])
        
        Fully composable — result is a valid HLLSet.
        """
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot XOR HLLs with different p_bits")
        
        result_core = self._core.symmetric_difference(other._core)
        return HLLSet(p_bits=self.p_bits, _core=result_core)
    
    def xor(self, other: HLLSet) -> HLLSet:
        """Alias for symmetric_difference()."""
        return self.symmetric_difference(other)
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    
    def cardinality(self) -> float:
        """Estimated cardinality."""
        return self._core.cardinality()
    
    def similarity(self, other: HLLSet) -> float:
        """Compute Jaccard similarity with another HLLSet (0.0 to 1.0)."""
        return self._core.jaccard_similarity(other._core)
    
    def cosine(self, other: HLLSet) -> float:
        """Cosine similarity."""
        return self._core.cosine_similarity(other._core)
    
    def dump_numpy(self) -> np.ndarray:
        """
        Get register vector as numpy array.
        
        Returns:
            np.ndarray of shape (2^p_bits,) with dtype uint32.
            Each register is a 32-bit BITMAP where bit k is set
            when an element with k trailing zeros was observed.
            
            This is NOT the traditional HLL register format (max zeros+1).
            Use bitwise operations: OR for union, AND for intersection.
        """
        return self._core.get_registers()
    
    def dump_roaring(self) -> bytes:
        """
        Get registers as compressed Roaring bitmap.
        
        Returns serialized Roaring bitmap with 10-50x compression ratio
        for typical sparse HLLSets.
        
        Returns:
            bytes: Serialized Roaring bitmap
        """
        return self._core.get_registers_roaring()
    
    @classmethod
    def from_roaring(cls, compressed_data: bytes, p_bits: int = P_BITS) -> HLLSet:
        """
        Create HLLSet from compressed Roaring bitmap.
        
        Args:
            compressed_data: Serialized Roaring bitmap
            p_bits: Precision bits (must match original HLLSet)
            
        Returns:
            New HLLSet instance
        """
        hll = cls(p_bits=p_bits)
        hll._core.set_registers_roaring(compressed_data)
        hll._compute_name()
        return hll
    
    def get_compression_stats(self) -> dict:
        """
        Get compression statistics for this HLLSet.
        
        Returns:
            dict with keys:
                - original_size: Uncompressed size in bytes
                - compressed_size: Roaring bitmap size in bytes
                - compression_ratio: original / compressed
                - non_zero_registers: Count of non-zero registers
        """
        return self._core.get_compression_stats()
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        """Content-addressed name."""
        return self._name if self._name is not None else ""
    
    @property
    def short_name(self) -> str:
        """Short name for display."""
        return self._name[:8] if self._name else "unknown"
    
    @property
    def backend(self) -> str:
        """Return which backend is being used."""
        return "C/Cython"
    
    # -------------------------------------------------------------------------
    # Python Protocols
    # -------------------------------------------------------------------------
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, HLLSet):
            return False
        return self.name == other.name
    
    def __repr__(self) -> str:
        return f"HLLSet({self.short_name}..., |A|≈{self.cardinality():.1f}, backend={self.backend})"


# Export
__all__ = [
    'HLLSet', 
    'HashConfig', 
    'HashType',
    'DEFAULT_HASH_CONFIG',
    'REGISTER_DTYPE',
    'compute_sha1', 
    'C_BACKEND_AVAILABLE',
    'murmur_hash64a',
]
