"""
HLLSetRedis — Redis-backed HLLSet using native hllset module.

This module provides a Redis-native implementation of HLLSet that:
1. Uses the native hllset Rust module for all operations
2. Stores data using internal Roaring Bitmaps
3. Maintains API compatibility with core.hllset.HLLSet
4. Enables distributed/persistent HLLSet algebra

Usage:
    import redis
    from core.hllset_redis import HLLSetRedis, RedisClientManager
    
    r = redis.Redis()
    RedisClientManager.set_default(r)
    
    # Create from tokens
    hll = HLLSetRedis.from_batch(['hello', 'world'])
    
    # Operations
    hll2 = HLLSetRedis.from_batch(['world', 'test'])
    hll_union = hll.union(hll2)
    
    # Queries
    print(hll_union.cardinality())
    print(hll.similarity(hll2))
"""

from __future__ import annotations
from typing import Set, Tuple, Union, List, Optional, Iterable
from dataclasses import dataclass
import hashlib
import struct
import redis as redis_lib

# Import from parent package hllset for hash config compatibility
from ..hllset import (
    HashConfig,
    HashType,
    DEFAULT_HASH_CONFIG,
    murmur_hash64a,
    P_BITS,
    SHARED_SEED,
)


# =============================================================================
# CONSTANTS
# =============================================================================

PREFIX_HLLSET = "hllset:"
PREFIX_META = "hllset:meta:"


# =============================================================================
# REDIS CLIENT MANAGER
# =============================================================================

class RedisClientManager:
    """
    Manages Redis connection for HLLSetRedis.
    
    Can be configured globally or per-instance.
    """
    
    _default_client: Optional[redis_lib.Redis] = None
    
    @classmethod
    def set_default(cls, client: redis_lib.Redis):
        """Set the default Redis client for all HLLSetRedis instances."""
        cls._default_client = client
    
    @classmethod
    def get_default(cls) -> redis_lib.Redis:
        """Get the default Redis client, creating one if needed."""
        if cls._default_client is None:
            cls._default_client = redis_lib.Redis(
                host='127.0.0.1',  # Use IP, not 'localhost' (IPv6 issues)
                port=6379,
                decode_responses=False
            )
        return cls._default_client
    
    @classmethod
    def ensure_module_loaded(cls, client: redis_lib.Redis) -> bool:
        """
        Check if hllset module is loaded in Redis.
        
        Returns True if module is available.
        """
        try:
            modules = client.execute_command('MODULE', 'LIST')
            for mod in modules:
                name = mod[1].decode('utf-8') if isinstance(mod[1], bytes) else mod[1]
                if name.lower() == 'hllset':
                    return True
            return False
        except redis_lib.ResponseError:
            return False


# =============================================================================
# HLLSET REDIS IMPLEMENTATION
# =============================================================================

class HLLSetRedis:
    """
    Redis-backed HLLSet using native hllset module.
    
    Data is stored in Redis using the hllset native type. All operations
    are performed server-side via module commands.
    
    Attributes:
        key: Redis key for this HLLSet (hllset:{sha1})
        redis: Redis client instance
        p_bits: Precision bits (default 10)
        seed: Hash seed (default 42)
    """
    
    # Class-level default configuration
    _default_config: HashConfig = DEFAULT_HASH_CONFIG
    
    def __init__(
        self,
        key: Optional[str] = None,
        redis_client: Optional[redis_lib.Redis] = None,
        p_bits: int = P_BITS,
        seed: int = SHARED_SEED,
        _config: Optional[HashConfig] = None
    ):
        """
        Create HLLSetRedis.
        
        Args:
            key: Redis key (if None, represents empty set)
            redis_client: Redis client (uses default if None)
            p_bits: Precision bits
            seed: Hash seed
            _config: Explicit hash config
        """
        self.redis = redis_client or RedisClientManager.get_default()
        self.key = key
        
        if _config is not None:
            self._config = _config
        else:
            self._config = HashConfig(
                hash_type=HashType.MURMUR3,
                p_bits=p_bits,
                seed=seed
            )
        
        self.p_bits = self._config.p_bits
        self.seed = self._config.seed
    
    @property
    def config(self) -> HashConfig:
        """Get the hash configuration."""
        return self._config
    
    @property
    def name(self) -> str:
        """Content-addressed name (SHA1)."""
        if self.key is None:
            return ""
        # Extract SHA1 from key (hllset:{sha1})
        if self.key.startswith(PREFIX_HLLSET):
            return self.key[len(PREFIX_HLLSET):]
        return self.key
    
    @property
    def short_name(self) -> str:
        """Short name for display."""
        name = self.name
        return name[:8] if name else "empty"
    
    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------
    
    @classmethod
    def from_batch(
        cls,
        tokens: Union[List[str], Set[str], Iterable[str]],
        redis_client: Optional[redis_lib.Redis] = None,
        p_bits: int = P_BITS,
        seed: int = SHARED_SEED,
        config: Optional[HashConfig] = None
    ) -> HLLSetRedis:
        """
        Create HLLSetRedis from a batch of tokens.
        
        This is the primary factory method. Tokens are hashed and
        stored in Redis via the native hllset module.
        
        Args:
            tokens: Batch of tokens (list, set, or iterable)
            redis_client: Redis client (uses default if None)
            p_bits: Precision bits
            seed: Hash seed
            config: Optional HashConfig
            
        Returns:
            New HLLSetRedis instance
        """
        if not isinstance(tokens, list):
            tokens = list(tokens)
        
        r = redis_client or RedisClientManager.get_default()
        
        # Use hash config
        actual_seed = config.seed if config else seed
        actual_p_bits = config.p_bits if config else p_bits
        
        if not tokens:
            # Empty set - return with None key
            return cls(key=None, redis_client=r, p_bits=actual_p_bits, seed=actual_seed)
        
        # Use HLLSET.CREATE with tokens - module will hash them
        key = r.execute_command('HLLSET.CREATE', *tokens)
        if isinstance(key, bytes):
            key = key.decode('utf-8')
        
        return cls(key=key, redis_client=r, p_bits=actual_p_bits, seed=actual_seed)
    
    @classmethod
    def from_hashes(
        cls,
        hashes: List[int],
        redis_client: Optional[redis_lib.Redis] = None,
        p_bits: int = P_BITS,
        seed: int = SHARED_SEED
    ) -> HLLSetRedis:
        """
        Create HLLSetRedis from pre-computed 64-bit hashes.
        
        Args:
            hashes: List of 64-bit hash values
            redis_client: Redis client
            p_bits: Precision bits
            seed: Hash seed
            
        Returns:
            New HLLSetRedis instance
        """
        r = redis_client or RedisClientManager.get_default()
        
        if not hashes:
            return cls(key=None, redis_client=r, p_bits=p_bits, seed=seed)
        
        # Use HLLSET.CREATEHASH with hex-encoded hashes
        hex_hashes = [format(h & 0xFFFFFFFFFFFFFFFF, '016x') for h in hashes]
        key = r.execute_command('HLLSET.CREATEHASH', *hex_hashes)
        if isinstance(key, bytes):
            key = key.decode('utf-8')
        
        return cls(key=key, redis_client=r, p_bits=p_bits, seed=seed)
    
    @classmethod
    def from_key(
        cls,
        key: str,
        redis_client: Optional[redis_lib.Redis] = None
    ) -> HLLSetRedis:
        """
        Create HLLSetRedis from existing Redis key.
        
        Args:
            key: Redis key (hllset:{sha1})
            redis_client: Redis client
            
        Returns:
            HLLSetRedis instance pointing to existing data
        """
        r = redis_client or RedisClientManager.get_default()
        
        # Verify key exists using HLLSET.EXISTS
        exists = r.execute_command('HLLSET.EXISTS', key)
        if not exists:
            raise KeyError(f"HLLSet key not found: {key}")
        
        return cls(key=key, redis_client=r)
    
    @classmethod
    def merge(cls, hlls: List[HLLSetRedis]) -> HLLSetRedis:
        """
        Merge multiple HLLSetRedis via union.
        
        Args:
            hlls: List of HLLSetRedis instances
            
        Returns:
            New HLLSetRedis containing union of all inputs
        """
        if not hlls:
            return cls()
        
        if len(hlls) == 1:
            return hlls[0]
        
        return cls.bulk_union(hlls)
    
    @classmethod
    def bulk_union(cls, hlls: List[HLLSetRedis]) -> HLLSetRedis:
        """
        Bulk union of multiple HLLSetRedis.
        
        Uses HLLSET.MERGE for efficiency.
        """
        if not hlls:
            return cls()
        
        if len(hlls) == 1:
            return hlls[0]
        
        r = hlls[0].redis
        keys = [h.key for h in hlls if h.key]
        
        if len(keys) < 2:
            return hlls[0] if keys else cls(redis_client=r)
        
        # Use HLLSET.MERGE for bulk union
        result_key = r.execute_command('HLLSET.MERGE', *keys)
        if isinstance(result_key, bytes):
            result_key = result_key.decode('utf-8')
        
        return cls(key=result_key, redis_client=r)
    
    # -------------------------------------------------------------------------
    # Set Operations (return new instances)
    # -------------------------------------------------------------------------
    
    def union(self, other: HLLSetRedis) -> HLLSetRedis:
        """
        Union with another HLLSetRedis (A | B).
        
        Returns new instance.
        """
        if not self.key or not other.key:
            return self if self.key else other
        
        result_key = self.redis.execute_command('HLLSET.UNION', self.key, other.key)
        if isinstance(result_key, bytes):
            result_key = result_key.decode('utf-8')
        
        return HLLSetRedis(key=result_key, redis_client=self.redis)
    
    def intersect(self, other: HLLSetRedis) -> HLLSetRedis:
        """
        Intersection with another HLLSetRedis (A & B).
        
        Returns new instance.
        """
        if not self.key or not other.key:
            return HLLSetRedis(redis_client=self.redis)
        
        result_key = self.redis.execute_command('HLLSET.INTER', self.key, other.key)
        if isinstance(result_key, bytes):
            result_key = result_key.decode('utf-8')
        
        return HLLSetRedis(key=result_key, redis_client=self.redis)
    
    def diff(self, other: HLLSetRedis) -> HLLSetRedis:
        """
        Difference A - B (A & ~B).
        
        Returns new instance with elements in self but not other.
        """
        if not self.key:
            return HLLSetRedis(redis_client=self.redis)
        
        if not other.key:
            return self
        
        result_key = self.redis.execute_command('HLLSET.DIFF', self.key, other.key)
        if isinstance(result_key, bytes):
            result_key = result_key.decode('utf-8')
        
        return HLLSetRedis(key=result_key, redis_client=self.redis)
    
    def symmetric_difference(self, other: HLLSetRedis) -> HLLSetRedis:
        """
        Symmetric difference (A ^ B).
        
        Returns new instance with elements in exactly one set.
        """
        if not self.key:
            return other
        
        if not other.key:
            return self
        
        result_key = self.redis.execute_command('HLLSET.XOR', self.key, other.key)
        if isinstance(result_key, bytes):
            result_key = result_key.decode('utf-8')
        
        return HLLSetRedis(key=result_key, redis_client=self.redis)
    
    def xor(self, other: HLLSetRedis) -> HLLSetRedis:
        """Alias for symmetric_difference()."""
        return self.symmetric_difference(other)
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    
    def cardinality(self) -> float:
        """Estimated cardinality."""
        if not self.key:
            return 0.0
        
        result = self.redis.execute_command('HLLSET.CARD', self.key)
        if isinstance(result, bytes):
            result = result.decode('utf-8')
        return float(result)
    
    def similarity(self, other: HLLSetRedis) -> float:
        """Jaccard similarity (0.0 to 1.0)."""
        if not self.key or not other.key:
            return 0.0
        
        result = self.redis.execute_command('HLLSET.SIM', self.key, other.key)
        if isinstance(result, bytes):
            result = result.decode('utf-8')
        return float(result)
    
    def cosine(self, other: HLLSetRedis) -> float:
        """Cosine similarity (computed from cardinalities)."""
        if not self.key or not other.key:
            return 0.0
        
        # Compute cosine from cardinalities: |A ∩ B| / sqrt(|A| * |B|)
        card_a = self.cardinality()
        card_b = other.cardinality()
        if card_a == 0 or card_b == 0:
            return 0.0
        
        inter = self.intersect(other)
        card_inter = inter.cardinality()
        
        import math
        return card_inter / math.sqrt(card_a * card_b)
    
    def info(self) -> dict:
        """Get detailed information about this HLLSet."""
        if not self.key:
            return {'key': None, 'cardinality': 0, 'bits_set': 0}
        
        result = self.redis.execute_command('HLLSET.INFO', self.key)
        
        # Convert flat list to dict
        info = {}
        for i in range(0, len(result), 2):
            key = result[i]
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            value = result[i + 1]
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            elif isinstance(value, list):
                # Nested metadata
                nested = {}
                for j in range(0, len(value), 2):
                    nk = value[j]
                    nv = value[j + 1]
                    if isinstance(nk, bytes):
                        nk = nk.decode('utf-8')
                    if isinstance(nv, bytes):
                        nv = nv.decode('utf-8')
                    nested[nk] = nv
                value = nested
            info[key] = value
        
        return info
    
    # -------------------------------------------------------------------------
    # Data Export/Import
    # -------------------------------------------------------------------------
    
    def dump_positions(self) -> List[Tuple[int, int]]:
        """
        Get all register positions and their values.
        
        Returns list of (register_index, zero_count) tuples.
        Useful for debugging and export.
        """
        if not self.key:
            return []
        
        result = self.redis.execute_command('HLLSET.DUMP', self.key)
        # Result is array of [bucket, value] pairs
        return [(int(pair[0]), int(pair[1])) for pair in result]
    
    @classmethod
    def from_positions(
        cls,
        positions: List[int],
        redis_client: Optional[redis_lib.Redis] = None
    ) -> HLLSetRedis:
        """
        Create HLLSetRedis from bit positions.
        
        Useful for import from other sources.
        
        Note: This creates a new HLLSet but positions are not directly
        loadable via current module API. Use from_batch or from_hashes instead.
        """
        r = redis_client or RedisClientManager.get_default()
        
        if not positions:
            return cls(key=None, redis_client=r)
        
        # TODO: Implement HLLSET.LOAD in Rust module for direct position loading
        # For now, this is a placeholder
        return cls(key=None, redis_client=r)
    
    # -------------------------------------------------------------------------
    # Management
    # -------------------------------------------------------------------------
    
    def delete(self) -> bool:
        """Delete this HLLSet from Redis."""
        if not self.key:
            return False
        
        result = self.redis.execute_command('HLLSET.DEL', self.key)
        if result:
            self.key = None
        return bool(result)
    
    def exists(self) -> bool:
        """Check if this HLLSet exists in Redis."""
        if not self.key:
            return False
        return bool(self.redis.execute_command('HLLSET.EXISTS', self.key))
    
    # -------------------------------------------------------------------------
    # Python Protocols
    # -------------------------------------------------------------------------
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, HLLSetRedis):
            return False
        return self.name == other.name
    
    def __repr__(self) -> str:
        card = self.cardinality() if self.key else 0
        return f"HLLSetRedis({self.short_name}..., |A|≈{card:.1f}, backend=Redis)"
    
    # -------------------------------------------------------------------------
    # Operator Overloads
    # -------------------------------------------------------------------------
    
    def __or__(self, other: HLLSetRedis) -> HLLSetRedis:
        """Union operator (|)."""
        return self.union(other)
    
    def __and__(self, other: HLLSetRedis) -> HLLSetRedis:
        """Intersection operator (&)."""
        return self.intersect(other)
    
    def __sub__(self, other: HLLSetRedis) -> HLLSetRedis:
        """Difference operator (-)."""
        return self.diff(other)
    
    def __xor__(self, other: HLLSetRedis) -> HLLSetRedis:
        """Symmetric difference operator (^)."""
        return self.symmetric_difference(other)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_redis_modules(redis_client: Optional[redis_lib.Redis] = None) -> dict:
    """
    Check which Redis modules are loaded.
    
    Returns dict with module names and versions.
    """
    r = redis_client or RedisClientManager.get_default()
    
    result = {}
    try:
        modules = r.execute_command('MODULE', 'LIST')
        for mod in modules:
            name = mod[1].decode('utf-8') if isinstance(mod[1], bytes) else mod[1]
            version = mod[3]
            result[name] = version
    except:
        pass
    
    return result


def load_functions(redis_client: Optional[redis_lib.Redis] = None, lua_path: Optional[str] = None):
    """
    DEPRECATED: Lua functions no longer needed.
    
    The native hllset module now provides all functionality.
    This function is kept for backwards compatibility but does nothing.
    """
    import warnings
    warnings.warn(
        "load_functions() is deprecated. The native hllset module provides all functionality.",
        DeprecationWarning,
        stacklevel=2
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'HLLSetRedis',
    'RedisClientManager',
    'check_redis_modules',
    'load_functions',  # Deprecated but kept for compatibility
]
