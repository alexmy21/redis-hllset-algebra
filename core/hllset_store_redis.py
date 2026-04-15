"""
HLLSet Store Redis — Redis-native HLLSet Registry and Derivation Tracking

This module provides Redis-backed storage for HLLSets with:
- RediSearch indexing for metadata queries
- Derivation graph (LUT) stored in Redis
- Support for base and compound HLLSets
- Collision-aware design (same as TokenLUT)

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          HLLSet Store (Redis)                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  hllstore:entry:<sha1>     Hash    Base HLLSet data + metadata          │
    │  hllstore:lut:<sha1>       Hash    Derivation record                    │
    │  hllstore:idx              Index   RediSearch on metadata               │
    └─────────────────────────────────────────────────────────────────────────┘

Schema for hllstore:entry:<sha1>:
    - sha1: TEXT (40-char hex, also the key suffix)
    - source: TEXT (document/session identifier)
    - cardinality: NUMERIC (estimated count)
    - created_at: NUMERIC (Unix timestamp)
    - updated_at: NUMERIC (Unix timestamp)
    - layer: NUMERIC (0=token, 1=sentence, 2=document, etc.)
    - is_base: NUMERIC (1 if base, 0 if compound)
    - tags: TAG (comma-separated labels)
    - metadata: TEXT (JSON for extensibility)

Schema for hllstore:lut:<sha1>:
    - operation: TEXT (base, union, intersect, diff, xor)
    - operands: TEXT (JSON array of operand sha1s)
    - timestamp: NUMERIC
    - metadata: TEXT (JSON)

Usage:
    from core.hllset_store_redis import HLLSetStoreRedis
    
    store = HLLSetStoreRedis(redis_client)
    store.create_index()
    
    # Register base HLLSet
    sha1 = store.register_base(hllset, source="doc1", tags=["corpus", "v1"])
    
    # Query by metadata
    recent = store.query_by_time(since=time.time() - 3600)
    by_source = store.query_by_source("doc1")
    by_tag = store.query_by_tag("corpus")
    
    # Compound operations (derivation tracked)
    sha3 = store.union(sha1, sha2)
    
    # Retrieve (computes on-the-fly if compound)
    hll = store.get(sha3)
    
    # Trace derivation
    deriv = store.get_derivation(sha3)
    bases = store.get_bases(sha3)

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional, Iterator, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import hashlib
import redis
from redis.commands.search.field import TextField, NumericField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from .hllset import HLLSet


# =============================================================================
# Type Definitions
# =============================================================================

HLLSetID = str  # SHA1 hex string (40 chars)


class Operation(Enum):
    """Ring/lattice operations."""
    BASE = "base"           # Not derived; this is a base HLLSet
    UNION = "union"         # Lattice join (∨, OR)
    INTERSECT = "intersect" # Lattice meet (∧, AND)
    DIFF = "diff"           # Set difference (A \ B)
    XOR = "xor"             # Ring addition (symmetric difference)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HLLSetEntry:
    """
    Entry in the HLLSet store (metadata + derivation info).
    
    Attributes:
        sha1: SHA1 hash of the HLLSet (40-char hex)
        source: Source identifier (document, session, etc.)
        cardinality: Estimated cardinality
        created_at: Creation timestamp
        updated_at: Last update timestamp
        layer: Semantic layer (0=token, 1=sentence, 2=document, etc.)
        is_base: True if this is a base HLLSet (not derived)
        tags: List of tags for categorization
        metadata: Additional metadata (JSON-serializable)
    """
    sha1: str
    source: str = ""
    cardinality: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0
    layer: int = 0
    is_base: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dict for Redis storage."""
        return {
            'sha1': self.sha1,
            'source': self.source,
            'cardinality': self.cardinality,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'layer': self.layer,
            'is_base': 1 if self.is_base else 0,
            'tags': ",".join(self.tags),
            'metadata': json.dumps(self.metadata),
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'HLLSetEntry':
        """Create from Redis hash dict."""
        # Handle bytes vs string
        def decode(v):
            return v.decode('utf-8') if isinstance(v, bytes) else v
        
        tags_raw = decode(d.get('tags', ''))
        tags = [t.strip() for t in tags_raw.split(',') if t.strip()]
        
        metadata_raw = decode(d.get('metadata', '{}'))
        metadata = json.loads(metadata_raw) if metadata_raw else {}
        
        is_base_raw = d.get('is_base', 1)
        if isinstance(is_base_raw, bytes):
            is_base_raw = is_base_raw.decode('utf-8')
        is_base = int(is_base_raw) == 1
        
        return cls(
            sha1=decode(d.get('sha1', '')),
            source=decode(d.get('source', '')),
            cardinality=float(decode(d.get('cardinality', 0))),
            created_at=float(decode(d.get('created_at', 0))),
            updated_at=float(decode(d.get('updated_at', 0))),
            layer=int(decode(d.get('layer', 0))),
            is_base=is_base,
            tags=tags,
            metadata=metadata,
        )


@dataclass
class Derivation:
    """
    How a compound HLLSet was derived.
    
    For base HLLSets: operation=BASE, operands=[]
    For compounds: operation specifies how, operands are the input IDs
    """
    operation: Operation
    operands: List[HLLSetID] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_base(self) -> bool:
        return self.operation == Operation.BASE
    
    def to_dict(self) -> Dict:
        """Convert to dict for Redis storage."""
        return {
            'operation': self.operation.value,
            'operands': json.dumps(self.operands),
            'timestamp': self.timestamp,
            'metadata': json.dumps(self.metadata),
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Derivation':
        """Create from Redis hash dict."""
        def decode(v):
            return v.decode('utf-8') if isinstance(v, bytes) else v
        
        operands_raw = decode(d.get('operands', '[]'))
        operands = json.loads(operands_raw) if operands_raw else []
        
        metadata_raw = decode(d.get('metadata', '{}'))
        metadata = json.loads(metadata_raw) if metadata_raw else {}
        
        return cls(
            operation=Operation(decode(d.get('operation', 'base'))),
            operands=operands,
            timestamp=float(decode(d.get('timestamp', 0))),
            metadata=metadata,
        )


# =============================================================================
# Default Index and Prefix
# =============================================================================

DEFAULT_INDEX = "hllstore:idx"
DEFAULT_ENTRY_PREFIX = "hllstore:entry:"
DEFAULT_LUT_PREFIX = "hllstore:lut:"
DEFAULT_DATA_PREFIX = "hllstore:data:"  # Raw HLLSet bytes


# =============================================================================
# HLLSet Store Redis
# =============================================================================

class HLLSetStoreRedis:
    """
    Redis-backed HLLSet store with RediSearch indexing.
    
    Provides:
    - Registration and retrieval of HLLSets
    - Metadata-based queries via RediSearch
    - Derivation tracking for compound HLLSets
    - On-the-fly reconstruction of compound HLLSets
    
    Best Practices (matching TokenLUT patterns):
    - 64-bit SHA1 prefix as key suffix (first 16 hex chars for fast lookup)
    - Full SHA1 stored as field for uniqueness
    - TagField for efficient tag queries
    - Derivation LUT separate from entry metadata
    """
    
    def __init__(
        self,
        client: redis.Redis,
        index_name: str = DEFAULT_INDEX,
        entry_prefix: str = DEFAULT_ENTRY_PREFIX,
        lut_prefix: str = DEFAULT_LUT_PREFIX,
        data_prefix: str = DEFAULT_DATA_PREFIX,
        p_bits: int = 10,
    ):
        """
        Initialize HLLSet store with Redis connection.
        
        Args:
            client: Redis client instance
            index_name: RediSearch index name
            entry_prefix: Key prefix for entry metadata
            lut_prefix: Key prefix for derivation LUT
            data_prefix: Key prefix for raw HLLSet bytes
            p_bits: Precision bits for HLLSets
        """
        self.client = client
        self.index_name = index_name
        self.entry_prefix = entry_prefix
        self.lut_prefix = lut_prefix
        self.data_prefix = data_prefix
        self.p_bits = p_bits
        
        # In-memory cache for frequently accessed HLLSets
        self._hllset_cache: Dict[HLLSetID, HLLSet] = {}
        self._cache_max_size = 100
    
    # =========================================================================
    # Index Management
    # =========================================================================
    
    def create_index(self, drop_existing: bool = False) -> bool:
        """
        Create RediSearch index for HLLSet entries.
        
        Args:
            drop_existing: If True, drop existing index first
            
        Returns:
            True if index was created, False if it already exists
        """
        try:
            if drop_existing:
                try:
                    self.client.ft(self.index_name).dropindex(delete_documents=True)
                except redis.ResponseError:
                    pass
            
            # Define schema
            schema = (
                TextField("sha1"),
                TextField("source"),
                NumericField("cardinality", sortable=True),
                NumericField("created_at", sortable=True),
                NumericField("updated_at", sortable=True),
                NumericField("layer", sortable=True),
                NumericField("is_base", sortable=True),
                TagField("tags", separator=","),
                TextField("metadata"),
            )
            
            definition = IndexDefinition(
                prefix=[self.entry_prefix],
                index_type=IndexType.HASH
            )
            
            self.client.ft(self.index_name).create_index(
                schema,
                definition=definition
            )
            return True
            
        except redis.ResponseError as e:
            if "Index already exists" in str(e):
                return False
            raise
    
    def drop_index(self, delete_documents: bool = True):
        """Drop the RediSearch index."""
        try:
            self.client.ft(self.index_name).dropindex(delete_documents=delete_documents)
        except redis.ResponseError:
            pass
    
    def index_exists(self) -> bool:
        """Check if index exists."""
        try:
            self.client.ft(self.index_name).info()
            return True
        except redis.ResponseError:
            return False
    
    # =========================================================================
    # ID Computation
    # =========================================================================
    
    @staticmethod
    def compute_id(hll: HLLSet) -> HLLSetID:
        """
        Compute SHA1 ID for an HLLSet.
        
        The ID is deterministic: same bit-vector → same ID.
        """
        numpy_bytes = hll.dump_numpy().tobytes()
        return hashlib.sha1(numpy_bytes).hexdigest()
    
    # =========================================================================
    # Base HLLSet Registration
    # =========================================================================
    
    def register_base(
        self,
        hll: HLLSet,
        source: str = "",
        layer: int = 0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> HLLSetID:
        """
        Register a base HLLSet (persisted to storage).
        
        Args:
            hll: The HLLSet to register
            source: Source identifier (e.g., document name)
            layer: Semantic layer (0=token, 1=sentence, etc.)
            tags: Optional tags for categorization
            metadata: Optional additional metadata
        
        Returns:
            The SHA1 ID of the registered HLLSet
        """
        sha1 = self.compute_id(hll)
        now = time.time()
        
        # Check if already registered
        existing = self.get_entry(sha1)
        if existing is not None:
            # Update timestamp
            self._update_entry_timestamp(sha1, now)
            return sha1
        
        # Create entry
        entry = HLLSetEntry(
            sha1=sha1,
            source=source,
            cardinality=hll.cardinality(),
            created_at=now,
            updated_at=now,
            layer=layer,
            is_base=True,
            tags=tags or [],
            metadata=metadata or {},
        )
        
        # Create derivation (BASE)
        derivation = Derivation(
            operation=Operation.BASE,
            operands=[],
            timestamp=now,
            metadata={'source': source},
        )
        
        # Store entry, derivation, and raw data
        pipe = self.client.pipeline()
        
        # Entry metadata
        entry_key = f"{self.entry_prefix}{sha1}"
        pipe.hset(entry_key, mapping=entry.to_dict())
        
        # Derivation LUT
        lut_key = f"{self.lut_prefix}{sha1}"
        pipe.hset(lut_key, mapping=derivation.to_dict())
        
        # Raw HLLSet bytes
        data_key = f"{self.data_prefix}{sha1}"
        pipe.set(data_key, hll.dump_numpy().tobytes())
        
        pipe.execute()
        
        # Cache
        self._cache_hllset(sha1, hll)
        
        return sha1
    
    def _update_entry_timestamp(self, sha1: HLLSetID, timestamp: float):
        """Update the updated_at timestamp for an entry."""
        entry_key = f"{self.entry_prefix}{sha1}"
        self.client.hset(entry_key, 'updated_at', timestamp)
    
    def _cache_hllset(self, sha1: HLLSetID, hll: HLLSet):
        """Add HLLSet to cache with LRU eviction."""
        if len(self._hllset_cache) >= self._cache_max_size:
            # Simple eviction: remove oldest
            oldest = next(iter(self._hllset_cache))
            del self._hllset_cache[oldest]
        self._hllset_cache[sha1] = hll
    
    # =========================================================================
    # Compound HLLSet Operations
    # =========================================================================
    
    def union(self, *sha1s: HLLSetID, metadata: Optional[Dict] = None) -> HLLSetID:
        """
        Compute union of HLLSets and record derivation.
        
        The result is NOT stored as raw bytes (computed on-the-fly).
        Only the derivation is recorded in the LUT.
        """
        return self._compound_operation(Operation.UNION, list(sha1s), metadata)
    
    def intersect(self, *sha1s: HLLSetID, metadata: Optional[Dict] = None) -> HLLSetID:
        """
        Compute intersection of HLLSets and record derivation.
        """
        return self._compound_operation(Operation.INTERSECT, list(sha1s), metadata)
    
    def diff(self, sha1_a: HLLSetID, sha1_b: HLLSetID, metadata: Optional[Dict] = None) -> HLLSetID:
        """
        Compute set difference A \ B and record derivation.
        """
        return self._compound_operation(Operation.DIFF, [sha1_a, sha1_b], metadata)
    
    def xor(self, sha1_a: HLLSetID, sha1_b: HLLSetID, metadata: Optional[Dict] = None) -> HLLSetID:
        """
        Compute symmetric difference (XOR) and record derivation.
        """
        return self._compound_operation(Operation.XOR, [sha1_a, sha1_b], metadata)
    
    def _compound_operation(
        self,
        operation: Operation,
        operands: List[HLLSetID],
        metadata: Optional[Dict],
    ) -> HLLSetID:
        """
        Perform a compound operation and record derivation.
        
        1. Compute the result HLLSet
        2. Compute its SHA1
        3. Record derivation in LUT
        4. Store entry metadata (but NOT raw bytes)
        """
        # Get operand HLLSets
        hlls = [self.get(sha1) for sha1 in operands]
        if any(h is None for h in hlls):
            missing = [sha1 for sha1, h in zip(operands, hlls) if h is None]
            raise ValueError(f"Missing HLLSets: {missing}")
        
        # Compute result
        if operation == Operation.UNION:
            result = hlls[0].copy()
            for h in hlls[1:]:
                result = result | h
        elif operation == Operation.INTERSECT:
            result = hlls[0].copy()
            for h in hlls[1:]:
                result = result & h
        elif operation == Operation.DIFF:
            result = hlls[0] - hlls[1]
        elif operation == Operation.XOR:
            result = hlls[0] ^ hlls[1]
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Compute SHA1 of result
        sha1 = self.compute_id(result)
        now = time.time()
        
        # Check if already exists
        existing = self.get_derivation(sha1)
        if existing is not None:
            return sha1
        
        # Create entry (compound, not base)
        entry = HLLSetEntry(
            sha1=sha1,
            source=f"{operation.value}({','.join(operands[:3])}{'...' if len(operands) > 3 else ''})",
            cardinality=result.cardinality(),
            created_at=now,
            updated_at=now,
            layer=max(self.get_entry(op).layer for op in operands if self.get_entry(op)),
            is_base=False,
            tags=[],
            metadata=metadata or {},
        )
        
        # Create derivation
        derivation = Derivation(
            operation=operation,
            operands=operands,
            timestamp=now,
            metadata=metadata or {},
        )
        
        # Store entry and derivation (but NOT raw bytes)
        pipe = self.client.pipeline()
        
        entry_key = f"{self.entry_prefix}{sha1}"
        pipe.hset(entry_key, mapping=entry.to_dict())
        
        lut_key = f"{self.lut_prefix}{sha1}"
        pipe.hset(lut_key, mapping=derivation.to_dict())
        
        pipe.execute()
        
        # Cache the computed result
        self._cache_hllset(sha1, result)
        
        return sha1
    
    # =========================================================================
    # Retrieval
    # =========================================================================
    
    def get(self, sha1: HLLSetID) -> Optional[HLLSet]:
        """
        Retrieve an HLLSet by SHA1.
        
        For base HLLSets: loads from storage
        For compound HLLSets: reconstructs from derivation
        """
        # Check cache
        if sha1 in self._hllset_cache:
            return self._hllset_cache[sha1]
        
        # Try loading raw bytes (base HLLSet)
        data_key = f"{self.data_prefix}{sha1}"
        data = self.client.get(data_key)
        
        if data:
            hll = HLLSet.from_bytes(data)
            self._cache_hllset(sha1, hll)
            return hll
        
        # Try reconstructing from derivation (compound HLLSet)
        derivation = self.get_derivation(sha1)
        if derivation and not derivation.is_base():
            return self._reconstruct(derivation)
        
        return None
    
    def _reconstruct(self, derivation: Derivation) -> Optional[HLLSet]:
        """Reconstruct HLLSet from derivation."""
        hlls = [self.get(op) for op in derivation.operands]
        if any(h is None for h in hlls):
            return None
        
        if derivation.operation == Operation.UNION:
            result = hlls[0].copy()
            for h in hlls[1:]:
                result = result | h
            return result
        elif derivation.operation == Operation.INTERSECT:
            result = hlls[0].copy()
            for h in hlls[1:]:
                result = result & h
            return result
        elif derivation.operation == Operation.DIFF:
            return hlls[0] - hlls[1]
        elif derivation.operation == Operation.XOR:
            return hlls[0] ^ hlls[1]
        
        return None
    
    def get_entry(self, sha1: HLLSetID) -> Optional[HLLSetEntry]:
        """Get entry metadata by SHA1."""
        entry_key = f"{self.entry_prefix}{sha1}"
        data = self.client.hgetall(entry_key)
        
        if not data:
            return None
        
        return HLLSetEntry.from_dict(data)
    
    def get_derivation(self, sha1: HLLSetID) -> Optional[Derivation]:
        """Get derivation by SHA1."""
        lut_key = f"{self.lut_prefix}{sha1}"
        data = self.client.hgetall(lut_key)
        
        if not data:
            return None
        
        return Derivation.from_dict(data)
    
    def exists(self, sha1: HLLSetID) -> bool:
        """Check if HLLSet exists (either base or compound)."""
        entry_key = f"{self.entry_prefix}{sha1}"
        return self.client.exists(entry_key) > 0
    
    # =========================================================================
    # Derivation Graph Queries
    # =========================================================================
    
    def is_base(self, sha1: HLLSetID) -> bool:
        """Check if SHA1 refers to a base HLLSet."""
        deriv = self.get_derivation(sha1)
        return deriv is not None and deriv.is_base()
    
    def get_bases(self, sha1: HLLSetID) -> Set[HLLSetID]:
        """
        Recursively find all base HLLSets that contribute to this ID.
        
        Walks the derivation graph back to the roots.
        """
        deriv = self.get_derivation(sha1)
        if deriv is None:
            return set()
        
        if deriv.is_base():
            return {sha1}
        
        bases = set()
        for operand_id in deriv.operands:
            bases.update(self.get_bases(operand_id))
        
        return bases
    
    def get_dependents(self, sha1: HLLSetID) -> List[HLLSetID]:
        """
        Find all compound HLLSets that depend on this ID.
        
        This requires scanning all derivations (expensive for large stores).
        Consider maintaining a reverse index for frequent use.
        """
        dependents = []
        
        # Scan all LUT entries
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor, match=f"{self.lut_prefix}*", count=1000)
            
            for key in keys:
                data = self.client.hgetall(key)
                if data:
                    deriv = Derivation.from_dict(data)
                    if sha1 in deriv.operands:
                        # Extract SHA1 from key
                        dep_sha1 = key.decode() if isinstance(key, bytes) else key
                        dep_sha1 = dep_sha1[len(self.lut_prefix):]
                        dependents.append(dep_sha1)
            
            if cursor == 0:
                break
        
        return dependents
    
    def derivation_depth(self, sha1: HLLSetID) -> int:
        """Compute depth of derivation (0 for base, 1+ for compounds)."""
        deriv = self.get_derivation(sha1)
        if deriv is None or deriv.is_base():
            return 0
        
        return 1 + max(
            (self.derivation_depth(op_id) for op_id in deriv.operands),
            default=0
        )
    
    # =========================================================================
    # RediSearch Queries
    # =========================================================================
    
    def query_by_source(self, source: str, limit: int = 100) -> List[HLLSetEntry]:
        """Find HLLSets by source."""
        escaped = source.replace("-", "\\-").replace("@", "\\@")
        query = Query(f"@source:{escaped}").paging(0, limit)
        results = self.client.ft(self.index_name).search(query)
        return [HLLSetEntry.from_dict(doc.__dict__) for doc in results.docs]
    
    def query_by_tag(self, *tags: str, limit: int = 100) -> List[HLLSetEntry]:
        """Find HLLSets by tags (OR query)."""
        if not tags:
            return []
        
        escaped = [t.replace(",", "\\,").replace("-", "\\-") for t in tags]
        tag_query = "|".join(escaped)
        query = Query(f"@tags:{{{tag_query}}}").paging(0, limit)
        results = self.client.ft(self.index_name).search(query)
        return [HLLSetEntry.from_dict(doc.__dict__) for doc in results.docs]
    
    def query_by_time(
        self,
        since: Optional[float] = None,
        until: Optional[float] = None,
        field: str = "created_at",
        limit: int = 100,
    ) -> List[HLLSetEntry]:
        """Find HLLSets by time range."""
        since_str = str(since) if since else "-inf"
        until_str = str(until) if until else "+inf"
        
        query = Query(f"@{field}:[{since_str} {until_str}]")
        query = query.sort_by(field, asc=False).paging(0, limit)
        results = self.client.ft(self.index_name).search(query)
        return [HLLSetEntry.from_dict(doc.__dict__) for doc in results.docs]
    
    def query_by_layer(self, layer: int, limit: int = 100) -> List[HLLSetEntry]:
        """Find HLLSets by semantic layer."""
        query = Query(f"@layer:[{layer} {layer}]").paging(0, limit)
        results = self.client.ft(self.index_name).search(query)
        return [HLLSetEntry.from_dict(doc.__dict__) for doc in results.docs]
    
    def query_base_only(self, limit: int = 100) -> List[HLLSetEntry]:
        """Find only base HLLSets (not compounds)."""
        query = Query("@is_base:[1 1]").paging(0, limit)
        results = self.client.ft(self.index_name).search(query)
        return [HLLSetEntry.from_dict(doc.__dict__) for doc in results.docs]
    
    def query_by_cardinality(
        self,
        min_card: Optional[float] = None,
        max_card: Optional[float] = None,
        limit: int = 100,
    ) -> List[HLLSetEntry]:
        """Find HLLSets by cardinality range."""
        min_str = str(min_card) if min_card is not None else "-inf"
        max_str = str(max_card) if max_card is not None else "+inf"
        
        query = Query(f"@cardinality:[{min_str} {max_str}]")
        query = query.sort_by("cardinality", asc=False).paging(0, limit)
        results = self.client.ft(self.index_name).search(query)
        return [HLLSetEntry.from_dict(doc.__dict__) for doc in results.docs]
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def stats(self) -> Dict:
        """Return store statistics."""
        try:
            info = self.client.ft(self.index_name).info()
            total = int(info.get('num_docs', 0))
        except redis.ResponseError:
            total = 0
        
        # Count bases vs compounds
        base_count = 0
        compound_count = 0
        
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor, match=f"{self.entry_prefix}*", count=1000)
            for key in keys:
                is_base = self.client.hget(key, 'is_base')
                if is_base:
                    if isinstance(is_base, bytes):
                        is_base = is_base.decode()
                    if int(is_base) == 1:
                        base_count += 1
                    else:
                        compound_count += 1
            if cursor == 0:
                break
        
        return {
            'total_entries': total,
            'base_count': base_count,
            'compound_count': compound_count,
            'cache_size': len(self._hllset_cache),
            'cache_max': self._cache_max_size,
        }
    
    # =========================================================================
    # Bulk Operations
    # =========================================================================
    
    def register_batch(
        self,
        hllsets: List[Tuple[HLLSet, str, int, List[str]]],
        pipeline_size: int = 100,
    ) -> List[HLLSetID]:
        """
        Register multiple base HLLSets efficiently.
        
        Args:
            hllsets: List of (hllset, source, layer, tags) tuples
            pipeline_size: Batch size for pipelining
            
        Returns:
            List of SHA1 IDs
        """
        sha1s = []
        now = time.time()
        
        pipe = self.client.pipeline()
        count = 0
        
        for hll, source, layer, tags in hllsets:
            sha1 = self.compute_id(hll)
            sha1s.append(sha1)
            
            entry = HLLSetEntry(
                sha1=sha1,
                source=source,
                cardinality=hll.cardinality(),
                created_at=now,
                updated_at=now,
                layer=layer,
                is_base=True,
                tags=tags,
                metadata={},
            )
            
            derivation = Derivation(
                operation=Operation.BASE,
                operands=[],
                timestamp=now,
                metadata={'source': source},
            )
            
            # Entry
            entry_key = f"{self.entry_prefix}{sha1}"
            pipe.hset(entry_key, mapping=entry.to_dict())
            
            # Derivation
            lut_key = f"{self.lut_prefix}{sha1}"
            pipe.hset(lut_key, mapping=derivation.to_dict())
            
            # Raw data
            data_key = f"{self.data_prefix}{sha1}"
            pipe.set(data_key, hll.dump_numpy().tobytes())
            
            count += 1
            if count % pipeline_size == 0:
                pipe.execute()
                pipe = self.client.pipeline()
        
        if count % pipeline_size != 0:
            pipe.execute()
        
        return sha1s
    
    def all_ids(self, base_only: bool = False) -> Iterator[HLLSetID]:
        """Iterate over all registered HLLSet IDs."""
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor, match=f"{self.entry_prefix}*", count=1000)
            
            for key in keys:
                sha1 = key.decode() if isinstance(key, bytes) else key
                sha1 = sha1[len(self.entry_prefix):]
                
                if base_only:
                    if self.is_base(sha1):
                        yield sha1
                else:
                    yield sha1
            
            if cursor == 0:
                break
