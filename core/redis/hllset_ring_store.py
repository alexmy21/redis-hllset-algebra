"""
HLLSet Ring Store — XOR Ring Algebra with Base-Only Storage

This module implements the algebraic HLLSet store where:
- Only BASE HLLSets are stored (leaves in the derivation DAG)
- All other HLLSets are expressed as XOR of bases (reconstructed on-the-fly)
- Ring decomposition determines the unique basis representation
- Temporal W lattice tracks evolution with base reuse

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      HLLSet Ring Store (Redis)                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Persistent:                                                            │
    │    hllring:base:<sha1>     STRING   Raw bytes (ONLY for bases)          │
    │    hllring:lut:<sha1>      HASH     Derivation: {op, bases:[...]}       │
    │    hllring:meta:<sha1>     HASH     Metadata (source, time, etc)        │
    │    hllring:ring:<ring_id>  HASH     Ring state (basis SHA1s, rank)      │
    │    hllring:refcount:<sha1> STRING   Reference count for eviction        │
    │    hllring:W:<ring_id>:<t> HASH     W commit snapshot at time t         │
    │    hllring:idx             INDEX    RediSearch on metadata              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Ephemeral (session cache):                                             │
    │    LRU cache of reconstructed compounds                                 │
    └─────────────────────────────────────────────────────────────────────────┘

Key Principle:
    Every HLLSet H can be uniquely expressed as XOR of basis elements:
        H = B₁ ⊕ B₂ ⊕ ... ⊕ Bₖ
    
    This expression is stored in the LUT, and H is reconstructed on demand.

Ring Decomposition:
    When ingesting a new HLLSet:
    1. Check if it's linearly dependent on current basis
    2. If independent: add as new base, store raw bytes
    3. If dependent: express as XOR of existing bases, store only LUT entry

Usage:
    from core.hllset_ring_store import HLLSetRingStore
    
    store = HLLSetRingStore(redis_client)
    store.create_index()
    
    # Initialize a ring for decomposition
    ring = store.init_ring("session:ring1")
    
    # Ingest tokens (auto-decompose into bases)
    result = store.ingest(ring, "hello")
    # result.is_new_base, result.sha1, result.expression
    
    # Ingest HLLSet directly
    result = store.decompose(ring, hllset, source="doc1")
    
    # Retrieve (reconstructs from XOR of bases)
    hll = store.get(sha1)
    
    # W lattice commits
    store.commit_W(ring, t=1)
    store.commit_W(ring, t=2)
    diff = store.diff_W(ring, t1=1, t2=2)

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional, Iterator, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import json
import time
import hashlib
import numpy as np
import redis
from redis.commands.search.field import TextField, NumericField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from ..hllset import HLLSet


# =============================================================================
# Type Definitions
# =============================================================================

HLLSetID = str  # SHA1 hex string (40 chars)


class Operation(Enum):
    """Ring operations."""
    BASE = "base"   # Fundamental base HLLSet (stored)
    XOR = "xor"     # XOR of bases (reconstructed)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DecomposeResult:
    """Result of ring decomposition."""
    sha1: HLLSetID                      # SHA1 of the HLLSet
    is_new_base: bool                   # True if this is a new independent base
    expression: List[HLLSetID]          # XOR expression: [base_sha1, ...]
    rank_before: int                    # Ring rank before decomposition
    rank_after: int                     # Ring rank after decomposition
    
    @property
    def is_dependent(self) -> bool:
        """True if HLLSet was expressible in existing bases."""
        return not self.is_new_base


@dataclass
class RingState:
    """
    State of an HLLSet ring (basis for decomposition).
    
    The ring maintains a set of linearly independent HLLSets (bases).
    Any new HLLSet can be expressed as XOR of these bases.
    """
    ring_id: str
    basis_sha1s: List[HLLSetID] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    p_bits: int = 10
    
    @property
    def rank(self) -> int:
        """Number of independent bases."""
        return len(self.basis_sha1s)
    
    def to_dict(self) -> Dict:
        return {
            'ring_id': self.ring_id,
            'basis_sha1s': json.dumps(self.basis_sha1s),
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'p_bits': self.p_bits,
            'rank': self.rank,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'RingState':
        def decode(v):
            return v.decode('utf-8') if isinstance(v, bytes) else v
        
        def get_field(name: str, default=''):
            # Handle both bytes and string keys
            return d.get(name, d.get(name.encode('utf-8'), default))
        
        basis_raw = decode(get_field('basis_sha1s', '[]'))
        basis_sha1s = json.loads(basis_raw) if basis_raw else []
        
        return cls(
            ring_id=decode(get_field('ring_id', '')),
            basis_sha1s=basis_sha1s,
            created_at=float(decode(get_field('created_at', 0))),
            updated_at=float(decode(get_field('updated_at', 0))),
            p_bits=int(decode(get_field('p_bits', 10))),
        )


@dataclass
class WCommit:
    """
    W lattice commit snapshot.
    
    Represents the ring state at a specific time point.
    """
    ring_id: str
    time_index: int
    basis_sha1s: List[HLLSetID]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'ring_id': self.ring_id,
            'time_index': self.time_index,
            'basis_sha1s': json.dumps(self.basis_sha1s),
            'timestamp': self.timestamp,
            'metadata': json.dumps(self.metadata),
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'WCommit':
        def decode(v):
            return v.decode('utf-8') if isinstance(v, bytes) else v
        
        def get_field(name: str, default=''):
            return d.get(name, d.get(name.encode('utf-8'), default))
        
        return cls(
            ring_id=decode(get_field('ring_id', '')),
            time_index=int(decode(get_field('time_index', 0))),
            basis_sha1s=json.loads(decode(get_field('basis_sha1s', '[]'))),
            timestamp=float(decode(get_field('timestamp', 0))),
            metadata=json.loads(decode(get_field('metadata', '{}'))),
        )


@dataclass
class WDiff:
    """Difference between two W commits."""
    t1: int
    t2: int
    added_bases: List[HLLSetID]      # Bases in t2 but not t1
    removed_bases: List[HLLSetID]    # Bases in t1 but not t2
    shared_bases: List[HLLSetID]     # Bases in both
    
    @property
    def delta_rank(self) -> int:
        """Change in rank from t1 to t2."""
        return len(self.added_bases) - len(self.removed_bases)


@dataclass
class HLLSetMeta:
    """Metadata for an HLLSet entry."""
    sha1: HLLSetID
    source: str = ""
    cardinality: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0
    is_base: bool = True
    refcount: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'sha1': self.sha1,
            'source': self.source,
            'cardinality': self.cardinality,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'is_base': 1 if self.is_base else 0,
            'refcount': self.refcount,
            'tags': ",".join(self.tags),
            'metadata': json.dumps(self.metadata),
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'HLLSetMeta':
        def decode(v):
            return v.decode('utf-8') if isinstance(v, bytes) else v
        
        def get_field(name: str, default=''):
            return d.get(name, d.get(name.encode('utf-8'), default))
        
        tags_raw = decode(get_field('tags', ''))
        tags = [t.strip() for t in tags_raw.split(',') if t.strip()]
        
        return cls(
            sha1=decode(get_field('sha1', '')),
            source=decode(get_field('source', '')),
            cardinality=float(decode(get_field('cardinality', 0))),
            created_at=float(decode(get_field('created_at', 0))),
            updated_at=float(decode(get_field('updated_at', 0))),
            is_base=int(decode(get_field('is_base', 1))) == 1,
            refcount=int(decode(get_field('refcount', 0))),
            tags=tags,
            metadata=json.loads(decode(get_field('metadata', '{}'))),
        )


@dataclass 
class Derivation:
    """XOR derivation of an HLLSet."""
    operation: Operation
    bases: List[HLLSetID] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'operation': self.operation.value,
            'bases': json.dumps(self.bases),
            'timestamp': self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Derivation':
        def decode(v):
            return v.decode('utf-8') if isinstance(v, bytes) else v
        
        def get_field(name: str, default=''):
            return d.get(name, d.get(name.encode('utf-8'), default))
        
        return cls(
            operation=Operation(decode(get_field('operation', 'base'))),
            bases=json.loads(decode(get_field('bases', '[]'))),
            timestamp=float(decode(get_field('timestamp', 0))),
        )


# =============================================================================
# LRU Cache for Reconstructed HLLSets
# =============================================================================

class LRUCache:
    """Simple LRU cache for reconstructed HLLSets."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: OrderedDict[HLLSetID, HLLSet] = OrderedDict()
    
    def get(self, sha1: HLLSetID) -> Optional[HLLSet]:
        if sha1 in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(sha1)
            return self._cache[sha1]
        return None
    
    def put(self, sha1: HLLSetID, hll: HLLSet):
        if sha1 in self._cache:
            self._cache.move_to_end(sha1)
        else:
            if len(self._cache) >= self.max_size:
                # Evict oldest
                self._cache.popitem(last=False)
            self._cache[sha1] = hll
    
    def invalidate(self, sha1: HLLSetID):
        self._cache.pop(sha1, None)
    
    def clear(self):
        self._cache.clear()
    
    def __len__(self) -> int:
        return len(self._cache)


# =============================================================================
# Key Prefixes
# =============================================================================

DEFAULT_PREFIX = "hllring:"
KEY_BASE = "base:"      # Raw bytes for base HLLSets
KEY_LUT = "lut:"        # Derivation (XOR expression)
KEY_META = "meta:"      # Metadata
KEY_RING = "ring:"      # Ring state
KEY_REFCOUNT = "ref:"   # Reference counts
KEY_W = "W:"            # W commits
INDEX_NAME = "hllring:idx"


# =============================================================================
# HLLSet Ring Store
# =============================================================================

class HLLSetRingStore:
    """
    XOR Ring-based HLLSet Store.
    
    Key principles:
    1. Only BASE HLLSets are stored as raw bytes
    2. All other HLLSets are XOR expressions of bases
    3. Ring decomposition determines unique representation
    4. Compounds are reconstructed on-the-fly with LRU caching
    
    Usage:
        store = HLLSetRingStore(redis_client)
        store.create_index()
        
        # Initialize ring
        ring = store.init_ring("session1")
        
        # Ingest (auto-decompose)
        result = store.ingest(ring, "hello world")
        
        # Get (reconstructs if compound)
        hll = store.get(result.sha1)
    """
    
    def __init__(
        self,
        client: redis.Redis,
        prefix: str = DEFAULT_PREFIX,
        cache_size: int = 100,
        p_bits: int = 10,
        use_rust: Optional[bool] = None,
    ):
        self.client = client
        self.prefix = prefix
        self.p_bits = p_bits
        self._cache = LRUCache(cache_size)
        
        # In-memory ring matrices for Gaussian elimination
        # (Used when Rust module not available)
        self._ring_matrices: Dict[str, np.ndarray] = {}
        self._ring_pivots: Dict[str, List[int]] = {}
        
        # Determine if Rust ring commands are available
        if use_rust is None:
            self._use_rust = self._detect_rust_ring_commands()
        else:
            self._use_rust = use_rust
    
    def _detect_rust_ring_commands(self) -> bool:
        """Check if Rust RING commands are available in the loaded module."""
        try:
            # Try calling a Rust ring command - it will fail gracefully if not loaded
            result = self.client.execute_command("HLLSET.RING.RANK", "nonexistent")
            # If we get here (even with "not found" response), the command exists
            return True
        except redis.ResponseError as e:
            # Command doesn't exist if we get "unknown command"
            if "unknown command" in str(e).lower():
                return False
            # Other errors (like "ring not found") mean the command exists
            return True
        except Exception:
            return False
    
    @property
    def using_rust(self) -> bool:
        """Return True if using Rust ring commands."""
        return self._use_rust
    
    # =========================================================================
    # Key Helpers
    # =========================================================================
    
    def _key(self, kind: str, suffix: str) -> str:
        return f"{self.prefix}{kind}{suffix}"
    
    # =========================================================================
    # Index Management
    # =========================================================================
    
    def create_index(self, drop_existing: bool = False) -> bool:
        """Create RediSearch index for metadata queries."""
        try:
            if drop_existing:
                try:
                    self.client.ft(INDEX_NAME).dropindex(delete_documents=True)
                except redis.ResponseError:
                    pass
            
            schema = (
                TextField("sha1"),
                TextField("source"),
                NumericField("cardinality", sortable=True),
                NumericField("created_at", sortable=True),
                NumericField("updated_at", sortable=True),
                NumericField("is_base", sortable=True),
                NumericField("refcount", sortable=True),
                TagField("tags", separator=","),
            )
            
            definition = IndexDefinition(
                prefix=[f"{self.prefix}{KEY_META}"],
                index_type=IndexType.HASH
            )
            
            self.client.ft(INDEX_NAME).create_index(schema, definition=definition)
            return True
            
        except redis.ResponseError as e:
            if "Index already exists" in str(e):
                return False
            raise
    
    # =========================================================================
    # SHA1 Computation
    # =========================================================================
    
    @staticmethod
    def compute_sha1(hll: HLLSet) -> HLLSetID:
        """Compute SHA1 of HLLSet (deterministic)."""
        return hashlib.sha1(hll.dump_numpy().tobytes()).hexdigest()
    
    # =========================================================================
    # Ring Management
    # =========================================================================
    
    def init_ring(self, ring_id: str, p_bits: Optional[int] = None) -> RingState:
        """
        Initialize a new ring for decomposition.
        
        Args:
            ring_id: Unique identifier for the ring
            p_bits: Precision bits (default: store's p_bits)
            
        Returns:
            RingState object
        """
        p = p_bits or self.p_bits
        
        if self._use_rust:
            return self._init_ring_rust(ring_id, p)
        else:
            return self._init_ring_python(ring_id, p)
    
    def _init_ring_rust(self, ring_id: str, p_bits: int) -> RingState:
        """Initialize ring using Rust RING.INIT command."""
        try:
            result = self.client.execute_command(
                "HLLSET.RING.INIT", ring_id, "P_BITS", p_bits
            )
            # Result format: ["OK", ring_id, rank, created_at]
            state = RingState(
                ring_id=ring_id,
                basis_sha1s=[],
                created_at=time.time(),
                updated_at=time.time(),
                p_bits=p_bits,
            )
            return state
        except redis.ResponseError as e:
            raise RuntimeError(f"Rust RING.INIT failed: {e}")
    
    def _init_ring_python(self, ring_id: str, p_bits: int) -> RingState:
        """Initialize ring using Python (in-memory matrices)."""
        num_registers = 1 << p_bits
        
        state = RingState(
            ring_id=ring_id,
            basis_sha1s=[],
            created_at=time.time(),
            updated_at=time.time(),
            p_bits=p_bits,
        )
        
        # Initialize in-memory matrix for Gaussian elimination
        # Matrix shape: (max_rank, num_registers * 32) for bit positions
        # We'll grow this dynamically
        self._ring_matrices[ring_id] = np.zeros((0, num_registers * 32), dtype=np.uint8)
        self._ring_pivots[ring_id] = []
        
        # Store in Redis
        key = self._key(KEY_RING, ring_id)
        self.client.hset(key, mapping=state.to_dict())
        
        return state
    
    def get_ring(self, ring_id: str) -> Optional[RingState]:
        """Get ring state by ID."""
        key = self._key(KEY_RING, ring_id)
        data = self.client.hgetall(key)
        if not data:
            return None
        return RingState.from_dict(data)
    
    def _load_ring_matrix(self, ring_id: str) -> bool:
        """Load ring matrix from bases (for reconstruction after restart)."""
        state = self.get_ring(ring_id)
        if not state:
            return False
        
        num_registers = 1 << state.p_bits
        num_bits = num_registers * 32
        
        # Reconstruct matrix from stored bases
        if state.basis_sha1s:
            rows = []
            for sha1 in state.basis_sha1s:
                hll = self._get_base(sha1)
                if hll:
                    row = self._hllset_to_bitvector(hll, num_bits)
                    rows.append(row)
            
            if rows:
                self._ring_matrices[ring_id] = np.array(rows, dtype=np.uint8)
                # Recompute pivots
                self._ring_pivots[ring_id] = self._compute_pivots(
                    self._ring_matrices[ring_id]
                )
        else:
            self._ring_matrices[ring_id] = np.zeros((0, num_bits), dtype=np.uint8)
            self._ring_pivots[ring_id] = []
        
        return True
    
    def _hllset_to_bitvector(self, hll: HLLSet, num_bits: int) -> np.ndarray:
        """Convert HLLSet to flat bit vector for ring operations."""
        # Flatten the (num_registers, 32) tensor to (num_registers * 32,)
        tensor = hll.dump_numpy()
        bitvec = np.zeros(num_bits, dtype=np.uint8)
        
        for reg in range(tensor.shape[0]):
            zeros = tensor[reg]
            if zeros > 0:
                bit_idx = reg * 32 + (zeros - 1)
                if bit_idx < num_bits:
                    bitvec[bit_idx] = 1
        
        return bitvec
    
    def _bitvector_to_positions(self, bitvec: np.ndarray) -> List[Tuple[int, int]]:
        """Convert bit vector back to (reg, zeros) positions."""
        positions = []
        nonzero = np.nonzero(bitvec)[0]
        for bit_idx in nonzero:
            reg = bit_idx // 32
            zeros = (bit_idx % 32) + 1
            positions.append((reg, zeros))
        return positions
    
    def _compute_pivots(self, matrix: np.ndarray) -> List[int]:
        """Compute pivot columns for the matrix."""
        pivots = []
        for row_idx in range(matrix.shape[0]):
            row = matrix[row_idx]
            nonzero = np.nonzero(row)[0]
            if len(nonzero) > 0:
                pivots.append(int(nonzero[0]))
            else:
                pivots.append(-1)
        return pivots
    
    # =========================================================================
    # Ring Decomposition (Gaussian Elimination over GF(2))
    # =========================================================================
    
    def decompose(
        self,
        ring_id: str,
        hll: HLLSet,
        source: str = "",
        tags: Optional[List[str]] = None,
    ) -> DecomposeResult:
        """
        Decompose an HLLSet into XOR of basis elements.
        
        Uses Gaussian elimination over GF(2) to find the unique representation.
        If Rust module is loaded, uses server-side decomposition for efficiency.
        
        Args:
            ring_id: Ring to use for decomposition
            hll: HLLSet to decompose
            source: Source identifier
            tags: Optional tags
            
        Returns:
            DecomposeResult with SHA1, expression, and whether it's a new base
        """
        if self._use_rust:
            return self._decompose_rust(ring_id, hll, source, tags)
        else:
            return self._decompose_python(ring_id, hll, source, tags)
    
    def _decompose_rust(
        self,
        ring_id: str,
        hll: HLLSet,
        source: str = "",
        tags: Optional[List[str]] = None,
    ) -> DecomposeResult:
        """
        Decompose using Rust RING.DECOMPOSE command.
        
        Server-side decomposition eliminates round-trips for Gaussian elimination.
        """
        sha1 = self.compute_sha1(hll)
        now = time.time()
        
        # Check if already decomposed (local check to save round-trip)
        existing = self.get_derivation(sha1)
        if existing:
            state = self.get_ring(ring_id)
            return DecomposeResult(
                sha1=sha1,
                is_new_base=existing.operation == Operation.BASE,
                expression=existing.bases if existing.operation == Operation.XOR else [sha1],
                rank_before=state.rank if state else 0,
                rank_after=state.rank if state else 0,
            )
        
        # First, store the HLLSet in Redis for the Rust command to access
        temp_key = f"{self.prefix}temp:{sha1}"
        self.client.execute_command("HLLSET.CREATE", temp_key, "P_BITS", hll.p_bits)
        
        # Build the HLLSet in Redis from our local data
        # We need to transfer the bitvector data
        # For now, use numpy serialization
        data = hll.dump_numpy().tobytes()
        self.client.set(f"{temp_key}:data", data)
        
        try:
            # Call RING.DECOMPOSE
            result = self.client.execute_command(
                "HLLSET.RING.DECOMPOSE", ring_id, temp_key, "SOURCE", source
            )
            
            # Parse result: [sha1, is_new_base, rank_before, rank_after, expr1, expr2, ...]
            if isinstance(result, (list, tuple)) and len(result) >= 4:
                result_sha1 = result[0].decode() if isinstance(result[0], bytes) else str(result[0])
                is_new_base = bool(result[1])
                rank_before = int(result[2])
                rank_after = int(result[3])
                expression = [
                    r.decode() if isinstance(r, bytes) else str(r)
                    for r in result[4:]
                ]
                
                # Store metadata locally
                meta = HLLSetMeta(
                    sha1=result_sha1,
                    source=source,
                    cardinality=hll.cardinality(),
                    created_at=now,
                    updated_at=now,
                    is_base=is_new_base,
                    refcount=0,
                    tags=tags or [],
                )
                self._store_meta(result_sha1, meta)
                
                # Store derivation
                op = Operation.BASE if is_new_base else Operation.XOR
                deriv = Derivation(operation=op, bases=expression, timestamp=now)
                self._store_derivation(result_sha1, deriv)
                
                return DecomposeResult(
                    sha1=result_sha1,
                    is_new_base=is_new_base,
                    expression=expression,
                    rank_before=rank_before,
                    rank_after=rank_after,
                )
            else:
                raise RuntimeError(f"Unexpected RING.DECOMPOSE result format: {result}")
                
        finally:
            # Cleanup temp key
            self.client.delete(temp_key)
            self.client.delete(f"{temp_key}:data")
    
    def _decompose_python(
        self,
        ring_id: str,
        hll: HLLSet,
        source: str = "",
        tags: Optional[List[str]] = None,
    ) -> DecomposeResult:
        """
        Decompose using Python (in-memory Gaussian elimination).
        
        Original implementation for when Rust module is not available.
        """
        # Ensure ring is loaded
        if ring_id not in self._ring_matrices:
            if not self._load_ring_matrix(ring_id):
                raise ValueError(f"Ring {ring_id} not found")
        
        state = self.get_ring(ring_id)
        sha1 = self.compute_sha1(hll)
        now = time.time()
        
        # Check if already decomposed
        existing = self.get_derivation(sha1)
        if existing:
            return DecomposeResult(
                sha1=sha1,
                is_new_base=existing.operation == Operation.BASE,
                expression=existing.bases if existing.operation == Operation.XOR else [sha1],
                rank_before=state.rank,
                rank_after=state.rank,
            )
        
        # Convert to bit vector
        num_bits = self._ring_matrices[ring_id].shape[1] if self._ring_matrices[ring_id].size > 0 else (1 << state.p_bits) * 32
        bitvec = self._hllset_to_bitvector(hll, num_bits)
        
        rank_before = state.rank
        
        # Gaussian elimination to express in terms of basis
        expression, residual = self._gaussian_reduce(ring_id, bitvec)
        
        if np.any(residual):
            # New independent element - add to basis
            is_new_base = True
            
            # Add to matrix
            if self._ring_matrices[ring_id].size == 0:
                self._ring_matrices[ring_id] = residual.reshape(1, -1)
            else:
                self._ring_matrices[ring_id] = np.vstack([
                    self._ring_matrices[ring_id],
                    residual
                ])
            
            # Update pivots
            nonzero = np.nonzero(residual)[0]
            pivot = int(nonzero[0]) if len(nonzero) > 0 else -1
            self._ring_pivots[ring_id].append(pivot)
            
            # Update state
            state.basis_sha1s.append(sha1)
            state.updated_at = now
            
            # Store base HLLSet bytes
            self._store_base(sha1, hll)
            
            # Store derivation as BASE
            deriv = Derivation(operation=Operation.BASE, bases=[sha1], timestamp=now)
            expression_result = [sha1]
            
        else:
            # Dependent - express as XOR of existing bases
            is_new_base = False
            
            # expression contains indices into basis
            expression_sha1s = [state.basis_sha1s[i] for i in expression]
            
            # Store derivation as XOR
            deriv = Derivation(operation=Operation.XOR, bases=expression_sha1s, timestamp=now)
            expression_result = expression_sha1s
            
            # Increment refcounts for used bases
            self._increment_refcounts(expression_sha1s)
        
        # Store metadata
        meta = HLLSetMeta(
            sha1=sha1,
            source=source,
            cardinality=hll.cardinality(),
            created_at=now,
            updated_at=now,
            is_base=is_new_base,
            refcount=0,
            tags=tags or [],
        )
        self._store_meta(sha1, meta)
        self._store_derivation(sha1, deriv)
        
        # Update ring state in Redis
        ring_key = self._key(KEY_RING, ring_id)
        self.client.hset(ring_key, mapping=state.to_dict())
        
        return DecomposeResult(
            sha1=sha1,
            is_new_base=is_new_base,
            expression=expression_result,
            rank_before=rank_before,
            rank_after=state.rank,
        )
    
    def _gaussian_reduce(
        self,
        ring_id: str,
        bitvec: np.ndarray
    ) -> Tuple[List[int], np.ndarray]:
        """
        Reduce bitvec using Gaussian elimination over GF(2).
        
        Returns:
            (expression, residual) where:
            - expression: list of basis indices used
            - residual: remaining bits (non-zero if independent)
        """
        matrix = self._ring_matrices[ring_id]
        pivots = self._ring_pivots[ring_id]
        
        expression = []
        residual = bitvec.copy()
        
        for row_idx, pivot in enumerate(pivots):
            if pivot < 0:
                continue
            
            if residual[pivot] == 1:
                # XOR with this basis row
                residual = (residual + matrix[row_idx]) % 2
                expression.append(row_idx)
        
        return expression, residual
    
    # =========================================================================
    # Ingest (Token → HLLSet → Decompose)
    # =========================================================================
    
    def ingest(
        self,
        ring_id: str,
        token: str,
        source: str = "",
        tags: Optional[List[str]] = None,
    ) -> DecomposeResult:
        """
        Ingest a token: create HLLSet and decompose into ring.
        
        Args:
            ring_id: Ring to use
            token: Token string
            source: Source identifier
            tags: Optional tags
            
        Returns:
            DecomposeResult
        """
        if self._use_rust:
            return self._ingest_rust(ring_id, token, source, tags)
        else:
            # Create HLLSet from token using from_batch
            hll = HLLSet.from_batch([token], p_bits=self.p_bits)
            return self._decompose_python(ring_id, hll, source=source, tags=tags)
    
    def _ingest_rust(
        self,
        ring_id: str,
        token: str,
        source: str = "",
        tags: Optional[List[str]] = None,
    ) -> DecomposeResult:
        """Ingest using Rust RING.INGEST command (server-side HLLSet creation)."""
        # Build command args
        args = ["HLLSET.RING.INGEST", ring_id, token]
        if source:
            args.extend(["SOURCE", source])
        if tags:
            args.extend(["TAGS", ",".join(tags)])
        
        # Execute
        result = self.client.execute_command(*args)
        
        # Parse result: [sha1, is_new_base, num_bases, base1, base2, ...]
        # Rust format: 0=sha1, 1=is_new_base(0/1), 2=num_bases, 3+=expression
        if isinstance(result, (list, tuple)) and len(result) >= 3:
            result_sha1 = result[0].decode() if isinstance(result[0], bytes) else str(result[0])
            is_new_base = bool(result[1])
            num_bases = int(result[2])
            expression = [
                r.decode() if isinstance(r, bytes) else str(r)
                for r in result[3:]
            ]
            
            # Get current rank from ring state
            ring_state = self.get_ring(ring_id)
            rank = ring_state.rank if ring_state else num_bases
            
            return DecomposeResult(
                sha1=result_sha1,
                is_new_base=is_new_base,
                expression=expression if expression else [result_sha1],
                rank_before=rank - (1 if is_new_base else 0),
                rank_after=rank,
            )
        else:
            raise ValueError(f"Unexpected RING.INGEST result: {result}")
    
    def ingest_batch(
        self,
        ring_id: str,
        tokens: List[str],
        source: str = "",
    ) -> List[DecomposeResult]:
        """Ingest multiple tokens."""
        results = []
        for token in tokens:
            result = self.ingest(ring_id, token, source=source)
            results.append(result)
        return results
    
    # =========================================================================
    # Storage Operations
    # =========================================================================
    
    def _store_base(self, sha1: HLLSetID, hll: HLLSet):
        """Store base HLLSet bytes as numpy array."""
        key = self._key(KEY_BASE, sha1)
        self.client.set(key, hll.dump_numpy().tobytes())
    
    def _get_base(self, sha1: HLLSetID) -> Optional[HLLSet]:
        """Get base HLLSet from storage.
        
        Handles both:
        - Python storage: raw bytes (STRING)
        - Rust storage: native hllset-rs type
        """
        import numpy as np
        key = self._key(KEY_BASE, sha1)
        
        # Check key type
        key_type = self.client.type(key)
        if isinstance(key_type, bytes):
            key_type = key_type.decode()
        
        if key_type == 'none':
            return None
        elif key_type == 'string':
            # Python format: raw bytes
            data = self.client.get(key)
            if data:
                registers = np.frombuffer(data, dtype=np.uint32)
                hll = HLLSet(p_bits=self.p_bits)
                hll._core.set_registers(registers)
                hll._compute_name()
                return hll
            return None
        elif key_type == 'hllset-rs' or key_type.startswith('hllset'):
            # Rust format: native HLLSet type, use HLLSET.DUMP
            # DUMP returns sparse format: [(register_index, value), ...]
            result = self.client.execute_command("HLLSET.DUMP", key)
            if result and isinstance(result, list):
                # Create dense array from sparse representation
                num_registers = 1 << self.p_bits
                registers = np.zeros(num_registers, dtype=np.uint32)
                for entry in result:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        idx, val = int(entry[0]), int(entry[1])
                        if 0 <= idx < num_registers:
                            registers[idx] = val
                
                hll = HLLSet(p_bits=self.p_bits)
                hll._core.set_registers(registers)
                hll._compute_name()
                return hll
            return None
        else:
            # Unknown type
            return None
    
    def _store_meta(self, sha1: HLLSetID, meta: HLLSetMeta):
        """Store metadata."""
        key = self._key(KEY_META, sha1)
        self.client.hset(key, mapping=meta.to_dict())
    
    def _store_derivation(self, sha1: HLLSetID, deriv: Derivation):
        """Store derivation."""
        key = self._key(KEY_LUT, sha1)
        self.client.hset(key, mapping=deriv.to_dict())
    
    def _increment_refcounts(self, sha1s: List[HLLSetID]):
        """Increment reference counts for bases."""
        pipe = self.client.pipeline()
        for sha1 in sha1s:
            key = self._key(KEY_REFCOUNT, sha1)
            pipe.incr(key)
        pipe.execute()
    
    def _decrement_refcounts(self, sha1s: List[HLLSetID]):
        """Decrement reference counts for bases."""
        pipe = self.client.pipeline()
        for sha1 in sha1s:
            key = self._key(KEY_REFCOUNT, sha1)
            pipe.decr(key)
        pipe.execute()
    
    # =========================================================================
    # Retrieval (with XOR Reconstruction)
    # =========================================================================
    
    def get(self, sha1: HLLSetID) -> Optional[HLLSet]:
        """
        Get HLLSet by SHA1.
        
        For bases: returns stored bytes
        For compounds: reconstructs from XOR of bases (with caching)
        """
        # Check cache
        cached = self._cache.get(sha1)
        if cached:
            return cached
        
        # Check if it's a base
        base = self._get_base(sha1)
        if base:
            self._cache.put(sha1, base)
            return base
        
        # Reconstruct from derivation
        deriv = self.get_derivation(sha1)
        if not deriv:
            return None
        
        if deriv.operation == Operation.BASE:
            # Should have been found above
            return None
        
        # XOR reconstruction
        result = self._reconstruct_xor(deriv.bases)
        if result:
            self._cache.put(sha1, result)
        
        return result
    
    def _reconstruct_xor(self, base_sha1s: List[HLLSetID]) -> Optional[HLLSet]:
        """Reconstruct HLLSet from XOR of bases."""
        if not base_sha1s:
            return None
        
        # Get first base
        result = self._get_base(base_sha1s[0])
        if not result:
            return None
        
        result = result.copy()
        
        # XOR with remaining bases
        for sha1 in base_sha1s[1:]:
            base = self._get_base(sha1)
            if not base:
                return None
            result = result ^ base
        
        return result
    
    def get_derivation(self, sha1: HLLSetID) -> Optional[Derivation]:
        """Get derivation by SHA1."""
        key = self._key(KEY_LUT, sha1)
        data = self.client.hgetall(key)
        if not data:
            return None
        return Derivation.from_dict(data)
    
    def get_meta(self, sha1: HLLSetID) -> Optional[HLLSetMeta]:
        """Get metadata by SHA1."""
        key = self._key(KEY_META, sha1)
        data = self.client.hgetall(key)
        if not data:
            return None
        return HLLSetMeta.from_dict(data)
    
    def exists(self, sha1: HLLSetID) -> bool:
        """Check if HLLSet exists."""
        key = self._key(KEY_LUT, sha1)
        return self.client.exists(key) > 0
    
    def is_base(self, sha1: HLLSetID) -> bool:
        """Check if SHA1 is a base HLLSet."""
        deriv = self.get_derivation(sha1)
        return deriv is not None and deriv.operation == Operation.BASE
    
    # =========================================================================
    # W Lattice Commits
    # =========================================================================
    
    def commit_W(
        self,
        ring_id: str,
        time_index: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> WCommit:
        """
        Create W lattice commit (snapshot of ring state).
        
        Args:
            ring_id: Ring to snapshot
            time_index: Optional time index (auto-increments if not provided)
            metadata: Optional metadata
            
        Returns:
            WCommit object
        """
        if self._use_rust:
            return self._commit_W_rust(ring_id, time_index, metadata)
        else:
            return self._commit_W_python(ring_id, time_index, metadata)
    
    def _commit_W_rust(
        self,
        ring_id: str,
        time_index: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> WCommit:
        """Create W commit using Rust W.COMMIT command."""
        try:
            args = ["HLLSET.W.COMMIT", ring_id]
            if time_index is not None:
                args.extend(["TIME", str(time_index)])
            if metadata:
                args.extend(["META", json.dumps(metadata)])
            
            result = self.client.execute_command(*args)
            
            # Rust returns just the time index as an integer
            if isinstance(result, int):
                t_idx = result
            elif isinstance(result, (list, tuple)) and len(result) >= 1:
                t_idx = int(result[0])
            else:
                raise RuntimeError(f"Unexpected W.COMMIT result: {result}")
            
            # Fetch the stored commit from Redis to get full data
            w_key = f"{self.prefix}W:{ring_id}:{t_idx}"
            data = self.client.hgetall(w_key)
            if data:
                commit = WCommit.from_dict(data)
                commit.metadata = metadata or {}
                return commit
            else:
                # Fallback: construct from current ring state
                state = self.get_ring(ring_id)
                return WCommit(
                    ring_id=ring_id,
                    time_index=t_idx,
                    basis_sha1s=state.basis_sha1s if state else [],
                    timestamp=time.time(),
                    metadata=metadata or {},
                )
                
        except redis.ResponseError as e:
            raise RuntimeError(f"Rust W.COMMIT failed: {e}")
    
    def _commit_W_python(
        self,
        ring_id: str,
        time_index: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> WCommit:
        """Create W commit using Python (Redis hash storage)."""
        state = self.get_ring(ring_id)
        if not state:
            raise ValueError(f"Ring {ring_id} not found")
        
        # Auto-increment time index
        if time_index is None:
            time_index = self._get_next_W_index(ring_id)
        
        commit = WCommit(
            ring_id=ring_id,
            time_index=time_index,
            basis_sha1s=list(state.basis_sha1s),
            timestamp=time.time(),
            metadata=metadata or {},
        )
        
        # Store commit
        key = self._key(KEY_W, f"{ring_id}:{time_index}")
        self.client.hset(key, mapping=commit.to_dict())
        
        return commit
        
        return commit
    
    def _get_next_W_index(self, ring_id: str) -> int:
        """Get next W commit index for a ring."""
        pattern = self._key(KEY_W, f"{ring_id}:*")
        keys = list(self.client.scan_iter(match=pattern, count=1000))
        
        if not keys:
            return 0
        
        max_idx = -1
        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            idx_str = key_str.split(':')[-1]
            try:
                idx = int(idx_str)
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
        
        return max_idx + 1
    
    def get_W(self, ring_id: str, time_index: int) -> Optional[WCommit]:
        """Get W commit by ring and time index."""
        key = self._key(KEY_W, f"{ring_id}:{time_index}")
        data = self.client.hgetall(key)
        if not data:
            return None
        return WCommit.from_dict(data)
    
    def diff_W(self, ring_id: str, t1: int, t2: int) -> Optional[WDiff]:
        """
        Compute difference between two W commits.
        
        Args:
            ring_id: Ring ID
            t1: Earlier time index
            t2: Later time index
            
        Returns:
            WDiff showing added/removed/shared bases
        """
        if self._use_rust:
            return self._diff_W_rust(ring_id, t1, t2)
        else:
            return self._diff_W_python(ring_id, t1, t2)
    
    def _diff_W_rust(self, ring_id: str, t1: int, t2: int) -> Optional[WDiff]:
        """Compute W diff using Rust W.DIFF command."""
        try:
            result = self.client.execute_command(
                "HLLSET.W.DIFF", ring_id, str(t1), str(t2)
            )
            
            # Parse result: { added: [...], removed: [...], shared: [...], delta_rank: int }
            if isinstance(result, dict):
                return WDiff(
                    t1=t1,
                    t2=t2,
                    added_bases=[r.decode() if isinstance(r, bytes) else str(r) for r in result.get(b'added', [])],
                    removed_bases=[r.decode() if isinstance(r, bytes) else str(r) for r in result.get(b'removed', [])],
                    shared_bases=[r.decode() if isinstance(r, bytes) else str(r) for r in result.get(b'shared', [])],
                )
            elif isinstance(result, (list, tuple)):
                # Alternate format: [t1, t2, added_count, [added...], removed_count, [removed...], shared_count, [shared...]]
                # For now, fall back to Python implementation
                return self._diff_W_python(ring_id, t1, t2)
            else:
                return self._diff_W_python(ring_id, t1, t2)
                
        except redis.ResponseError:
            return self._diff_W_python(ring_id, t1, t2)
    
    def _diff_W_python(self, ring_id: str, t1: int, t2: int) -> Optional[WDiff]:
        """Compute W diff using Python."""
        w1 = self.get_W(ring_id, t1)
        w2 = self.get_W(ring_id, t2)
        
        if not w1 or not w2:
            return None
        
        set1 = set(w1.basis_sha1s)
        set2 = set(w2.basis_sha1s)
        
        return WDiff(
            t1=t1,
            t2=t2,
            added_bases=list(set2 - set1),
            removed_bases=list(set1 - set2),
            shared_bases=list(set1 & set2),
        )
    
    def list_W_commits(self, ring_id: str) -> List[WCommit]:
        """List all W commits for a ring."""
        pattern = self._key(KEY_W, f"{ring_id}:*")
        commits = []
        
        for key in self.client.scan_iter(match=pattern, count=1000):
            data = self.client.hgetall(key)
            if data:
                commits.append(WCommit.from_dict(data))
        
        # Sort by time index
        commits.sort(key=lambda c: c.time_index)
        return commits
    
    # =========================================================================
    # Eviction
    # =========================================================================
    
    def evict_unreferenced(self, dry_run: bool = True) -> List[HLLSetID]:
        """
        Evict bases with refcount=0 that are not in any ring.
        
        Args:
            dry_run: If True, just return candidates without evicting
            
        Returns:
            List of evicted (or candidate) SHA1s
        """
        candidates = []
        
        # Scan all bases
        pattern = self._key(KEY_BASE, "*")
        for key in self.client.scan_iter(match=pattern, count=1000):
            key_str = key.decode() if isinstance(key, bytes) else key
            sha1 = key_str.split(':')[-1]
            
            # Check refcount
            ref_key = self._key(KEY_REFCOUNT, sha1)
            refcount = self.client.get(ref_key)
            
            if refcount is None or int(refcount) <= 0:
                # Check if in any active ring
                in_ring = self._is_in_any_ring(sha1)
                if not in_ring:
                    candidates.append(sha1)
        
        if not dry_run:
            for sha1 in candidates:
                self._evict_base(sha1)
        
        return candidates
    
    def _is_in_any_ring(self, sha1: HLLSetID) -> bool:
        """Check if SHA1 is in any active ring's basis."""
        pattern = self._key(KEY_RING, "*")
        for key in self.client.scan_iter(match=pattern, count=100):
            data = self.client.hgetall(key)
            if data:
                state = RingState.from_dict(data)
                if sha1 in state.basis_sha1s:
                    return True
        return False
    
    def _evict_base(self, sha1: HLLSetID):
        """Evict a base HLLSet."""
        pipe = self.client.pipeline()
        pipe.delete(self._key(KEY_BASE, sha1))
        pipe.delete(self._key(KEY_LUT, sha1))
        pipe.delete(self._key(KEY_META, sha1))
        pipe.delete(self._key(KEY_REFCOUNT, sha1))
        pipe.execute()
        
        self._cache.invalidate(sha1)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def stats(self) -> Dict:
        """Return store statistics."""
        # Count bases
        base_count = 0
        for _ in self.client.scan_iter(match=self._key(KEY_BASE, "*"), count=1000):
            base_count += 1
        
        # Count all entries (including compounds)
        entry_count = 0
        for _ in self.client.scan_iter(match=self._key(KEY_LUT, "*"), count=1000):
            entry_count += 1
        
        # Count rings
        ring_count = 0
        for _ in self.client.scan_iter(match=self._key(KEY_RING, "*"), count=1000):
            ring_count += 1
        
        return {
            'base_count': base_count,
            'compound_count': entry_count - base_count,
            'total_entries': entry_count,
            'ring_count': ring_count,
            'cache_size': len(self._cache),
        }
    
    # =========================================================================
    # Query Methods (via RediSearch)
    # =========================================================================
    
    def query_bases(self, limit: int = 100) -> List[HLLSetMeta]:
        """Query only base HLLSets."""
        query = Query("@is_base:[1 1]").paging(0, limit)
        results = self.client.ft(INDEX_NAME).search(query)
        return [HLLSetMeta.from_dict(doc.__dict__) for doc in results.docs]
    
    def query_by_source(self, source: str, limit: int = 100) -> List[HLLSetMeta]:
        """Query by source."""
        escaped = source.replace("-", "\\-").replace("@", "\\@")
        query = Query(f"@source:{escaped}").paging(0, limit)
        results = self.client.ft(INDEX_NAME).search(query)
        return [HLLSetMeta.from_dict(doc.__dict__) for doc in results.docs]
    
    def query_by_tag(self, *tags: str, limit: int = 100) -> List[HLLSetMeta]:
        """Query by tags."""
        if not tags:
            return []
        escaped = [t.replace(",", "\\,").replace("-", "\\-") for t in tags]
        tag_query = "|".join(escaped)
        query = Query(f"@tags:{{{tag_query}}}").paging(0, limit)
        results = self.client.ft(INDEX_NAME).search(query)
        return [HLLSetMeta.from_dict(doc.__dict__) for doc in results.docs]
