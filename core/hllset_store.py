"""
HLLSet Store — Persistent Base HLLSets + Derivation LUT

Architecture:
    - Base HLLSets: stored as roaring bitmaps in key-value store (SHA1 → bytes)
    - Compound HLLSets: NOT stored; generated on-the-fly from LUT
    - HLLSet LUT: derivation graph mapping SHA1 → (operation, [operand_SHA1s])
    - Ephemeral Lattice: in-memory lattice; changes recorded to LUT

This follows the principle: "explicit only base HLLSets, all others generated on the fly"

The ring structure enables reconstruction:
    H₄ = H₁ ∪ H₂  →  LUT["H₄"] = ("union", ["H₁", "H₂"])
    H₅ = H₁ ⊕ H₃  →  LUT["H₅"] = ("xor", ["H₁", "H₃"])

Storage format:
    Key:   SHA1 hash (20 bytes, hex = 40 chars)
    Value: RoaringBitmap serialization (compressed)

Usage:
    store = HLLSetStore()
    
    # Register base HLLSets
    id1 = store.register_base(hll1, source="doc1")
    id2 = store.register_base(hll2, source="doc2")
    
    # Compound operations (not stored, but LUT records derivation)
    id3 = store.union(id1, id2)      # LUT: id3 → ("union", [id1, id2])
    id4 = store.intersect(id1, id2)  # LUT: id4 → ("intersect", [id1, id2])
    
    # Reconstruct any HLLSet from its ID
    hll3 = store.get(id3)  # Computes H1 ∪ H2 on-the-fly
    
    # Ephemeral lattice for complex operations
    with store.lattice() as lat:
        lat.add(id1, id2, id3)
        # ... operations ...
        # Changes recorded to LUT on exit
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, List, Tuple, Optional, Set, Union, 
    Protocol, Iterator, Any, Callable
)
from enum import Enum
from pathlib import Path
import json
import time

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


@dataclass(frozen=True)
class Derivation:
    """
    How a compound HLLSet was derived.
    
    For base HLLSets: operation=BASE, operands=[]
    For compounds: operation specifies how, operands are the input IDs
    """
    operation: Operation
    operands: Tuple[HLLSetID, ...]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_base(self) -> bool:
        return self.operation == Operation.BASE
    
    def to_dict(self) -> Dict:
        return {
            "operation": self.operation.value,
            "operands": list(self.operands),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Derivation':
        return cls(
            operation=Operation(d["operation"]),
            operands=tuple(d["operands"]),
            timestamp=d.get("timestamp", 0.0),
            metadata=d.get("metadata", {}),
        )


# =============================================================================
# Storage Backend Protocol
# =============================================================================

class StorageBackend(Protocol):
    """Protocol for key-value storage backends."""
    
    def get(self, key: str) -> Optional[bytes]:
        """Get value by key."""
        ...
    
    def put(self, key: str, value: bytes) -> None:
        """Store key-value pair."""
        ...
    
    def delete(self, key: str) -> None:
        """Delete key."""
        ...
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...
    
    def keys(self, prefix: str = "") -> Iterator[str]:
        """Iterate over keys with optional prefix."""
        ...


class InMemoryBackend:
    """Simple in-memory storage backend for testing."""
    
    def __init__(self):
        self._store: Dict[str, bytes] = {}
    
    def get(self, key: str) -> Optional[bytes]:
        return self._store.get(key)
    
    def put(self, key: str, value: bytes) -> None:
        self._store[key] = value
    
    def delete(self, key: str) -> None:
        self._store.pop(key, None)
    
    def exists(self, key: str) -> bool:
        return key in self._store
    
    def keys(self, prefix: str = "") -> Iterator[str]:
        for k in self._store.keys():
            if k.startswith(prefix):
                yield k
    
    def __len__(self) -> int:
        return len(self._store)


# =============================================================================
# HLLSet LUT (Derivation Graph)
# =============================================================================

class HLLSetLUT:
    """
    Look-Up Table mapping HLLSet IDs to their derivations.
    
    This is the "derivation graph" that tracks how compound HLLSets
    are constructed from base HLLSets.
    
    Structure:
        {
            "abc123...": Derivation(BASE, [], metadata={"source": "doc1"}),
            "def456...": Derivation(UNION, ["abc123...", "xyz789..."]),
            ...
        }
    """
    
    def __init__(self, backend: Optional[StorageBackend] = None):
        self._backend = backend or InMemoryBackend()
        self._cache: Dict[HLLSetID, Derivation] = {}
        self._reverse_index: Dict[HLLSetID, Set[HLLSetID]] = {}
    
    def register(self, hllset_id: HLLSetID, derivation: Derivation) -> None:
        """Register a derivation for an HLLSet ID."""
        self._cache[hllset_id] = derivation
        
        # Update reverse index: operand → {compounds that use it}
        for operand_id in derivation.operands:
            if operand_id not in self._reverse_index:
                self._reverse_index[operand_id] = set()
            self._reverse_index[operand_id].add(hllset_id)
        
        # Persist to backend
        key = f"lut:{hllset_id}"
        value = json.dumps(derivation.to_dict()).encode('utf-8')
        self._backend.put(key, value)
    
    def lookup(self, hllset_id: HLLSetID) -> Optional[Derivation]:
        """Look up derivation for an HLLSet ID."""
        if hllset_id in self._cache:
            return self._cache[hllset_id]
        
        # Try backend
        key = f"lut:{hllset_id}"
        data = self._backend.get(key)
        if data:
            derivation = Derivation.from_dict(json.loads(data.decode('utf-8')))
            self._cache[hllset_id] = derivation
            return derivation
        
        return None
    
    def is_base(self, hllset_id: HLLSetID) -> bool:
        """Check if ID refers to a base HLLSet."""
        deriv = self.lookup(hllset_id)
        return deriv is not None and deriv.is_base()
    
    def get_dependents(self, hllset_id: HLLSetID) -> Set[HLLSetID]:
        """Get all compound HLLSets that depend on this ID."""
        return self._reverse_index.get(hllset_id, set())
    
    def get_bases(self, hllset_id: HLLSetID) -> Set[HLLSetID]:
        """
        Recursively find all base HLLSets that contribute to this ID.
        
        Walks the derivation graph back to the roots.
        """
        deriv = self.lookup(hllset_id)
        if deriv is None:
            return set()
        
        if deriv.is_base():
            return {hllset_id}
        
        bases = set()
        for operand_id in deriv.operands:
            bases.update(self.get_bases(operand_id))
        
        return bases
    
    def derivation_depth(self, hllset_id: HLLSetID) -> int:
        """
        Compute depth of derivation (0 for base, 1+ for compounds).
        """
        deriv = self.lookup(hllset_id)
        if deriv is None or deriv.is_base():
            return 0
        
        return 1 + max(
            (self.derivation_depth(op_id) for op_id in deriv.operands),
            default=0
        )
    
    def all_ids(self) -> Iterator[HLLSetID]:
        """Iterate over all registered HLLSet IDs."""
        for key in self._backend.keys("lut:"):
            yield key[4:]  # Strip "lut:" prefix
    
    def base_ids(self) -> Iterator[HLLSetID]:
        """Iterate over base HLLSet IDs only."""
        for hllset_id in self.all_ids():
            if self.is_base(hllset_id):
                yield hllset_id


# =============================================================================
# HLLSet Store (Main Interface)
# =============================================================================

class HLLSetStore:
    """
    Main storage interface for HLLSets.
    
    Design:
        - Base HLLSets: stored as roaring bitmaps
        - Compound HLLSets: computed on-the-fly from LUT
        - All operations recorded in LUT for reproducibility
    
    Usage:
        store = HLLSetStore()
        
        # Register base HLLSets (these get persisted)
        id1 = store.register_base(hll1, source="document_1")
        id2 = store.register_base(hll2, source="document_2")
        
        # Operations (compounds NOT persisted, but LUT updated)
        id3 = store.union(id1, id2)
        
        # Retrieve (computes on-the-fly if compound)
        hll3 = store.get(id3)
    """
    
    def __init__(
        self, 
        backend: Optional[StorageBackend] = None,
        p_bits: int = 10,
    ):
        self._backend = backend or InMemoryBackend()
        self._lut = HLLSetLUT(self._backend)
        self._p_bits = p_bits
        
        # In-memory cache for frequently accessed HLLSets
        self._hllset_cache: Dict[HLLSetID, HLLSet] = {}
        self._cache_max_size = 100
    
    # -------------------------------------------------------------------------
    # ID Computation
    # -------------------------------------------------------------------------
    
    @staticmethod
    def compute_id(hll: HLLSet) -> HLLSetID:
        """
        Compute SHA1 ID for an HLLSet.
        
        The ID is deterministic: same bit-vector → same ID.
        """
        # Use numpy array bytes for consistent hashing (always available)
        numpy_bytes = hll.dump_numpy().tobytes()
        return hashlib.sha1(numpy_bytes).hexdigest()
    
    # -------------------------------------------------------------------------
    # Base HLLSet Operations
    # -------------------------------------------------------------------------
    
    def register_base(
        self, 
        hll: HLLSet, 
        source: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> HLLSetID:
        """
        Register a base HLLSet (persisted to storage).
        
        Args:
            hll: The HLLSet to register
            source: Optional source identifier (e.g., document name)
            metadata: Optional additional metadata
        
        Returns:
            The SHA1 ID of the registered HLLSet
        """
        hllset_id = self.compute_id(hll)
        
        # Check if already registered
        if self._lut.lookup(hllset_id) is not None:
            return hllset_id
        
        # Store as numpy bytes (always available, roaring is optional)
        numpy_bytes = hll.dump_numpy().tobytes()
        self._backend.put(f"hll:{hllset_id}", numpy_bytes)
        
        # Record in LUT as base
        meta = metadata or {}
        if source:
            meta["source"] = source
        meta["p_bits"] = hll.p_bits  # Store p_bits for reconstruction
        
        derivation = Derivation(
            operation=Operation.BASE,
            operands=(),
            metadata=meta,
        )
        self._lut.register(hllset_id, derivation)
        
        # Cache
        self._cache_put(hllset_id, hll)
        
        return hllset_id
    
    def _cache_put(self, hllset_id: HLLSetID, hll: HLLSet) -> None:
        """Add to cache with LRU eviction."""
        if len(self._hllset_cache) >= self._cache_max_size:
            # Simple eviction: remove first item
            first_key = next(iter(self._hllset_cache))
            del self._hllset_cache[first_key]
        self._hllset_cache[hllset_id] = hll
    
    # -------------------------------------------------------------------------
    # Retrieval (On-the-fly reconstruction)
    # -------------------------------------------------------------------------
    
    def get(self, hllset_id: HLLSetID) -> Optional[HLLSet]:
        """
        Retrieve an HLLSet by ID.
        
        - For base HLLSets: loads from storage
        - For compounds: reconstructs from LUT on-the-fly
        """
        import numpy as np
        
        # Check cache
        if hllset_id in self._hllset_cache:
            return self._hllset_cache[hllset_id]
        
        deriv = self._lut.lookup(hllset_id)
        if deriv is None:
            return None
        
        if deriv.is_base():
            # Load from storage (numpy bytes format)
            data = self._backend.get(f"hll:{hllset_id}")
            if data is None:
                return None
            
            # Get p_bits from metadata or use default
            p_bits = deriv.metadata.get("p_bits", self._p_bits)
            
            # Reconstruct from numpy bytes
            arr = np.frombuffer(data, dtype=np.uint32)
            hll = HLLSet(p_bits=p_bits)
            hll._core.set_registers(arr)
            
            self._cache_put(hllset_id, hll)
            return hll
        
        # Reconstruct compound on-the-fly
        hll = self._reconstruct(deriv)
        if hll:
            self._cache_put(hllset_id, hll)
        return hll
    
    def _reconstruct(self, deriv: Derivation) -> Optional[HLLSet]:
        """Reconstruct a compound HLLSet from its derivation."""
        if deriv.is_base():
            raise ValueError("Cannot reconstruct base HLLSet from derivation alone")
        
        # Get operands recursively
        operands = []
        for op_id in deriv.operands:
            hll = self.get(op_id)
            if hll is None:
                return None
            operands.append(hll)
        
        if len(operands) < 2:
            return operands[0] if operands else None
        
        # Apply operation
        result = operands[0]
        for operand in operands[1:]:
            if deriv.operation == Operation.UNION:
                result = result.union(operand)
            elif deriv.operation == Operation.INTERSECT:
                result = result.intersect(operand)
            elif deriv.operation == Operation.DIFF:
                result = result.diff(operand)
            elif deriv.operation == Operation.XOR:
                result = result.xor(operand)
        
        return result
    
    # -------------------------------------------------------------------------
    # Compound Operations (Not persisted, but LUT updated)
    # -------------------------------------------------------------------------
    
    def _register_compound(
        self, 
        result: HLLSet, 
        operation: Operation, 
        operand_ids: List[HLLSetID],
    ) -> HLLSetID:
        """Register a compound operation in the LUT."""
        result_id = self.compute_id(result)
        
        # Only register if not already known
        if self._lut.lookup(result_id) is None:
            derivation = Derivation(
                operation=operation,
                operands=tuple(operand_ids),
            )
            self._lut.register(result_id, derivation)
        
        # Cache the result
        self._cache_put(result_id, result)
        
        return result_id
    
    def union(self, id1: HLLSetID, id2: HLLSetID) -> HLLSetID:
        """Compute union and register in LUT."""
        hll1 = self.get(id1)
        hll2 = self.get(id2)
        if hll1 is None or hll2 is None:
            raise ValueError(f"Unknown HLLSet ID: {id1 if hll1 is None else id2}")
        
        result = hll1.union(hll2)
        return self._register_compound(result, Operation.UNION, [id1, id2])
    
    def intersect(self, id1: HLLSetID, id2: HLLSetID) -> HLLSetID:
        """Compute intersection and register in LUT."""
        hll1 = self.get(id1)
        hll2 = self.get(id2)
        if hll1 is None or hll2 is None:
            raise ValueError(f"Unknown HLLSet ID: {id1 if hll1 is None else id2}")
        
        result = hll1.intersect(hll2)
        return self._register_compound(result, Operation.INTERSECT, [id1, id2])
    
    def diff(self, id1: HLLSetID, id2: HLLSetID) -> HLLSetID:
        """Compute difference (id1 \\ id2) and register in LUT."""
        hll1 = self.get(id1)
        hll2 = self.get(id2)
        if hll1 is None or hll2 is None:
            raise ValueError(f"Unknown HLLSet ID: {id1 if hll1 is None else id2}")
        
        result = hll1.diff(hll2)
        return self._register_compound(result, Operation.DIFF, [id1, id2])
    
    def xor(self, id1: HLLSetID, id2: HLLSetID) -> HLLSetID:
        """Compute XOR (symmetric difference) and register in LUT."""
        hll1 = self.get(id1)
        hll2 = self.get(id2)
        if hll1 is None or hll2 is None:
            raise ValueError(f"Unknown HLLSet ID: {id1 if hll1 is None else id2}")
        
        result = hll1.xor(hll2)
        return self._register_compound(result, Operation.XOR, [id1, id2])
    
    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------
    
    def exists(self, hllset_id: HLLSetID) -> bool:
        """Check if an HLLSet ID is known."""
        return self._lut.lookup(hllset_id) is not None
    
    def is_base(self, hllset_id: HLLSetID) -> bool:
        """Check if ID refers to a base HLLSet."""
        return self._lut.is_base(hllset_id)
    
    def get_derivation(self, hllset_id: HLLSetID) -> Optional[Derivation]:
        """Get the derivation for an HLLSet ID."""
        return self._lut.lookup(hllset_id)
    
    def get_bases(self, hllset_id: HLLSetID) -> Set[HLLSetID]:
        """Get all base HLLSets that contribute to this ID."""
        return self._lut.get_bases(hllset_id)
    
    def trace_derivation(self, hllset_id: HLLSetID) -> List[Tuple[HLLSetID, Derivation]]:
        """
        Trace the full derivation tree for an HLLSet.
        
        Returns list of (id, derivation) pairs in topological order
        (bases first, compound last).
        """
        visited = set()
        result = []
        
        def visit(hid: HLLSetID):
            if hid in visited:
                return
            visited.add(hid)
            
            deriv = self._lut.lookup(hid)
            if deriv is None:
                return
            
            # Visit operands first
            for op_id in deriv.operands:
                visit(op_id)
            
            result.append((hid, deriv))
        
        visit(hllset_id)
        return result
    
    def all_base_ids(self) -> Iterator[HLLSetID]:
        """Iterate over all base HLLSet IDs."""
        return self._lut.base_ids()
    
    def all_ids(self) -> Iterator[HLLSetID]:
        """Iterate over all HLLSet IDs (base and compound)."""
        return self._lut.all_ids()
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        base_count = sum(1 for _ in self._lut.base_ids())
        total_count = sum(1 for _ in self._lut.all_ids())
        
        return {
            "base_hllsets": base_count,
            "compound_hllsets": total_count - base_count,
            "total_entries": total_count,
            "cache_size": len(self._hllset_cache),
            "cache_max_size": self._cache_max_size,
        }


# =============================================================================
# Ephemeral Lattice (In-Memory Operations)
# =============================================================================

class EphemeralLattice:
    """
    In-memory lattice for complex multi-HLLSet operations.
    
    Created via store.lattice() context manager. All changes
    are recorded to the store's LUT on exit.
    
    Usage:
        with store.lattice() as lat:
            # Load HLLSets into lattice
            lat.add(id1, id2, id3)
            
            # Perform operations
            id4 = lat.union(id1, id2)
            id5 = lat.intersect(id4, id3)
            
            # Changes recorded to LUT automatically
    """
    
    def __init__(self, store: HLLSetStore):
        self._store = store
        self._local_hllsets: Dict[HLLSetID, HLLSet] = {}
        self._new_derivations: List[Tuple[HLLSetID, Derivation]] = []
    
    def add(self, *hllset_ids: HLLSetID) -> None:
        """Load HLLSets into the lattice."""
        for hid in hllset_ids:
            if hid not in self._local_hllsets:
                hll = self._store.get(hid)
                if hll:
                    self._local_hllsets[hid] = hll
    
    def get(self, hllset_id: HLLSetID) -> Optional[HLLSet]:
        """Get HLLSet from lattice or store."""
        if hllset_id in self._local_hllsets:
            return self._local_hllsets[hllset_id]
        return self._store.get(hllset_id)
    
    def _register_local(
        self, 
        result: HLLSet, 
        operation: Operation, 
        operand_ids: List[HLLSetID],
    ) -> HLLSetID:
        """Register a new HLLSet locally (will be committed on exit)."""
        result_id = HLLSetStore.compute_id(result)
        self._local_hllsets[result_id] = result
        
        derivation = Derivation(
            operation=operation,
            operands=tuple(operand_ids),
        )
        self._new_derivations.append((result_id, derivation))
        
        return result_id
    
    def union(self, id1: HLLSetID, id2: HLLSetID) -> HLLSetID:
        """Compute union within lattice."""
        hll1 = self.get(id1)
        hll2 = self.get(id2)
        if hll1 is None or hll2 is None:
            raise ValueError("HLLSet not found in lattice")
        
        result = hll1.union(hll2)
        return self._register_local(result, Operation.UNION, [id1, id2])
    
    def intersect(self, id1: HLLSetID, id2: HLLSetID) -> HLLSetID:
        """Compute intersection within lattice."""
        hll1 = self.get(id1)
        hll2 = self.get(id2)
        if hll1 is None or hll2 is None:
            raise ValueError("HLLSet not found in lattice")
        
        result = hll1.intersect(hll2)
        return self._register_local(result, Operation.INTERSECT, [id1, id2])
    
    def xor(self, id1: HLLSetID, id2: HLLSetID) -> HLLSetID:
        """Compute XOR within lattice."""
        hll1 = self.get(id1)
        hll2 = self.get(id2)
        if hll1 is None or hll2 is None:
            raise ValueError("HLLSet not found in lattice")
        
        result = hll1.xor(hll2)
        return self._register_local(result, Operation.XOR, [id1, id2])
    
    def commit(self) -> int:
        """Commit all new derivations to the store's LUT."""
        for hllset_id, derivation in self._new_derivations:
            self._store._lut.register(hllset_id, derivation)
            # Cache in store
            if hllset_id in self._local_hllsets:
                self._store._cache_put(hllset_id, self._local_hllsets[hllset_id])
        
        count = len(self._new_derivations)
        self._new_derivations.clear()
        return count
    
    def __enter__(self) -> 'EphemeralLattice':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.commit()


# Add lattice method to HLLSetStore
def _lattice(self) -> EphemeralLattice:
    """Create an ephemeral lattice for complex operations."""
    return EphemeralLattice(self)

HLLSetStore.lattice = _lattice


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Types
    "HLLSetID",
    "Operation",
    "Derivation",
    
    # Storage
    "StorageBackend",
    "InMemoryBackend",
    "HLLSetLUT",
    "HLLSetStore",
    
    # Lattice
    "EphemeralLattice",
]
