"""
HLL Lattice — Temporal lattice of HLLSet observations (the W lattice).

Implements the lattice structure from the manuscript (§2.3 of INTRODUCTION):

    The set of all HLLSet fingerprints (for fixed m, b) is partially ordered
    by bitwise inclusion:
        A ≤ B  ⟺  R_A ∧ ¬R_B = 0

    This is a distributive lattice (meet = ∧, join = ∨) with bottom ∅
    and top the all-ones matrix. We call this the W lattice.

Temporal structure:
    Each time-step t produces a LatticeNode:
        Lattice(t) = { T₁(t), T₂(t), ..., Tₖ(t) }   ← individual HLLSets
                   ↓ merge (bitwise OR)
                 M(t) = ∪ Tᵢ(t)                         ← merged snapshot
                   ↓ content-address
               ID(t) = SHA1(M(t).registers)             ← node identity

    Temporal ordering: Lattice(t₁) ⊑ Lattice(t₂)  ⟺  M(t₁) ⊆ M(t₂)

Corollary 2.4 (from manuscript):
    In a closed environment where additions and deletions balance,
    the W lattice is INVARIANT — its structure is conserved.

Design:
    - LatticeNode is immutable once created
    - HLLLattice is an explicit object (no singletons)
    - In-memory storage with Protocol interface for external persistence
    - All operations return new objects (functional style)

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import (
    List, Tuple, Optional, Dict, Set, Any, Iterator,
    Protocol, runtime_checkable,
)
from dataclasses import dataclass, field
import time
import hashlib
import numpy as np

from .hllset import HLLSet, compute_sha1
from .bss import bss, BSSPair, morphism_graph


# =============================================================================
# Lattice Node
# =============================================================================

@dataclass(frozen=True)
class LatticeNode:
    """
    Immutable node in the temporal W lattice.
    
    Content-addressed by SHA1 of merged register state.
    Once created, a node never changes.
    
    Attributes:
        node_id: SHA1 hash of merged registers (content address)
        timestamp: When this node was created (wall-clock or logical)
        merged: The merged HLLSet (union of all components)
        cardinality: Estimated cardinality of merged set
        popcount: Total set bits in merged registers
        component_ids: SHA1 IDs of the individual HLLSets that were merged
        parent_ids: IDs of causal predecessor nodes (empty for roots)
        metadata: Application-defined metadata dict
    """
    node_id: str
    timestamp: float
    merged: HLLSet
    cardinality: float
    popcount: int
    component_ids: Tuple[str, ...]
    parent_ids: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LatticeNode):
            return NotImplemented
        return self.node_id == other.node_id

    def is_subset_of(self, other: 'LatticeNode') -> bool:
        """Check if self ⊆ other in the lattice partial order."""
        # A ≤ B ⟺ R_A ∧ ¬R_B = 0
        diff = self.merged.diff(other.merged)
        return diff.cardinality() < 1.0  # Near-zero (accounting for HLL error)

    def __repr__(self) -> str:
        return (
            f"LatticeNode({self.node_id[:8]}..., t={self.timestamp:.1f}, "
            f"|M|≈{self.cardinality:.0f}, pop={self.popcount})"
        )


# =============================================================================
# Storage Protocol (for external persistence)
# =============================================================================

@runtime_checkable
class LatticeStorage(Protocol):
    """
    Protocol for external lattice persistence.
    
    Applications can implement this to back the lattice with
    SQLite, RocksDB, files, or any other store.
    The default HLLLattice uses in-memory dicts.
    """

    def store_node(self, node: LatticeNode) -> None: ...
    def load_node(self, node_id: str) -> Optional[LatticeNode]: ...
    def list_node_ids(self) -> List[str]: ...
    def node_count(self) -> int: ...


# =============================================================================
# In-Memory Storage (default)
# =============================================================================

class InMemoryStorage:
    """Default in-memory storage for the lattice."""

    def __init__(self):
        self._nodes: Dict[str, LatticeNode] = {}

    def store_node(self, node: LatticeNode) -> None:
        self._nodes[node.node_id] = node

    def load_node(self, node_id: str) -> Optional[LatticeNode]:
        return self._nodes.get(node_id)

    def list_node_ids(self) -> List[str]:
        return list(self._nodes.keys())

    def node_count(self) -> int:
        return len(self._nodes)


# =============================================================================
# HLL Lattice
# =============================================================================

class HLLLattice:
    """
    Temporal lattice of HLLSet observations (the W lattice).
    
    Provides:
        - Append new nodes (with automatic merge + SHA1 content-address)
        - Query by time range
        - Lattice join (∨ = union) and meet (∧ = intersection) of nodes
        - Cumulative merge up to time t
        - Delta between time-steps (what changed?)
        - Morphism graph via BSS thresholds
    
    The lattice is a LIBRARY — applications drive when nodes are created.
    
    Usage:
        lattice = HLLLattice(p_bits=10)
        
        # Append observations
        hll1 = HLLSet.from_batch(["hello", "world"])
        hll2 = HLLSet.from_batch(["foo", "bar"])
        node = lattice.append([hll1, hll2], timestamp=1.0)
        
        # Query
        print(lattice.cumulative(t=1.0))
        print(lattice.node_by_id(node.node_id))
        
        # Lattice operations
        node_a = lattice.append([hll_a])
        node_b = lattice.append([hll_b])
        joined = lattice.join(node_a, node_b)  # ∨ = union
        met = lattice.meet(node_a, node_b)      # ∧ = intersection
    """

    def __init__(
        self,
        p_bits: int = 10,
        storage: Optional[LatticeStorage] = None,
    ):
        """
        Create a new lattice.
        
        Args:
            p_bits: HLL precision bits
            storage: Optional external storage (default: in-memory)
        """
        self._p_bits = p_bits
        self._storage = storage or InMemoryStorage()

        # Temporal index: sorted list of (timestamp, node_id) pairs
        self._timeline: List[Tuple[float, str]] = []

    @property
    def p_bits(self) -> int:
        return self._p_bits

    @property
    def node_count(self) -> int:
        return self._storage.node_count()

    # =========================================================================
    # Node Creation
    # =========================================================================

    def append(
        self,
        hllsets: List[HLLSet],
        timestamp: Optional[float] = None,
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LatticeNode:
        """
        Append a new node to the lattice from a collection of HLLSets.
        
        The HLLSets are merged via union, content-addressed by SHA1,
        and stored as an immutable node.
        
        Args:
            hllsets: List of HLLSet observations for this time-step
            timestamp: Logical or wall-clock time (default: time.time())
            parent_ids: Causal predecessors (default: last node)
            metadata: Application-defined metadata
            
        Returns:
            New immutable LatticeNode
        """
        ts = timestamp if timestamp is not None else time.time()

        # Merge all HLLSets via union
        if not hllsets:
            merged = HLLSet(p_bits=self._p_bits)
        elif len(hllsets) == 1:
            merged = hllsets[0]
        else:
            merged = HLLSet.merge(hllsets)

        # Content address
        registers = merged.dump_numpy()
        node_id = compute_sha1(registers)

        # Popcount (conserved quantity)
        popcount = sum(int(r).bit_count() for r in registers)

        # Component IDs
        component_ids = tuple(h.name for h in hllsets)

        # Parent IDs: default to the most recent node
        if parent_ids is None and self._timeline:
            parent_ids = [self._timeline[-1][1]]
        elif parent_ids is None:
            parent_ids = []

        node = LatticeNode(
            node_id=node_id,
            timestamp=ts,
            merged=merged,
            cardinality=merged.cardinality(),
            popcount=popcount,
            component_ids=component_ids,
            parent_ids=tuple(parent_ids),
            metadata=metadata or {},
        )

        # Store
        self._storage.store_node(node)
        self._timeline.append((ts, node_id))
        self._timeline.sort(key=lambda x: x[0])

        return node

    def append_tokens(
        self,
        token_batches: List[List[str]],
        timestamp: Optional[float] = None,
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LatticeNode:
        """
        Convenience: append from raw token batches.
        
        Each batch becomes one HLLSet, all are merged into the node.
        """
        hllsets = [
            HLLSet.from_batch(batch, p_bits=self._p_bits)
            for batch in token_batches
        ]
        return self.append(hllsets, timestamp, parent_ids, metadata)

    # =========================================================================
    # Node Lookup
    # =========================================================================

    def node_by_id(self, node_id: str) -> Optional[LatticeNode]:
        """Retrieve a node by its content-addressed ID."""
        return self._storage.load_node(node_id)

    def nodes_in_range(
        self, t_start: float, t_end: float
    ) -> List[LatticeNode]:
        """
        Get all nodes with timestamps in [t_start, t_end].
        
        Args:
            t_start: Start of time range (inclusive)
            t_end: End of time range (inclusive)
            
        Returns:
            List of LatticeNode ordered by timestamp
        """
        result = []
        for ts, nid in self._timeline:
            if ts < t_start:
                continue
            if ts > t_end:
                break
            node = self._storage.load_node(nid)
            if node is not None:
                result.append(node)
        return result

    def latest_node(self) -> Optional[LatticeNode]:
        """Get the most recent node."""
        if not self._timeline:
            return None
        _, nid = self._timeline[-1]
        return self._storage.load_node(nid)

    def all_nodes(self) -> List[LatticeNode]:
        """Get all nodes ordered by timestamp."""
        result = []
        for _, nid in self._timeline:
            node = self._storage.load_node(nid)
            if node is not None:
                result.append(node)
        return result

    # =========================================================================
    # Lattice Operations
    # =========================================================================

    def join(self, node_a: LatticeNode, node_b: LatticeNode) -> LatticeNode:
        """
        Lattice join: a ∨ b = union of merged HLLSets.
        
        The result is a new node whose merged HLLSet is the union.
        It is NOT automatically stored — call append() to persist.
        
        Args:
            node_a: First node
            node_b: Second node
            
        Returns:
            New LatticeNode (not yet stored in the lattice)
        """
        merged = node_a.merged.union(node_b.merged)
        registers = merged.dump_numpy()
        node_id = compute_sha1(registers)
        popcount = sum(int(r).bit_count() for r in registers)

        return LatticeNode(
            node_id=node_id,
            timestamp=max(node_a.timestamp, node_b.timestamp),
            merged=merged,
            cardinality=merged.cardinality(),
            popcount=popcount,
            component_ids=node_a.component_ids + node_b.component_ids,
            parent_ids=(node_a.node_id, node_b.node_id),
            metadata={'operation': 'join'},
        )

    def meet(self, node_a: LatticeNode, node_b: LatticeNode) -> LatticeNode:
        """
        Lattice meet: a ∧ b = intersection of merged HLLSets.
        
        The result captures what is common to both nodes.
        
        Args:
            node_a: First node
            node_b: Second node
            
        Returns:
            New LatticeNode (not yet stored in the lattice)
        """
        merged = node_a.merged.intersect(node_b.merged)
        registers = merged.dump_numpy()
        node_id = compute_sha1(registers)
        popcount = sum(int(r).bit_count() for r in registers)

        return LatticeNode(
            node_id=node_id,
            timestamp=min(node_a.timestamp, node_b.timestamp),
            merged=merged,
            cardinality=merged.cardinality(),
            popcount=popcount,
            component_ids=(),
            parent_ids=(node_a.node_id, node_b.node_id),
            metadata={'operation': 'meet'},
        )

    # =========================================================================
    # Temporal Aggregation
    # =========================================================================

    def cumulative(self, t: Optional[float] = None) -> HLLSet:
        """
        Cumulative merge: ∪ M(τ) for all τ ≤ t.
        
        This gives the "everything observed up to time t" HLLSet.
        
        Args:
            t: Time cutoff (default: +∞, i.e., all nodes)
            
        Returns:
            HLLSet representing the union of all nodes up to time t
        """
        hllsets = []
        for ts, nid in self._timeline:
            if t is not None and ts > t:
                break
            node = self._storage.load_node(nid)
            if node is not None:
                hllsets.append(node.merged)

        if not hllsets:
            return HLLSet(p_bits=self._p_bits)

        return HLLSet.merge(hllsets)

    def delta(self, t1: float, t2: float) -> HLLSet:
        """
        Compute what changed between two time points: M(t₂) \\ M(t₁).
        
        "What new information appeared between t₁ and t₂?"
        
        Args:
            t1: Earlier time point
            t2: Later time point
            
        Returns:
            HLLSet of new information (cumulative(t2) \\ cumulative(t1))
        """
        cum_t1 = self.cumulative(t1)
        cum_t2 = self.cumulative(t2)
        return cum_t2.diff(cum_t1)

    def delta_nodes(
        self, node_a: LatticeNode, node_b: LatticeNode
    ) -> HLLSet:
        """
        Compute difference between two specific nodes: M(b) \\ M(a).
        """
        return node_b.merged.diff(node_a.merged)

    # =========================================================================
    # W-Graph (Morphism Structure)
    # =========================================================================

    def build_w_graph(
        self,
        tau_threshold: float = 0.7,
        rho_threshold: float = 0.3,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> Dict:
        """
        Build the W-graph (morphism graph) from lattice nodes.
        
        Uses BSS to determine directed edges between nodes:
            edge i → j exists iff BSS_τ(i→j) ≥ τ and BSS_ρ(i→j) ≤ ρ
        
        This is the "active cognitive model" from §2.3 of the manuscript.
        
        Args:
            tau_threshold: Minimum inclusion for morphism
            rho_threshold: Maximum exclusion for morphism
            t_start: Optional time range start
            t_end: Optional time range end
            
        Returns:
            Morphism graph dict (from bss.morphism_graph)
        """
        if t_start is not None and t_end is not None:
            nodes = self.nodes_in_range(t_start, t_end)
        else:
            nodes = self.all_nodes()

        if not nodes:
            return {
                'edges': [], 'adjacency': {}, 'labels': [],
                'node_count': 0, 'edge_count': 0,
            }

        hllsets = [n.merged for n in nodes]
        labels = [n.node_id[:8] for n in nodes]

        return morphism_graph(
            hllsets,
            tau_threshold=tau_threshold,
            rho_threshold=rho_threshold,
            labels=labels,
        )

    # =========================================================================
    # Conservation Analysis
    # =========================================================================

    def popcount_series(self) -> List[Tuple[float, int]]:
        """
        Time series of popcount (the Noether-conserved quantity).
        
        Returns:
            List of (timestamp, popcount) pairs, ordered by time
        """
        result = []
        for ts, nid in self._timeline:
            node = self._storage.load_node(nid)
            if node is not None:
                result.append((ts, node.popcount))
        return result

    def cardinality_series(self) -> List[Tuple[float, float]]:
        """
        Time series of estimated cardinality.
        
        Returns:
            List of (timestamp, cardinality) pairs, ordered by time
        """
        result = []
        for ts, nid in self._timeline:
            node = self._storage.load_node(nid)
            if node is not None:
                result.append((ts, node.cardinality))
        return result

    # =========================================================================
    # Lattice Partial Order
    # =========================================================================

    def compare(
        self, node_a: LatticeNode, node_b: LatticeNode
    ) -> str:
        """
        Compare two nodes in the lattice partial order.
        
        Returns:
            "a⊆b" if a is subset of b
            "b⊆a" if b is subset of a
            "a=b" if equal
            "a∥b" if incomparable (neither is subset of other)
        """
        a_sub_b = node_a.is_subset_of(node_b)
        b_sub_a = node_b.is_subset_of(node_a)

        if a_sub_b and b_sub_a:
            return "a=b"
        elif a_sub_b:
            return "a⊆b"
        elif b_sub_a:
            return "b⊆a"
        else:
            return "a∥b"

    # =========================================================================
    # Stats
    # =========================================================================

    def stats(self) -> Dict[str, Any]:
        """Lattice statistics."""
        nodes = self.all_nodes()
        if not nodes:
            return {
                'node_count': 0,
                'p_bits': self._p_bits,
                'time_span': 0.0,
            }

        cards = [n.cardinality for n in nodes]
        pops = [n.popcount for n in nodes]
        times = [n.timestamp for n in nodes]

        return {
            'node_count': len(nodes),
            'p_bits': self._p_bits,
            'time_span': max(times) - min(times) if len(times) > 1 else 0.0,
            'cardinality_range': (min(cards), max(cards)),
            'popcount_range': (min(pops), max(pops)),
            'mean_cardinality': float(np.mean(cards)),
            'mean_popcount': float(np.mean(pops)),
        }

    def __repr__(self) -> str:
        return (
            f"HLLLattice(p={self._p_bits}, nodes={self.node_count})"
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'LatticeNode',
    'LatticeStorage',
    'InMemoryStorage',
    'HLLLattice',
]
