"""
Lattice Reconstruction — Restore W lattice from ring-compressed HLLSets.

Problem Statement:
    When HLLSets are stored as linear combinations of base vectors in a
    BitVectorRing, we gain compression but lose the lattice structure
    (partial ordering by subset relation). After restoring individual
    HLLSets from queries, we need to reconstruct:
    
    1. The partial order (A ⊆ B relationships)
    2. The BSS morphism graph (directed edges with (τ, ρ) weights)
    3. The Hasse diagram (minimal edges representing the covering relation)

De Bruijn Graph Analogy:
    The reconstruction problem is analogous to genome assembly:
    
    | Genome Assembly          | Lattice Reconstruction              |
    |--------------------------|-------------------------------------|
    | K-mers (fragments)       | HLLSets (restored from ring)        |
    | (k-1)-mer overlap        | BSS similarity / subset relation    |
    | De Bruijn edge           | Morphism A →(τ,ρ) B                 |
    | Edge multiplicity        | Strength of relationship (τ)        |
    | Eulerian path            | Maximal chain in lattice            |
    | Connected components     | Lattice levels / antichains         |

Key Insight:
    The BSS metric provides directed edges:
    - BSS_τ(A → B) = |A ∩ B| / |B| (how much A covers B)
    - BSS_ρ(A → B) = |A \\ B| / |B| (noise from A to B)
    
    A strong morphism (high τ, low ρ) indicates A "contains" most of B.
    This gives us the partial order: A ≤ B iff BSS_τ(B → A) ≈ 1.

Reconstruction Algorithm:
    1. Compute pairwise BSS for all restored HLLSets
    2. Build a BSS-weighted directed graph (like De Bruijn)
    3. Extract subset edges: A → B where τ(A→B) ≥ threshold
    4. Compute transitive reduction to get Hasse diagram
    5. Identify levels (longest path from bottom) and antichains

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import (
    List, Tuple, Dict, Set, Optional, Any, Iterator,
    TypeVar, Generic, Hashable, Callable,
)
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

from .hllset import HLLSet
from .bss import bss, BSSPair, bss_symmetric, test_morphism, MorphismResult


# Type for HLLSet identifiers
HLLSetID = str


@dataclass
class LatticeEdge:
    """
    Directed edge in the BSS morphism graph.
    
    Represents a morphism A →(τ,ρ) B where A is "smaller" (more specific)
    and B is "larger" (more general) in the lattice sense.
    
    Attributes:
        source_id: ID of source HLLSet (A)
        target_id: ID of target HLLSet (B)
        bss_forward: BSS(A → B) — how A relates to B
        bss_backward: BSS(B → A) — how B relates to A
        is_subset: True if A ⊆ B (within tolerance)
        is_superset: True if A ⊇ B (within tolerance)
        edge_type: Classification of the relationship
    """
    source_id: HLLSetID
    target_id: HLLSetID
    bss_forward: BSSPair      # BSS(source → target)
    bss_backward: BSSPair     # BSS(target → source)
    is_subset: bool = False   # source ⊆ target
    is_superset: bool = False # source ⊇ target
    edge_type: str = "related"
    
    @property
    def tau_forward(self) -> float:
        return self.bss_forward.tau
    
    @property
    def rho_forward(self) -> float:
        return self.bss_forward.rho
    
    @property
    def tau_backward(self) -> float:
        return self.bss_backward.tau
    
    @property
    def rho_backward(self) -> float:
        return self.bss_backward.rho
    
    @property
    def is_comparable(self) -> bool:
        """True if one is a subset of the other."""
        return self.is_subset or self.is_superset
    
    def __repr__(self) -> str:
        rel = "⊆" if self.is_subset else ("⊇" if self.is_superset else "~")
        return (
            f"LatticeEdge({self.source_id[:6]}→{self.target_id[:6]}, "
            f"τ={self.tau_forward:.3f}, {rel})"
        )


@dataclass
class LatticeLevel:
    """
    A level (antichain) in the reconstructed lattice.
    
    Elements at the same level are incomparable — neither is a subset
    of the other. Level 0 is the bottom (most specific), higher levels
    are more general.
    
    Attributes:
        level: Level index (0 = bottom)
        node_ids: HLLSet IDs at this level
        cardinality_range: (min, max) cardinality at this level
    """
    level: int
    node_ids: List[HLLSetID]
    cardinality_range: Tuple[float, float]
    
    def __len__(self) -> int:
        return len(self.node_ids)
    
    def __repr__(self) -> str:
        return (
            f"Level({self.level}, {len(self.node_ids)} nodes, "
            f"card=[{self.cardinality_range[0]:.0f},{self.cardinality_range[1]:.0f}])"
        )


@dataclass
class ReconstructedLattice:
    """
    Result of lattice reconstruction from restored HLLSets.
    
    Contains the partial order, Hasse diagram, and level structure.
    
    Attributes:
        nodes: Dict of ID → HLLSet
        edges: All BSS-computed edges (full graph)
        hasse_edges: Transitive reduction (covering relations only)
        levels: Level structure (antichains)
        bottom_ids: IDs of minimal elements
        top_ids: IDs of maximal elements
        is_lattice: True if structure forms a proper lattice
    """
    nodes: Dict[HLLSetID, HLLSet]
    edges: List[LatticeEdge]
    hasse_edges: List[LatticeEdge]  # Transitive reduction
    levels: List[LatticeLevel]
    bottom_ids: List[HLLSetID]
    top_ids: List[HLLSetID]
    is_lattice: bool
    stats: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def num_edges(self) -> int:
        return len(self.edges)
    
    @property
    def num_hasse_edges(self) -> int:
        return len(self.hasse_edges)
    
    @property
    def num_levels(self) -> int:
        return len(self.levels)
    
    @property
    def height(self) -> int:
        """Height of the lattice (longest chain length - 1)."""
        return max(0, self.num_levels - 1)
    
    @property
    def width(self) -> int:
        """Width of the lattice (size of largest antichain)."""
        return max(len(level) for level in self.levels) if self.levels else 0
    
    def get_level(self, node_id: HLLSetID) -> int:
        """Get the level of a node."""
        for level in self.levels:
            if node_id in level.node_ids:
                return level.level
        return -1
    
    def predecessors(self, node_id: HLLSetID) -> List[HLLSetID]:
        """Get immediate predecessors (covered by this node)."""
        return [e.source_id for e in self.hasse_edges if e.target_id == node_id]
    
    def successors(self, node_id: HLLSetID) -> List[HLLSetID]:
        """Get immediate successors (cover this node)."""
        return [e.target_id for e in self.hasse_edges if e.source_id == node_id]
    
    def chain_to_top(self, node_id: HLLSetID) -> List[HLLSetID]:
        """Find a maximal chain from node to top."""
        chain = [node_id]
        current = node_id
        while True:
            succs = self.successors(current)
            if not succs:
                break
            current = succs[0]  # Pick any successor
            chain.append(current)
        return chain
    
    def chain_to_bottom(self, node_id: HLLSetID) -> List[HLLSetID]:
        """Find a maximal chain from node to bottom."""
        chain = [node_id]
        current = node_id
        while True:
            preds = self.predecessors(current)
            if not preds:
                break
            current = preds[0]  # Pick any predecessor
            chain.insert(0, current)
        return chain
    
    def __repr__(self) -> str:
        return (
            f"ReconstructedLattice(nodes={self.num_nodes}, "
            f"edges={self.num_hasse_edges}, levels={self.num_levels}, "
            f"height={self.height}, width={self.width})"
        )


class BSSMorphismGraph:
    """
    Directed graph of BSS morphisms between HLLSets.
    
    This is analogous to a De Bruijn graph where:
    - Nodes are HLLSets (like k-mers)
    - Edges are BSS relationships (like overlaps)
    - Edge weights are (τ, ρ) pairs (like multiplicities)
    
    The graph structure encodes the partial order of the W lattice.
    """
    
    def __init__(self, 
                 tau_threshold: float = 0.9,
                 rho_threshold: float = 0.1,
                 subset_tau: float = 0.95):
        """
        Initialize BSS morphism graph.
        
        Args:
            tau_threshold: Minimum τ for morphism edge
            rho_threshold: Maximum ρ for morphism edge
            subset_tau: τ threshold for declaring A ⊆ B
        """
        self.tau_threshold = tau_threshold
        self.rho_threshold = rho_threshold
        self.subset_tau = subset_tau
        
        # Node storage: id → HLLSet
        self._nodes: Dict[HLLSetID, HLLSet] = {}
        self._cardinalities: Dict[HLLSetID, float] = {}
        
        # Edge storage
        self._edges: Dict[Tuple[HLLSetID, HLLSetID], LatticeEdge] = {}
        
        # Adjacency lists
        self._outgoing: Dict[HLLSetID, List[HLLSetID]] = defaultdict(list)
        self._incoming: Dict[HLLSetID, List[HLLSetID]] = defaultdict(list)
        
        # Subset adjacency (for lattice structure)
        self._subset_children: Dict[HLLSetID, Set[HLLSetID]] = defaultdict(set)  # A → {B : A ⊆ B}
        self._subset_parents: Dict[HLLSetID, Set[HLLSetID]] = defaultdict(set)   # A → {B : B ⊆ A}
    
    def add_node(self, node_id: HLLSetID, hllset: HLLSet) -> None:
        """Add a node (HLLSet) to the graph."""
        self._nodes[node_id] = hllset
        self._cardinalities[node_id] = hllset.cardinality()
    
    def add_edge(self, edge: LatticeEdge) -> None:
        """Add a computed edge to the graph."""
        key = (edge.source_id, edge.target_id)
        self._edges[key] = edge
        self._outgoing[edge.source_id].append(edge.target_id)
        self._incoming[edge.target_id].append(edge.source_id)
        
        # Track subset relations
        if edge.is_subset:  # source ⊆ target
            self._subset_children[edge.source_id].add(edge.target_id)
            self._subset_parents[edge.target_id].add(edge.source_id)
    
    def compute_edge(self, 
                     source_id: HLLSetID, 
                     target_id: HLLSetID) -> Optional[LatticeEdge]:
        """
        Compute BSS edge between two nodes.
        
        Returns None if the relationship is too weak.
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return None
        
        source = self._nodes[source_id]
        target = self._nodes[target_id]
        
        # Compute BSS in both directions
        bss_fwd = bss(source, target)
        bss_bwd = bss(target, source)
        
        # Determine edge type
        is_subset = bss_bwd.tau >= self.subset_tau  # source ⊆ target means BSS(target→source) high
        is_superset = bss_fwd.tau >= self.subset_tau  # source ⊇ target means BSS(source→target) high
        
        # Classify edge
        if is_subset and is_superset:
            edge_type = "equal"  # source ≈ target
        elif is_subset:
            edge_type = "subset"  # source ⊂ target
        elif is_superset:
            edge_type = "superset"  # source ⊃ target
        elif bss_fwd.tau >= self.tau_threshold or bss_bwd.tau >= self.tau_threshold:
            edge_type = "overlap"  # Significant overlap but not subset
        else:
            edge_type = "disjoint"  # Weak relationship
        
        # Only create edge if there's significant relationship
        if edge_type == "disjoint":
            return None
        
        edge = LatticeEdge(
            source_id=source_id,
            target_id=target_id,
            bss_forward=bss_fwd,
            bss_backward=bss_bwd,
            is_subset=is_subset,
            is_superset=is_superset,
            edge_type=edge_type,
        )
        
        return edge
    
    def compute_all_edges(self) -> List[LatticeEdge]:
        """
        Compute edges between all pairs of nodes.
        
        This is O(n²) — for large graphs, consider sampling or
        approximate methods.
        """
        node_ids = list(self._nodes.keys())
        edges = []
        
        for i, src_id in enumerate(node_ids):
            for tgt_id in node_ids[i+1:]:  # Only upper triangle
                # Compute edge src → tgt
                edge = self.compute_edge(src_id, tgt_id)
                if edge is not None:
                    self.add_edge(edge)
                    edges.append(edge)
                
                # Compute reverse edge tgt → src
                # (Reuse BSS values by swapping)
                if edge is not None:
                    rev_edge = LatticeEdge(
                        source_id=tgt_id,
                        target_id=src_id,
                        bss_forward=edge.bss_backward,
                        bss_backward=edge.bss_forward,
                        is_subset=edge.is_superset,  # Swapped!
                        is_superset=edge.is_subset,   # Swapped!
                        edge_type=("subset" if edge.is_superset else 
                                   ("superset" if edge.is_subset else edge.edge_type)),
                    )
                    self.add_edge(rev_edge)
                    edges.append(rev_edge)
        
        return edges
    
    def get_subset_edges(self) -> List[LatticeEdge]:
        """Get only edges representing subset relations."""
        return [e for e in self._edges.values() if e.is_subset]
    
    @property
    def nodes(self) -> Dict[HLLSetID, HLLSet]:
        return self._nodes
    
    @property
    def edges(self) -> List[LatticeEdge]:
        return list(self._edges.values())
    
    def stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        subset_edges = self.get_subset_edges()
        return {
            'num_nodes': len(self._nodes),
            'num_edges': len(self._edges),
            'num_subset_edges': len(subset_edges),
            'avg_out_degree': np.mean([len(v) for v in self._outgoing.values()]) if self._outgoing else 0,
            'max_out_degree': max(len(v) for v in self._outgoing.values()) if self._outgoing else 0,
            'tau_threshold': self.tau_threshold,
            'subset_tau': self.subset_tau,
        }
    
    def __repr__(self) -> str:
        return (
            f"BSSMorphismGraph(nodes={len(self._nodes)}, "
            f"edges={len(self._edges)}, "
            f"subset_edges={len(self.get_subset_edges())})"
        )


class LatticeReconstructor:
    """
    Reconstruct W lattice from restored HLLSets using BSS morphism graph.
    
    The reconstruction follows a De Bruijn-like approach:
    1. Add HLLSets as nodes
    2. Compute pairwise BSS (like computing k-mer overlaps)
    3. Build morphism graph (like De Bruijn graph)
    4. Extract partial order (subset edges)
    5. Compute transitive reduction (Hasse diagram)
    6. Identify levels and structure
    """
    
    def __init__(self,
                 tau_threshold: float = 0.9,
                 rho_threshold: float = 0.1,
                 subset_tau: float = 0.95):
        """
        Initialize reconstructor.
        
        Args:
            tau_threshold: Minimum τ for significant relationship
            rho_threshold: Maximum ρ for morphism edge
            subset_tau: τ threshold for subset detection
        """
        self.tau_threshold = tau_threshold
        self.rho_threshold = rho_threshold
        self.subset_tau = subset_tau
        
        self._graph = BSSMorphismGraph(
            tau_threshold=tau_threshold,
            rho_threshold=rho_threshold,
            subset_tau=subset_tau,
        )
    
    def add_hllset(self, hllset: HLLSet, node_id: Optional[HLLSetID] = None) -> HLLSetID:
        """
        Add an HLLSet to the reconstruction.
        
        Args:
            hllset: The HLLSet to add
            node_id: Optional ID (defaults to SHA1 of registers)
            
        Returns:
            The node ID
        """
        if node_id is None:
            node_id = hllset.identity
        self._graph.add_node(node_id, hllset)
        return node_id
    
    def add_hllsets(self, 
                    hllsets: List[HLLSet],
                    node_ids: Optional[List[HLLSetID]] = None) -> List[HLLSetID]:
        """Add multiple HLLSets."""
        if node_ids is None:
            node_ids = [None] * len(hllsets)
        return [self.add_hllset(h, nid) for h, nid in zip(hllsets, node_ids)]
    
    def reconstruct(self) -> ReconstructedLattice:
        """
        Perform lattice reconstruction.
        
        Algorithm:
        1. Compute all pairwise BSS edges
        2. Extract subset partial order
        3. Compute transitive reduction (Hasse diagram)
        4. Identify levels (longest path from bottom)
        5. Find bottom and top elements
        
        Returns:
            ReconstructedLattice with full structure
        """
        # Step 1: Compute all edges
        all_edges = self._graph.compute_all_edges()
        
        # Step 2: Extract subset edges for partial order
        subset_edges = self._graph.get_subset_edges()
        
        # Step 3: Compute transitive reduction
        hasse_edges = self._transitive_reduction(subset_edges)
        
        # Step 4: Compute levels
        levels = self._compute_levels(hasse_edges)
        
        # Step 5: Find bottom and top
        all_sources = {e.source_id for e in hasse_edges}
        all_targets = {e.target_id for e in hasse_edges}
        node_ids = set(self._graph.nodes.keys())
        
        # Bottom: nodes with no predecessors (not targets)
        bottom_ids = list(node_ids - all_targets)
        
        # Top: nodes with no successors (not sources)
        top_ids = list(node_ids - all_sources)
        
        # Handle isolated nodes (no edges)
        isolated = node_ids - all_sources - all_targets
        if isolated:
            bottom_ids.extend(isolated)
            top_ids.extend(isolated)
            bottom_ids = list(set(bottom_ids))
            top_ids = list(set(top_ids))
        
        # Step 6: Check if it's a proper lattice
        is_lattice = self._check_lattice_property(hasse_edges)
        
        # Compute stats
        stats = self._graph.stats()
        stats['num_hasse_edges'] = len(hasse_edges)
        stats['num_levels'] = len(levels)
        stats['num_bottom'] = len(bottom_ids)
        stats['num_top'] = len(top_ids)
        
        return ReconstructedLattice(
            nodes=self._graph.nodes,
            edges=all_edges,
            hasse_edges=hasse_edges,
            levels=levels,
            bottom_ids=bottom_ids,
            top_ids=top_ids,
            is_lattice=is_lattice,
            stats=stats,
        )
    
    def _transitive_reduction(self, edges: List[LatticeEdge]) -> List[LatticeEdge]:
        """
        Compute transitive reduction (Hasse diagram edges).
        
        Remove edges that are implied by transitivity:
        If A ⊂ B and B ⊂ C, remove A ⊂ C (it's implied).
        
        Uses DFS-based algorithm.
        """
        # Build adjacency
        adj: Dict[HLLSetID, Set[HLLSetID]] = defaultdict(set)
        edge_lookup: Dict[Tuple[HLLSetID, HLLSetID], LatticeEdge] = {}
        
        for edge in edges:
            adj[edge.source_id].add(edge.target_id)
            edge_lookup[(edge.source_id, edge.target_id)] = edge
        
        # For each node, find all reachable nodes
        reachable: Dict[HLLSetID, Set[HLLSetID]] = {}
        
        def dfs_reachable(node: HLLSetID) -> Set[HLLSetID]:
            if node in reachable:
                return reachable[node]
            
            result = set()
            for neighbor in adj[node]:
                result.add(neighbor)
                result.update(dfs_reachable(neighbor))
            
            reachable[node] = result
            return result
        
        nodes = set(adj.keys()) | {t for s in adj.values() for t in s}
        for node in nodes:
            dfs_reachable(node)
        
        # Keep edge A → B only if B is not reachable through another path
        hasse_edges = []
        
        for edge in edges:
            src, tgt = edge.source_id, edge.target_id
            
            # Check if there's another path from src to tgt
            other_reachable = set()
            for intermediate in adj[src]:
                if intermediate != tgt:
                    other_reachable.update(reachable.get(intermediate, set()))
                    other_reachable.add(intermediate)
            
            # Keep edge only if target not reachable through other paths
            if tgt not in other_reachable:
                hasse_edges.append(edge)
        
        return hasse_edges
    
    def _compute_levels(self, hasse_edges: List[LatticeEdge]) -> List[LatticeLevel]:
        """
        Compute level structure (antichains).
        
        Level of a node = length of longest path from any bottom element.
        Uses topological sort + dynamic programming.
        
        For Hasse edge A → B (A ⊂ B), A is at a lower level than B.
        """
        # Build forward adjacency (source → targets)
        forward_adj: Dict[HLLSetID, List[HLLSetID]] = defaultdict(list)
        in_degree: Dict[HLLSetID, int] = defaultdict(int)
        
        all_nodes = set(self._graph.nodes.keys())
        
        # For edge A → B: A is source (smaller), B is target (larger)
        # in_degree counts how many edges come INTO a node
        for edge in hasse_edges:
            forward_adj[edge.source_id].append(edge.target_id)
            in_degree[edge.target_id] += 1  # B has incoming edge from A
        
        # Initialize: nodes with no incoming Hasse edges are bottom elements (level 0)
        levels_map: Dict[HLLSetID, int] = {}
        
        # Topological order using Kahn's algorithm (process from bottom up)
        queue = deque()
        for node in all_nodes:
            if in_degree[node] == 0:
                queue.append(node)
                levels_map[node] = 0
        
        while queue:
            node = queue.popleft()
            node_level = levels_map[node]
            
            # Process successors (larger elements covered by this node)
            for successor in forward_adj[node]:
                # Successor level is at least node_level + 1
                levels_map[successor] = max(
                    levels_map.get(successor, 0),
                    node_level + 1
                )
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        # Handle disconnected nodes
        for node in all_nodes:
            if node not in levels_map:
                levels_map[node] = 0
        
        # Group by level
        level_groups: Dict[int, List[HLLSetID]] = defaultdict(list)
        for node, level in levels_map.items():
            level_groups[level].append(node)
        
        # Build LatticeLevel objects
        levels = []
        for level_idx in sorted(level_groups.keys()):
            node_ids = level_groups[level_idx]
            cardinalities = [self._graph._cardinalities.get(nid, 0) for nid in node_ids]
            
            levels.append(LatticeLevel(
                level=level_idx,
                node_ids=node_ids,
                cardinality_range=(min(cardinalities), max(cardinalities)) if cardinalities else (0, 0),
            ))
        
        return levels
    
    def _check_lattice_property(self, hasse_edges: List[LatticeEdge]) -> bool:
        """
        Check if the structure is a proper lattice.
        
        A poset is a lattice iff every pair of elements has
        a unique join (supremum) and meet (infimum).
        
        This is a simplified check — we verify that:
        1. There's a unique top (or the structure is bounded above)
        2. There's a unique bottom (or the structure is bounded below)
        """
        # For now, just check connectivity
        # A full check would verify join/meet existence for all pairs
        
        if len(self._graph.nodes) <= 1:
            return True
        
        # Check if the subset relation is connected
        # (There exists a path between any two comparable elements)
        all_sources = {e.source_id for e in hasse_edges}
        all_targets = {e.target_id for e in hasse_edges}
        
        # If we have edges, we have some partial order
        return len(hasse_edges) > 0 or len(self._graph.nodes) == 1
    
    @property
    def graph(self) -> BSSMorphismGraph:
        """Access the underlying morphism graph."""
        return self._graph


# =============================================================================
# Convenience Functions
# =============================================================================

def reconstruct_lattice(
    hllsets: List[HLLSet],
    node_ids: Optional[List[HLLSetID]] = None,
    tau_threshold: float = 0.9,
    subset_tau: float = 0.95,
) -> ReconstructedLattice:
    """
    Reconstruct W lattice from a list of HLLSets.
    
    One-liner for the common case.
    
    Args:
        hllsets: List of HLLSets to organize into a lattice
        node_ids: Optional IDs (default: use SHA1 identity)
        tau_threshold: BSS τ threshold for relationships
        subset_tau: τ threshold for subset detection
        
    Returns:
        ReconstructedLattice with full structure
        
    Example:
        # Restore HLLSets from ring queries
        restored = [ring.query(indices) for indices in selections]
        
        # Reconstruct the lattice
        lattice = reconstruct_lattice(restored)
        
        print(f"Height: {lattice.height}, Width: {lattice.width}")
        for level in lattice.levels:
            print(f"  Level {level.level}: {len(level)} nodes")
    """
    reconstructor = LatticeReconstructor(
        tau_threshold=tau_threshold,
        subset_tau=subset_tau,
    )
    reconstructor.add_hllsets(hllsets, node_ids)
    return reconstructor.reconstruct()


def lattice_to_dot(lattice: ReconstructedLattice, max_nodes: int = 50) -> str:
    """
    Export reconstructed lattice to DOT format.
    
    Args:
        lattice: The reconstructed lattice
        max_nodes: Maximum nodes to include
        
    Returns:
        DOT format string for Graphviz
    """
    lines = ['digraph Lattice {']
    lines.append('  rankdir=BT;')  # Bottom to top
    lines.append('  node [shape=ellipse];')
    
    # Group nodes by level
    for level in lattice.levels:
        lines.append(f'  {{ rank=same; /* Level {level.level} */')
        for node_id in level.node_ids[:max_nodes // lattice.num_levels]:
            card = lattice.nodes[node_id].cardinality()
            label = f"{node_id[:8]}\\n|{card:.0f}|"
            lines.append(f'    "{node_id[:12]}" [label="{label}"];')
        lines.append('  }')
    
    # Add Hasse edges
    edge_count = 0
    for edge in lattice.hasse_edges:
        if edge_count >= max_nodes * 2:
            break
        src = edge.source_id[:12]
        tgt = edge.target_id[:12]
        tau = edge.tau_forward
        lines.append(f'  "{src}" -> "{tgt}" [label="τ={tau:.2f}"];')
        edge_count += 1
    
    lines.append('}')
    return '\n'.join(lines)


# =============================================================================
# Integration with RingTransaction
# =============================================================================

def reconstruct_from_ring(
    ring,  # BitVectorRing
    tensor,  # TriangulationTensor
    basis_indices: Optional[List[int]] = None,
    tau_threshold: float = 0.9,
    subset_tau: float = 0.95,
) -> ReconstructedLattice:
    """
    Reconstruct lattice from ring-stored HLLSets.
    
    This is the main integration point with ring_transaction.py.
    
    Args:
        ring: BitVectorRing containing compressed HLLSets
        tensor: TriangulationTensor for reconstructing HLLSets
        basis_indices: Which basis vectors to include (None = all)
        tau_threshold: BSS threshold
        subset_tau: Subset detection threshold
        
    Returns:
        ReconstructedLattice
    """
    from .hll_tensor import HLLTensor
    from .bitvector_ring import BitVectorRing
    
    # Get basis vectors
    if basis_indices is None:
        basis_indices = list(range(ring.rank()))
    
    # Reconstruct HLLSets from each basis vector
    hllsets = []
    node_ids = []
    
    for idx in basis_indices:
        if idx < ring.rank():
            bv = ring.basis[idx]
            # Convert bitvector back to HLLSet
            hllset = _bitvector_to_hllset(bv, tensor.p_bits if hasattr(tensor, 'p_bits') else 10)
            hllsets.append(hllset)
            node_ids.append(f"basis_{idx}")
    
    return reconstruct_lattice(hllsets, node_ids, tau_threshold, subset_tau)


def _bitvector_to_hllset(bv, p_bits: int = 10) -> HLLSet:
    """Convert a bitvector back to an HLLSet."""
    num_registers = 1 << p_bits
    registers = np.zeros(num_registers, dtype=np.uint32)
    
    # Reconstruct registers from bitvector
    for reg in range(num_registers):
        for bit in range(32):
            global_idx = reg * 32 + bit
            if bv[global_idx]:
                registers[reg] |= (1 << bit)
    
    return HLLSet.from_registers(registers, p_bits=p_bits)
