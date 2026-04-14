"""
HLLSet De Bruijn Graph — Two-Level Sequence & Evolution Recovery

This module provides a unified architecture for:

1. **Token-Level De Bruijn** (classical):
   - Nodes = bigrams, Edges = trigrams
   - Eulerian path → original token order

2. **HLLSet-Level De Bruijn** (novel):
   - Nodes = HLLSet states
   - Edges = (D, R, N) transformation triples
   - Path reconstruction → evolution order

The key insight: D, R, N are themselves HLLSets with trigram structure,
so they support full token recovery. This makes HLLSet lattices **fully invertible**.

Architecture:
```
    HLLSet Collection (unordered)
              │
              ▼ BSS adjacency graph
    ┌─────────────────────────┐
    │  HLLSet De Bruijn Graph │
    │    Nodes = HLLSet states│
    │    Edges = (D, R, N)    │
    └─────────────────────────┘
              │
              ▼ Find evolution path
    ┌─────────────────────────┐
    │  Ordered Lattice        │
    │    H₀ → H₁ → H₂ → ...   │
    └─────────────────────────┘
              │
              ▼ For each edge (D, R, N)
    ┌─────────────────────────┐
    │  Token-Level De Bruijn  │
    │    D → deleted tokens   │
    │    R → retained tokens  │
    │    N → novel tokens     │
    └─────────────────────────┘
```

IICA Properties:
- **Immutable**: All results are frozen dataclasses
- **Idempotent**: Same inputs → same SHA1 → same results
- **Composable**: Every function returns HLLSet-compatible types
- **Algebraic**: D, R, N support full ring/lattice operations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, List, Dict, Tuple, Optional, Set
from enum import Enum

from .hllset import HLLSet
from .bss import bss, BSSPair
from .disambiguation import DisambiguationEngine


# =============================================================================
# HLLSet-Level (D, R, N) Transformation Triple
# =============================================================================

class DRNTriple(NamedTuple):
    """
    Transformation triple for HLLSet transition: hll_1 → hll_2.
    
    Decomposition: hll_2 = (hll_1 \\ D) ∪ R ∪ N
    
    Attributes:
        deleted: D — tokens removed from hll_1 (hll_1 \\ hll_2)
        retained: R — tokens preserved (hll_1 ∩ hll_2)
        novel: N — tokens added to hll_2 (hll_2 \\ hll_1)
    
    Each component is an HLLSet, supporting:
        - Algebraic operations (union, intersect, diff)
        - Cardinality estimation
        - Token recovery via De Bruijn (if trigrams stored)
    """
    deleted: HLLSet
    retained: HLLSet
    novel: HLLSet
    
    @property
    def deleted_card(self) -> float:
        return self.deleted.cardinality()
    
    @property
    def retained_card(self) -> float:
        return self.retained.cardinality()
    
    @property
    def novel_card(self) -> float:
        return self.novel.cardinality()
    
    def is_growth(self) -> bool:
        """True if more tokens added than deleted (forward evolution)."""
        return self.novel_card > self.deleted_card
    
    def is_decay(self) -> bool:
        """True if more tokens deleted than added (backward evolution)."""
        return self.deleted_card > self.novel_card
    
    def net_change(self) -> float:
        """Net cardinality change: |N| - |D|."""
        return self.novel_card - self.deleted_card


@dataclass(frozen=True)
class FullDRNTriple:
    """
    Full invertible transformation with token recovery.
    
    Each component includes:
        - HLLSet (for algebraic operations)
        - Original tokens in order (for verification)
        - DisambiguationEngine (for De Bruijn recovery)
    
    This enables LOSSLESS round-trip: HLLSet → tokens → HLLSet
    """
    deleted_hll: HLLSet
    deleted_tokens: Tuple[str, ...]
    
    retained_hll: HLLSet
    retained_tokens: Tuple[str, ...]
    
    novel_hll: HLLSet
    novel_tokens: Tuple[str, ...]
    
    @property
    def drn(self) -> DRNTriple:
        """Get the basic DRN triple (HLLSets only)."""
        return DRNTriple(self.deleted_hll, self.retained_hll, self.novel_hll)


# =============================================================================
# HLLSet-Level De Bruijn Graph
# =============================================================================

@dataclass(frozen=True)
class HLLSetEdge:
    """
    Directed edge in the HLLSet De Bruijn graph.
    
    Represents a valid transition from source → target HLLSet,
    with the (D, R, N) triple as the edge label.
    """
    source_idx: int
    target_idx: int
    bss_pair: BSSPair
    drn: DRNTriple
    
    @property
    def tau(self) -> float:
        return self.bss_pair.tau
    
    @property
    def rho(self) -> float:
        return self.bss_pair.rho


@dataclass
class HLLSetDeBruijnGraph:
    """
    De Bruijn-like graph over a collection of HLLSets.
    
    Nodes: HLLSet states (indexed by position in input list)
    Edges: (D, R, N) triples where BSS meets adjacency threshold
    
    Use cases:
        - Recover evolution order from unordered HLLSet collection
        - Compute transformation history between states
        - Find similar/related HLLSets via graph traversal
    """
    nodes: List[HLLSet]
    labels: List[str]
    edges: List[HLLSetEdge]
    tau_min: float
    rho_max: float
    p_bits: int
    
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def num_edges(self) -> int:
        return len(self.edges)
    
    def out_edges(self, node_idx: int) -> List[HLLSetEdge]:
        """Get all outgoing edges from a node."""
        return [e for e in self.edges if e.source_idx == node_idx]
    
    def in_edges(self, node_idx: int) -> List[HLLSetEdge]:
        """Get all incoming edges to a node."""
        return [e for e in self.edges if e.target_idx == node_idx]
    
    def out_degree(self, node_idx: int) -> int:
        return len(self.out_edges(node_idx))
    
    def in_degree(self, node_idx: int) -> int:
        return len(self.in_edges(node_idx))
    
    def adjacency_list(self) -> Dict[int, List[Tuple[int, float]]]:
        """Get adjacency list: node → [(neighbor, tau), ...]"""
        adj = {i: [] for i in range(self.num_nodes)}
        for e in self.edges:
            adj[e.source_idx].append((e.target_idx, e.tau))
        return adj
    
    def to_dot(self, title: str = "HLLSetDeBruijn") -> str:
        """Generate DOT graph for visualization."""
        lines = [
            f"digraph {title} {{",
            "  rankdir=LR;",
            "  node [shape=ellipse];",
        ]
        
        # Add nodes
        for i, (hll, label) in enumerate(zip(self.nodes, self.labels)):
            card = hll.cardinality()
            lines.append(f'  H{i} [label="{label}\\n|H|≈{card:.0f}"];')
        
        # Add edges with (D, R, N) labels
        for e in self.edges:
            d, r, n = e.drn.deleted_card, e.drn.retained_card, e.drn.novel_card
            label = f"τ={e.tau:.2f}\\nD:{d:.0f} R:{r:.0f} N:{n:.0f}"
            lines.append(f'  H{e.source_idx} -> H{e.target_idx} [label="{label}"];')
        
        lines.append("}")
        return "\n".join(lines)


# =============================================================================
# Core Functions
# =============================================================================

def decompose_transition(hll_1: HLLSet, hll_2: HLLSet) -> DRNTriple:
    """
    Compute (D, R, N) triple for transition hll_1 → hll_2.
    
    Args:
        hll_1: Source HLLSet
        hll_2: Target HLLSet
    
    Returns:
        DRNTriple with:
            D = hll_1 \\ hll_2 (deleted)
            R = hll_1 ∩ hll_2 (retained)
            N = hll_2 \\ hll_1 (novel)
    
    Invariant: hll_2 ≈ (hll_1 \\ D) ∪ R ∪ N
    """
    retained = hll_1.intersect(hll_2)
    deleted = hll_1.diff(hll_2)
    novel = hll_2.diff(hll_1)
    return DRNTriple(deleted, retained, novel)


def full_decompose_transition(
    tokens_1: List[str],
    tokens_2: List[str],
    p_bits: int = 10
) -> FullDRNTriple:
    """
    Compute full (D, R, N) triple with token recovery support.
    
    This preserves the original token ORDER within each component,
    enabling De Bruijn recovery of the actual tokens (not just HLLSets).
    
    Args:
        tokens_1: Source token sequence
        tokens_2: Target token sequence
        p_bits: Precision bits for HLLSets
    
    Returns:
        FullDRNTriple with HLLSets and ordered token lists
    """
    set_1 = set(tokens_1)
    set_2 = set(tokens_2)
    
    # Compute token sets
    deleted_set = set_1 - set_2
    retained_set = set_1 & set_2
    novel_set = set_2 - set_1
    
    # Preserve original order by filtering from source sequences
    deleted_ordered = tuple(t for t in tokens_1 if t in deleted_set)
    retained_ordered = tuple(t for t in tokens_1 if t in retained_set)
    novel_ordered = tuple(t for t in tokens_2 if t in novel_set)
    
    # Create HLLSets
    deleted_hll = HLLSet.from_batch(deleted_ordered, p_bits=p_bits) if deleted_ordered else HLLSet.empty(p_bits)
    retained_hll = HLLSet.from_batch(retained_ordered, p_bits=p_bits) if retained_ordered else HLLSet.empty(p_bits)
    novel_hll = HLLSet.from_batch(novel_ordered, p_bits=p_bits) if novel_ordered else HLLSet.empty(p_bits)
    
    return FullDRNTriple(
        deleted_hll, deleted_ordered,
        retained_hll, retained_ordered,
        novel_hll, novel_ordered
    )


def verify_reconstruction(hll_1: HLLSet, hll_2: HLLSet, drn: DRNTriple, tolerance: float = 2.0) -> bool:
    """
    Verify that hll_2 ≈ (hll_1 \\ D) ∪ R ∪ N.
    
    Args:
        hll_1: Source HLLSet
        hll_2: Target HLLSet
        drn: The (D, R, N) triple
        tolerance: Cardinality difference tolerance
    
    Returns:
        True if reconstruction is within tolerance
    """
    step1 = hll_1.diff(drn.deleted)
    step2 = step1.union(drn.retained)
    reconstructed = step2.union(drn.novel)
    return abs(reconstructed.cardinality() - hll_2.cardinality()) < tolerance


def build_hllset_debruijn(
    hllsets: List[HLLSet],
    labels: Optional[List[str]] = None,
    tau_min: float = 0.2,
    rho_max: float = 0.8,
    p_bits: int = 10
) -> HLLSetDeBruijnGraph:
    """
    Build a De Bruijn-like graph over a collection of HLLSets.
    
    Edges are created where BSS(source → target) meets the thresholds:
        τ ≥ tau_min (sufficient overlap)
        ρ ≤ rho_max (not too much extraneous)
    
    Args:
        hllsets: List of HLLSets (nodes)
        labels: Optional labels for each HLLSet
        tau_min: Minimum τ for edge creation
        rho_max: Maximum ρ for edge creation
        p_bits: Precision bits
    
    Returns:
        HLLSetDeBruijnGraph with nodes and edges
    """
    n = len(hllsets)
    if labels is None:
        labels = [f"H{i}" for i in range(n)]
    
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            pair = bss(hllsets[i], hllsets[j])
            
            if pair.tau >= tau_min and pair.rho <= rho_max:
                drn = decompose_transition(hllsets[i], hllsets[j])
                edges.append(HLLSetEdge(i, j, pair, drn))
    
    return HLLSetDeBruijnGraph(
        nodes=hllsets,
        labels=labels,
        edges=edges,
        tau_min=tau_min,
        rho_max=rho_max,
        p_bits=p_bits
    )


def find_evolution_path(
    graph: HLLSetDeBruijnGraph,
    start_idx: Optional[int] = None
) -> List[int]:
    """
    Find a plausible evolution path through the HLLSet graph.
    
    Strategy:
        1. If start not given, choose node with highest source score
           (out_degree - in_degree)
        2. Greedily follow highest-τ edges to unvisited nodes
    
    This is analogous to finding an Eulerian path in a De Bruijn graph.
    
    Args:
        graph: The HLLSet De Bruijn graph
        start_idx: Optional starting node index
    
    Returns:
        List of node indices representing the evolution path
    """
    n = graph.num_nodes
    adj = graph.adjacency_list()
    
    # Sort each adjacency list by τ descending
    for i in adj:
        adj[i].sort(key=lambda x: -x[1])
    
    # Find start node if not provided
    if start_idx is None:
        # Score = out_degree - in_degree (higher = more "source-like")
        source_score = {
            i: graph.out_degree(i) - graph.in_degree(i)
            for i in range(n)
        }
        start_idx = max(source_score, key=source_score.get)
    
    # Greedy path following highest τ
    path = [start_idx]
    visited = {start_idx}
    
    while True:
        current = path[-1]
        candidates = [(j, tau) for (j, tau) in adj[current] if j not in visited]
        if not candidates:
            break
        next_node = max(candidates, key=lambda x: x[1])[0]
        path.append(next_node)
        visited.add(next_node)
    
    return path


def recover_tokens_from_drn(
    drn: FullDRNTriple,
    p_bits: int = 10
) -> Dict[str, Dict]:
    """
    Recover original token order from each (D, R, N) component using De Bruijn.
    
    This demonstrates that HLLSet transitions are FULLY INVERTIBLE when
    trigram structure is preserved.
    
    Args:
        drn: Full DRN triple with original tokens
        p_bits: Precision bits
    
    Returns:
        Dict with recovery results for D, R, N:
            {"D": {"original": [...], "recovered": [...], "match": bool}, ...}
    """
    results = {}
    
    for name, tokens in [
        ("D", drn.deleted_tokens),
        ("R", drn.retained_tokens),
        ("N", drn.novel_tokens)
    ]:
        if not tokens:
            results[name] = {"original": [], "recovered": [], "match": True}
            continue
        
        # Create engine and ingest with boundary markers
        engine = DisambiguationEngine(p_bits=p_bits)
        engine.ingest_tokens(list(tokens), max_n=3, add_boundaries=True)
        
        # Recover via De Bruijn
        recovered = engine.restore_token_order()
        
        results[name] = {
            "original": list(tokens),
            "recovered": recovered,
            "match": recovered == list(tokens)
        }
    
    return results


# =============================================================================
# Evolution Analysis
# =============================================================================

class EvolutionPhase(Enum):
    """Phase of evolution between two HLLSets."""
    GROWTH = "growth"      # |N| > |D|
    DECAY = "decay"        # |D| > |N|
    STABLE = "stable"      # |N| ≈ |D|
    REPLACEMENT = "replacement"  # High |D| and |N|, low |R|


def classify_transition(drn: DRNTriple, stability_threshold: float = 0.1) -> EvolutionPhase:
    """
    Classify the evolution phase of a transition.
    
    Args:
        drn: The (D, R, N) triple
        stability_threshold: Relative difference threshold for "stable"
    
    Returns:
        EvolutionPhase classification
    """
    d, r, n = drn.deleted_card, drn.retained_card, drn.novel_card
    total = d + r + n
    
    if total == 0:
        return EvolutionPhase.STABLE
    
    # Check for replacement: low retention relative to changes
    if r < (d + n) * 0.5:
        return EvolutionPhase.REPLACEMENT
    
    # Check for stability
    if abs(n - d) / max(total, 1) < stability_threshold:
        return EvolutionPhase.STABLE
    
    # Growth or decay
    return EvolutionPhase.GROWTH if n > d else EvolutionPhase.DECAY


@dataclass(frozen=True)
class EvolutionSummary:
    """Summary of evolution through an HLLSet sequence."""
    path: Tuple[int, ...]
    transitions: Tuple[DRNTriple, ...]
    phases: Tuple[EvolutionPhase, ...]
    total_deleted: float
    total_novel: float
    total_retained_avg: float
    
    @property
    def net_growth(self) -> float:
        return self.total_novel - self.total_deleted
    
    @property
    def dominant_phase(self) -> EvolutionPhase:
        from collections import Counter
        counts = Counter(self.phases)
        return counts.most_common(1)[0][0]


def analyze_evolution(
    graph: HLLSetDeBruijnGraph,
    path: List[int]
) -> EvolutionSummary:
    """
    Analyze the evolution along a path through the HLLSet graph.
    
    Args:
        graph: The HLLSet De Bruijn graph
        path: Node indices representing the evolution path
    
    Returns:
        EvolutionSummary with statistics
    """
    transitions = []
    phases = []
    
    for i in range(len(path) - 1):
        src, tgt = path[i], path[i + 1]
        drn = decompose_transition(graph.nodes[src], graph.nodes[tgt])
        transitions.append(drn)
        phases.append(classify_transition(drn))
    
    total_deleted = sum(t.deleted_card for t in transitions)
    total_novel = sum(t.novel_card for t in transitions)
    total_retained = sum(t.retained_card for t in transitions)
    avg_retained = total_retained / len(transitions) if transitions else 0
    
    return EvolutionSummary(
        path=tuple(path),
        transitions=tuple(transitions),
        phases=tuple(phases),
        total_deleted=total_deleted,
        total_novel=total_novel,
        total_retained_avg=avg_retained
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core types
    "DRNTriple",
    "FullDRNTriple",
    "HLLSetEdge",
    "HLLSetDeBruijnGraph",
    
    # Core functions
    "decompose_transition",
    "full_decompose_transition",
    "verify_reconstruction",
    "build_hllset_debruijn",
    "find_evolution_path",
    "recover_tokens_from_drn",
    
    # Evolution analysis
    "EvolutionPhase",
    "classify_transition",
    "EvolutionSummary",
    "analyze_evolution",
]
