"""
HLLSet Transformer — Complement-based temporal attention over the W lattice.

Implements the transformer architecture for HLLSet-based retrieval:

    Query → Encode → Complement Attention → Graph Disambiguation → LLM Handoff

This module orchestrates EXISTING infrastructure:
    - hll_lattice.py:     W lattice with cumulative(), delta(), nodes_in_range()
    - bss.py:             BSS τ/ρ for attention weights
    - markov_hll.py:      HLLMarkovChain for Bayesian selection
    - noether.py:         Conservation-based convergence
    - disambiguation.py:  Token recovery
    - hllset_debruijn.py: Sequence reconstruction

Architecture (following ring_transaction.py pattern):

    ┌─ forward() ──────────────────────────────────────────────────┐
    │  transformer = HLLSetTransformer(lattice=lat)                │
    │                                                              │
    │  # Encode query                                              │
    │  query_hll = transformer.encode("user query text")           │
    │                                                              │
    │  # Backward propagation through time                         │
    │  result = transformer.forward(query_hll)                     │
    │                                                              │
    │  # Result contains graph of HLLSets with τ edges             │
    └──────────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌─ TransformerResult (frozen) ─────────────────────────────────┐
    │  .final_context      — accumulated HLLSet                    │
    │  .attention_trace    — per-level attention records           │
    │  .hllset_graph       — extracted HLLSets with τ edges        │
    │  .convergence        — Noether diagnostics                   │
    │  .depths_traversed   — how deep we went                      │
    └──────────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌─ user decides ───────────────────────────────────────────────┐
    │  # Disambiguate graph (preserves τ edges)                    │
    │  doc_graph = result.disambiguate(engine, registry)           │
    │                                                              │
    │  # Hand off to external LLM                                  │
    │  prompt = doc_graph.to_llm_prompt()                          │
    │  response = llm.complete(prompt)                             │
    │                                                              │
    │  # Re-ingest (close the loop)                                │
    │  lattice.append([HLLSet.from_batch(response.tokens)])        │
    └──────────────────────────────────────────────────────────────┘

The Complement-Based Algorithm:
    
    Level W(t):   M(t) = lattice.cumulative(t)    — level's merged HLLSet
                  C(t) = Q ∩ M(t)                  — initial context
                  
    Level W(t-1): Δ = M(t-1) \\ C(t)              — COMPLEMENT: what's new
                  if |Δ| ≈ 0: STOP                — nothing new to learn
                  τ(Q → Δ) → select from Δ       — relevance scoring
                  C(t-1) = C(t) ∪ selected        — accumulate
                  Noether check: Φ = |selected|   — conservation tracking
                  
    Repeat until: Δ ≈ ∅ OR Noether converged OR max_depth

Key Insight: O(D) complexity, not O(D×n), because we work with
level-merged HLLSets and complements, not individual nodes.

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import (
    List, Tuple, Optional, Dict, Any, Union, NamedTuple,
)
from dataclasses import dataclass, field
from enum import Enum
import time

from .hllset import HLLSet, compute_sha1
from .hll_lattice import HLLLattice, LatticeNode
from .bss import bss, BSSPair
from .noether import NoetherEvolution, SteeringDiagnostics, SteeringPhase


# =============================================================================
# Enums & Value Types
# =============================================================================

class TransformerPhase(Enum):
    """Phase of transformer inference."""
    ENCODING = "encoding"
    PROPAGATING = "propagating"
    CONVERGED = "converged"
    EXHAUSTED = "exhausted"      # Complement empty
    MAX_DEPTH = "max_depth"


class ConvergenceReason(Enum):
    """Why the transformer stopped propagating."""
    COMPLEMENT_EXHAUSTED = "complement_exhausted"
    NOETHER_CONVERGED = "noether_converged"
    RELEVANCE_ZERO = "relevance_zero"
    MAX_DEPTH_REACHED = "max_depth_reached"
    LATTICE_BOUNDARY = "lattice_boundary"


@dataclass(frozen=True)
class AttentionRecord:
    """
    Immutable record of one attention step.
    
    Analogous to IngestRecord in ring_transaction.py but for
    HLLSet-level attention over W lattice levels.
    
    BSS metrics recorded:
        τ (tau): Coverage - how much of query is in complement
        ρ (rho): Noise - how much of complement is NOT in query
    """
    depth: int
    level_time: float
    level_id: str                 # SHA1(level_merged) — content address
    complement: HLLSet            # Δ = M(t-d) \ C(previous)
    complement_cardinality: float
    selected: HLLSet              # What we chose to add
    selection_cardinality: float  # = information gain Φ
    tau_to_query: float           # τ(Q → Δ) - coverage
    rho_noise: float              # ρ(Q → Δ) - noise ratio
    accumulated_context: HLLSet   # C after this step
    timestamp: float
    
    @property
    def is_exhausted(self) -> bool:
        """Was the complement effectively empty?"""
        return self.complement_cardinality < 1.0
    
    @property
    def is_noisy(self) -> bool:
        """Is the complement noisy (high ρ)?"""
        return self.rho_noise > 0.5
    
    @property
    def information_gain(self) -> float:
        """Alias for selection_cardinality (Noether flux)."""
        return self.selection_cardinality
    
    def __repr__(self) -> str:
        return (
            f"AttentionRecord(d={self.depth}, level={self.level_id[:8]}…, "
            f"|Δ|≈{self.complement_cardinality:.0f}, "
            f"τ={self.tau_to_query:.3f}, ρ={self.rho_noise:.3f}, Φ={self.information_gain:.0f})"
        )


@dataclass(frozen=True)
class HLLSetEdge:
    """Edge between HLLSets preserving BSS relationship."""
    source_id: str
    target_id: str
    tau: float       # τ(source → target) = P(source | target)
    rho: float       # ρ(source → target) = noise ratio
    
    def __repr__(self) -> str:
        return f"Edge({self.source_id[:8]}→{self.target_id[:8]}, τ={self.tau:.3f})"


@dataclass(frozen=True)
class HLLSetGraph:
    """
    Graph of extracted HLLSets with τ-weighted edges.
    
    This is the output of the transformer before disambiguation.
    Preserving this structure allows the LLM to understand
    relationships between documents, not just their content.
    """
    nodes: Dict[str, HLLSet]      # node_id → HLLSet
    edges: Tuple[HLLSetEdge, ...] # Directed edges with τ weights
    query_id: str                 # SHA1 of query HLLSet
    
    @property
    def node_count(self) -> int:
        return len(self.nodes)
    
    @property
    def edge_count(self) -> int:
        return len(self.edges)
    
    def merged(self) -> HLLSet:
        """Merge all nodes into single HLLSet (loses structure)."""
        if not self.nodes:
            return HLLSet(p_bits=10)
        return HLLSet.merge(list(self.nodes.values()))
    
    def __repr__(self) -> str:
        return f"HLLSetGraph({self.node_count} nodes, {self.edge_count} edges)"


# =============================================================================
# Phase 1: Collected Context (Lattice Traversal Output)
# =============================================================================

@dataclass
class CollectedHLLSet:
    """
    An HLLSet collected during Phase 1 lattice traversal.
    
    Stores provenance information for each collected set.
    """
    hll_id: str           # SHA1 content address
    hllset: HLLSet        # The actual HLLSet
    level_time: float     # W lattice timestamp
    depth: int            # How many levels back from query
    tau_to_query: float   # τ(query → this) at collection time
    rho_noise: float      # ρ at collection time
    
    @property
    def cardinality(self) -> float:
        return self.hllset.cardinality()
    
    def __repr__(self) -> str:
        return f"CollectedHLLSet({self.hll_id[:8]}…, t={self.level_time}, τ={self.tau_to_query:.3f})"


@dataclass
class CollectedContext:
    """
    Phase 1 Output: Collection of HLLSets from lattice traversal.
    
    This is a CLEAN separation of concerns:
    - Phase 1 (collect_context): Uses LATTICE TOPOLOGY to collect HLLSets
    - Phase 2 (build_markov_chain): Uses BSS to compute TRANSITION PROBABILITIES
    
    The collected HLLSets are stored individually with provenance,
    NOT merged into a blob. This preserves document identity.
    """
    query_id: str                              # SHA1 of query HLLSet
    query_hll: HLLSet                          # Original query
    collected: Dict[str, CollectedHLLSet]      # id → CollectedHLLSet
    traversal_order: List[str]                 # Order of collection (for debugging)
    stop_reason: ConvergenceReason             # Why collection stopped
    flux_history: Tuple[float, ...]            # Noether flux at each step
    
    def add(self, hll: HLLSet, level_time: float, depth: int, 
            tau: float, rho: float) -> str:
        """Add an HLLSet to collection, return its ID."""
        hll_id = compute_sha1(hll.dump_numpy())
        
        if hll_id not in self.collected:
            self.collected[hll_id] = CollectedHLLSet(
                hll_id=hll_id,
                hllset=hll,
                level_time=level_time,
                depth=depth,
                tau_to_query=tau,
                rho_noise=rho,
            )
            self.traversal_order.append(hll_id)
        
        return hll_id
    
    def merged(self) -> HLLSet:
        """Merge all collected HLLSets (loses individual identity)."""
        if not self.collected:
            return self.query_hll
        hllsets = [c.hllset for c in self.collected.values()]
        return HLLSet.merge(hllsets)
    
    def get_hllsets(self) -> List[HLLSet]:
        """Get list of collected HLLSets (preserves identity)."""
        return [c.hllset for c in self.collected.values()]
    
    def get_ids(self) -> set:
        """Get set of collected HLLSet IDs."""
        return set(self.collected.keys())
    
    @property
    def size(self) -> int:
        return len(self.collected)
    
    @property
    def total_cardinality(self) -> float:
        return self.merged().cardinality()
    
    def __repr__(self) -> str:
        return f"CollectedContext({self.size} HLLSets, |merged|≈{self.total_cardinality:.0f})"


# =============================================================================
# Phase 2: HLL Markov Chain (BSS-based transitions)
# =============================================================================

@dataclass(frozen=True)
class MarkovTransition:
    """
    A transition in the HLL Markov Chain.
    
    Transition probability is derived from BSS:
        P(j|i) = BSS(i,j).τ / Σ_k BSS(i,k).τ
    
    This is SEMANTICALLY meaningful:
    - High τ(i→j) means state j contains much of state i's content
    - Normalized to form proper probability distribution
    """
    from_id: str          # Source state (HLLSet ID)
    to_id: str            # Target state (HLLSet ID)
    tau: float            # τ(from → to) = raw BSS coverage
    rho: float            # ρ(from → to) = noise ratio
    probability: float    # P(to|from) = normalized transition probability
    
    def __repr__(self) -> str:
        return f"Transition({self.from_id[:8]}→{self.to_id[:8]}, P={self.probability:.3f}, τ={self.tau:.3f})"


@dataclass
class HLLMarkovChain:
    """
    Phase 2 Output: Markov Chain over collected HLLSets.
    
    ARCHITECTURE INSIGHT:
    - States = collected HLLSets (from Phase 1)
    - Transitions = BSS-derived probabilities (computed in Phase 2)
    
    This separation means:
    1. Collection strategy (lattice topology) is INDEPENDENT from
    2. Transition semantics (BSS similarity)
    
    Benefits:
    - Can traverse lattice once, then compute many different MC views
    - BSS gives true SEMANTIC transitions, not just adjacency
    - Full connectivity: any state can transition to any other
    - Proper conditional probability: P(j|i) = τ(i→j) / Σ_k τ(i→k)
    """
    query_id: str                                      # SHA1 of query
    states: Dict[str, HLLSet]                          # state_id → HLLSet
    transitions: Dict[str, List[MarkovTransition]]     # from_id → [transitions]
    initial_distribution: Dict[str, float]             # P(initial state)
    
    @property
    def state_count(self) -> int:
        return len(self.states)
    
    @property
    def transition_count(self) -> int:
        return sum(len(t) for t in self.transitions.values())
    
    def transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Get transition matrix as nested dict.
        
        Returns:
            matrix[from_id][to_id] = P(to|from)
        """
        matrix: Dict[str, Dict[str, float]] = {}
        for from_id, trans_list in self.transitions.items():
            matrix[from_id] = {t.to_id: t.probability for t in trans_list}
        return matrix
    
    def neighbors(self, state_id: str, min_prob: float = 0.01) -> List[Tuple[str, float]]:
        """
        Get reachable states from given state.
        
        Returns:
            List of (state_id, probability) pairs
        """
        if state_id not in self.transitions:
            return []
        return [
            (t.to_id, t.probability) 
            for t in self.transitions[state_id]
            if t.probability >= min_prob
        ]
    
    def most_likely_path(self, start_id: str, length: int = 3) -> List[str]:
        """
        Get most likely path from start state.
        
        Greedy: always picks highest probability transition.
        """
        path = [start_id]
        current = start_id
        
        for _ in range(length):
            neighbors = self.neighbors(current)
            if not neighbors:
                break
            # Pick highest probability neighbor not yet in path
            for next_id, prob in sorted(neighbors, key=lambda x: -x[1]):
                if next_id not in path:
                    path.append(next_id)
                    current = next_id
                    break
            else:
                break  # All neighbors already visited
        
        return path
    
    def stationary_distribution(self, iterations: int = 100) -> Dict[str, float]:
        """
        Estimate stationary distribution via power iteration.
        
        This gives the "importance" of each state in the MC.
        """
        if not self.states:
            return {}
        
        # Start uniform
        n = len(self.states)
        dist = {sid: 1.0/n for sid in self.states}
        
        matrix = self.transition_matrix()
        
        for _ in range(iterations):
            new_dist: Dict[str, float] = {sid: 0.0 for sid in self.states}
            for from_id, row in matrix.items():
                for to_id, prob in row.items():
                    new_dist[to_id] += dist[from_id] * prob
            
            # Normalize
            total = sum(new_dist.values())
            if total > 0:
                dist = {k: v/total for k, v in new_dist.items()}
        
        return dist
    
    def to_hllset_graph(self) -> 'HLLSetGraph':
        """
        Convert Markov Chain to HLLSetGraph for compatibility.
        
        Uses transition probabilities as edge weights.
        """
        edges: List[HLLSetEdge] = []
        
        for from_id, trans_list in self.transitions.items():
            for t in trans_list:
                # Use probability as tau (it's normalized)
                edges.append(HLLSetEdge(
                    source_id=t.from_id,
                    target_id=t.to_id,
                    tau=t.tau,
                    rho=t.rho,
                ))
        
        return HLLSetGraph(
            nodes=dict(self.states),
            edges=tuple(edges),
            query_id=self.query_id,
        )
    
    def __repr__(self) -> str:
        return f"HLLMarkovChain({self.state_count} states, {self.transition_count} transitions)"


# =============================================================================
# Legacy MarkovState (for backward compatibility with forward_mc)
# =============================================================================

@dataclass
class MarkovState:
    """
    State in the W-Markov Chain traversal.
    
    NOTE: This is the LEGACY coupled implementation.
    For the new 2-phase approach, use:
      1. collect_context() → CollectedContext
      2. build_markov_chain() → HLLMarkovChain
    """
    collected_ids: set           # SHA1 IDs of collected HLLSets
    collected_hllsets: dict      # id → HLLSet (for final merge)
    transitions: list            # (from_id, to_id, tau) edges
    current_level_id: str        # Current position in W
    depth: int                   # How far back we've gone
    
    def add_hllset(self, hll: HLLSet, source_id: Optional[str] = None, tau: float = 1.0) -> str:
        """Add an HLLSet to the collection, return its ID."""
        hll_id = compute_sha1(hll.dump_numpy())
        
        if hll_id not in self.collected_ids:
            self.collected_ids.add(hll_id)
            self.collected_hllsets[hll_id] = hll
            
            # Record transition edge (lattice adjacency based)
            if source_id is not None:
                self.transitions.append((source_id, hll_id, tau))
        
        return hll_id
    
    def merged_context(self) -> HLLSet:
        """Merge all collected HLLSets (for Noether check or final output)."""
        if not self.collected_hllsets:
            return HLLSet(p_bits=10)
        return HLLSet.merge(list(self.collected_hllsets.values()))
    
    @property
    def collection_size(self) -> int:
        return len(self.collected_ids)
    
    def __repr__(self) -> str:
        return f"MarkovState(collected={self.collection_size}, depth={self.depth})"


@dataclass(frozen=True)
class TransformerResult:
    """
    Immutable result of transformer forward pass.
    
    Analogous to TransactionResult in ring_transaction.py.
    """
    result_id: str                          # SHA1(final_context)
    query: HLLSet                           # Original query
    final_context: HLLSet                   # Accumulated context
    hllset_graph: HLLSetGraph               # Extracted HLLSets with τ edges
    attention_trace: Tuple[AttentionRecord, ...]
    depths_traversed: int
    total_information_gain: float
    convergence_reason: ConvergenceReason
    noether_flux_history: Tuple[float, ...]
    timestamp: float
    
    @property
    def is_converged(self) -> bool:
        return self.convergence_reason in (
            ConvergenceReason.NOETHER_CONVERGED,
            ConvergenceReason.COMPLEMENT_EXHAUSTED,
        )
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Transformer: {self.depths_traversed} levels, "
            f"+{self.total_information_gain:.0f} info, "
            f"{self.convergence_reason.value}"
        )
    
    def disambiguate(
        self,
        engine,           # DisambiguationEngine
        ngram_registry=None,  # GlobalNGramRegistry
    ) -> 'DocumentGraph':
        """
        Disambiguate HLLSet graph to document graph.
        
        Preserves τ-edge structure through disambiguation.
        """
        return _disambiguate_graph(self.hllset_graph, engine, ngram_registry)
    
    def __repr__(self) -> str:
        card = self.final_context.cardinality()
        return (
            f"TransformerResult({self.result_id[:8]}…, "
            f"|C|≈{card:.0f}, d={self.depths_traversed}, "
            f"{self.convergence_reason.value})"
        )


# =============================================================================
# Document Graph (for post-disambiguation)
# =============================================================================

@dataclass(frozen=True)
class DocumentNode:
    """A disambiguated document with provenance."""
    node_id: str          # SHA1 of source HLLSet
    text: str             # Reconstructed text
    tokens: Tuple[str, ...]
    cardinality: float
    source_depth: int     # Which W level it came from


@dataclass(frozen=True)
class DocumentEdge:
    """Edge between documents preserving BSS relationship."""
    source_id: str
    target_id: str
    tau: float
    rho: float


@dataclass
class DocumentGraph:
    """
    Graph of disambiguated documents with τ-weighted edges.
    
    This is the final output before LLM handoff.
    """
    nodes: Dict[str, DocumentNode]
    edges: List[DocumentEdge]
    query_text: str
    
    def to_llm_prompt(self, max_chars: int = 8000) -> str:
        """
        Format graph as prompt for external LLM.
        
        Includes relationship strengths so LLM understands
        how documents relate to each other.
        """
        lines = [
            f"Query: {self.query_text}",
            "",
            f"Found {len(self.nodes)} related documents with the following relationships:",
            "",
        ]
        
        # Sort nodes by centrality (sum of incoming τ)
        centrality: Dict[str, float] = {nid: 0.0 for nid in self.nodes}
        for edge in self.edges:
            centrality[edge.target_id] += edge.tau
        
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: centrality[n.node_id],
            reverse=True
        )
        
        char_count = sum(len(line) for line in lines)
        
        for i, node in enumerate(sorted_nodes, 1):
            # Find outgoing edges
            outgoing = [e for e in self.edges if e.source_id == node.node_id]
            edge_str = ", ".join(
                f"→Doc{list(self.nodes.keys()).index(e.target_id)+1}(τ={e.tau:.2f})"
                for e in outgoing[:3]
            ) if outgoing else "standalone"
            
            header = f"[Document {i}] (id={node.node_id[:8]}, relations: {edge_str})"
            
            # Truncate text if needed
            remaining = max_chars - char_count - len(header) - 100
            text = node.text[:remaining] + "..." if len(node.text) > remaining else node.text
            
            lines.append(header)
            lines.append(text)
            lines.append("")
            
            char_count += len(header) + len(text) + 2
            if char_count > max_chars:
                lines.append(f"[{len(sorted_nodes) - i} more documents truncated]")
                break
        
        lines.append("Please synthesize these documents into a coherent response to the query.")
        
        return "\n".join(lines)


# =============================================================================
# HLLSet Transformer
# =============================================================================

class HLLSetTransformer:
    """
    Complement-based transformer over the temporal W lattice.
    
    Orchestrates existing modules:
        - HLLLattice: cumulative(), delta(), nodes_in_range()
        - bss: τ/ρ computation for attention
        - NoetherEvolution: conservation-based convergence
    
    Usage::
    
        transformer = HLLSetTransformer(lattice=lat)
        
        # Forward pass
        result = transformer.forward("what is the capital of France")
        
        # Examine attention trace
        for rec in result.attention_trace:
            print(f"Level {rec.depth}: τ={rec.tau_to_query:.3f}, Φ={rec.information_gain:.0f}")
        
        # Disambiguate (requires engine)
        doc_graph = result.disambiguate(engine, ngram_registry)
        
        # Hand off to LLM
        prompt = doc_graph.to_llm_prompt()
    
    The transformer is STATELESS — each forward() call is independent.
    Like RingTransaction, it's a library component with no side effects.
    
    Temperature Control:
        Similar to LLM temperature, controls selection strictness:
        - temperature=0.0: Only highest-τ content (most focused)
        - temperature=0.5: Balanced selection (default)
        - temperature=1.0: Include all above-threshold content (exploratory)
        
        Mathematically: effective_threshold = tau_threshold * (1 - temperature)
        
    BSS Thresholds:
        - tau_threshold: Minimum τ(query → target) for relevance
        - rho_threshold: Maximum ρ (noise ratio) to tolerate
        
        τ measures "how much of query is in target" (coverage)
        ρ measures "how much of target is NOT in query" (noise)
        
        High τ, low ρ = focused, relevant content
        High τ, high ρ = relevant but noisy content
        Low τ = irrelevant content
    """
    
    def __init__(
        self,
        lattice: HLLLattice,
        p_bits: int = 10,
        tau_threshold: float = 0.05,
        rho_threshold: float = 1.0,
        temperature: float = 0.5,
        max_depth: int = 10,
        convergence_window: int = 3,
        convergence_tolerance: float = 0.05,
        complement_exhaustion_threshold: float = 1.0,
    ):
        """
        Create transformer.
        
        Args:
            lattice: W lattice to query (read-only)
            p_bits: HLL precision bits
            tau_threshold: Minimum τ for relevance (0.0 to 1.0)
                          Higher = stricter relevance requirement
            rho_threshold: Maximum ρ (noise) to tolerate (0.0 to 1.0)
                          Lower = less noise tolerance, more focused
            temperature: Selection temperature (0.0 to 1.0)
                        0.0 = strict (only best matches)
                        0.5 = balanced (default)
                        1.0 = exploratory (include more context)
            max_depth: Maximum propagation depth
            convergence_window: Window for Noether stability check
            convergence_tolerance: Tolerance for flux stability
            complement_exhaustion_threshold: |Δ| below which complement is "empty"
        """
        self._lattice = lattice
        self._p_bits = p_bits
        self._tau_threshold = tau_threshold
        self._rho_threshold = rho_threshold
        self._temperature = max(0.0, min(1.0, temperature))  # Clamp to [0, 1]
        self._max_depth = max_depth
        self._conv_window = convergence_window
        self._conv_tol = convergence_tolerance
        self._exhaust_threshold = complement_exhaustion_threshold
    
    @property
    def temperature(self) -> float:
        """Current temperature setting."""
        return self._temperature
    
    @temperature.setter
    def temperature(self, value: float):
        """Set temperature (clamped to [0, 1])."""
        self._temperature = max(0.0, min(1.0, value))
    
    @property
    def tau_threshold(self) -> float:
        """Current τ threshold."""
        return self._tau_threshold
    
    @tau_threshold.setter
    def tau_threshold(self, value: float):
        """Set τ threshold (clamped to [0, 1])."""
        self._tau_threshold = max(0.0, min(1.0, value))
    
    @property
    def rho_threshold(self) -> float:
        """Current ρ threshold."""
        return self._rho_threshold
    
    @rho_threshold.setter
    def rho_threshold(self, value: float):
        """Set ρ threshold (clamped to [0, 1])."""
        self._rho_threshold = max(0.0, min(1.0, value))
    
    def _effective_tau_threshold(self) -> float:
        """
        Compute effective τ threshold based on temperature.
        
        At temp=0: threshold = tau_threshold (strictest)
        At temp=1: threshold = 0 (most permissive)
        At temp=0.5: threshold = tau_threshold * 0.5
        """
        return self._tau_threshold * (1.0 - self._temperature)
    
    def _compute_relevance_score(
        self, 
        query: HLLSet, 
        target: HLLSet
    ) -> Tuple[float, float, float]:
        """
        Compute bidirectional relevance score between query and target.
        
        Returns:
            (score, tau_forward, tau_reverse) where:
            - score: Combined relevance (0-1), higher = more relevant
            - tau_forward: τ(query → target) = how much of query is in target
            - tau_reverse: τ(target → query) = how much of target is in query
        
        The Problem with Asymmetric τ:
            Given query Q=4 tokens, target T=100 tokens:
            - τ(Q→T) can be 1.0 (Q fully covered by T) 
            - τ(T→Q) will be ~0.04 (only 4% of T is in Q)
            
            If we use only τ(Q→T), EVERYTHING seems relevant!
            If we use only τ(T→Q), NOTHING seems relevant!
            
        Solution - Harmonic Mean:
            score = 2 * τ_fwd * τ_rev / (τ_fwd + τ_rev)
            
            This penalizes when either direction is weak:
            - τ_fwd=1.0, τ_rev=0.04 → score=0.077 (low - T has lots of noise)
            - τ_fwd=0.8, τ_rev=0.6  → score=0.686 (high - mutual relevance)
            - τ_fwd=1.0, τ_rev=1.0  → score=1.0 (perfect match)
        """
        query_card = query.cardinality()
        target_card = target.cardinality()
        
        # Handle edge cases
        if query_card < 1.0 or target_card < 1.0:
            return (0.0, 0.0, 0.0)
        
        # Compute intersection
        intersection = query.intersect(target)
        intersection_card = intersection.cardinality()
        
        # Bidirectional τ
        tau_forward = intersection_card / query_card   # τ(Q→T)
        tau_reverse = intersection_card / target_card  # τ(T→Q)
        
        # Harmonic mean (penalizes asymmetry)
        if tau_forward + tau_reverse < 1e-9:
            score = 0.0
        else:
            score = 2.0 * tau_forward * tau_reverse / (tau_forward + tau_reverse)
        
        return (score, tau_forward, tau_reverse)
    
    def _expand_query_from_level(
        self,
        prompt: HLLSet,
        level_merged: HLLSet,
        level_nodes: Optional[List[HLLSet]] = None,
    ) -> HLLSet:
        """
        Expand query by finding all related content at a level.
        
        Strategy:
            1. Find intersection of prompt with level (seed tokens)
            2. For each node at level, compute bidirectional score
            3. Select nodes with score > threshold (temperature-adjusted)
            4. Return union of selected nodes
            
        This ensures the query contains ALL tokens from relevant documents,
        not just the 4 tokens in the original prompt.
        
        Args:
            prompt: Original prompt HLLSet
            level_merged: Cumulative HLLSet at this level
            level_nodes: Individual HLLSets at this level (optional)
            
        Returns:
            Expanded query HLLSet
        """
        # If no individual nodes, use intersection approach
        if level_nodes is None:
            # Get what prompt shares with level
            intersection = prompt.intersect(level_merged)
            int_card = intersection.cardinality()
            
            if int_card < 1.0:
                # No overlap - return prompt as-is
                return prompt
            
            # Compute bidirectional score
            score, tau_fwd, tau_rev = self._compute_relevance_score(prompt, level_merged)
            
            # Temperature-adjusted selection
            # At temp=1.0, take full level if ANY overlap
            # At temp=0.0, require strong bidirectional relevance
            effective_score_threshold = 0.1 * (1.0 - self._temperature)
            
            if score > effective_score_threshold or tau_fwd > 0.5:
                # Return intersection expanded to include related content
                # The key insight: if prompt tokens exist in level, 
                # the whole level's relevant content should be in query
                return intersection.union(prompt)
            else:
                return prompt
        
        # With individual nodes: score each and select best
        selected = []
        effective_threshold = self._tau_threshold * (1.0 - self._temperature * 0.5)
        
        for node in level_nodes:
            score, tau_fwd, tau_rev = self._compute_relevance_score(prompt, node)
            
            # Select if bidirectional score is good OR forward τ is high
            if score > effective_threshold or tau_fwd > 0.3:
                selected.append(node)
        
        if not selected:
            return prompt
        
        # Union of selected nodes becomes the expanded query
        expanded = HLLSet.merge(selected)
        return expanded.union(prompt)
    
    # -------------------------------------------------------------------------
    # Encoding
    # -------------------------------------------------------------------------
    
    def encode(self, text: Union[str, List[str]]) -> HLLSet:
        """
        Encode text to query HLLSet.
        
        Args:
            text: String or list of tokens
            
        Returns:
            HLLSet representing the query
        """
        if isinstance(text, str):
            tokens = text.lower().split()
        else:
            tokens = text
        return HLLSet.from_batch(tokens, p_bits=self._p_bits)
    
    # -------------------------------------------------------------------------
    # 2-Phase Architecture: Collection + Markov Chain
    # -------------------------------------------------------------------------
    
    def collect_context(
        self,
        query: Union[str, List[str], HLLSet],
        current_time: Optional[float] = None,
        time_step: float = 1.0,
        max_depth: Optional[int] = None,
    ) -> CollectedContext:
        """
        PHASE 1: Collect relevant HLLSets using lattice topology.
        
        This method ONLY collects HLLSets — it does NOT build transitions.
        Separation of concerns:
        - Phase 1 (this): Use LATTICE STRUCTURE to find relevant content
        - Phase 2 (build_markov_chain): Use BSS to compute SEMANTIC transitions
        
        Collection Strategy:
            1. Start at current_time, work backward through W lattice
            2. At each level, check if query tokens are present (τ > threshold)
            3. If relevant, add the WHOLE level's HLLSet to collection
            4. Stop when: Noether converged, lattice boundary, or max_depth
        
        Args:
            query: Prompt (text, tokens, or HLLSet)
            current_time: Starting time (default: latest)
            time_step: Time delta between levels
            max_depth: Override default max_depth
            
        Returns:
            CollectedContext with individual HLLSets (not merged)
        """
        # 1. Encode prompt
        if isinstance(query, (str, list)):
            query_hll = self.encode(query)
        else:
            query_hll = query
        
        query_id = compute_sha1(query_hll.dump_numpy())
        query_card = query_hll.cardinality()
        
        # 2. Get lattice bounds
        if current_time is None:
            latest = self._lattice.latest_node()
            current_time = latest.timestamp if latest else 0.0
        
        earliest = self._get_earliest_time()
        depth_limit = max_depth if max_depth is not None else self._max_depth
        
        # 3. Initialize CollectedContext
        context = CollectedContext(
            query_id=query_id,
            query_hll=query_hll,
            collected={},
            traversal_order=[],
            stop_reason=ConvergenceReason.MAX_DEPTH_REACHED,
            flux_history=(),
        )
        
        # Add query itself as first collected item
        context.add(query_hll, current_time, depth=0, tau=1.0, rho=0.0)
        
        # 4. Tracking
        flux_history: List[float] = []
        prev_total_card = query_card
        
        # Temperature-adjusted threshold
        min_tau = self._tau_threshold * (1.0 - self._temperature)
        
        # 5. Lattice Traversal (backward through time)
        for depth in range(0, depth_limit + 1):
            target_time = current_time - depth * time_step
            
            # Boundary check
            if target_time < earliest:
                context.stop_reason = ConvergenceReason.LATTICE_BOUNDARY
                break
            
            # Get level's HLLSet
            nodes_at_time = self._lattice.nodes_in_range(
                target_time - 0.001, target_time + 0.001
            )
            
            if not nodes_at_time:
                flux_history.append(0.0)
                continue
            
            level_node = nodes_at_time[0]
            level_hll = level_node.merged
            level_card = level_hll.cardinality()
            
            if level_card < 1.0:
                flux_history.append(0.0)
                continue
            
            # Compute relevance: τ(query → level)
            intersection = query_hll.intersect(level_hll)
            overlap_card = intersection.cardinality()
            
            tau = overlap_card / query_card if query_card > 0 else 0.0
            tau_reverse = overlap_card / level_card if level_card > 0 else 0.0
            rho = 1.0 - tau_reverse
            
            # SELECTION: Is this level relevant to query?
            if tau >= min_tau or overlap_card >= 1.0:
                # Relevant! Add to collection
                context.add(level_hll, target_time, depth, tau, rho)
                
                # Compute information gain
                new_total_card = context.total_cardinality
                gain = new_total_card - prev_total_card
                flux_history.append(gain)
                prev_total_card = new_total_card
            else:
                # Not relevant, skip
                flux_history.append(0.0)
            
            # Noether convergence check
            if self._check_noether_convergence(flux_history):
                context.stop_reason = ConvergenceReason.NOETHER_CONVERGED
                break
        
        # Update flux history
        context = CollectedContext(
            query_id=context.query_id,
            query_hll=context.query_hll,
            collected=context.collected,
            traversal_order=context.traversal_order,
            stop_reason=context.stop_reason,
            flux_history=tuple(flux_history),
        )
        
        return context
    
    def build_markov_chain(
        self,
        collected: CollectedContext,
        min_transition_prob: float = 0.01,
    ) -> HLLMarkovChain:
        """
        PHASE 2: Build Markov Chain using BSS as transition probabilities.
        
        This method computes SEMANTIC transitions between collected HLLSets
        using BSS (Binary Sketch Similarity).
        
        Key Insight - BSS as Conditional Probability:
            τ(i→j) = |i ∩ j| / |i| = P(token in j | token in i)
            
            Normalized transition probability:
            P(j|i) = τ(i→j) / Σ_k τ(i→k)
        
        This gives FULL CONNECTIVITY: any state can transition to any other,
        weighted by semantic similarity. Unlike lattice-based transitions
        which only connect adjacent levels.
        
        Args:
            collected: CollectedContext from Phase 1
            min_transition_prob: Minimum probability to include edge
            
        Returns:
            HLLMarkovChain with BSS-derived transitions
        """
        # 1. Extract states (HLLSets)
        states: Dict[str, HLLSet] = {}
        for hll_id, collected_hll in collected.collected.items():
            states[hll_id] = collected_hll.hllset
        
        if len(states) < 2:
            # Need at least 2 states for meaningful MC
            return HLLMarkovChain(
                query_id=collected.query_id,
                states=states,
                transitions={},
                initial_distribution={collected.query_id: 1.0} if states else {},
            )
        
        # 2. Compute pairwise BSS for all states
        state_ids = list(states.keys())
        transitions: Dict[str, List[MarkovTransition]] = {sid: [] for sid in state_ids}
        
        for from_id in state_ids:
            from_hll = states[from_id]
            
            # Compute τ to all other states
            raw_scores: List[Tuple[str, float, float]] = []  # (to_id, tau, rho)
            
            for to_id in state_ids:
                if to_id == from_id:
                    continue
                
                to_hll = states[to_id]
                tau, rho = bss(from_hll, to_hll)
                
                if tau > 0:  # Only non-zero transitions
                    raw_scores.append((to_id, tau, rho))
            
            if not raw_scores:
                continue
            
            # 3. Normalize to get proper probability distribution
            total_tau = sum(tau for _, tau, _ in raw_scores)
            
            if total_tau > 0:
                for to_id, tau, rho in raw_scores:
                    prob = tau / total_tau
                    
                    if prob >= min_transition_prob:
                        transitions[from_id].append(MarkovTransition(
                            from_id=from_id,
                            to_id=to_id,
                            tau=tau,
                            rho=rho,
                            probability=prob,
                        ))
        
        # 4. Initial distribution: favor query, then by τ to query
        initial: Dict[str, float] = {}
        query_id = collected.query_id
        
        if query_id in states:
            # Query state gets highest initial probability
            initial[query_id] = 0.5
            
            # Distribute rest by τ to query
            other_ids = [sid for sid in state_ids if sid != query_id]
            if other_ids:
                query_hll = states[query_id]
                taus = []
                for sid in other_ids:
                    tau, _ = bss(query_hll, states[sid])
                    taus.append(tau)
                
                total_tau = sum(taus)
                if total_tau > 0:
                    for sid, tau in zip(other_ids, taus):
                        initial[sid] = 0.5 * (tau / total_tau)
                else:
                    # Uniform fallback
                    for sid in other_ids:
                        initial[sid] = 0.5 / len(other_ids)
        else:
            # No query in states - uniform distribution
            for sid in state_ids:
                initial[sid] = 1.0 / len(state_ids)
        
        return HLLMarkovChain(
            query_id=query_id,
            states=states,
            transitions=transitions,
            initial_distribution=initial,
        )
    
    def forward_2phase(
        self,
        query: Union[str, List[str], HLLSet],
        current_time: Optional[float] = None,
        time_step: float = 1.0,
    ) -> Tuple[CollectedContext, HLLMarkovChain, TransformerResult]:
        """
        Full 2-phase forward pass: collect then build Markov Chain.
        
        This is the RECOMMENDED entry point for the new architecture.
        
        Phase 1: collect_context() - Lattice topology traversal
        Phase 2: build_markov_chain() - BSS-based transitions
        
        Returns:
            (CollectedContext, HLLMarkovChain, TransformerResult)
            
        Example:
            context, mc, result = transformer.forward_2phase("capital of france")
            
            # Phase 1 output: individual HLLSets with provenance
            print(f"Collected {context.size} HLLSets")
            
            # Phase 2 output: Markov Chain with semantic transitions
            print(f"MC has {mc.transition_count} transitions")
            path = mc.most_likely_path(context.query_id, length=3)
            
            # Compatible result for downstream
            doc_graph = result.disambiguate(engine, registry)
        """
        # Phase 1: Collect
        collected = self.collect_context(query, current_time, time_step)
        
        # Phase 2: Build Markov Chain
        mc = self.build_markov_chain(collected)
        
        # Build TransformerResult for compatibility
        hllset_graph = mc.to_hllset_graph()
        final_context = collected.merged()
        result_id = compute_sha1(final_context.dump_numpy())
        
        # Build attention trace from collected items
        trace: List[AttentionRecord] = []
        for hll_id in collected.traversal_order:
            c = collected.collected[hll_id]
            trace.append(AttentionRecord(
                depth=c.depth,
                level_time=c.level_time,
                level_id=c.hll_id,
                complement=c.hllset,  # Use hllset as complement for now
                complement_cardinality=c.cardinality,
                selected=c.hllset,
                selection_cardinality=c.cardinality,
                tau_to_query=c.tau_to_query,
                rho_noise=c.rho_noise,
                accumulated_context=final_context,
                timestamp=time.time(),
            ))
        
        result = TransformerResult(
            result_id=result_id,
            query=collected.query_hll,
            final_context=final_context,
            hllset_graph=hllset_graph,
            attention_trace=tuple(trace),
            depths_traversed=len(trace),
            total_information_gain=sum(collected.flux_history),
            convergence_reason=collected.stop_reason,
            noether_flux_history=collected.flux_history,
            timestamp=time.time(),
        )
        
        return collected, mc, result
    
    # -------------------------------------------------------------------------
    # Forward Pass (Original - Coupled)
    # -------------------------------------------------------------------------
    
    def forward(
        self,
        query: Union[str, List[str], HLLSet],
        current_time: Optional[float] = None,
        time_step: float = 1.0,
    ) -> TransformerResult:
        """
        Forward pass: propagate query backward through W lattice.
        
        CRITICAL INSIGHT:
            Query "capital of france" exists in the corpus. When we diff
            the level with context, the complement contains ALL tokens 
            EXCEPT query tokens — including "paris", "eiffel" (the ANSWERS).
            
            We need to INCLUDE content from levels that CONTAIN query tokens,
            not exclude it. The check should be:
            
                "Does this level contain query-related content?"
            
            NOT:
            
                "Does the complement overlap with query?"
        
        Algorithm:
            1. Start with prompt, find which levels contain prompt tokens
            2. For each such level, include ALL content from that level
            3. Temperature controls: low=only exact token matches, high=whole level
        
        Args:
            query: Prompt (text, tokens, or HLLSet) to select from W(t)
            current_time: Starting time (default: latest in lattice)
            time_step: Time delta between levels
            
        Returns:
            TransformerResult with enriched context and HLLSet graph
        """
        start_time = time.time()
        
        # 1. Encode prompt if needed
        if isinstance(query, (str, list)):
            prompt_hll = self.encode(query)
            query_text = query if isinstance(query, str) else " ".join(query)
        else:
            prompt_hll = query
            query_text = ""
        
        prompt_card = prompt_hll.cardinality()
        
        # 2. Get lattice time bounds
        if current_time is None:
            latest = self._lattice.latest_node()
            current_time = latest.timestamp if latest else 0.0
        
        earliest = self._get_earliest_time()
        
        # 3. Initialize: context starts with prompt
        context = prompt_hll
        query_id = compute_sha1(prompt_hll.dump_numpy())
        
        # 4. Tracking
        trace: List[AttentionRecord] = []
        flux_history: List[float] = []
        extracted_hllsets: Dict[str, HLLSet] = {}
        convergence_reason = ConvergenceReason.MAX_DEPTH_REACHED
        
        # Temperature-adjusted thresholds
        # At temp=0: require strong overlap
        # At temp=1: include any level with any overlap
        min_overlap_ratio = self._tau_threshold * (1.0 - self._temperature)
        
        # 5. Backward propagation through time
        #    Start at depth=0 (current level) and work backward
        #    Use nodes_in_range to get each level's actual content
        
        for depth in range(0, self._max_depth + 1):
            target_time = current_time - depth * time_step
            
            # Check lattice boundary
            if target_time < earliest:
                convergence_reason = ConvergenceReason.LATTICE_BOUNDARY
                break
            
            # Get the actual node(s) at this timestamp
            # Use a small range to find the exact node
            nodes_at_time = self._lattice.nodes_in_range(
                target_time - 0.001, target_time + 0.001
            )
            
            if not nodes_at_time:
                # No node at this exact time - skip
                continue
            
            # Use the merged HLLSet of the first node at this time
            # This is the ACTUAL content at this level, not cumulative
            level_node = nodes_at_time[0]
            level_delta = level_node.merged
            level_id = level_node.node_id
            level_card = level_delta.cardinality()
            
            if level_card < 1.0:
                # Empty level
                trace.append(self._make_attention_record(
                    depth, target_time, level_id,
                    level_delta, HLLSet(p_bits=self._p_bits),
                    0.0, 1.0, context
                ))
                flux_history.append(0.0)
                continue
            
            # KEY METRIC: How much of PROMPT is contained in this level?
            # τ(prompt → level) = overlap / prompt_size
            # If high, this level has content related to our query
            intersection = prompt_hll.intersect(level_delta)
            overlap_card = intersection.cardinality()
            
            if prompt_card > 0:
                tau_prompt_in_level = overlap_card / prompt_card
            else:
                tau_prompt_in_level = 0.0
            
            # Also compute reverse: what fraction of level overlaps with prompt
            if level_card > 0:
                tau_level_in_prompt = overlap_card / level_card
            else:
                tau_level_in_prompt = 0.0
            
            # Noise ratio: how much of level is NOT in prompt (high = level is broad)
            rho = 1.0 - tau_level_in_prompt
            
            # What's new in this level that we don't have yet
            new_content = level_delta.diff(context)
            new_card = new_content.cardinality()
            
            # SELECTION LOGIC:
            # If prompt tokens exist in this level, it's relevant!
            # Temperature controls HOW MUCH of the level we take:
            #   temp=0.0: only the overlapping tokens
            #   temp=0.5: overlap + some surrounding context  
            #   temp=1.0: entire level (exploratory)
            
            if tau_prompt_in_level < min_overlap_ratio and overlap_card < 1.0:
                # No meaningful overlap - skip this level
                trace.append(self._make_attention_record(
                    depth, target_time, level_id,
                    new_content, HLLSet(p_bits=self._p_bits),
                    tau_prompt_in_level, rho, context
                ))
                flux_history.append(0.0)
                
                if self._check_noether_convergence(flux_history):
                    convergence_reason = ConvergenceReason.NOETHER_CONVERGED
                    break
                continue
            
            # Level is relevant! Select based on temperature:
            if self._temperature > 0.8:
                # High temp: take entire level (exploratory)
                selected = new_content
            elif self._temperature > 0.4:
                # Medium temp: take entire level if good overlap, else just new content
                if tau_prompt_in_level > 0.3:
                    selected = new_content  # Good overlap, take all new content
                else:
                    # Moderate overlap - take the intersection expanded
                    selected = new_content
            else:
                # Low temp: only take content that directly overlaps
                # But wait - the overlap IS in the prompt, not new!
                # We need the NEW content that RELATES to prompt
                # Since we can't easily determine that, take new content
                # but from levels with high overlap only
                if tau_prompt_in_level > 0.5:
                    selected = new_content
                else:
                    # Skip - not enough direct overlap for strict selection
                    selected = HLLSet(p_bits=self._p_bits)
            
            selected_card = selected.cardinality()
            
            # REVISED NOISE HANDLING:
            # High ρ only matters when τ is LOW - it means "unrelated broad content"
            # High ρ with HIGH τ means "relevant content with extra context" - that's OK!
            # 
            # Only filter when: low τ AND high ρ (truly irrelevant noise)
            # With τ=0.71 and ρ=0.91, we SHOULD include the content
            if tau_prompt_in_level < 0.3 and rho > self._rho_threshold and selected_card > 0:
                # Low overlap AND high noise - truly irrelevant, skip
                selected = HLLSet(p_bits=self._p_bits)
                selected_card = 0
            
            # Update context
            if selected_card > 0:
                new_context = context.union(selected)
                
                # Record
                trace.append(self._make_attention_record(
                    depth, target_time, level_id,
                    new_content, selected,
                    tau_prompt_in_level, rho, new_context
                ))
                flux_history.append(selected_card)
                
                # Store extracted HLLSet
                selected_id = compute_sha1(selected.dump_numpy())
                extracted_hllsets[selected_id] = selected
                
                context = new_context
            else:
                trace.append(self._make_attention_record(
                    depth, target_time, level_id,
                    new_content, HLLSet(p_bits=self._p_bits),
                    tau_prompt_in_level, rho, context
                ))
                flux_history.append(0.0)
            
            # Check Noether convergence
            if self._check_noether_convergence(flux_history):
                convergence_reason = ConvergenceReason.NOETHER_CONVERGED
                break
            
            # Check if we've collected everything
            if new_card < self._exhaust_threshold:
                convergence_reason = ConvergenceReason.COMPLEMENT_EXHAUSTED
                break
        
        # 6. Build HLLSet graph
        hllset_graph = self._build_graph(extracted_hllsets, query_id)
        
        # 7. Compute result
        result_id = compute_sha1(context.dump_numpy())
        total_gain = sum(flux_history)
        
        return TransformerResult(
            result_id=result_id,
            query=prompt_hll,
            final_context=context,
            hllset_graph=hllset_graph,
            attention_trace=tuple(trace),
            depths_traversed=len(trace),
            total_information_gain=total_gain,
            convergence_reason=convergence_reason,
            noether_flux_history=tuple(flux_history),
            timestamp=time.time() - start_time,
        )
    
    # -------------------------------------------------------------------------
    # Forward Pass (Markov Chain View)
    # -------------------------------------------------------------------------
    
    def forward_mc(
        self,
        query: Union[str, List[str], HLLSet],
        current_time: Optional[float] = None,
        time_step: float = 1.0,
    ) -> TransformerResult:
        """
        Forward pass using W-Markov Chain architecture.
        
        KEY DIFFERENCES FROM forward():
        1. W(t) treated as Markov Chain, not just lattice
        2. Dynamically build W-MC moving from top level backward
        3. Track collected HLLSets as SET OF SHA1 IDs (not merged blob)
        4. Union of HLLSets only for Noether check, keep IDs separate
        
        Algorithm:
            state = MarkovState(collected={query_id})
            
            for each level going backward:
                level_hll = lattice.node_at(t).merged
                
                # Compute τ(query → level) - does this level relate to query?
                if τ > threshold:
                    # This level is relevant - add it to our collection
                    state.add_hllset(level_hll, source=current_level_id, τ=τ)
                    
                # Complement = what's in level but NOT in our collection
                complement = level_hll.diff(state.merged_context())
                
                # If complement is empty, this level is exhausted
                if |complement| ≈ 0: continue
                
                # Noether check: has information gain stabilized?
                merged = state.merged_context()
                if noether_converged(merged): break
        
        Returns:
            TransformerResult with preserved HLLSet graph
        """
        start_time = time.time()
        
        # 1. Encode prompt
        if isinstance(query, (str, list)):
            prompt_hll = self.encode(query)
        else:
            prompt_hll = query
        
        prompt_id = compute_sha1(prompt_hll.dump_numpy())
        prompt_card = prompt_hll.cardinality()
        
        # 2. Get lattice bounds
        if current_time is None:
            latest = self._lattice.latest_node()
            current_time = latest.timestamp if latest else 0.0
        
        earliest = self._get_earliest_time()
        
        # 3. Initialize Markov State
        state = MarkovState(
            collected_ids={prompt_id},
            collected_hllsets={prompt_id: prompt_hll},
            transitions=[],
            current_level_id=prompt_id,
            depth=0,
        )
        
        # 4. Tracking
        trace: List[AttentionRecord] = []
        flux_history: List[float] = []
        convergence_reason = ConvergenceReason.MAX_DEPTH_REACHED
        
        # Temperature-adjusted threshold
        min_tau = self._tau_threshold * (1.0 - self._temperature)
        
        # 5. W-Markov Chain traversal (backward through time)
        prev_merged_card = prompt_card
        
        for depth in range(0, self._max_depth + 1):
            target_time = current_time - depth * time_step
            
            # Boundary check
            if target_time < earliest:
                convergence_reason = ConvergenceReason.LATTICE_BOUNDARY
                break
            
            # Get level's HLLSet
            nodes_at_time = self._lattice.nodes_in_range(
                target_time - 0.001, target_time + 0.001
            )
            
            if not nodes_at_time:
                continue
            
            level_node = nodes_at_time[0]
            level_hll = level_node.merged
            level_id = level_node.node_id
            level_card = level_hll.cardinality()
            
            if level_card < 1.0:
                continue
            
            # MARKOV TRANSITION CHECK:
            # τ(prompt → level) = how much of prompt is in this level
            intersection = prompt_hll.intersect(level_hll)
            overlap_card = intersection.cardinality()
            
            tau_prompt = overlap_card / prompt_card if prompt_card > 0 else 0.0
            tau_level = overlap_card / level_card if level_card > 0 else 0.0
            rho = 1.0 - tau_level
            
            # COMPLEMENT: What's in this level that we DON'T have yet?
            current_merged = state.merged_context()
            complement = level_hll.diff(current_merged)
            complement_card = complement.cardinality()
            
            # If no new content, this level is exhausted for us
            if complement_card < self._exhaust_threshold:
                trace.append(self._make_attention_record(
                    depth, target_time, level_id,
                    complement, HLLSet(p_bits=self._p_bits),
                    tau_prompt, rho, current_merged
                ))
                flux_history.append(0.0)
                
                if self._check_noether_convergence(flux_history):
                    convergence_reason = ConvergenceReason.NOETHER_CONVERGED
                    break
                continue
            
            # SELECTION: Does this level relate to our query?
            # High τ_prompt means query tokens are in this level
            if tau_prompt >= min_tau or overlap_card >= 1.0:
                # This level is RELEVANT - add the WHOLE level to our collection
                # (not just the complement, but the level's original HLLSet)
                new_id = state.add_hllset(
                    level_hll, 
                    source_id=state.current_level_id,
                    tau=tau_prompt
                )
                
                # Update current position in Markov Chain
                state.current_level_id = new_id
                state.depth = depth
                
                # Record the selection (complement is what was NEW)
                new_merged = state.merged_context()
                information_gain = new_merged.cardinality() - prev_merged_card
                
                trace.append(self._make_attention_record(
                    depth, target_time, level_id,
                    complement, level_hll,  # selected = whole level
                    tau_prompt, rho, new_merged
                ))
                flux_history.append(information_gain)
                prev_merged_card = new_merged.cardinality()
                
            else:
                # Low relevance - skip this level
                trace.append(self._make_attention_record(
                    depth, target_time, level_id,
                    complement, HLLSet(p_bits=self._p_bits),
                    tau_prompt, rho, current_merged
                ))
                flux_history.append(0.0)
            
            # Noether convergence check
            if self._check_noether_convergence(flux_history):
                convergence_reason = ConvergenceReason.NOETHER_CONVERGED
                break
        
        # 6. Build graph from collected HLLSets (preserves individual IDs!)
        hllset_graph = self._build_graph_from_state(state, prompt_id)
        
        # 7. Final merged context (only for output)
        final_context = state.merged_context()
        result_id = compute_sha1(final_context.dump_numpy())
        
        return TransformerResult(
            result_id=result_id,
            query=prompt_hll,
            final_context=final_context,
            hllset_graph=hllset_graph,
            attention_trace=tuple(trace),
            depths_traversed=len(trace),
            total_information_gain=sum(flux_history),
            convergence_reason=convergence_reason,
            noether_flux_history=tuple(flux_history),
            timestamp=time.time() - start_time,
        )
    
    def _build_graph_from_state(
        self,
        state: MarkovState,
        query_id: str,
    ) -> HLLSetGraph:
        """
        Build HLLSetGraph from Markov state.
        
        Uses the collected_hllsets and transitions to create
        a graph with proper provenance.
        """
        # Convert transitions to edges
        edges: List[HLLSetEdge] = []
        
        for from_id, to_id, tau in state.transitions:
            # Get ρ from BSS
            if from_id in state.collected_hllsets and to_id in state.collected_hllsets:
                _, rho = bss(
                    state.collected_hllsets[from_id],
                    state.collected_hllsets[to_id]
                )
            else:
                rho = 0.0
            
            edges.append(HLLSetEdge(from_id, to_id, tau, rho))
        
        return HLLSetGraph(
            nodes=dict(state.collected_hllsets),
            edges=tuple(edges),
            query_id=query_id,
        )
    
    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    
    def _get_earliest_time(self) -> float:
        """Get earliest timestamp in lattice."""
        nodes = self._lattice.all_nodes()
        if not nodes:
            return 0.0
        return min(n.timestamp for n in nodes)
    
    def _make_attention_record(
        self,
        depth: int,
        level_time: float,
        level_id: str,
        complement: HLLSet,
        selected: HLLSet,
        tau: float,
        rho: float,
        accumulated: HLLSet,
    ) -> AttentionRecord:
        """Create immutable attention record."""
        return AttentionRecord(
            depth=depth,
            level_time=level_time,
            level_id=level_id,
            complement=complement,
            complement_cardinality=complement.cardinality(),
            selected=selected,
            selection_cardinality=selected.cardinality(),
            tau_to_query=tau,
            rho_noise=rho,
            accumulated_context=accumulated,
            timestamp=time.time(),
        )
    
    def _check_noether_convergence(self, flux_history: List[float]) -> bool:
        """
        Check if flux has stabilized (Noether conservation).
        
        Converged when recent flux is consistently near zero.
        """
        if len(flux_history) < self._conv_window:
            return False
        
        recent = flux_history[-self._conv_window:]
        max_flux = max(abs(f) for f in recent)
        
        # Use relative tolerance if we have meaningful context
        return max_flux < self._exhaust_threshold
    
    def _build_graph(
        self,
        hllsets: Dict[str, HLLSet],
        query_id: str,
    ) -> HLLSetGraph:
        """
        Build HLLSet graph with τ edges.
        
        Computes pairwise BSS for all extracted HLLSets.
        """
        edges: List[HLLSetEdge] = []
        
        node_ids = list(hllsets.keys())
        for i, id_a in enumerate(node_ids):
            for id_b in node_ids[i+1:]:
                hll_a = hllsets[id_a]
                hll_b = hllsets[id_b]
                
                # Bidirectional BSS
                tau_ab, rho_ab = bss(hll_a, hll_b)
                tau_ba, rho_ba = bss(hll_b, hll_a)
                
                # Add edges above threshold
                if tau_ab > self._tau_threshold:
                    edges.append(HLLSetEdge(id_a, id_b, tau_ab, rho_ab))
                if tau_ba > self._tau_threshold:
                    edges.append(HLLSetEdge(id_b, id_a, tau_ba, rho_ba))
        
        return HLLSetGraph(
            nodes=hllsets,
            edges=tuple(edges),
            query_id=query_id,
        )


# =============================================================================
# Disambiguation Helper
# =============================================================================

def _disambiguate_graph(
    hllset_graph: HLLSetGraph,
    engine,           # DisambiguationEngine
    ngram_registry,   # Optional GlobalNGramRegistry
) -> DocumentGraph:
    """
    Disambiguate HLLSet graph to document graph.
    
    Preserves τ-edge structure through disambiguation.
    """
    doc_nodes: Dict[str, DocumentNode] = {}
    
    for node_id, hllset in hllset_graph.nodes.items():
        # Recover tokens
        tokens = engine.disambiguate_hllset(hllset)
        
        # Recover sequence ordering if registry available
        if ngram_registry is not None:
            try:
                from .hllset_debruijn import restore_sequence_debruijn
                text = restore_sequence_debruijn(tokens, ngram_registry)
            except Exception:
                text = " ".join(tokens)
        else:
            text = " ".join(tokens)
        
        doc_nodes[node_id] = DocumentNode(
            node_id=node_id,
            text=text,
            tokens=tuple(tokens),
            cardinality=hllset.cardinality(),
            source_depth=0,  # Could track from attention trace
        )
    
    # Convert edges (same structure, just different node types)
    doc_edges = [
        DocumentEdge(
            source_id=e.source_id,
            target_id=e.target_id,
            tau=e.tau,
            rho=e.rho,
        )
        for e in hllset_graph.edges
    ]
    
    return DocumentGraph(
        nodes=doc_nodes,
        edges=doc_edges,
        query_text="",  # Could pass through from result
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def transformer_forward(
    lattice: HLLLattice,
    query: Union[str, List[str], HLLSet],
    **kwargs,
) -> TransformerResult:
    """
    One-call transformer forward pass.
    
    Convenience wrapper around HLLSetTransformer.
    """
    transformer = HLLSetTransformer(lattice, **kwargs)
    return transformer.forward(query)
