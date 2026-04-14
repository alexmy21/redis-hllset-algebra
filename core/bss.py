"""
Bell State Similarity (BSS) — Directed similarity metric and categorical morphisms.

Implements the BSS framework from the manuscript (§3):

    BSS_τ(A → B) = |A ∩ B| / |B|      (inclusion: how much of B is covered by A)
    BSS_ρ(A → B) = |A \\ B| / |B|      (exclusion: how much of A is extraneous to B)

A morphism A →(τ,ρ) B exists iff:
    BSS_τ(A → B) ≥ τ   and   BSS_ρ(A → B) ≤ ρ

Properties (Proposition 3.1):
    1. 0 ≤ BSS_τ ≤ 1,  0 ≤ BSS_ρ ≤ 1
    2. BSS_τ + BSS_ρ ≤ 1  (equality when A ⊆ B, up to estimation error)
    3. BSS_τ(A→A) = 1,  BSS_ρ(A→A) = 0  (identity morphism always exists)

The BSS pair (τ, ρ) gives a complete directed relationship picture:
    - High τ, low ρ: A covers B well with little noise  (strong morphism)
    - Low τ, high ρ: A is mostly disjoint from B         (no morphism)
    - High τ, high ρ: A covers B but adds much noise      (noisy inclusion)

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Dict, NamedTuple
from dataclasses import dataclass
import numpy as np


# =============================================================================
# BSS Result Types
# =============================================================================

class BSSPair(NamedTuple):
    """
    A (τ, ρ) pair from BSS computation.
    
    Attributes:
        tau: Inclusion score — fraction of B covered by A ∩ B
        rho: Exclusion score — fraction of extraneous A relative to B
    """
    tau: float
    rho: float


@dataclass(frozen=True)
class MorphismResult:
    """
    Result of testing whether a morphism A →(τ,ρ) B exists.
    
    Attributes:
        exists: Whether the morphism exists given the thresholds
        bss: The computed BSS pair
        source_card: Estimated cardinality of source A
        target_card: Estimated cardinality of target B
        intersection_card: Estimated cardinality of A ∩ B
        difference_card: Estimated cardinality of A \\ B
        tau_threshold: τ threshold used for the test
        rho_threshold: ρ threshold used for the test
    """
    exists: bool
    bss: BSSPair
    source_card: float
    target_card: float
    intersection_card: float
    difference_card: float
    tau_threshold: float
    rho_threshold: float

    @property
    def tau(self) -> float:
        return self.bss.tau

    @property
    def rho(self) -> float:
        return self.bss.rho

    @property
    def margin_tau(self) -> float:
        """How far above the τ threshold (positive = passes)."""
        return self.bss.tau - self.tau_threshold

    @property
    def margin_rho(self) -> float:
        """How far below the ρ threshold (positive = passes)."""
        return self.rho_threshold - self.bss.rho

    def __repr__(self) -> str:
        arrow = "→" if self.exists else "↛"
        return (
            f"Morphism({arrow}, τ={self.bss.tau:.4f}≥{self.tau_threshold}, "
            f"ρ={self.bss.rho:.4f}≤{self.rho_threshold})"
        )


# =============================================================================
# Core BSS Computation
# =============================================================================

def bss(source, target) -> BSSPair:
    """
    Compute Bell State Similarity from source A to target B.
    
    BSS_τ(A → B) = |A ∩ B| / |B|
    BSS_ρ(A → B) = |A \\ B| / |B|
    
    Args:
        source: HLLSet A (the "query" or "candidate")
        target: HLLSet B (the "reference" or "context")
        
    Returns:
        BSSPair(tau, rho) — the directed similarity scores
        
    Note:
        BSS is NOT symmetric: bss(A, B) ≠ bss(B, A) in general.
        This directionality is by design — it captures "how well does
        A cover B" vs "how well does B cover A".
    """
    target_card = target.cardinality()

    if target_card <= 0.0:
        # Degenerate case: empty target
        return BSSPair(tau=0.0, rho=1.0)

    # A ∩ B
    intersection = source.intersect(target)
    intersection_card = intersection.cardinality()

    # A \\ B
    difference = source.diff(target)
    difference_card = difference.cardinality()

    tau = min(1.0, max(0.0, intersection_card / target_card))
    rho = min(1.0, max(0.0, difference_card / target_card))

    return BSSPair(tau=tau, rho=rho)


def bss_symmetric(a, b) -> Tuple[BSSPair, BSSPair]:
    """
    Compute BSS in both directions: A→B and B→A.
    
    Returns:
        (bss_a_to_b, bss_b_to_a)
    """
    return bss(a, b), bss(b, a)


# =============================================================================
# Morphism Testing
# =============================================================================

def test_morphism(
    source,
    target,
    tau_threshold: float = 0.7,
    rho_threshold: float = 0.3,
) -> MorphismResult:
    """
    Test whether a morphism A →(τ,ρ) B exists.
    
    A morphism exists iff:
        BSS_τ(A → B) ≥ τ_threshold   (sufficient inclusion)
        BSS_ρ(A → B) ≤ ρ_threshold   (bounded exclusion)
    
    Args:
        source: HLLSet A
        target: HLLSet B
        tau_threshold: Minimum inclusion score (default 0.7)
        rho_threshold: Maximum exclusion score (default 0.3)
        
    Returns:
        MorphismResult with full diagnostics
        
    Raises:
        ValueError: If thresholds are invalid (must have 0 ≤ ρ < τ ≤ 1)
        
    Example:
        >>> result = test_morphism(hll_a, hll_b, tau_threshold=0.8, rho_threshold=0.1)
        >>> if result.exists:
        ...     print(f"A covers B with τ={result.tau:.3f}, noise ρ={result.rho:.3f}")
    """
    if not (0 <= rho_threshold < tau_threshold <= 1):
        raise ValueError(
            f"Must have 0 ≤ ρ < τ ≤ 1, got ρ={rho_threshold}, τ={tau_threshold}"
        )

    pair = bss(source, target)

    source_card = source.cardinality()
    target_card = target.cardinality()
    intersection = source.intersect(target)
    difference = source.diff(target)

    exists = pair.tau >= tau_threshold and pair.rho <= rho_threshold

    return MorphismResult(
        exists=exists,
        bss=pair,
        source_card=source_card,
        target_card=target_card,
        intersection_card=intersection.cardinality(),
        difference_card=difference.cardinality(),
        tau_threshold=tau_threshold,
        rho_threshold=rho_threshold,
    )


# =============================================================================
# BSS Matrix — Pairwise directed similarity
# =============================================================================

def bss_matrix(hllsets: List, labels: Optional[List[str]] = None) -> Dict:
    """
    Compute pairwise BSS matrix for a collection of HLLSets.
    
    Returns both τ-matrix and ρ-matrix. Entry (i,j) is BSS(i → j).
    
    Args:
        hllsets: List of HLLSet objects
        labels: Optional names for each HLLSet
        
    Returns:
        Dict with keys:
            'tau_matrix': np.ndarray of shape (n, n) — inclusion scores
            'rho_matrix': np.ndarray of shape (n, n) — exclusion scores
            'labels': List[str] — labels used
            'morphisms': List[Tuple] — (i, j, tau, rho) for all strong morphisms
    """
    n = len(hllsets)
    if labels is None:
        labels = [f"S{i}" for i in range(n)]

    tau_mat = np.zeros((n, n), dtype=np.float64)
    rho_mat = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            pair = bss(hllsets[i], hllsets[j])
            tau_mat[i, j] = pair.tau
            rho_mat[i, j] = pair.rho

    return {
        'tau_matrix': tau_mat,
        'rho_matrix': rho_mat,
        'labels': labels,
    }


def morphism_graph(
    hllsets: List,
    tau_threshold: float = 0.7,
    rho_threshold: float = 0.3,
    labels: Optional[List[str]] = None,
) -> Dict:
    """
    Build the directed morphism graph for a collection of HLLSets.
    
    An edge i → j exists iff BSS_τ(i→j) ≥ τ and BSS_ρ(i→j) ≤ ρ.
    
    This is the W-graph / categorical structure from the manuscript.
    
    Args:
        hllsets: List of HLLSet objects
        tau_threshold: Minimum inclusion score for morphism
        rho_threshold: Maximum exclusion score for morphism
        labels: Optional names for each HLLSet
        
    Returns:
        Dict with keys:
            'edges': List of (source_idx, target_idx, tau, rho) tuples
            'adjacency': Dict[int, List[int]] — adjacency list
            'labels': List[str]
            'node_count': int
            'edge_count': int
    """
    n = len(hllsets)
    if labels is None:
        labels = [f"S{i}" for i in range(n)]

    edges = []
    adjacency: Dict[int, List[int]] = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # Identity morphism always exists, skip
            pair = bss(hllsets[i], hllsets[j])
            if pair.tau >= tau_threshold and pair.rho <= rho_threshold:
                edges.append((i, j, pair.tau, pair.rho))
                adjacency[i].append(j)

    return {
        'edges': edges,
        'adjacency': adjacency,
        'labels': labels,
        'node_count': n,
        'edge_count': len(edges),
    }


# =============================================================================
# Tensor-level BSS (register-by-register analysis)
# =============================================================================

def bss_from_registers(
    source_registers: np.ndarray,
    target_registers: np.ndarray,
) -> BSSPair:
    """
    Compute BSS directly from register arrays (bypass HLLSet objects).
    
    Useful for performance-critical paths and lattice operations.
    Uses popcount as a cardinality proxy (not HLL estimation).
    
    Args:
        source_registers: uint32 array of A's registers
        target_registers: uint32 array of B's registers
        
    Returns:
        BSSPair(tau, rho) based on bit-level overlap
    """
    # Popcount of B
    target_pop = sum(int(r).bit_count() for r in target_registers)
    if target_pop == 0:
        return BSSPair(tau=0.0, rho=1.0)

    # A ∩ B = AND
    intersection = source_registers & target_registers
    intersection_pop = sum(int(r).bit_count() for r in intersection)

    # A \\ B = AND-NOT
    difference = source_registers & ~target_registers
    difference_pop = sum(int(r).bit_count() for r in difference)

    tau = intersection_pop / target_pop
    rho = difference_pop / target_pop

    return BSSPair(tau=min(1.0, tau), rho=min(1.0, rho))


# =============================================================================
# Convenience / Analysis
# =============================================================================

def bss_summary(source, target) -> str:
    """
    Human-readable summary of the BSS relationship A → B.
    """
    pair = bss(source, target)
    s_card = source.cardinality()
    t_card = target.cardinality()
    i_card = source.intersect(target).cardinality()

    lines = [
        f"BSS(A → B):",
        f"  |A| ≈ {s_card:.1f},  |B| ≈ {t_card:.1f},  |A∩B| ≈ {i_card:.1f}",
        f"  τ (inclusion) = {pair.tau:.4f}",
        f"  ρ (exclusion) = {pair.rho:.4f}",
        f"  τ + ρ         = {pair.tau + pair.rho:.4f}",
    ]

    if pair.tau > 0.9 and pair.rho < 0.1:
        lines.append("  → A nearly contains B (strong inclusion)")
    elif pair.tau < 0.1:
        lines.append("  → A and B are nearly disjoint")
    elif pair.rho < 0.05:
        lines.append("  → A ⊆ B (up to estimation error)")

    return "\n".join(lines)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'BSSPair',
    'MorphismResult',
    'bss',
    'bss_symmetric',
    'test_morphism',
    'bss_matrix',
    'morphism_graph',
    'bss_from_registers',
    'bss_summary',
]
