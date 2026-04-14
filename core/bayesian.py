"""
Bayesian Interpretation Layer for HLLSet Algebra.

This module provides a Bayesian probabilistic interpretation of HLLSet
operations, complementing the existing evolutionary (Noether) interpretation.

The Two Interpretations of R(t+1) = [R(t) \\ D(t)] ∪ N(t)
==========================================================

**Evolution interpretation** (noether.py, evolution.py):
    The state transition is a *physical fact*. The system changed.
    Conservation laws govern the popcount. Flux Φ(t) measures growth/decay.
    Branches are parallel realities. Merges combine histories.

**Bayesian interpretation** (this module):
    The state transition is a *belief update*. Our confidence changed.
    Bayes' theorem governs conditional probabilities.
    The universe U defines the context for inference.
    Priors become posteriors through evidence.

The two interpretations are *complementary* — like wave and particle
descriptions of the same quantum reality. They agree on the *data*
(same HLLSets, same operations) but disagree on the *meaning*:

- Evolution says "the system grew by 10 tokens" (absolute)
- Bayesian says "entity A's probability decreased from 0.8 to 0.6"
  (relative to a growing universe)

They can *compete* — see `interpretation_divergence()` — in cases where
the evolutionary view sees stability (Φ ≈ 0, popcount conserved) but
the Bayesian view sees dramatic shifts (conditional probabilities
changing as the universe composition shifts).

Key Insight (§ SGS Universe from the Bayesian Analysis document):
    "There is no 'universal' universe. What was deemed acceptable
    a few years ago may no longer hold today."

This means the *same* HLLSet A can have different probabilities
depending on which universe U we condition on — head, tail slice,
or cumulative. The evolution interpretation has no such ambiguity:
the state is what it is.

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
import math
import numpy as np


# =============================================================================
# Data Types
# =============================================================================

class BayesianResult(NamedTuple):
    """Result of a Bayesian probability computation."""
    value: float          # The probability value P(·)
    numerator_card: float # |numerator set| (intersection or set itself)
    denominator_card: float # |denominator set| (universe or conditioning set)
    description: str      # Human-readable description


@dataclass
class BayesTheoremResult:
    """
    Full Bayes' theorem verification:
        P(A|B) = P(B|A) · P(A) / P(B)
    """
    p_a_given_b: float        # P(A|B) — posterior
    p_b_given_a: float        # P(B|A) — likelihood
    p_a: float                # P(A) — prior
    p_b: float                # P(B) — evidence
    bayes_rhs: float          # P(B|A)·P(A)/P(B) — RHS of Bayes' theorem
    consistency_error: float  # |LHS - RHS| (should be ~0 for exact sets)

    def is_consistent(self, tolerance: float = 0.15) -> bool:
        """Check Bayes consistency within HLL approximation error."""
        return self.consistency_error < tolerance

    def __repr__(self) -> str:
        return (
            f"BayesTheorem(P(A|B)={self.p_a_given_b:.4f}, "
            f"P(B|A)·P(A)/P(B)={self.bayes_rhs:.4f}, "
            f"error={self.consistency_error:.4f})"
        )


@dataclass
class InterpretationComparison:
    """
    Side-by-side comparison of evolutionary and Bayesian interpretations
    of the same state transition.
    """
    # Evolution view
    flux: float                   # Φ(t) = |N| - |D|
    popcount_delta: int           # Δ popcount
    evolution_verdict: str        # "growth" / "decay" / "balanced"

    # Bayesian view
    prior_probability: float      # P_t(A) before transition
    posterior_probability: float  # P_{t+1}(A) after transition
    probability_delta: float      # P_{t+1}(A) - P_t(A)
    bayesian_verdict: str         # "strengthened" / "weakened" / "stable"

    # Divergence
    interpretations_agree: bool   # Do both views say the same thing?
    divergence_note: str          # Explanation when they disagree

    def __repr__(self) -> str:
        agree = "AGREE" if self.interpretations_agree else "DIVERGE"
        return (
            f"Comparison({agree}: "
            f"evo={self.evolution_verdict}(Φ={self.flux:+.1f}), "
            f"bayes={self.bayesian_verdict}(ΔP={self.probability_delta:+.4f}))"
        )


@dataclass
class TemporalBayesRecord:
    """Record of Bayesian probability at a point in time."""
    timestamp: float
    p_a: float              # P(A) in the current universe
    universe_card: float    # |U(t)|
    entity_card: float      # |A(t)| or |A ∩ U(t)|
    surprise: float         # -log₂(P(A)) — Shannon surprise


# =============================================================================
# Core Bayesian Functions
# =============================================================================

def prior(entity, universe) -> BayesianResult:
    """
    Compute prior probability P(A) = |A| / |U|.

    This is the simplest Bayesian quantity: how much of the universe
    does entity A occupy?

    Args:
        entity: HLLSet A
        universe: HLLSet U (must contain A, conceptually)

    Returns:
        BayesianResult with P(A)

    Note:
        Because HLLSets use probabilistic cardinality, P(A) can
        exceed 1.0 in edge cases. This is epistemic uncertainty —
        a feature, not a bug, in the Bayesian interpretation.
    """
    a_card = entity.cardinality()
    u_card = universe.cardinality()

    if u_card <= 0:
        return BayesianResult(0.0, a_card, u_card, "P(A)=0 (empty universe)")

    p = a_card / u_card
    return BayesianResult(p, a_card, u_card, f"P(A) = |A|/|U| = {a_card:.1f}/{u_card:.1f}")


def conditional(a, b) -> BayesianResult:
    """
    Compute conditional probability P(A|B) = |A ∩ B| / |B|.

    "Given that we observe B, what is the probability of A?"

    Args:
        a: HLLSet A (the hypothesis)
        b: HLLSet B (the evidence / conditioning set)

    Returns:
        BayesianResult with P(A|B)
    """
    intersection = a.intersect(b)
    ab_card = intersection.cardinality()
    b_card = b.cardinality()

    if b_card <= 0:
        return BayesianResult(0.0, ab_card, b_card, "P(A|B)=0 (empty B)")

    p = ab_card / b_card
    return BayesianResult(p, ab_card, b_card, f"P(A|B) = |A∩B|/|B| = {ab_card:.1f}/{b_card:.1f}")


def joint(a, b, universe) -> BayesianResult:
    """
    Compute joint probability P(A ∩ B) = |A ∩ B| / |U|.

    Args:
        a: HLLSet A
        b: HLLSet B
        universe: HLLSet U

    Returns:
        BayesianResult with P(A ∩ B)
    """
    intersection = a.intersect(b)
    ab_card = intersection.cardinality()
    u_card = universe.cardinality()

    if u_card <= 0:
        return BayesianResult(0.0, ab_card, u_card, "P(A∩B)=0 (empty universe)")

    p = ab_card / u_card
    return BayesianResult(p, ab_card, u_card, f"P(A∩B) = |A∩B|/|U| = {ab_card:.1f}/{u_card:.1f}")


def bayes_theorem(a, b, universe) -> BayesTheoremResult:
    """
    Verify Bayes' theorem: P(A|B) = P(B|A) · P(A) / P(B).

    This function computes both sides independently and measures
    the consistency error. For exact sets, error = 0. For HLLSets,
    the HLL approximation error creates a small discrepancy that
    represents *epistemic uncertainty* about set membership.

    Args:
        a: HLLSet A
        b: HLLSet B
        universe: HLLSet U

    Returns:
        BayesTheoremResult with both sides and error
    """
    p_a_given_b = conditional(a, b).value
    p_b_given_a = conditional(b, a).value
    p_a = prior(a, universe).value
    p_b = prior(b, universe).value

    # RHS of Bayes' theorem
    if p_b > 0:
        bayes_rhs = p_b_given_a * p_a / p_b
    else:
        bayes_rhs = 0.0

    error = abs(p_a_given_b - bayes_rhs)

    return BayesTheoremResult(
        p_a_given_b=p_a_given_b,
        p_b_given_a=p_b_given_a,
        p_a=p_a,
        p_b=p_b,
        bayes_rhs=bayes_rhs,
        consistency_error=error,
    )


# =============================================================================
# Information-Theoretic Measures
# =============================================================================

def surprise(entity, universe) -> float:
    """
    Shannon surprise (self-information): S(A) = -log₂(P(A)).

    Low probability → high surprise. This measures how "unexpected"
    entity A is in the context of universe U.

    Args:
        entity: HLLSet A
        universe: HLLSet U

    Returns:
        Surprise in bits. Returns inf for P(A)=0.
    """
    p = prior(entity, universe).value
    if p <= 0:
        return float('inf')
    if p >= 1.0:
        return 0.0
    return -math.log2(p)


def entropy_of_partition(entities: list, universe) -> float:
    """
    Shannon entropy of a partition: H = -Σ P(Aᵢ) log₂ P(Aᵢ).

    Measures the uncertainty in the system when the universe is
    partitioned into entities. Maximum entropy = uniform distribution.

    Args:
        entities: List of HLLSets forming a (possibly overlapping) partition
        universe: HLLSet U

    Returns:
        Entropy in bits
    """
    h = 0.0
    for entity in entities:
        p = prior(entity, universe).value
        if 0 < p < 1:
            h -= p * math.log2(p)
        # p=0 or p≥1 contribute 0 to entropy
    return h


def kl_divergence(entities: list, universe_p, universe_q) -> float:
    """
    KL divergence D_KL(P || Q) = Σ P(Aᵢ) log₂(P(Aᵢ)/Q(Aᵢ)).

    Measures how different the Bayesian view is under two different
    universes. This is the key measure for temporal evolution:
    how much did our beliefs change when the universe shifted
    from U_p to U_q?

    This directly addresses the SGS insight: the same entities
    have different probabilities under different "local universes."

    Args:
        entities: List of HLLSets (the hypotheses)
        universe_p: HLLSet U_p (reference universe, e.g., t-1)
        universe_q: HLLSet U_q (comparison universe, e.g., t)

    Returns:
        KL divergence in bits (always ≥ 0)
    """
    kl = 0.0
    for entity in entities:
        p = prior(entity, universe_p).value
        q = prior(entity, universe_q).value
        if p > 0 and q > 0:
            kl += p * math.log2(p / q)
    return kl


def bayesian_surprise_temporal(entity, universe_before, universe_after) -> float:
    """
    Bayesian surprise: how much did the probability of entity A change
    when the universe evolved from U_before to U_after?

    ΔS = S_after(A) - S_before(A) = log₂(P_before(A)) - log₂(P_after(A))

    Positive ΔS = entity became more surprising (less probable)
    Negative ΔS = entity became less surprising (more probable)

    This quantity has no analogue in the evolution interpretation.

    Args:
        entity: HLLSet A
        universe_before: HLLSet U(t)
        universe_after: HLLSet U(t+1)

    Returns:
        Change in surprise (bits)
    """
    s_before = surprise(entity, universe_before)
    s_after = surprise(entity, universe_after)

    # Handle infinities gracefully
    if math.isinf(s_before) and math.isinf(s_after):
        return 0.0
    if math.isinf(s_before):
        return -float('inf')  # Went from impossible to possible
    if math.isinf(s_after):
        return float('inf')   # Went from possible to impossible

    return s_after - s_before


# =============================================================================
# Temporal Bayesian Analysis (connects to HLLLattice — Tutorial 03)
# =============================================================================

def temporal_posterior(lattice, entity, t: float) -> BayesianResult:
    """
    Compute P_t(A) = |A ∩ U(t)| / |U(t)| using the lattice's cumulative
    universe at time t.

    This is the Bayesian "posterior" in the temporal sense: given
    everything we've observed up to time t, what is the probability
    of entity A?

    Args:
        lattice: HLLLattice instance
        entity: HLLSet A
        t: Time point

    Returns:
        BayesianResult with P_t(A)
    """
    universe_t = lattice.cumulative(t=t)
    intersection = entity.intersect(universe_t)
    a_card = intersection.cardinality()
    u_card = universe_t.cardinality()

    if u_card <= 0:
        return BayesianResult(0.0, a_card, u_card, f"P_{t}(A)=0 (empty universe at t={t})")

    p = a_card / u_card
    return BayesianResult(
        p, a_card, u_card,
        f"P_{t}(A) = |A∩U({t})|/|U({t})| = {a_card:.1f}/{u_card:.1f}"
    )


def temporal_trajectory(lattice, entity, timestamps: List[float]) -> List[TemporalBayesRecord]:
    """
    Track how P(A) evolves over a sequence of time points.

    This produces the "Bayesian trajectory" — the time-series of
    an entity's probability. The evolution interpretation would
    show cardinality; the Bayesian interpretation shows probability.

    Args:
        lattice: HLLLattice instance
        entity: HLLSet A
        timestamps: Time points to evaluate

    Returns:
        List of TemporalBayesRecord
    """
    records = []
    for t in timestamps:
        result = temporal_posterior(lattice, entity, t)
        s = -math.log2(result.value) if 0 < result.value < 1 else 0.0
        records.append(TemporalBayesRecord(
            timestamp=t,
            p_a=result.value,
            universe_card=result.denominator_card,
            entity_card=result.numerator_card,
            surprise=s,
        ))
    return records


# =============================================================================
# Evolution vs. Bayesian: Complementarity and Competition
# =============================================================================

def interpretation_divergence(
    entity,
    state_before,
    state_after,
    universe_before,
    universe_after,
    additions=None,
    deletions=None,
) -> InterpretationComparison:
    """
    Compare evolutionary and Bayesian interpretations of the SAME
    state transition. This is the core "competition" function.

    The two interpretations can DIVERGE in interesting ways:

    Case 1 — Agreement (common):
        Evolution: Φ > 0, system grew
        Bayesian: P(A) increased (entity grew faster than universe)

    Case 2 — Evolution stable, Bayesian shifting:
        Evolution: Φ ≈ 0, balanced (conservation holds)
        Bayesian: P(A) decreased (universe composition shifted,
                  even though total size is stable)

    Case 3 — Evolution growing, Bayesian weakening:
        Evolution: Φ > 0, system expanding
        Bayesian: P(A) decreased (universe grew faster than A)
        The "dilution effect" — growth can weaken beliefs.

    Case 4 — Evolution decaying, Bayesian strengthening:
        Evolution: Φ < 0, system contracting
        Bayesian: P(A) increased (universe shrank faster than A)
        The "concentration effect" — decay can strengthen beliefs.

    Args:
        entity: HLLSet A (the entity being tracked)
        state_before: HLLSet R(t)
        state_after: HLLSet R(t+1)
        universe_before: HLLSet U(t)
        universe_after: HLLSet U(t+1)
        additions: HLLSet N(t) (optional, for flux computation)
        deletions: HLLSet D(t) (optional, for flux computation)

    Returns:
        InterpretationComparison
    """
    # --- Evolution view ---
    if additions is not None and deletions is not None:
        n_card = additions.cardinality()
        d_card = deletions.cardinality()
        flux = n_card - d_card
    else:
        # Estimate from state change
        flux = state_after.cardinality() - state_before.cardinality()

    pop_before = _popcount(state_before)
    pop_after = _popcount(state_after)
    popcount_delta = pop_after - pop_before

    if flux > 1.0:
        evo_verdict = "growth"
    elif flux < -1.0:
        evo_verdict = "decay"
    else:
        evo_verdict = "balanced"

    # --- Bayesian view ---
    p_before = prior(entity, universe_before).value
    p_after = prior(entity, universe_after).value
    p_delta = p_after - p_before

    threshold = 0.02  # 2% change threshold
    if p_delta > threshold:
        bayes_verdict = "strengthened"
    elif p_delta < -threshold:
        bayes_verdict = "weakened"
    else:
        bayes_verdict = "stable"

    # --- Divergence analysis ---
    # They agree if both see improvement, both see decline, or both see stability
    evo_direction = 1 if flux > 1 else (-1 if flux < -1 else 0)
    bayes_direction = 1 if p_delta > threshold else (-1 if p_delta < -threshold else 0)

    agree = (evo_direction == bayes_direction) or (evo_direction == 0 and bayes_direction == 0)

    if agree:
        note = "Both interpretations agree on the direction of change."
    elif evo_direction > 0 and bayes_direction < 0:
        note = (
            "DILUTION: Evolution sees growth (Φ>0), but Bayesian sees weakening. "
            "The universe grew faster than the entity — absolute growth, "
            "relative decline. Like a company whose revenue grows but "
            "market share shrinks."
        )
    elif evo_direction < 0 and bayes_direction > 0:
        note = (
            "CONCENTRATION: Evolution sees decay (Φ<0), but Bayesian sees "
            "strengthening. The universe shrank faster than the entity — "
            "absolute decline, relative growth. Like a company that "
            "loses revenue but gains market share in a shrinking market."
        )
    elif evo_direction == 0 and bayes_direction != 0:
        note = (
            "HIDDEN SHIFT: Evolution sees balance (Φ≈0, conservation holds), "
            "but Bayesian detects a probability shift. The total quantity "
            "is conserved, but the composition changed. Like a balanced "
            "budget where spending priorities shifted dramatically."
        )
    else:
        note = (
            f"Divergence: Evolution={evo_verdict}, Bayesian={bayes_verdict}. "
            "The two interpretations see different aspects of the same transition."
        )

    return InterpretationComparison(
        flux=flux,
        popcount_delta=popcount_delta,
        evolution_verdict=evo_verdict,
        prior_probability=p_before,
        posterior_probability=p_after,
        probability_delta=p_delta,
        bayesian_verdict=bayes_verdict,
        interpretations_agree=agree,
        divergence_note=note,
    )


# =============================================================================
# De Bruijn Bayesian Path Selection (connects to Tutorial 08)
# =============================================================================

def edge_probability(edge_multiplicity: int, total_edges: int) -> float:
    """
    Compute the prior probability of a De Bruijn edge based on multiplicity.

    P(edge) = multiplicity / total_edge_count

    In the Bayesian interpretation, higher-multiplicity edges
    are more probable transitions. The evolution interpretation
    sees all edges as equally valid structural connections.

    Args:
        edge_multiplicity: How many times this k-mer appeared
        total_edges: Total edge count (with multiplicities)

    Returns:
        P(edge) ∈ (0, 1]
    """
    if total_edges <= 0:
        return 0.0
    return edge_multiplicity / total_edges


def path_log_likelihood(path_edges: list, graph) -> float:
    """
    Compute log-likelihood of a De Bruijn path.

    log P(path) = Σ log P(edge_i) = Σ log(mult_i / total)

    Higher log-likelihood → more probable reconstruction.
    The evolutionary interpretation doesn't assign probabilities
    to paths — it only checks structural validity (Eulerian property).

    Args:
        path_edges: List of (source, target, label) tuples
        graph: DeBruijnGraph instance

    Returns:
        Log-likelihood (negative; closer to 0 = more likely)
    """
    total = graph.total_edge_count
    if total <= 0:
        return float('-inf')

    ll = 0.0
    for edge_tuple in path_edges:
        edge = graph.get_edge(edge_tuple)
        if edge is None:
            return float('-inf')
        p = edge.multiplicity / total
        if p <= 0:
            return float('-inf')
        ll += math.log2(p)
    return ll


# =============================================================================
# Helpers
# =============================================================================

def _popcount(hllset) -> int:
    """Total set bits in all registers."""
    registers = hllset.dump_numpy()
    return sum(int(r).bit_count() for r in registers)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data types
    'BayesianResult',
    'BayesTheoremResult',
    'InterpretationComparison',
    'TemporalBayesRecord',
    # Core Bayesian
    'prior',
    'conditional',
    'joint',
    'bayes_theorem',
    # Information theory
    'surprise',
    'entropy_of_partition',
    'kl_divergence',
    'bayesian_surprise_temporal',
    # Temporal
    'temporal_posterior',
    'temporal_trajectory',
    # Competition
    'interpretation_divergence',
    # De Bruijn
    'edge_probability',
    'path_log_likelihood',
]
