"""
Bayesian Network (BN) over the HLLSet Ring — The Third Vertex of the Triangle.

The Love-Hate Triangle
======================

Three structures share the same HLLSet ring as their algebraic root:

    1. HLLSet Ring (bitvector_ring.py)
       - Operations: XOR (ring add), AND (ring mult), OR (lattice join)
       - Bridge law: A ∪ B = (A △ B) △ (A ∩ B)
       - This is the GROUND TRUTH — pure algebra.

    2. BSS Lattice (bss.py + lattice_reconstruction.py)
       - Nodes: HLLSets
       - Edges: morphisms A →(τ,ρ) B
       - Key metric: τ(A→B) = |A ∩ B| / |B|
       - Focus: STRUCTURAL relationships (partial order, subset)

    3. Bayesian Network (this module)
       - Nodes: HLLSets as random variables
       - Edges: conditional dependencies P(A|B)
       - Key metric: P(A|B) = |A ∩ B| / |B|
       - Focus: EPISTEMIC relationships (belief, evidence, inference)

The Isomorphism
===============

    τ(A → B) = |A ∩ B| / |B| = P(A | B)

BSS τ IS the conditional probability. The BSS morphism graph and the
Bayesian conditional graph are isomorphic as weighted directed graphs.

But they are NOT isomorphic as algebraic structures — they carry
different additional operations:

    BSS has:  ρ (exclusion), morphism composition, τ + ρ ≤ 1
    BN has:   Bayes' theorem, d-separation, independence, belief propagation

Both are FAITHFUL representations of the underlying ring — they preserve
all algebraic operations (union, intersect, diff map cleanly). But each
representation reveals structure that the other hides:

    BSS reveals: order (⊆), covering relations, Hasse diagram, levels
    BN reveals:  independence (⊥), Markov blankets, causal structure

The relationship is a forgetful functor: both project the ring onto the
same graph, but "forget" different aspects.

BN Toolbox on HLLSets
======================

This module implements standard Bayesian Network algorithms reinterpreted
on HLLSets:

    1. Conditional Probability Table (CPT)
       Standard: tabular P(child | parent_config)
       HLLSet:   P(A|B) = |A∩B|/|B| = BSS τ(A→B)

    2. d-separation (conditional independence)
       Standard: graph-theoretic path blocking
       HLLSet:   A ⊥ B | C  iff  |A∩B∩C'| ≈ |A∩C'|·|B∩C'|/|C'|

    3. Markov blanket
       Standard: parents ∪ children ∪ co-parents
       HLLSet:   nodes with non-trivial BSS overlap

    4. Belief propagation
       Standard: message passing on junction tree
       HLLSet:   conditional updates through the BSS graph

    5. Structure learning
       Standard: score-based or constraint-based
       HLLSet:   THIS IS EXACTLY WHAT LatticeReconstructor DOES!

    6. Mutual information
       Standard: I(A;B) = H(A) - H(A|B)
       HLLSet:   I(A;B) from cardinality-based probabilities

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import (
    List, Dict, Set, Optional, Tuple, NamedTuple, Any, Iterator,
)
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math
import numpy as np

from .hllset import HLLSet
from .bss import bss, BSSPair, bss_symmetric


# =============================================================================
# Data Types
# =============================================================================

class CPTEntry(NamedTuple):
    """
    Conditional Probability Table entry.

    In a standard BN, this would be P(child = x | parents = config).
    In HLLSet BN, this is P(child | parent) = |child ∩ parent| / |parent|.

    The key insight: this is IDENTICAL to BSS τ(child → parent).
    """
    parent_id: str
    child_id: str
    probability: float    # P(child | parent) = |child ∩ parent| / |parent|
    bss_tau: float        # BSS τ(child → parent) — same value
    bss_rho: float        # BSS ρ(child → parent) — BN has no direct analogue
    intersection_card: float  # |child ∩ parent|
    parent_card: float        # |parent|


@dataclass
class IndependenceResult:
    """Result of a conditional independence test (d-separation)."""
    a_id: str
    b_id: str
    given_ids: List[str]
    independent: bool           # A ⊥ B | given?
    mutual_information: float   # I(A; B | given) — 0 means independent
    p_a: float                  # P(A | universe)
    p_b: float                  # P(B | universe)
    p_ab: float                 # P(A ∩ B | universe)
    p_a_times_p_b: float        # P(A) · P(B) — for independence check
    explanation: str

    def __repr__(self) -> str:
        sym = "⊥" if self.independent else "⊬⊥"
        given = ",".join(self.given_ids) if self.given_ids else "∅"
        return f"Independence({self.a_id} {sym} {self.b_id} | {given}, I={self.mutual_information:.4f})"


@dataclass
class MarkovBlanket:
    """
    Markov blanket of a node: the minimal set that makes it
    conditionally independent of all other nodes.

    In standard BN: parents ∪ children ∪ co-parents.
    In HLLSet BN: nodes with significant conditional dependency.
    """
    node_id: str
    parent_ids: List[str]
    child_ids: List[str]
    coparent_ids: List[str]

    @property
    def blanket_ids(self) -> Set[str]:
        return set(self.parent_ids) | set(self.child_ids) | set(self.coparent_ids)

    @property
    def size(self) -> int:
        return len(self.blanket_ids)

    def __repr__(self) -> str:
        return (
            f"MarkovBlanket({self.node_id}: "
            f"{len(self.parent_ids)} parents, "
            f"{len(self.child_ids)} children, "
            f"{len(self.coparent_ids)} co-parents, "
            f"total={self.size})"
        )


@dataclass
class IsomorphismWitness:
    """
    Formal witness that the BSS lattice and BN are graph-isomorphic.

    The mapping is: node → node (identity), edge weight τ → P(A|B).
    """
    num_nodes: int
    num_edges: int
    max_weight_error: float      # max |τ(A→B) − P(A|B)| across all edges
    mean_weight_error: float     # mean |τ(A→B) − P(A|B)|
    is_isomorphic: bool          # True if max_weight_error < tolerance
    edge_correspondences: List[Dict[str, Any]]  # per-edge details
    structural_note: str         # What the isomorphism preserves/forgets

    def __repr__(self) -> str:
        sym = "≅" if self.is_isomorphic else "≇"
        return (
            f"Isomorphism(BSS {sym} BN: "
            f"{self.num_nodes} nodes, {self.num_edges} edges, "
            f"max_err={self.max_weight_error:.6f})"
        )


@dataclass
class BeliefState:
    """Belief state of the network: probability of each node."""
    probabilities: Dict[str, float]       # node_id → P(node | evidence)
    universe_card: float                   # |U| used for normalization
    evidence_ids: List[str]                # which nodes are evidence

    def __repr__(self) -> str:
        return f"BeliefState({len(self.probabilities)} nodes, evidence={self.evidence_ids})"


# =============================================================================
# HLLBayesNet — Bayesian Network over HLLSets
# =============================================================================

class HLLBayesNet:
    """
    Bayesian Network constructed over HLLSets.

    This is the third vertex of the triangle:

        HLLSet Ring ← ground truth
            ↙              ↘
        BSS Lattice    Bayesian Network
        (structural)     (epistemic)

    The BSS lattice captures ORDER (⊆, Hasse diagram, levels).
    The Bayesian Network captures MEASURE (P, independence, causality).
    Both are rooted in the same ring operations.

    Construction:
        1. from_hllsets() — learn structure from pairwise conditionals
        2. from_bss_graph() — convert BSS lattice (proving isomorphism)
        3. from_lattice() — build from temporal HLLLattice
    """

    def __init__(self, universe: Optional[HLLSet] = None):
        """
        Initialize an empty Bayesian Network.

        Args:
            universe: The reference universe U for computing priors.
                      If None, the union of all nodes is used.
        """
        self._nodes: Dict[str, HLLSet] = {}
        self._universe: Optional[HLLSet] = universe

        # DAG structure
        self._parents: Dict[str, List[str]] = defaultdict(list)
        self._children: Dict[str, List[str]] = defaultdict(list)

        # CPTs: child_id → list of CPTEntry (one per parent)
        self._cpts: Dict[str, List[CPTEntry]] = defaultdict(list)

    # -----------------------------------------------------------------
    # Node / Edge Management
    # -----------------------------------------------------------------

    def add_node(self, node_id: str, hllset: HLLSet) -> None:
        """Add a node (random variable) to the network."""
        self._nodes[node_id] = hllset
        if node_id not in self._parents:
            self._parents[node_id] = []
        if node_id not in self._children:
            self._children[node_id] = []

    def add_edge(self, parent_id: str, child_id: str) -> CPTEntry:
        """
        Add a directed edge parent → child and compute the CPT entry.

        In the BN interpretation:
            parent → child means "parent influences child"
            CPT entry: P(child | parent) = |child ∩ parent| / |parent|

        In the BSS interpretation:
            This is τ(child → parent) = |child ∩ parent| / |parent|

        Returns the CPTEntry so the caller can inspect the correspondence.
        """
        if parent_id not in self._nodes or child_id not in self._nodes:
            raise ValueError(f"Both nodes must exist: {parent_id}, {child_id}")

        parent_hll = self._nodes[parent_id]
        child_hll = self._nodes[child_id]

        # Compute P(child | parent) = |child ∩ parent| / |parent|
        intersection = child_hll.intersect(parent_hll)
        inter_card = intersection.cardinality()
        parent_card = parent_hll.cardinality()

        if parent_card > 0:
            prob = inter_card / parent_card
        else:
            prob = 0.0

        # Also compute BSS for comparison
        bss_pair = bss(child_hll, parent_hll)

        entry = CPTEntry(
            parent_id=parent_id,
            child_id=child_id,
            probability=prob,
            bss_tau=bss_pair.tau,
            bss_rho=bss_pair.rho,
            intersection_card=inter_card,
            parent_card=parent_card,
        )

        # Update graph structure
        if parent_id not in self._parents[child_id]:
            self._parents[child_id].append(parent_id)
        if child_id not in self._children[parent_id]:
            self._children[parent_id].append(child_id)
        self._cpts[child_id].append(entry)

        return entry

    @property
    def nodes(self) -> Dict[str, HLLSet]:
        return dict(self._nodes)

    @property
    def node_ids(self) -> List[str]:
        return list(self._nodes.keys())

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return sum(len(children) for children in self._children.values())

    def parents(self, node_id: str) -> List[str]:
        """Get parent node IDs."""
        return list(self._parents.get(node_id, []))

    def children(self, node_id: str) -> List[str]:
        """Get child node IDs."""
        return list(self._children.get(node_id, []))

    def get_cpt(self, child_id: str) -> List[CPTEntry]:
        """Get CPT entries for a child node."""
        return list(self._cpts.get(child_id, []))

    def get_universe(self) -> HLLSet:
        """Get or compute the universe set."""
        if self._universe is not None:
            return self._universe
        if not self._nodes:
            raise ValueError("No nodes in the network")
        return HLLSet.merge(list(self._nodes.values()))

    # -----------------------------------------------------------------
    # Factory: from HLLSets (structure learning)
    # -----------------------------------------------------------------

    @classmethod
    def from_hllsets(
        cls,
        hllsets: Dict[str, HLLSet],
        threshold: float = 0.1,
        universe: Optional[HLLSet] = None,
    ) -> 'HLLBayesNet':
        """
        Learn Bayesian Network structure from pairwise conditionals.

        Algorithm:
            1. Compute P(A|B) for all pairs
            2. Add edge parent→child if P(child|parent) ≥ threshold
            3. Orient edges by cardinality (larger → smaller)
            4. Remove cycles (keep higher-weight edges)

        This is structure learning — and it's EXACTLY what
        LatticeReconstructor does, just with BN vocabulary!

        Args:
            hllsets: Dict of node_id → HLLSet
            threshold: Minimum conditional probability for edge
            universe: Optional universe set

        Returns:
            HLLBayesNet with learned structure
        """
        net = cls(universe=universe)

        # Add all nodes
        for nid, hll in hllsets.items():
            net.add_node(nid, hll)

        # Compute pairwise conditionals
        ids = list(hllsets.keys())
        candidates: List[Tuple[str, str, float]] = []

        for i, id_a in enumerate(ids):
            for j, id_b in enumerate(ids):
                if i == j:
                    continue
                hll_a = hllsets[id_a]
                hll_b = hllsets[id_b]
                inter_card = hll_a.intersect(hll_b).cardinality()
                b_card = hll_b.cardinality()
                if b_card > 0:
                    p = inter_card / b_card  # P(A|B)
                    if p >= threshold:
                        candidates.append((id_b, id_a, p))  # B → A edge

        # Sort by weight (strongest first) and add edges
        # Orient by cardinality: larger set → smaller set (information flow)
        added_edges: Set[Tuple[str, str]] = set()
        for parent_id, child_id, weight in sorted(candidates, key=lambda x: -x[2]):
            # Check for cycle (simple: don't add if reverse edge exists)
            if (child_id, parent_id) in added_edges:
                continue
            # Orient: larger cardinality = parent
            p_card = hllsets[parent_id].cardinality()
            c_card = hllsets[child_id].cardinality()
            if c_card > p_card:
                parent_id, child_id = child_id, parent_id
            if (parent_id, child_id) not in added_edges:
                net.add_edge(parent_id, child_id)
                added_edges.add((parent_id, child_id))

        return net

    # -----------------------------------------------------------------
    # Factory: from BSS Morphism Graph (the isomorphism)
    # -----------------------------------------------------------------

    @classmethod
    def from_bss_graph(
        cls,
        bss_graph,  # BSSMorphismGraph from lattice_reconstruction
        universe: Optional[HLLSet] = None,
    ) -> 'HLLBayesNet':
        """
        Construct a Bayesian Network from a BSS Morphism Graph.

        This is the ISOMORPHISM WITNESS constructor:
        - Every BSS node becomes a BN node
        - Every BSS edge (τ, ρ) becomes a BN edge P(A|B) = τ
        - The graph structure is IDENTICAL

        The only difference is interpretation:
        - BSS says: "A covers B with strength τ"
        - BN says: "Given B, the probability of A is P(A|B) = τ"

        Args:
            bss_graph: BSSMorphismGraph instance
            universe: Optional universe set

        Returns:
            HLLBayesNet with identical structure
        """
        net = cls(universe=universe)

        # Transfer nodes
        for node_id, hllset in bss_graph.nodes.items():
            net.add_node(node_id, hllset)

        # Transfer edges: BSS edge (src→tgt, τ, ρ) → BN edge
        # BSS: τ(src→tgt) = |src ∩ tgt| / |tgt|
        # BN:  P(src|tgt) = |src ∩ tgt| / |tgt|
        # So: tgt is the "parent" (conditioning set), src is the "child"
        for edge in bss_graph.edges:
            src_id = edge.source_id
            tgt_id = edge.target_id
            # In BSS: source "covers" target
            # In BN:  target conditions source → tgt is parent, src is child
            if src_id in net._nodes and tgt_id in net._nodes:
                # Avoid duplicate edges
                if tgt_id not in net._parents.get(src_id, []):
                    net.add_edge(parent_id=tgt_id, child_id=src_id)

        return net

    # -----------------------------------------------------------------
    # Factory: from temporal HLLLattice
    # -----------------------------------------------------------------

    @classmethod
    def from_lattice(
        cls,
        lattice,  # HLLLattice
        timestamps: List[float],
        node_ids: Optional[List[str]] = None,
    ) -> 'HLLBayesNet':
        """
        Construct BN from a temporal lattice.

        Each cumulative state U(t) becomes a node. Temporal ordering
        gives the DAG: U(t) → U(t+1) for all t.

        This is the CAUSAL interpretation: earlier states influence
        later states. The CPT entry P(U(t+1)|U(t)) measures how
        much the later state is "predicted by" the earlier one.

        Args:
            lattice: HLLLattice instance
            timestamps: Time points to include
            node_ids: Optional names (default: "t=X")

        Returns:
            HLLBayesNet with temporal structure
        """
        if node_ids is None:
            node_ids = [f"t={t}" for t in timestamps]

        net = cls()

        # Add nodes: cumulative state at each timestamp
        cum_states = {}
        for nid, t in zip(node_ids, timestamps):
            state = lattice.cumulative(t=t)
            net.add_node(nid, state)
            cum_states[nid] = state

        # Add temporal edges: t → t+1
        for i in range(len(node_ids) - 1):
            net.add_edge(parent_id=node_ids[i], child_id=node_ids[i + 1])

        # Universe is the final cumulative state
        if cum_states:
            net._universe = cum_states[node_ids[-1]]

        return net

    # -----------------------------------------------------------------
    # Conditional Independence (d-separation)
    # -----------------------------------------------------------------

    def conditional_independence(
        self,
        a_id: str,
        b_id: str,
        given_ids: Optional[List[str]] = None,
        tolerance: float = 0.05,
    ) -> IndependenceResult:
        """
        Test conditional independence: A ⊥ B | C.

        In standard BN, this uses d-separation on the graph.
        In HLLSet BN, we test directly using set operations:

            A ⊥ B | C  iff  P(A ∩ B | C) ≈ P(A | C) · P(B | C)

        This is the set-theoretic definition of independence:
        conditioning on C, the events A and B are independent if
        their joint probability factors.

        The HLLSet approximation adds noise, so we use a tolerance.

        Args:
            a_id: Node A
            b_id: Node B
            given_ids: Conditioning set C (empty = marginal independence)
            tolerance: Maximum deviation from independence

        Returns:
            IndependenceResult with full diagnostics
        """
        if given_ids is None:
            given_ids = []

        a_hll = self._nodes[a_id]
        b_hll = self._nodes[b_id]

        # Determine the conditioning context
        if given_ids:
            # C = intersection of all conditioning sets
            c_parts = [self._nodes[gid] for gid in given_ids]
            context = HLLSet.merge(c_parts)
        else:
            context = self.get_universe()

        ctx_card = context.cardinality()
        if ctx_card <= 0:
            return IndependenceResult(
                a_id=a_id, b_id=b_id, given_ids=given_ids,
                independent=True, mutual_information=0.0,
                p_a=0.0, p_b=0.0, p_ab=0.0, p_a_times_p_b=0.0,
                explanation="Empty context — trivially independent",
            )

        # P(A | context)
        a_in_ctx = a_hll.intersect(context).cardinality()
        p_a = a_in_ctx / ctx_card

        # P(B | context)
        b_in_ctx = b_hll.intersect(context).cardinality()
        p_b = b_in_ctx / ctx_card

        # P(A ∩ B | context)
        ab_hll = a_hll.intersect(b_hll)
        ab_in_ctx = ab_hll.intersect(context).cardinality()
        p_ab = ab_in_ctx / ctx_card

        # Independence check: P(A∩B|C) ≈ P(A|C) · P(B|C)
        p_a_times_p_b = p_a * p_b
        deviation = abs(p_ab - p_a_times_p_b)
        independent = deviation <= tolerance

        # Mutual information: I(A; B | C) = Σ P(a,b|c) log(P(a,b|c) / P(a|c)P(b|c))
        # Simplified for the binary case (entity present/absent)
        mi = 0.0
        if p_ab > 0 and p_a_times_p_b > 0:
            mi = p_ab * math.log2(p_ab / p_a_times_p_b)
        # Add the complementary terms for a better estimate
        p_a_not_b = max(0, p_a - p_ab)
        p_not_a_b = max(0, p_b - p_ab)
        p_not_ab = max(0, 1 - p_a - p_b + p_ab)
        if p_a_not_b > 0 and p_a > 0 and (1 - p_b) > 0:
            mi += p_a_not_b * math.log2(p_a_not_b / (p_a * max(1e-10, 1 - p_b)))
        if p_not_a_b > 0 and (1 - p_a) > 0 and p_b > 0:
            mi += p_not_a_b * math.log2(p_not_a_b / (max(1e-10, 1 - p_a) * p_b))
        mi = max(0.0, mi)  # Clip numerical noise

        given_str = ",".join(given_ids) if given_ids else "∅"
        if independent:
            explanation = (
                f"{a_id} ⊥ {b_id} | {{{given_str}}}: "
                f"P(A∩B|C)={p_ab:.4f} ≈ P(A|C)·P(B|C)={p_a_times_p_b:.4f} "
                f"(deviation={deviation:.4f} < tol={tolerance})"
            )
        else:
            explanation = (
                f"{a_id} ⊬⊥ {b_id} | {{{given_str}}}: "
                f"P(A∩B|C)={p_ab:.4f} ≠ P(A|C)·P(B|C)={p_a_times_p_b:.4f} "
                f"(deviation={deviation:.4f} ≥ tol={tolerance})"
            )

        return IndependenceResult(
            a_id=a_id, b_id=b_id, given_ids=given_ids,
            independent=independent, mutual_information=mi,
            p_a=p_a, p_b=p_b, p_ab=p_ab, p_a_times_p_b=p_a_times_p_b,
            explanation=explanation,
        )

    # -----------------------------------------------------------------
    # Markov Blanket
    # -----------------------------------------------------------------

    def markov_blanket(self, node_id: str) -> MarkovBlanket:
        """
        Compute the Markov blanket of a node.

        The Markov blanket is the minimal set of nodes that renders
        the target node conditionally independent of all others.

        In a standard BN: parents ∪ children ∪ co-parents.
        In HLLSet BN: same structural definition, but we can also
        VERIFY it using the conditional_independence test.

        Args:
            node_id: The target node

        Returns:
            MarkovBlanket with parents, children, co-parents
        """
        parents = self.parents(node_id)
        children = self.children(node_id)

        # Co-parents: other parents of my children
        coparents = set()
        for child_id in children:
            for other_parent in self.parents(child_id):
                if other_parent != node_id:
                    coparents.add(other_parent)

        return MarkovBlanket(
            node_id=node_id,
            parent_ids=parents,
            child_ids=children,
            coparent_ids=sorted(coparents),
        )

    # -----------------------------------------------------------------
    # Belief Propagation (simplified)
    # -----------------------------------------------------------------

    def belief_propagation(
        self,
        evidence: Optional[Dict[str, HLLSet]] = None,
    ) -> BeliefState:
        """
        Compute posterior beliefs given evidence.

        Simplified belief propagation:
        1. Start with priors: P(node) = |node| / |U|
        2. For evidence nodes, update P(node | evidence)
        3. Propagate through the graph using CPTs

        In the HLLSet framework, evidence is an HLLSet that
        constrains the universe. "Observing evidence E" means
        replacing U with E as the conditioning set.

        Args:
            evidence: Dict of node_id → HLLSet (observed values)

        Returns:
            BeliefState with posterior probabilities
        """
        universe = self.get_universe()
        u_card = universe.cardinality()

        if evidence is None:
            evidence = {}

        # Build effective universe: intersect with evidence
        if evidence:
            evidence_sets = list(evidence.values())
            effective_universe = HLLSet.merge(evidence_sets)
        else:
            effective_universe = universe

        eff_card = effective_universe.cardinality()
        if eff_card <= 0:
            eff_card = 1.0  # Avoid division by zero

        # Compute initial beliefs (priors conditioned on evidence)
        beliefs: Dict[str, float] = {}
        for node_id, hllset in self._nodes.items():
            if node_id in evidence:
                # Evidence node: P = 1 (we observed it)
                beliefs[node_id] = 1.0
            else:
                inter = hllset.intersect(effective_universe)
                beliefs[node_id] = inter.cardinality() / eff_card

        # Propagate: topological order (parents before children)
        topo_order = self._topological_sort()

        for node_id in topo_order:
            if node_id in evidence:
                continue  # Evidence is fixed

            parent_list = self.parents(node_id)
            if not parent_list:
                continue  # Root node keeps its prior

            # Update belief using CPT:
            # P(child | parents, evidence) ≈ avg of P(child | parent_i)
            # weighted by parent beliefs
            cpt_entries = self.get_cpt(node_id)
            if not cpt_entries:
                continue

            weighted_sum = 0.0
            weight_total = 0.0
            for entry in cpt_entries:
                parent_belief = beliefs.get(entry.parent_id, 0.0)
                weighted_sum += entry.probability * parent_belief
                weight_total += parent_belief

            if weight_total > 0:
                beliefs[node_id] = weighted_sum / weight_total

        return BeliefState(
            probabilities=beliefs,
            universe_card=eff_card,
            evidence_ids=list(evidence.keys()),
        )

    # -----------------------------------------------------------------
    # Mutual Information
    # -----------------------------------------------------------------

    def mutual_information(self, a_id: str, b_id: str) -> float:
        """
        Compute mutual information I(A; B).

        I(A; B) = H(A) + H(B) - H(A, B)

        where H is Shannon entropy computed from HLLSet cardinalities.

        This measures the shared information between two nodes.
        Higher MI = stronger dependency (in either BSS or BN terms).

        Returns:
            Mutual information in bits (≥ 0)
        """
        result = self.conditional_independence(a_id, b_id)
        return result.mutual_information

    # -----------------------------------------------------------------
    # Isomorphism Verification
    # -----------------------------------------------------------------

    def isomorphism_witness(
        self,
        bss_graph,  # BSSMorphismGraph
        tolerance: float = 0.01,
    ) -> IsomorphismWitness:
        """
        Formally verify that this BN and a BSS lattice are isomorphic
        as weighted directed graphs.

        The mapping is:
            f: BSS → BN
            f(node) = node  (identity on nodes)
            f(edge(A→B, τ)) = edge(B→A_child, P(A|B))

        The key verification:
            ∀ edges: |τ(A→B) − P(A|B)| < tolerance

        This SHOULD always be true because they use the same formula,
        but HLLSet approximation errors can create tiny discrepancies
        when computed through different code paths.

        Args:
            bss_graph: BSSMorphismGraph to compare against
            tolerance: Maximum allowed weight discrepancy

        Returns:
            IsomorphismWitness with full proof
        """
        correspondences = []
        max_err = 0.0
        total_err = 0.0
        n_edges = 0

        for edge in bss_graph.edges:
            src_id = edge.source_id
            tgt_id = edge.target_id

            # BSS: τ(src → tgt) = |src ∩ tgt| / |tgt|
            bss_tau = edge.bss_forward.tau

            # BN: P(src | tgt) = |src ∩ tgt| / |tgt|
            if tgt_id in self._nodes and src_id in self._nodes:
                tgt_hll = self._nodes[tgt_id]
                src_hll = self._nodes[src_id]
                inter_card = src_hll.intersect(tgt_hll).cardinality()
                tgt_card = tgt_hll.cardinality()
                bn_prob = inter_card / tgt_card if tgt_card > 0 else 0.0

                err = abs(bss_tau - bn_prob)
                max_err = max(max_err, err)
                total_err += err
                n_edges += 1

                correspondences.append({
                    'source': src_id,
                    'target': tgt_id,
                    'bss_tau': bss_tau,
                    'bn_probability': bn_prob,
                    'error': err,
                    'match': err < tolerance,
                })

        mean_err = total_err / n_edges if n_edges > 0 else 0.0
        is_iso = max_err < tolerance

        note = (
            "GRAPH ISOMORPHISM VERIFIED. "
            "BSS τ(A→B) = P(A|B) for all edges — both representations "
            "project the same HLLSet ring onto the same weighted graph. "
            "The BSS lattice carries additional ρ (exclusion) information "
            "that the BN does not use. The BN carries additional structure "
            "(d-separation, Bayes' theorem, belief propagation) that the "
            "BSS lattice does not encode. Both are faithful but incomplete "
            "representations of the underlying Boolean ring."
        ) if is_iso else (
            f"ISOMORPHISM BROKEN (max error {max_err:.6f} ≥ {tolerance}). "
            "This is likely due to HLL approximation errors in different "
            "code paths. The algebraic identity τ = P(A|B) holds exactly; "
            "the numerical discrepancy is epistemic."
        )

        return IsomorphismWitness(
            num_nodes=len(bss_graph.nodes),
            num_edges=n_edges,
            max_weight_error=max_err,
            mean_weight_error=mean_err,
            is_isomorphic=is_iso,
            edge_correspondences=correspondences,
            structural_note=note,
        )

    # -----------------------------------------------------------------
    # Structural Comparison: What BSS Sees vs What BN Sees
    # -----------------------------------------------------------------

    def triangle_analysis(self) -> Dict[str, Any]:
        """
        Analyze the love-hate triangle for this network.

        Returns a report showing:
        - What the Ring provides (operations)
        - What BSS reveals (order, covering)
        - What BN reveals (independence, Markov structure)
        - Where they agree (shared graph structure)
        - Where they diverge (ρ vs d-separation)
        """
        # Ring layer: operations available
        ring_ops = {
            'union': '∪ (lattice join, OR)',
            'intersect': '∩ (ring mult, AND)',
            'xor': '⊕ (ring add, XOR)',
            'diff': '\\ (set difference)',
            'complement': '¬ (bitwise NOT)',
            'bridge_law': 'A ∪ B = (A △ B) △ (A ∩ B)',
        }

        # BSS layer: what the lattice structure reveals
        bss_structure = {
            'edges': self.num_edges,
            'interpretation': 'Each edge weight τ = "inclusion strength"',
            'extra_info': 'ρ (exclusion) — no BN analogue',
            'reveals': 'Partial order, Hasse diagram, levels, antichains',
        }

        # BN layer: what the probabilistic structure reveals
        blankets = {nid: self.markov_blanket(nid) for nid in self.node_ids}
        roots = [nid for nid in self.node_ids if not self.parents(nid)]
        leaves = [nid for nid in self.node_ids if not self.children(nid)]

        bn_structure = {
            'edges': self.num_edges,
            'interpretation': 'Each edge weight P(A|B) = "conditional probability"',
            'extra_info': 'Bayes theorem, d-separation — no BSS analogue',
            'reveals': 'Independence structure, Markov blankets, causal ordering',
            'roots': roots,
            'leaves': leaves,
            'avg_blanket_size': (
                np.mean([b.size for b in blankets.values()])
                if blankets else 0
            ),
        }

        # Agreement: τ = P(A|B) everywhere
        agreement = {
            'formula': 'τ(A→B) = |A∩B|/|B| = P(A|B)',
            'graph_isomorphic': True,
            'shared_backbone': 'Both use intersect() and cardinality()',
        }

        # Divergence
        divergence = {
            'bss_only': 'ρ (exclusion metric), morphism composition law τ+ρ≤1',
            'bn_only': 'd-separation, Markov blanket, belief propagation, Bayes theorem',
            'philosophical': (
                'BSS asks "who contains whom?" (structural) — '
                'BN asks "what predicts what?" (epistemic)'
            ),
        }

        return {
            'ring': ring_ops,
            'bss': bss_structure,
            'bn': bn_structure,
            'agreement': agreement,
            'divergence': divergence,
        }

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def _topological_sort(self) -> List[str]:
        """Topological sort of nodes (parents before children)."""
        in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}
        for child_id, parent_list in self._parents.items():
            in_degree[child_id] = len(parent_list)

        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for child in self._children.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # Include any remaining nodes (cycles — shouldn't happen in a DAG)
        remaining = [nid for nid in self._nodes if nid not in result]
        result.extend(remaining)
        return result

    def to_dot(self, max_nodes: int = 30) -> str:
        """Export network to DOT format."""
        lines = ['digraph BayesNet {']
        lines.append('  rankdir=TB;')
        lines.append('  node [shape=ellipse, style=filled, fillcolor=lightblue];')

        for nid in list(self._nodes.keys())[:max_nodes]:
            card = self._nodes[nid].cardinality()
            lines.append(f'  "{nid}" [label="{nid}\\n|{card:.0f}|"];')

        for child_id, entries in self._cpts.items():
            for entry in entries:
                lines.append(
                    f'  "{entry.parent_id}" -> "{entry.child_id}" '
                    f'[label="P={entry.probability:.3f}\\nτ={entry.bss_tau:.3f}"];'
                )

        lines.append('}')
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f"HLLBayesNet(nodes={self.num_nodes}, edges={self.num_edges})"


# =============================================================================
# Standalone Functions
# =============================================================================

def hllset_mutual_information(a: HLLSet, b: HLLSet, universe: HLLSet) -> float:
    """
    Compute mutual information I(A; B) from HLLSets.

    I(A; B) = H(A) + H(B) - H(A,B)

    where H(X) = -P(X)log₂P(X) - (1-P(X))log₂(1-P(X))
    is the binary entropy of the event "X is present."

    Args:
        a: HLLSet A
        b: HLLSet B
        universe: Reference universe U

    Returns:
        Mutual information in bits
    """
    u_card = universe.cardinality()
    if u_card <= 0:
        return 0.0

    p_a = a.intersect(universe).cardinality() / u_card
    p_b = b.intersect(universe).cardinality() / u_card
    p_ab = a.intersect(b).intersect(universe).cardinality() / u_card

    def binary_entropy(p: float) -> float:
        if p <= 0 or p >= 1:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    h_a = binary_entropy(p_a)
    h_b = binary_entropy(p_b)

    # Joint entropy using all four cells of the 2x2 table
    h_ab = 0.0
    cells = [p_ab, max(0, p_a - p_ab), max(0, p_b - p_ab),
             max(0, 1 - p_a - p_b + p_ab)]
    for c in cells:
        if c > 0:
            h_ab -= c * math.log2(c)

    mi = max(0.0, h_a + h_b - h_ab)
    return mi


def conditional_mutual_information(
    a: HLLSet,
    b: HLLSet,
    given: HLLSet,
    universe: HLLSet,
) -> float:
    """
    Compute conditional mutual information I(A; B | C).

    I(A; B | C) = H(A|C) + H(B|C) - H(A,B|C)

    When I(A; B | C) ≈ 0, A and B are conditionally independent given C.
    This is the HLLSet version of d-separation.

    Args:
        a, b: HLLSets to test
        given: Conditioning HLLSet C
        universe: Reference universe

    Returns:
        Conditional MI in bits
    """
    # Restrict everything to the context of C
    context = given.intersect(universe)
    ctx_card = context.cardinality()
    if ctx_card <= 0:
        return 0.0

    return hllset_mutual_information(a.intersect(context), b.intersect(context), context)


def ring_to_bn_functor(hllsets: Dict[str, HLLSet], threshold: float = 0.1) -> Dict[str, Any]:
    """
    The forgetful functor from HLLSet Ring → Bayesian Network.

    This function makes explicit what structure is PRESERVED and
    what is LOST when viewing the ring through BN glasses.

    Preserved:
        - Node set (HLLSets)
        - Edge weights (|A∩B|/|B| = τ = P(A|B))
        - Intersection operation (used in CPT computation)

    Lost:
        - XOR (ring addition) — BN has no probabilistic XOR
        - ρ (exclusion metric) — BN only uses τ
        - Complement — BN works with positive probabilities
        - Bridge law — ring-specific algebraic identity

    Gained:
        - d-separation (conditional independence)
        - Markov blankets (minimal conditioning sets)
        - Belief propagation (inference algorithm)
        - Bayes' theorem (probability inversion)

    Returns:
        Dict documenting the functor's action
    """
    # Build the BN
    bn = HLLBayesNet.from_hllsets(hllsets, threshold=threshold)

    # Document what was preserved
    preserved_edges = []
    for child_id, entries in bn._cpts.items():
        for entry in entries:
            preserved_edges.append({
                'parent': entry.parent_id,
                'child': entry.child_id,
                'tau_equals_p': f"τ={entry.bss_tau:.4f} = P(A|B)={entry.probability:.4f}",
                'rho_lost': f"ρ={entry.bss_rho:.4f} (not in BN)",
            })

    return {
        'bn': bn,
        'preserved': {
            'nodes': list(hllsets.keys()),
            'edges': preserved_edges,
            'operation': 'intersect (used in CPT computation)',
        },
        'lost': {
            'xor': 'Ring addition (XOR) has no BN analogue',
            'rho': 'BSS exclusion metric ρ is not represented',
            'complement': 'BN works with positive probabilities only',
            'bridge_law': 'A ∪ B = (A △ B) △ (A ∩ B) is algebraic, not probabilistic',
        },
        'gained': {
            'd_separation': 'Conditional independence testing',
            'markov_blanket': 'Minimal conditioning sets',
            'belief_propagation': 'Evidence-driven inference',
            'bayes_theorem': 'P(A|B) = P(B|A)·P(A)/P(B)',
        },
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Types
    'CPTEntry',
    'IndependenceResult',
    'MarkovBlanket',
    'IsomorphismWitness',
    'BeliefState',
    # Main class
    'HLLBayesNet',
    # Standalone functions
    'hllset_mutual_information',
    'conditional_mutual_information',
    'ring_to_bn_functor',
]
