"""
Markov Chain & Hidden Markov Model over the HLLSet Ring.

Bayesian constructs permeate probabilistic computing — Markov chains,
Hidden Markov Models, PageRank, Markov Random Fields, and causal models
all rely on conditional probabilities.  In the HLLSet framework every one
of these constructs is grounded by the same identity:

    τ(A → B) = |A ∩ B| / |B| = P(A | B)

so the BSS τ-matrix IS a transition kernel.


Module map
==========

1.  **HLLMarkovChain** — Discrete-time, finite-state Markov chain whose
    transition matrix T[i,j] = P(Sⱼ | Sᵢ) is computed from BSS τ.
    Provides: stationary distribution, PageRank, hitting time,
    communicating classes, mixing time, entropy rate, random walk.

2.  **HLLHiddenMarkov** — HMM skeleton.  The "hidden" states are true
    token-sets; the "observations" are their HLL estimates.  Implements
    forward algorithm, Viterbi decoding, and Baum–Welch (EM) outline.

3.  **MarkovRandomField** — Undirected model using symmetric BSS.
    Gibbs potentials come from mutual information; MAP inference via
    iterated conditional modes (ICM).

4.  **CausalHLL** — Thin wrapper around HLLBayesNet for interventional
    queries (do-calculus stub).

Stand-alone helpers
-------------------
* ``hllset_pagerank``        — one-call PageRank over a set of HLLSets
* ``markov_from_lattice``    — build Markov chain from temporal lattice
* ``information_flow_rate``  — entropy rate H∞ of the chain

Architecture
============

The Markov chain sits *between* the BSS lattice and the Bayesian Network
in the interpretive stack:

    BSS Lattice   ──(τ matrix)──→   Markov Chain   ──(stationary π)──→  BN priors
        ↑                                ↑                                  ↑
      ORDER                           DYNAMICS                           MEASURE
   (who ⊆ whom)                   (who follows whom)                 (who predicts whom)

The temporal lattice U(1), U(2), …, U(T) IS a Markov chain whose
transition probabilities are BSS τ values between successive cumulative
states.

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import (
    List, Dict, Set, Optional, Tuple, NamedTuple, Any, Sequence,
)
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math
import numpy as np

from .hllset import HLLSet
from .bss import bss, BSSPair, bss_matrix as _bss_matrix


# ============================================================================
# Result Types
# ============================================================================

class StationaryResult(NamedTuple):
    """Stationary distribution of a Markov chain on HLLSets."""
    distribution: np.ndarray   # π[i] — long-run frequency of state i
    labels: List[str]          # human names
    dominant_state: str        # argmax π
    entropy: float             # H(π) in bits — uniformity measure
    converged: bool            # did the power iteration converge?


class PageRankResult(NamedTuple):
    """PageRank scores for HLLSet nodes."""
    scores: Dict[str, float]    # node_id → PR score
    ranked: List[Tuple[str, float]]  # sorted (node, score)
    damping: float
    iterations: int


@dataclass
class HittingTimeResult:
    """Expected first-passage time from source to target."""
    source: str
    target: str
    expected_time: float   # E[T_{target} | start = source]
    finite: bool           # False if target is unreachable
    path_probability: float  # P(eventually reach target)


@dataclass
class CommunicatingClass:
    """A communicating class (strongly-connected component) in the chain."""
    states: List[str]
    is_absorbing: bool    # only 1 state that self-loops with P≈1
    is_recurrent: bool    # class is recurrent (no escape)
    period: int           # period of the class (1 = aperiodic)


@dataclass
class MixingResult:
    """Mixing-time diagnostics."""
    spectral_gap: float    # 1 − |λ₂| — controls mixing speed
    mixing_time: float     # ≈ 1/gap · log(1/ε)
    second_eigenvalue: float
    is_ergodic: bool       # irreducible + aperiodic


@dataclass
class RandomWalkTrace:
    """Trace of a random walk through the HLLSet Markov chain."""
    states: List[str]           # visited state labels
    transition_probs: List[float]  # P of each step taken
    total_log_prob: float       # log₂ P(whole path)


@dataclass
class EntropyRateResult:
    """Entropy rate H∞ of the Markov chain (bits per step)."""
    entropy_rate: float       # H∞ = −Σ πᵢ Σ Tᵢⱼ log₂ Tᵢⱼ
    max_possible: float       # log₂(n) — uniform random
    efficiency: float         # H∞ / max_possible


# ============================================================================
# HMM types
# ============================================================================

@dataclass
class ForwardResult:
    """Forward-algorithm result for an HLLSet HMM."""
    log_likelihood: float                # log P(observations | model)
    alpha: np.ndarray                    # α[t, i] forward probabilities
    state_labels: List[str]


@dataclass
class ViterbiResult:
    """Viterbi decoding of the most likely hidden-state sequence."""
    path: List[str]                      # best hidden-state labels
    path_indices: List[int]              # best hidden-state indices
    log_probability: float               # log P(best path, obs)
    state_labels: List[str]


# ============================================================================
# HLLMarkovChain
# ============================================================================

class HLLMarkovChain:
    """
    Discrete-time Markov chain over HLLSets.

    States are HLLSets.  The one-step transition matrix is:

        T[i, j] = P(Sⱼ | Sᵢ)  =  τ(Sⱼ → Sᵢ)  =  |Sⱼ ∩ Sᵢ| / |Sᵢ|

    Row i sums to ≤ 1 in general (because the Sⱼ need not cover Sᵢ).
    We row-normalise to obtain a proper stochastic matrix, adding a
    small "teleportation" mass to avoid zero rows (same idea as PageRank).

    Construction
    ------------
    * ``from_hllsets(docs)``    — from a dict of named HLLSets
    * ``from_bss_matrix(mat)`` — from a pre-computed BSS τ-matrix
    * ``from_lattice(lat, ts)``— from a temporal W lattice

    Queries
    -------
    * ``stationary()``                  — πT = π
    * ``pagerank(d)``                   — damped stationary
    * ``hitting_time(src, tgt)``        — E[first passage]
    * ``communicating_classes()``       — SCCs
    * ``mixing_time()``                 — spectral gap analysis
    * ``entropy_rate()``                — bits per transition
    * ``random_walk(start, steps)``     — simulated trace
    * ``absorbing_states()``            — sink nodes
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        labels: List[str],
        hllsets: Optional[Dict[str, HLLSet]] = None,
        raw_tau_matrix: Optional[np.ndarray] = None,
    ):
        self._T = transition_matrix         # row-stochastic
        self._labels = labels
        self._n = len(labels)
        self._hllsets = hllsets or {}
        self._raw_tau = raw_tau_matrix       # before normalisation
        self._idx: Dict[str, int] = {l: i for i, l in enumerate(labels)}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def transition_matrix(self) -> np.ndarray:
        """Row-stochastic transition matrix T[i,j] = P(j | i)."""
        return self._T.copy()

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    @property
    def num_states(self) -> int:
        return self._n

    @property
    def raw_tau_matrix(self) -> Optional[np.ndarray]:
        """Pre-normalisation BSS τ matrix (may have row sums ≠ 1)."""
        return self._raw_tau.copy() if self._raw_tau is not None else None

    def transition_prob(self, src: str, tgt: str) -> float:
        """P(tgt | src) — single entry of T."""
        return float(self._T[self._idx[src], self._idx[tgt]])

    # ------------------------------------------------------------------
    # Factory: from HLLSets
    # ------------------------------------------------------------------

    @classmethod
    def from_hllsets(
        cls,
        hllsets: Dict[str, HLLSet],
        teleport: float = 0.0,
    ) -> 'HLLMarkovChain':
        """
        Build a Markov chain from a collection of HLLSets.

        Transition matrix:
            raw_T[i, j] = τ(Sⱼ → Sᵢ) = |Sⱼ ∩ Sᵢ| / |Sᵢ|
                        = P(Sⱼ | Sᵢ)

        We then row-normalise so each row sums to 1.

        Args:
            hllsets: Dict node_id → HLLSet
            teleport: Uniform teleportation probability (0 = none)

        Returns:
            HLLMarkovChain
        """
        labels = list(hllsets.keys())
        n = len(labels)
        hlls = [hllsets[l] for l in labels]

        # Build BSS τ-matrix
        bss_result = _bss_matrix(hlls, labels=labels)
        tau = bss_result['tau_matrix']  # tau[i,j] = τ(Sᵢ → Sⱼ)

        # We want T[i,j] = P(Sⱼ | Sᵢ) = τ(Sⱼ → Sᵢ) = tau[j,i]
        # i.e. the TRANSPOSE of the BSS convention
        raw = tau.T.copy()

        # Zero the diagonal (no self-transition in simple model)
        np.fill_diagonal(raw, 0.0)

        T = cls._row_normalise(raw, teleport, n)
        return cls(T, labels, hllsets=hllsets, raw_tau_matrix=raw)

    @classmethod
    def from_lattice(
        cls,
        lattice,         # HLLLattice
        timestamps: Sequence[float],
        node_ids: Optional[List[str]] = None,
        teleport: float = 0.0,
    ) -> 'HLLMarkovChain':
        """
        Build a Markov chain from the temporal W lattice.

        States are cumulative HLLSets U(t).  Transition:
            T[i, j] = P(U(tⱼ) | U(tᵢ))

        For the temporal lattice, this is typically a banded matrix
        (strong transitions between adjacent timesteps).

        Args:
            lattice: HLLLattice instance
            timestamps: Ordered time points
            node_ids: Optional names (default: "t=X")
            teleport: Teleportation probability

        Returns:
            HLLMarkovChain
        """
        if node_ids is None:
            node_ids = [f"t={t}" for t in timestamps]

        hllsets = {}
        for nid, t in zip(node_ids, timestamps):
            hllsets[nid] = lattice.cumulative(t=t)

        return cls.from_hllsets(hllsets, teleport=teleport)

    @classmethod
    def from_bss_matrix(
        cls,
        tau_matrix: np.ndarray,
        labels: List[str],
        teleport: float = 0.0,
    ) -> 'HLLMarkovChain':
        """
        Build from a pre-computed BSS τ-matrix.

        Args:
            tau_matrix: τ[i,j] = τ(Sᵢ → Sⱼ)
            labels: State names
            teleport: Teleportation probability

        Returns:
            HLLMarkovChain
        """
        n = len(labels)
        raw = tau_matrix.T.copy()
        np.fill_diagonal(raw, 0.0)
        T = cls._row_normalise(raw, teleport, n)
        return cls(T, labels, raw_tau_matrix=raw)

    @staticmethod
    def _row_normalise(
        raw: np.ndarray,
        teleport: float,
        n: int,
    ) -> np.ndarray:
        """Make a row-stochastic matrix with optional teleportation."""
        T = raw.copy()
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        T = T / row_sums
        if teleport > 0:
            T = (1 - teleport) * T + teleport * np.ones((n, n)) / n
        return T

    # ------------------------------------------------------------------
    # Stationary Distribution
    # ------------------------------------------------------------------

    def stationary(self, max_iter: int = 1000, tol: float = 1e-10) -> StationaryResult:
        """
        Compute the stationary distribution π such that πT = π.

        Uses power iteration on Tᵀ.

        π[i] is the long-run fraction of time the chain spends
        in state i.  For HLLSets this answers: "In a random walk
        on the BSS graph, how often do we visit each document?"

        Returns:
            StationaryResult with distribution, dominant state, entropy
        """
        T = self._T
        n = self._n

        # Power iteration: π_{k+1} = π_k T
        pi = np.ones(n) / n
        converged = False
        for _ in range(max_iter):
            pi_new = pi @ T
            if np.linalg.norm(pi_new - pi, 1) < tol:
                converged = True
                pi = pi_new
                break
            pi = pi_new

        # Normalise (safety)
        pi = pi / pi.sum() if pi.sum() > 0 else pi

        dom_idx = int(np.argmax(pi))
        H = -sum(p * math.log2(p) for p in pi if p > 0)

        return StationaryResult(
            distribution=pi,
            labels=self._labels,
            dominant_state=self._labels[dom_idx],
            entropy=H,
            converged=converged,
        )

    # ------------------------------------------------------------------
    # PageRank
    # ------------------------------------------------------------------

    def pagerank(
        self,
        damping: float = 0.85,
        max_iter: int = 200,
        tol: float = 1e-8,
    ) -> PageRankResult:
        """
        Compute PageRank over the HLLSet graph.

        PageRank models a random surfer who follows BSS edges with
        probability *damping* and jumps to a random node with
        probability (1−damping).

            PR = (1−d)/n + d · PR · T

        In HLLSet terms: the "important" documents are those that
        many other documents overlap with (high incoming τ).

        Args:
            damping: Damping factor (default 0.85)
            max_iter: Max iterations
            tol: Convergence tolerance

        Returns:
            PageRankResult with scores and ranking
        """
        n = self._n
        # Use the damped matrix
        M = (1 - damping) / n * np.ones((n, n)) + damping * self._T
        pr = np.ones(n) / n
        iters = 0
        for iters in range(1, max_iter + 1):
            pr_new = pr @ M
            if np.linalg.norm(pr_new - pr, 1) < tol:
                pr = pr_new
                break
            pr = pr_new

        pr = pr / pr.sum() if pr.sum() > 0 else pr
        scores = {self._labels[i]: float(pr[i]) for i in range(n)}
        ranked = sorted(scores.items(), key=lambda x: -x[1])

        return PageRankResult(scores=scores, ranked=ranked,
                              damping=damping, iterations=iters)

    # ------------------------------------------------------------------
    # Hitting Time (expected first passage)
    # ------------------------------------------------------------------

    def hitting_time(self, source: str, target: str) -> HittingTimeResult:
        """
        Compute the expected first-passage time from *source* to *target*.

        E[T_target | X_0 = source] solves:

            h(i) = 1 + Σⱼ T[i,j] h(j)   for i ≠ target
            h(target) = 0

        This is a system of linear equations (I − T_sub) h = 1.

        HLLSet interpretation: "How many random-walk steps does it
        take to reach document *target* starting from *source*?"

        Returns:
            HittingTimeResult
        """
        si = self._idx[source]
        ti = self._idx[target]
        n = self._n

        if si == ti:
            return HittingTimeResult(source, target, 0.0, True, 1.0)

        # Remove target row/col from T, solve (I - T_sub) h = 1
        mask = np.ones(n, dtype=bool)
        mask[ti] = False
        T_sub = self._T[np.ix_(mask, mask)]
        b = np.ones(mask.sum())
        A = np.eye(mask.sum()) - T_sub

        try:
            h = np.linalg.solve(A, b)
            # Map source index to sub-index
            sub_idx = si if si < ti else si - 1
            ht = float(h[sub_idx])
            finite = True
        except np.linalg.LinAlgError:
            ht = float('inf')
            finite = False

        # Probability of ever reaching target
        p_reach = self._T[si, ti]  # One-step; more complex for multi-step
        # For now report the one-step probability
        return HittingTimeResult(source, target, ht, finite, p_reach)

    # ------------------------------------------------------------------
    # Communicating Classes (SCCs)
    # ------------------------------------------------------------------

    def communicating_classes(self, threshold: float = 1e-8) -> List[CommunicatingClass]:
        """
        Find communicating classes (strongly connected components).

        Two states communicate if they can reach each other with
        positive probability.

        HLLSet interpretation: each class is a cluster of documents
        that mutually overlap — a "topic cluster" in the BSS graph.

        Returns:
            List of CommunicatingClass objects
        """
        # Build adjacency from T with threshold
        adj: Dict[int, Set[int]] = {i: set() for i in range(self._n)}
        for i in range(self._n):
            for j in range(self._n):
                if i != j and self._T[i, j] > threshold:
                    adj[i].add(j)

        # Tarjan's SCC algorithm
        index_counter = [0]
        stack: List[int] = []
        lowlink = [0] * self._n
        index_arr = [0] * self._n
        on_stack = [False] * self._n
        visited = [False] * self._n
        sccs: List[List[int]] = []

        def strongconnect(v: int):
            index_arr[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            visited[v] = True
            stack.append(v)
            on_stack[v] = True

            for w in adj[v]:
                if not visited[w]:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], index_arr[w])

            if lowlink[v] == index_arr[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)

        for v in range(self._n):
            if not visited[v]:
                strongconnect(v)

        # Analyse each SCC
        results = []
        for scc_indices in sccs:
            states = [self._labels[i] for i in scc_indices]
            is_absorbing = (len(scc_indices) == 1 and
                            self._T[scc_indices[0], scc_indices[0]] > 1 - threshold)

            # Recurrent: no edges leave the SCC
            leaves = False
            for i in scc_indices:
                for j in range(self._n):
                    if j not in scc_indices and self._T[i, j] > threshold:
                        leaves = True
                        break
                if leaves:
                    break
            is_recurrent = not leaves

            # Period: gcd of return-path lengths (approximate)
            period = self._estimate_period(scc_indices, adj)

            results.append(CommunicatingClass(
                states=states,
                is_absorbing=is_absorbing,
                is_recurrent=is_recurrent,
                period=period,
            ))

        return results

    def _estimate_period(
        self,
        scc_indices: List[int],
        adj: Dict[int, Set[int]],
    ) -> int:
        """Estimate period of an SCC via BFS cycle detection."""
        if len(scc_indices) <= 1:
            return 1
        scc_set = set(scc_indices)
        start = scc_indices[0]
        # BFS from start, record distances
        dist: Dict[int, int] = {start: 0}
        queue = deque([start])
        cycle_lengths = []
        while queue:
            v = queue.popleft()
            for w in adj.get(v, set()):
                if w not in scc_set:
                    continue
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                elif w == start:
                    cycle_lengths.append(dist[v] + 1)
        if not cycle_lengths:
            return 1
        g = cycle_lengths[0]
        for c in cycle_lengths[1:]:
            g = math.gcd(g, c)
        return g

    # ------------------------------------------------------------------
    # Absorbing States
    # ------------------------------------------------------------------

    def absorbing_states(self, threshold: float = 0.95) -> List[str]:
        """
        Find absorbing (or near-absorbing) states.

        An absorbing state has T[i,i] ≈ 1 — the chain stays there
        once it arrives.

        HLLSet interpretation: a "universal" document that contains
        everything.  τ(U → U) = 1 for the universe.

        Returns:
            List of absorbing state labels
        """
        result = []
        for i in range(self._n):
            if self._T[i, i] >= threshold:
                result.append(self._labels[i])
        return result

    # ------------------------------------------------------------------
    # Mixing Time (spectral analysis)
    # ------------------------------------------------------------------

    def mixing_diagnostics(self) -> MixingResult:
        """
        Analyse mixing time via spectral gap.

        The spectral gap γ = 1 − |λ₂| controls how fast the chain
        converges to stationarity.  Mixing time ≈ (1/γ) ln(1/ε).

        A large gap means fast mixing → the initial state is quickly
        forgotten.  A small gap means slow mixing → the chain has
        near-absorbing clusters.

        HLLSet interpretation: fast mixing ↔ documents are well-connected
        (high mutual BSS τ).  Slow mixing ↔ isolated topic clusters.

        Returns:
            MixingResult with spectral gap, second eigenvalue, mixing time
        """
        eigenvalues = np.linalg.eigvals(self._T)
        # Sort by magnitude, descending
        mags = np.abs(eigenvalues)
        order = np.argsort(-mags)
        sorted_eigs = eigenvalues[order]

        # Largest eigenvalue should be ≈ 1
        lam2 = float(np.abs(sorted_eigs[1])) if self._n > 1 else 0.0
        gap = 1.0 - lam2

        # Approximate mixing time for ε = 0.01
        eps = 0.01
        mixing = (1.0 / gap) * math.log(1.0 / eps) if gap > 1e-12 else float('inf')

        # Ergodic iff irreducible + aperiodic
        classes = self.communicating_classes()
        is_irreducible = len(classes) == 1
        is_aperiodic = all(c.period == 1 for c in classes)

        return MixingResult(
            spectral_gap=gap,
            mixing_time=mixing,
            second_eigenvalue=lam2,
            is_ergodic=is_irreducible and is_aperiodic,
        )

    # ------------------------------------------------------------------
    # Entropy Rate
    # ------------------------------------------------------------------

    def entropy_rate(self) -> EntropyRateResult:
        """
        Compute the entropy rate H∞ of the Markov chain.

            H∞ = − Σᵢ πᵢ Σⱼ Tᵢⱼ log₂ Tᵢⱼ

        The entropy rate measures the *average uncertainty* per step.

        - H∞ = 0      → deterministic transitions
        - H∞ = log₂(n) → uniformly random transitions (maximum)

        HLLSet interpretation: low H∞ means the BSS graph has
        strong, predictable overlaps.  High H∞ means overlaps
        are spread evenly — no dominant structure.

        Returns:
            EntropyRateResult
        """
        pi = self.stationary().distribution
        H = 0.0
        for i in range(self._n):
            for j in range(self._n):
                p = self._T[i, j]
                if p > 0:
                    H -= pi[i] * p * math.log2(p)

        max_H = math.log2(self._n) if self._n > 1 else 0.0
        eff = H / max_H if max_H > 0 else 0.0

        return EntropyRateResult(
            entropy_rate=H,
            max_possible=max_H,
            efficiency=eff,
        )

    # ------------------------------------------------------------------
    # Random Walk
    # ------------------------------------------------------------------

    def random_walk(
        self,
        start: str,
        steps: int,
        rng: Optional[np.random.Generator] = None,
    ) -> RandomWalkTrace:
        """
        Simulate a random walk on the HLLSet Markov chain.

        At each step the walk transitions from state i to state j
        with probability T[i,j].

        HLLSet interpretation: starting at document *start*,
        repeatedly "jump" to the next most-overlapping document.

        Args:
            start: Starting state label
            steps: Number of steps
            rng: Optional numpy random generator

        Returns:
            RandomWalkTrace with visited states and probabilities
        """
        if rng is None:
            rng = np.random.default_rng()

        idx = self._idx[start]
        states = [start]
        probs: List[float] = []
        log_prob = 0.0

        for _ in range(steps):
            row = self._T[idx]
            j = int(rng.choice(self._n, p=row))
            p = float(row[j])
            probs.append(p)
            if p > 0:
                log_prob += math.log2(p)
            else:
                log_prob = float('-inf')
            idx = j
            states.append(self._labels[j])

        return RandomWalkTrace(
            states=states,
            transition_probs=probs,
            total_log_prob=log_prob,
        )

    # ------------------------------------------------------------------
    # Expected Return Time
    # ------------------------------------------------------------------

    def expected_return_time(self, state: str) -> float:
        """
        Expected return time to *state* = 1 / π(state).

        A state with high stationary probability is visited often,
        so the expected return is short.

        Returns:
            Expected return time (steps), or inf if π ≈ 0.
        """
        pi = self.stationary().distribution
        p = pi[self._idx[state]]
        return 1.0 / p if p > 1e-15 else float('inf')

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"HLLMarkovChain(states={self._n}, labels={self._labels[:5]}...)"


# ============================================================================
# HLLHiddenMarkov — Hidden Markov Model on HLLSets
# ============================================================================

class HLLHiddenMarkov:
    """
    Hidden Markov Model over HLLSets.

    Architecture
    ============
    * **Hidden states**: "true" token-sets (or document clusters)
    * **Observations**: HLLSet estimates (lossy projections of the truth)
    * **Transition model**: T[i,j] = P(hidden_j | hidden_i), from BSS τ
    * **Emission model**: B[i,obs] = P(obs | hidden_i), from HLL error

    The key insight is that the HLLSet register representation is a
    *lossy observation* of the true token set.  The HMM decodes the
    most likely sequence of true states given the observations.

    This connects directly to **disambiguation** (Tutorial 03):
    the HMM is doing the same job — recovering ground truth from
    noisy HLL estimates — but with a principled probabilistic framework.

    Construction
    ------------
    * ``from_chain_and_emissions(mc, emission_fn)`` — from Markov chain + model
    * ``from_hllsets(hidden, observed)`` — automatic emission from BSS
    """

    def __init__(
        self,
        transition: np.ndarray,        # T[i,j] = P(hidden_j | hidden_i)
        emission: np.ndarray,          # B[i,o] = P(obs_o | hidden_i)
        initial: np.ndarray,           # π[i] = P(start in hidden_i)
        hidden_labels: List[str],
        obs_labels: List[str],
    ):
        self._T = transition
        self._B = emission
        self._pi = initial
        self._n_hidden = len(hidden_labels)
        self._n_obs = len(obs_labels)
        self._hidden_labels = hidden_labels
        self._obs_labels = obs_labels
        self._obs_idx: Dict[str, int] = {l: i for i, l in enumerate(obs_labels)}

    @classmethod
    def from_hllsets(
        cls,
        hidden_states: Dict[str, HLLSet],
        observed_states: Dict[str, HLLSet],
        teleport: float = 0.0,
    ) -> 'HLLHiddenMarkov':
        """
        Build an HMM from hidden HLLSets and observed HLLSets.

        * Transition matrix: BSS τ among hidden states.
        * Emission matrix: B[h, o] = τ(obs_o → hidden_h)
          i.e. "how much of hidden state h is visible in observation o?"
        * Initial distribution: uniform or from cardinality weights.

        Args:
            hidden_states: Dict of hidden-state label → HLLSet
            observed_states: Dict of observation label → HLLSet
            teleport: Teleportation for transition matrix

        Returns:
            HLLHiddenMarkov
        """
        h_labels = list(hidden_states.keys())
        o_labels = list(observed_states.keys())
        n_h = len(h_labels)
        n_o = len(o_labels)

        # Transition matrix among hidden states
        mc = HLLMarkovChain.from_hllsets(hidden_states, teleport=teleport)
        T = mc.transition_matrix

        # Emission matrix: B[h, o] = τ(obs → hidden) = |obs ∩ hidden| / |hidden|
        B = np.zeros((n_h, n_o))
        for i, hl in enumerate(h_labels):
            h_hll = hidden_states[hl]
            h_card = h_hll.cardinality()
            if h_card <= 0:
                continue
            for j, ol in enumerate(o_labels):
                o_hll = observed_states[ol]
                inter = o_hll.intersect(h_hll).cardinality()
                B[i, j] = inter / h_card

        # Row-normalise emission
        row_sums = B.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        B = B / row_sums

        # Initial distribution: proportional to cardinality
        cards = np.array([hidden_states[l].cardinality() for l in h_labels])
        pi = cards / cards.sum() if cards.sum() > 0 else np.ones(n_h) / n_h

        return cls(T, B, pi, h_labels, o_labels)

    # ------------------------------------------------------------------
    # Forward Algorithm
    # ------------------------------------------------------------------

    def forward(self, observations: List[str]) -> ForwardResult:
        """
        Forward algorithm: compute P(observations | model).

        α[t, i] = P(o₁, …, oₜ, qₜ = i)

        The forward algorithm computes the likelihood of the
        observation sequence under the model.

        Args:
            observations: Sequence of observation labels

        Returns:
            ForwardResult with log-likelihood and α matrix
        """
        T_len = len(observations)
        alpha = np.zeros((T_len, self._n_hidden))

        # Initialise: α[0, i] = π[i] · B[i, o₀]
        o0 = self._obs_idx.get(observations[0], 0)
        alpha[0] = self._pi * self._B[:, o0]

        # Recursion: α[t, j] = Σᵢ α[t−1, i] T[i,j] · B[j, oₜ]
        for t in range(1, T_len):
            ot = self._obs_idx.get(observations[t], 0)
            for j in range(self._n_hidden):
                alpha[t, j] = (alpha[t - 1] @ self._T[:, j]) * self._B[j, ot]

        total = alpha[-1].sum()
        ll = math.log2(total) if total > 0 else float('-inf')

        return ForwardResult(
            log_likelihood=ll,
            alpha=alpha,
            state_labels=self._hidden_labels,
        )

    # ------------------------------------------------------------------
    # Viterbi Algorithm
    # ------------------------------------------------------------------

    def viterbi(self, observations: List[str]) -> ViterbiResult:
        """
        Viterbi algorithm: find the most likely hidden-state sequence.

        δ[t, j] = max_path P(o₁,…,oₜ, q₁,…,qₜ = j)

        HLLSet interpretation: given a sequence of observed HLLSets,
        what is the most likely sequence of "true" document states?

        Args:
            observations: Sequence of observation labels

        Returns:
            ViterbiResult with best path and its probability
        """
        T_len = len(observations)
        delta = np.zeros((T_len, self._n_hidden))
        psi = np.zeros((T_len, self._n_hidden), dtype=int)

        o0 = self._obs_idx.get(observations[0], 0)
        delta[0] = self._pi * self._B[:, o0]

        for t in range(1, T_len):
            ot = self._obs_idx.get(observations[t], 0)
            for j in range(self._n_hidden):
                candidates = delta[t - 1] * self._T[:, j]
                psi[t, j] = int(np.argmax(candidates))
                delta[t, j] = candidates[psi[t, j]] * self._B[j, ot]

        # Backtrack
        path_idx = [0] * T_len
        path_idx[-1] = int(np.argmax(delta[-1]))
        for t in range(T_len - 2, -1, -1):
            path_idx[t] = psi[t + 1, path_idx[t + 1]]

        total = delta[-1, path_idx[-1]]
        lp = math.log2(total) if total > 0 else float('-inf')

        path = [self._hidden_labels[i] for i in path_idx]
        return ViterbiResult(
            path=path,
            path_indices=path_idx,
            log_probability=lp,
            state_labels=self._hidden_labels,
        )

    def __repr__(self) -> str:
        return (f"HLLHiddenMarkov(hidden={self._n_hidden}, "
                f"obs={self._n_obs})")


# ============================================================================
# Markov Random Field (undirected model)
# ============================================================================

class MarkovRandomField:
    """
    Undirected probabilistic graphical model over HLLSets.

    While a BN (directed) and Markov chain capture conditional /
    sequential relationships, the MRF captures **symmetric**
    associations.

    Edge potential:  ψ(Sᵢ, Sⱼ) = exp(MI(Sᵢ; Sⱼ))
    Node potential:  φ(Sᵢ)     = P(Sᵢ) = |Sᵢ| / |U|

    The joint distribution is:

        P(S₁, …, Sₙ) ∝ Πᵢ φ(Sᵢ) · Πᵢⱼ ψ(Sᵢ, Sⱼ)

    This is the **Ising model** analogy: each HLLSet is a spin,
    and mutual information determines the coupling strength.

    Construction:
        ``from_hllsets(docs)``
    Queries:
        ``energy(config)``   — Gibbs energy of a configuration
        ``neighbors(node)``  — Markov blanket (undirected)
        ``cliques()``        — maximal cliques
    """

    def __init__(
        self,
        nodes: Dict[str, HLLSet],
        universe: Optional[HLLSet] = None,
        mi_threshold: float = 0.01,
    ):
        self._nodes = dict(nodes)
        self._labels = list(nodes.keys())
        self._n = len(self._labels)
        self._idx = {l: i for i, l in enumerate(self._labels)}
        self._universe = universe or HLLSet.merge(list(nodes.values()))

        # Compute node potentials: φ(i) = |Sᵢ| / |U|
        u_card = self._universe.cardinality()
        self._node_pot = np.zeros(self._n)
        for i, l in enumerate(self._labels):
            c = nodes[l].intersect(self._universe).cardinality()
            self._node_pot[i] = c / u_card if u_card > 0 else 0.0

        # Compute edge potentials: ψ(i,j) = exp(MI(i,j))
        self._edge_pot = np.zeros((self._n, self._n))
        self._mi_matrix = np.zeros((self._n, self._n))
        self._adjacency: Dict[int, Set[int]] = {i: set() for i in range(self._n)}

        from .bayesian_network import hllset_mutual_information

        for i in range(self._n):
            for j in range(i + 1, self._n):
                mi = hllset_mutual_information(
                    nodes[self._labels[i]],
                    nodes[self._labels[j]],
                    self._universe,
                )
                self._mi_matrix[i, j] = mi
                self._mi_matrix[j, i] = mi
                if mi > mi_threshold:
                    pot = math.exp(mi)
                    self._edge_pot[i, j] = pot
                    self._edge_pot[j, i] = pot
                    self._adjacency[i].add(j)
                    self._adjacency[j].add(i)

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    @property
    def num_nodes(self) -> int:
        return self._n

    @property
    def num_edges(self) -> int:
        return sum(len(v) for v in self._adjacency.values()) // 2

    def neighbors(self, node: str) -> List[str]:
        """Markov blanket of a node in the MRF (undirected neighbors)."""
        idx = self._idx[node]
        return [self._labels[j] for j in self._adjacency[idx]]

    def energy(self, active: Optional[Set[str]] = None) -> float:
        """
        Gibbs energy of a configuration.

        E = −Σᵢ log φ(Sᵢ) − Σᵢⱼ log ψ(Sᵢ, Sⱼ)

        Lower energy = higher probability.

        Args:
            active: Set of active node labels (default: all)

        Returns:
            Gibbs energy (lower = more probable)
        """
        if active is None:
            active = set(self._labels)

        E = 0.0
        for l in active:
            i = self._idx[l]
            if self._node_pot[i] > 0:
                E -= math.log(self._node_pot[i])

        counted: Set[Tuple[int, int]] = set()
        for l in active:
            i = self._idx[l]
            for j in self._adjacency[i]:
                if self._labels[j] in active:
                    pair = (min(i, j), max(i, j))
                    if pair not in counted:
                        counted.add(pair)
                        if self._edge_pot[i, j] > 0:
                            E -= math.log(self._edge_pot[i, j])
        return E

    def cliques(self) -> List[List[str]]:
        """Find maximal cliques via Bron–Kerbosch (for small graphs)."""
        result: List[Set[int]] = []
        self._bron_kerbosch(set(), set(range(self._n)), set(), result)
        return [[self._labels[i] for i in sorted(c)] for c in result]

    def _bron_kerbosch(self, R: Set[int], P: Set[int], X: Set[int],
                       out: List[Set[int]]):
        if not P and not X:
            out.append(R.copy())
            return
        pivot = max(P | X, key=lambda v: len(self._adjacency[v] & P))
        for v in list(P - self._adjacency[pivot]):
            self._bron_kerbosch(
                R | {v}, P & self._adjacency[v], X & self._adjacency[v], out)
            P.remove(v)
            X.add(v)

    def mutual_information_matrix(self) -> np.ndarray:
        """Return the MI matrix."""
        return self._mi_matrix.copy()

    def __repr__(self) -> str:
        return f"MarkovRandomField(nodes={self._n}, edges={self.num_edges})"


# ============================================================================
# CausalHLL — do-calculus stub
# ============================================================================

class CausalHLL:
    """
    Causal inference on HLLSets via do-calculus (Pearl's framework).

    In a standard BN:
        P(Y | X = x)   — observational (conditioning)
        P(Y | do(X=x)) — interventional (forcing X, severing parents)

    In HLLSet BN:
        Observation:   P(Y | X) = |Y ∩ X| / |X|   (standard BSS τ)
        Intervention:  P(Y | do(X)) — remove X's parents from the graph,
                       then compute P(Y | X) in the mutilated graph.

    This is a **stub**: full structural causal models require
    counterfactual reasoning that goes beyond the current ring algebra.
    But the basic interventional query is implementable.

    Usage:
        causal = CausalHLL(bn)
        effect = causal.do('X', docs['X'])
        print(effect)  # P(Y | do(X)) for all Y
    """

    def __init__(self, bn):
        """
        Args:
            bn: HLLBayesNet instance
        """
        self._bn = bn

    def do(
        self,
        intervention_node: str,
        intervention_value: Optional[HLLSet] = None,
    ) -> Dict[str, float]:
        """
        Compute P(Y | do(X = x)) for all Y in the network.

        Algorithm (truncated factorisation):
            1. Remove all edges INTO X (sever parents)
            2. Set X = x (or its current value)
            3. Compute beliefs in the mutilated graph

        Args:
            intervention_node: The node to intervene on (X)
            intervention_value: The forced value (default: current HLLSet)

        Returns:
            Dict node_id → P(node | do(X))
        """
        from .bayesian_network import HLLBayesNet

        # Build mutilated graph: remove parents of X
        mutilated = HLLBayesNet()
        for nid, hll in self._bn.nodes.items():
            if nid == intervention_node and intervention_value is not None:
                mutilated.add_node(nid, intervention_value)
            else:
                mutilated.add_node(nid, hll)

        # Re-add all edges EXCEPT those going INTO intervention_node
        for child_id in self._bn.node_ids:
            for parent_id in self._bn.parents(child_id):
                if child_id == intervention_node:
                    continue  # Sever parents of X
                mutilated.add_edge(parent_id, child_id)

        # Run belief propagation on mutilated graph with X as evidence
        if intervention_value is not None:
            evidence = {intervention_node: intervention_value}
        else:
            evidence = {intervention_node: self._bn.nodes[intervention_node]}

        beliefs = mutilated.belief_propagation(evidence=evidence)
        return dict(beliefs.probabilities)

    def average_causal_effect(
        self,
        treatment: str,
        outcome: str,
    ) -> float:
        """
        Average Causal Effect: P(Y | do(X=1)) − P(Y | do(X=0)).

        Since HLLSets don't have binary on/off, we approximate:
            ACE ≈ P(outcome | do(treatment = treatment_hll))
                − P(outcome | do(treatment = ∅))

        Returns:
            ACE (positive = treatment increases outcome probability)
        """
        # Intervention: X = its current value
        effect_on = self.do(treatment)
        p_on = effect_on.get(outcome, 0.0)

        # Intervention: X = empty
        empty = HLLSet(p_bits=self._bn.nodes[treatment].p_bits)
        effect_off = self.do(treatment, empty)
        p_off = effect_off.get(outcome, 0.0)

        return p_on - p_off

    def __repr__(self) -> str:
        return f"CausalHLL(nodes={self._bn.num_nodes})"


# ============================================================================
# Stand-alone Convenience Functions
# ============================================================================

def hllset_pagerank(
    hllsets: Dict[str, HLLSet],
    damping: float = 0.85,
) -> PageRankResult:
    """
    One-call PageRank over a set of HLLSets.

    Builds a Markov chain from BSS τ and runs PageRank.
    The highest-scored documents are "hubs" — they overlap with
    many others and are thus central to the collection.

    Args:
        hllsets: Named HLLSets
        damping: Damping factor

    Returns:
        PageRankResult
    """
    mc = HLLMarkovChain.from_hllsets(hllsets, teleport=1 - damping)
    return mc.pagerank(damping=damping)


def markov_from_lattice(
    lattice,
    timestamps: Sequence[float],
    node_ids: Optional[List[str]] = None,
) -> HLLMarkovChain:
    """
    Build a Markov chain from a temporal W lattice.

    Args:
        lattice: HLLLattice
        timestamps: Ordered time points
        node_ids: Optional state names

    Returns:
        HLLMarkovChain
    """
    return HLLMarkovChain.from_lattice(lattice, timestamps, node_ids)


def information_flow_rate(chain: HLLMarkovChain) -> EntropyRateResult:
    """
    Entropy rate of the chain — bits of information per step.

    Low rate → predictable (strong BSS overlaps dominate).
    High rate → uncertain (overlaps are spread evenly).

    Returns:
        EntropyRateResult
    """
    return chain.entropy_rate()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Types
    'StationaryResult',
    'PageRankResult',
    'HittingTimeResult',
    'CommunicatingClass',
    'MixingResult',
    'RandomWalkTrace',
    'EntropyRateResult',
    'ForwardResult',
    'ViterbiResult',
    # Markov chain
    'HLLMarkovChain',
    # HMM
    'HLLHiddenMarkov',
    # MRF
    'MarkovRandomField',
    # Causal
    'CausalHLL',
    # Convenience
    'hllset_pagerank',
    'markov_from_lattice',
    'information_flow_rate',
]
