"""
HLLSet Dynamics — Monitoring, Planning & Steering for HLLSet Systems

This module treats evolving HLLSet collections as **dynamic systems** where:

    State:       HLLSet configuration at time t
    Transition:  (D, R, N) transformation triple
    Observable:  BSS metrics (τ, ρ), Noether flux Φ, phase classification
    Control:     Targeted additions/deletions to steer toward target state

The De Bruijn graph structure provides:
    - **Monitoring**: Real-time observation of (D, R, N) changes
    - **Planning**: Reachability analysis via BSS adjacency
    - **Steering**: Feedback control to reach target states

Architecture:
```
                    ┌─────────────────────┐
                    │   Target State H*   │
                    └─────────────────────┘
                              ▲
                              │ steering signal
    ┌─────────────┐     ┌─────┴─────────┐     ┌─────────────┐
    │  Monitor    │───▶│   Controller  │───▶│   Actuator  │
    │  (D,R,N,Φ)  │     │  (plan path)  │     │  (apply δ)  │
    └─────────────┘     └───────────────┘     └─────────────┘
          ▲                                          │
          │                                          ▼
    ┌─────┴─────────────────────────────────────────────────┐
    │                   HLLSet System State                 │
    │         H(t) ──(D,R,N)──▶ H(t+1) ──▶ ...            │
    └───────────────────────────────────────────────────────┘
```

Key Concepts:
    - **Reachability**: Can we reach H_target from H_current via valid (D,R,N)?
    - **Cost**: |D| + |N| represents "effort" to make a transition
    - **Stability**: High |R| / low |D|+|N| indicates stable state
    - **Steering Law**: Use Noether flux to maintain conservation during steering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, List, Dict, Tuple, Optional, Set, Callable
from enum import Enum
import heapq

from .hllset import HLLSet
from .bss import bss, BSSPair
from .hllset_debruijn import DRNTriple
from .noether import NoetherEvolution, SteeringPhase, SteeringDiagnostics
from .hllset_debruijn import (
    DRNTriple, decompose_transition, EvolutionPhase, classify_transition
)


# =============================================================================
# System State & Observations
# =============================================================================

@dataclass(frozen=True)
class SystemObservation:
    """
    Complete observation of system state at time t.
    
    Combines HLLSet state with derived metrics for monitoring.
    """
    timestamp: float
    state: HLLSet
    cardinality: float
    
    # If we have a previous state, compute transition metrics
    drn: Optional[DRNTriple] = None
    bss_from_prev: Optional[BSSPair] = None
    phase: Optional[EvolutionPhase] = None
    
    # Noether conservation
    flux: float = 0.0
    cumulative_flux: float = 0.0
    
    @property
    def is_stable(self) -> bool:
        """True if phase is STABLE or no transition occurred."""
        return self.phase is None or self.phase == EvolutionPhase.STABLE
    
    @property
    def transition_cost(self) -> float:
        """Cost of transition: |D| + |N| (effort to change state)."""
        if self.drn is None:
            return 0.0
        return self.drn.deleted_card + self.drn.novel_card
    
    @property
    def retention_ratio(self) -> float:
        """Fraction of previous state retained: |R| / (|D| + |R|)."""
        if self.drn is None:
            return 1.0
        total = self.drn.deleted_card + self.drn.retained_card
        return self.drn.retained_card / total if total > 0 else 1.0


@dataclass
class SystemMonitor:
    """
    Real-time monitor for HLLSet dynamic system.
    
    Tracks state evolution, computes metrics, and detects anomalies.
    """
    p_bits: int = 10
    history: List[SystemObservation] = field(default_factory=list)
    noether: NoetherEvolution = field(default_factory=lambda: NoetherEvolution(p_bits=10))
    
    # Thresholds for anomaly detection
    flux_threshold: float = 50.0       # Alert if |Φ| exceeds this
    retention_threshold: float = 0.3   # Alert if retention drops below this
    
    def __post_init__(self):
        self.noether = NoetherEvolution(p_bits=self.p_bits)
    
    def observe(self, state: HLLSet, timestamp: float) -> SystemObservation:
        """
        Record new state observation and compute metrics.
        
        Args:
            state: Current HLLSet state
            timestamp: Time of observation
        
        Returns:
            SystemObservation with all computed metrics
        """
        prev = self.history[-1] if self.history else None
        
        # Compute transition metrics if we have previous state
        drn = None
        bss_pair = None
        phase = None
        
        if prev is not None:
            drn = decompose_transition(prev.state, state)
            bss_pair = bss(prev.state, state)
            phase = classify_transition(drn)
            
            # Noether step
            additions = drn.novel
            deletions = drn.deleted
            diag = self.noether.step(additions=additions, deletions=deletions)
            flux = diag.flux
            cumulative_flux = diag.cumulative_flux
        else:
            # First observation
            self.noether.step(additions=state)
            flux = state.cardinality()
            cumulative_flux = flux
        
        obs = SystemObservation(
            timestamp=timestamp,
            state=state,
            cardinality=state.cardinality(),
            drn=drn,
            bss_from_prev=bss_pair,
            phase=phase,
            flux=flux,
            cumulative_flux=cumulative_flux
        )
        
        self.history.append(obs)
        return obs
    
    def detect_anomalies(self, obs: SystemObservation) -> List[str]:
        """Check for anomalous conditions in observation."""
        anomalies = []
        
        if abs(obs.flux) > self.flux_threshold:
            anomalies.append(f"HIGH_FLUX: |Φ|={abs(obs.flux):.1f} > {self.flux_threshold}")
        
        if obs.retention_ratio < self.retention_threshold:
            anomalies.append(f"LOW_RETENTION: {obs.retention_ratio:.2f} < {self.retention_threshold}")
        
        if obs.phase == EvolutionPhase.REPLACEMENT:
            anomalies.append("REPLACEMENT_PHASE: High churn detected")
        
        return anomalies
    
    def summary(self) -> Dict:
        """Get summary statistics of system history."""
        if not self.history:
            return {"observations": 0}
        
        phases = [o.phase for o in self.history if o.phase is not None]
        costs = [o.transition_cost for o in self.history]
        retentions = [o.retention_ratio for o in self.history]
        
        from collections import Counter
        phase_counts = Counter(phases)
        
        return {
            "observations": len(self.history),
            "total_flux": self.history[-1].cumulative_flux if self.history else 0,
            "avg_transition_cost": sum(costs) / len(costs) if costs else 0,
            "avg_retention": sum(retentions) / len(retentions) if retentions else 1,
            "phase_distribution": {p.value: c for p, c in phase_counts.items()},
            "final_cardinality": self.history[-1].cardinality,
        }


# =============================================================================
# Planning: Reachability & Path Finding
# =============================================================================

@dataclass(frozen=True)
class TransitionPlan:
    """
    A planned transition from source to target state.
    
    Contains the required (D, R, N) operations and estimated cost.
    """
    source: HLLSet
    target: HLLSet
    drn: DRNTriple
    bss_pair: BSSPair
    phase: EvolutionPhase
    
    @property
    def cost(self) -> float:
        """Transition cost: |D| + |N|."""
        return self.drn.deleted_card + self.drn.novel_card
    
    @property
    def is_reachable(self) -> bool:
        """True if BSS indicates valid adjacency."""
        return self.bss_pair.tau > 0
    
    def describe(self) -> str:
        """Human-readable description of the plan."""
        return (
            f"Transition: delete ≈{self.drn.deleted_card:.0f}, "
            f"retain ≈{self.drn.retained_card:.0f}, "
            f"add ≈{self.drn.novel_card:.0f} tokens "
            f"(cost={self.cost:.0f}, τ={self.bss_pair.tau:.2f})"
        )


@dataclass(frozen=True)
class PathPlan:
    """
    A multi-step path through the state space.
    
    Represents a sequence of transitions to reach a target.
    """
    steps: Tuple[TransitionPlan, ...]
    total_cost: float
    
    @property
    def num_steps(self) -> int:
        return len(self.steps)
    
    @property
    def source(self) -> HLLSet:
        return self.steps[0].source if self.steps else None
    
    @property
    def target(self) -> HLLSet:
        return self.steps[-1].target if self.steps else None


def plan_transition(source: HLLSet, target: HLLSet) -> TransitionPlan:
    """
    Plan a direct transition from source to target.
    
    Args:
        source: Current state
        target: Desired state
    
    Returns:
        TransitionPlan with required operations
    """
    drn = decompose_transition(source, target)
    bss_pair = bss(source, target)
    phase = classify_transition(drn)
    
    return TransitionPlan(
        source=source,
        target=target,
        drn=drn,
        bss_pair=bss_pair,
        phase=phase
    )


def find_path(
    source: HLLSet,
    target: HLLSet,
    waypoints: List[HLLSet],
    tau_min: float = 0.1,
    max_cost: float = float('inf')
) -> Optional[PathPlan]:
    """
    Find optimal path from source to target through waypoints.
    
    Uses Dijkstra's algorithm with transition cost as edge weight.
    
    Args:
        source: Starting state
        target: Goal state
        waypoints: Intermediate states that can be visited
        tau_min: Minimum τ for valid transitions
        max_cost: Maximum total path cost
    
    Returns:
        PathPlan if path found, None otherwise
    """
    # Build graph: source + waypoints + target
    all_states = [source] + waypoints + [target]
    n = len(all_states)
    source_idx = 0
    target_idx = n - 1
    
    # Compute adjacency with costs
    # adj[i] = [(j, cost, plan), ...]
    adj = {i: [] for i in range(n)}
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            plan = plan_transition(all_states[i], all_states[j])
            if plan.bss_pair.tau >= tau_min:
                adj[i].append((j, plan.cost, plan))
    
    # Dijkstra's algorithm
    dist = {i: float('inf') for i in range(n)}
    prev = {i: None for i in range(n)}
    prev_plan = {i: None for i in range(n)}
    dist[source_idx] = 0
    
    # Priority queue: (distance, node)
    pq = [(0, source_idx)]
    visited = set()
    
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        
        if u == target_idx:
            break
        
        for v, cost, plan in adj[u]:
            if v in visited:
                continue
            new_dist = d + cost
            if new_dist < dist[v] and new_dist <= max_cost:
                dist[v] = new_dist
                prev[v] = u
                prev_plan[v] = plan
                heapq.heappush(pq, (new_dist, v))
    
    # Reconstruct path
    if dist[target_idx] == float('inf'):
        return None
    
    steps = []
    node = target_idx
    while prev[node] is not None:
        steps.append(prev_plan[node])
        node = prev[node]
    steps.reverse()
    
    return PathPlan(
        steps=tuple(steps),
        total_cost=dist[target_idx]
    )


# =============================================================================
# Steering: Feedback Control
# =============================================================================

class SteeringMode(Enum):
    """Control modes for system steering."""
    DIRECT = "direct"           # Single-step transition
    INCREMENTAL = "incremental" # Gradual approach
    CONSERVATIVE = "conservative"  # Minimize disruption


@dataclass
class SteeringAction:
    """
    Action to apply to steer system toward target.
    
    Specifies tokens to add and remove.
    """
    to_add: Tuple[str, ...]
    to_remove: Tuple[str, ...]
    estimated_cost: float
    description: str
    
    def is_null(self) -> bool:
        """True if no action needed (already at target)."""
        return len(self.to_add) == 0 and len(self.to_remove) == 0


@dataclass
class SystemController:
    """
    Feedback controller for HLLSet dynamic system.
    
    Computes steering actions to guide system toward target state.
    """
    p_bits: int = 10
    mode: SteeringMode = SteeringMode.INCREMENTAL
    
    # Control parameters
    max_change_per_step: float = 0.2  # Max fraction of state to change per step
    convergence_threshold: float = 0.95  # τ threshold for "at target"
    
    def compute_action(
        self,
        current: HLLSet,
        target: HLLSet,
        current_tokens: Optional[List[str]] = None,
        target_tokens: Optional[List[str]] = None
    ) -> SteeringAction:
        """
        Compute steering action to move current toward target.
        
        Args:
            current: Current system state
            target: Desired target state
            current_tokens: Optional known tokens in current (for precise control)
            target_tokens: Optional known tokens in target
        
        Returns:
            SteeringAction with tokens to add/remove
        """
        # Check if already at target
        bss_pair = bss(current, target)
        if bss_pair.tau >= self.convergence_threshold and bss_pair.rho < 0.1:
            return SteeringAction(
                to_add=(),
                to_remove=(),
                estimated_cost=0,
                description="Already at target (τ={:.2f})".format(bss_pair.tau)
            )
        
        # Compute required transition
        drn = decompose_transition(current, target)
        
        if self.mode == SteeringMode.DIRECT:
            # Full transition in one step
            to_add = target_tokens if target_tokens else ()
            to_remove = current_tokens if current_tokens else ()
            cost = drn.deleted_card + drn.novel_card
            desc = f"Direct transition: delete ≈{drn.deleted_card:.0f}, add ≈{drn.novel_card:.0f}"
            
        elif self.mode == SteeringMode.INCREMENTAL:
            # Partial transition respecting max_change_per_step
            total_change = drn.deleted_card + drn.novel_card
            max_change = current.cardinality() * self.max_change_per_step
            
            if total_change <= max_change:
                # Can do full transition
                to_add = target_tokens if target_tokens else ()
                to_remove = current_tokens if current_tokens else ()
                cost = total_change
                desc = f"Incremental (full): delete ≈{drn.deleted_card:.0f}, add ≈{drn.novel_card:.0f}"
            else:
                # Partial transition - prioritize by impact
                # In practice, would select subset of tokens
                ratio = max_change / total_change
                est_delete = drn.deleted_card * ratio
                est_add = drn.novel_card * ratio
                to_add = ()  # Would need token selection logic
                to_remove = ()
                cost = max_change
                desc = f"Incremental (partial {ratio:.0%}): ≈{est_delete:.0f} delete, ≈{est_add:.0f} add"
                
        else:  # CONSERVATIVE
            # Minimize disruption - only add, avoid deletion
            to_add = target_tokens if target_tokens else ()
            to_remove = ()
            cost = drn.novel_card
            desc = f"Conservative: add ≈{drn.novel_card:.0f} only (no deletion)"
        
        return SteeringAction(
            to_add=tuple(to_add) if to_add else (),
            to_remove=tuple(to_remove) if to_remove else (),
            estimated_cost=cost,
            description=desc
        )
    
    def compute_convergence_trajectory(
        self,
        current: HLLSet,
        target: HLLSet,
        max_steps: int = 10
    ) -> List[Dict]:
        """
        Simulate convergence trajectory under incremental control.
        
        Returns list of expected states showing convergence path.
        """
        trajectory = []
        state = current
        
        for step in range(max_steps):
            bss_pair = bss(state, target)
            drn = decompose_transition(state, target)
            
            trajectory.append({
                "step": step,
                "tau": bss_pair.tau,
                "rho": bss_pair.rho,
                "remaining_deletions": drn.deleted_card,
                "remaining_additions": drn.novel_card,
                "converged": bss_pair.tau >= self.convergence_threshold
            })
            
            if bss_pair.tau >= self.convergence_threshold:
                break
            
            # Simulate one step of incremental control
            # In practice, would actually apply the changes
            # Here we just estimate the trajectory
            change_ratio = min(1.0, self.max_change_per_step * (step + 1))
            
            # Simulate state moving toward target
            # (simplified - actual would require token-level operations)
            if change_ratio >= 1.0:
                state = target
        
        return trajectory


# =============================================================================
# Integrated Dynamic System
# =============================================================================

@dataclass
class HLLSetDynamicSystem:
    """
    Complete dynamic system with monitoring, planning, and steering.
    
    Integrates all components for controlling HLLSet evolution.
    """
    p_bits: int = 10
    monitor: SystemMonitor = field(default_factory=lambda: SystemMonitor(p_bits=10))
    controller: SystemController = field(default_factory=lambda: SystemController(p_bits=10))
    
    # State tracking
    current_state: Optional[HLLSet] = None
    target_state: Optional[HLLSet] = None
    time: float = 0.0
    
    def __post_init__(self):
        self.monitor = SystemMonitor(p_bits=self.p_bits)
        self.controller = SystemController(p_bits=self.p_bits)
    
    def set_target(self, target: HLLSet):
        """Set the target state for steering."""
        self.target_state = target
    
    def step(self, new_state: HLLSet, dt: float = 1.0) -> Dict:
        """
        Process one time step of the dynamic system.
        
        Args:
            new_state: Observed state at this time step
            dt: Time delta since last step
        
        Returns:
            Dict with observation, anomalies, and steering action
        """
        self.time += dt
        self.current_state = new_state
        
        # Monitor
        obs = self.monitor.observe(new_state, self.time)
        anomalies = self.monitor.detect_anomalies(obs)
        
        # Plan/Steer (if target set)
        action = None
        distance_to_target = None
        if self.target_state is not None:
            action = self.controller.compute_action(new_state, self.target_state)
            bss_pair = bss(new_state, self.target_state)
            distance_to_target = 1.0 - bss_pair.tau
        
        return {
            "time": self.time,
            "observation": obs,
            "anomalies": anomalies,
            "steering_action": action,
            "distance_to_target": distance_to_target,
        }
    
    def status(self) -> Dict:
        """Get current system status."""
        return {
            "time": self.time,
            "current_cardinality": self.current_state.cardinality() if self.current_state else 0,
            "target_set": self.target_state is not None,
            "at_target": self._at_target(),
            "monitor_summary": self.monitor.summary(),
        }
    
    def _at_target(self) -> bool:
        if self.current_state is None or self.target_state is None:
            return False
        bss_pair = bss(self.current_state, self.target_state)
        return bss_pair.tau >= self.controller.convergence_threshold


# =============================================================================
# Bernoulli Map & Symbolic Dynamics
# =============================================================================
# 
# Connection to ergodic theory (Wikipedia):
#   "The Bernoulli map (also called the 2x mod 1 map) is an ergodic dynamical
#    system. Trajectories correspond to walks in the De Bruijn graph, where
#    each real x in [0,1) maps to the vertex corresponding to the first n
#    digits in the base-m representation of x."
#
# For HLLSets (which are bit vectors of size 2^P × 32):
#   - State space: {0,1}^N where N = 2^P × 32
#   - Shift map: analogous to Bernoulli 2x mod 1
#   - De Bruijn walks: trajectories through state space
#   - Ergodicity: system explores all reachable states uniformly
#
# Key insight: (D, R, N) is a GENERALIZED SHIFT on the bit vector
#   D = bits to clear (shift out)
#   R = bits to retain (invariant subspace)  
#   N = bits to set (shift in)

from .bitvector_ring import BitVector


@dataclass(frozen=True)
class BitVectorState:
    """
    State representation as a bit vector (m-adic number).
    
    This is the fundamental representation for symbolic dynamics.
    Both HLLSet and raw BitVector can be wrapped.
    """
    vector: BitVector
    n_bits: int
    
    @classmethod
    def from_hllset(cls, hll: HLLSet) -> 'BitVectorState':
        """Convert HLLSet to BitVectorState."""
        import numpy as np
        bv = BitVector.from_numpy(hll.dump_numpy())
        num_registers = 1 << hll.p_bits  # 2^p_bits registers
        return cls(vector=bv, n_bits=num_registers * 32)
    
    @classmethod
    def from_bitvector(cls, bv: BitVector, n_bits: int) -> 'BitVectorState':
        """Wrap raw BitVector."""
        return cls(vector=bv, n_bits=n_bits)
    
    @property
    def popcount(self) -> int:
        """Number of 1-bits (Hamming weight)."""
        return self.vector.popcount()
    
    @property
    def density(self) -> float:
        """Fraction of bits set: popcount / n_bits."""
        return self.popcount / self.n_bits if self.n_bits > 0 else 0.0
    
    def xor(self, other: 'BitVectorState') -> 'BitVectorState':
        """XOR (ring addition): symmetric difference."""
        return BitVectorState(self.vector ^ other.vector, self.n_bits)
    
    def and_(self, other: 'BitVectorState') -> 'BitVectorState':
        """AND (ring multiplication): intersection."""
        return BitVectorState(self.vector & other.vector, self.n_bits)
    
    def or_(self, other: 'BitVectorState') -> 'BitVectorState':
        """OR: union."""
        return BitVectorState(self.vector | other.vector, self.n_bits)
    
    def hamming_distance(self, other: 'BitVectorState') -> int:
        """Hamming distance: number of differing bits."""
        return (self.vector ^ other.vector).popcount()
    
    def normalized_distance(self, other: 'BitVectorState') -> float:
        """Normalized Hamming distance in [0, 1]."""
        return self.hamming_distance(other) / self.n_bits if self.n_bits > 0 else 0.0


@dataclass(frozen=True)
class ShiftTransition:
    """
    Generalized shift transition on bit vectors.
    
    Represents the transformation: v' = (v AND NOT clear_mask) OR set_mask
    
    This is the bit-level analog of (D, R, N):
        clear_mask = bits to delete (D)
        set_mask   = bits to add (N)
        unchanged  = bits retained (R)
    """
    clear_mask: BitVector   # Bits to clear (D analog)
    set_mask: BitVector     # Bits to set (N analog)
    n_bits: int
    
    @classmethod
    def from_states(cls, source: BitVectorState, target: BitVectorState) -> 'ShiftTransition':
        """
        Compute shift transition from source → target.
        
        clear_mask = source AND NOT target  (bits in source but not target)
        set_mask   = target AND NOT source  (bits in target but not source)
        """
        clear = source.vector & ~target.vector
        set_ = target.vector & ~source.vector
        return cls(clear_mask=clear, set_mask=set_, n_bits=source.n_bits)
    
    @classmethod
    def from_drn(cls, drn: DRNTriple, p_bits: int = 10) -> 'ShiftTransition':
        """Convert (D, R, N) triple to shift transition."""
        import numpy as np
        n_bits = (1 << p_bits) * 32
        clear = BitVector.from_numpy(drn.deleted.dump_numpy())
        set_ = BitVector.from_numpy(drn.novel.dump_numpy())
        return cls(clear_mask=clear, set_mask=set_, n_bits=n_bits)
    
    def apply(self, state: BitVectorState) -> BitVectorState:
        """Apply shift transition to state."""
        new_vec = (state.vector & ~self.clear_mask) | self.set_mask
        return BitVectorState(new_vec, self.n_bits)
    
    @property
    def hamming_cost(self) -> int:
        """Total bits changed: |clear| + |set|."""
        return self.clear_mask.popcount() + self.set_mask.popcount()
    
    @property
    def is_identity(self) -> bool:
        """True if no bits change."""
        return self.clear_mask.is_zero() and self.set_mask.is_zero()


class BernoulliAnalyzer:
    """
    Analyzer for Bernoulli map / symbolic dynamics properties.
    
    Treats HLLSet/BitVector evolution as trajectories in the
    symbolic dynamical system induced by the De Bruijn graph.
    """
    
    def __init__(self, n_bits: int):
        self.n_bits = n_bits
        self.trajectory: List[BitVectorState] = []
        self.transitions: List[ShiftTransition] = []
    
    def observe(self, state: BitVectorState) -> None:
        """Add state to trajectory."""
        if self.trajectory:
            trans = ShiftTransition.from_states(self.trajectory[-1], state)
            self.transitions.append(trans)
        self.trajectory.append(state)
    
    def observe_hllset(self, hll: HLLSet) -> None:
        """Add HLLSet to trajectory."""
        self.observe(BitVectorState.from_hllset(hll))
    
    def observe_bitvector(self, bv: BitVector) -> None:
        """Add raw BitVector to trajectory."""
        self.observe(BitVectorState.from_bitvector(bv, self.n_bits))
    
    # -------------------------------------------------------------------------
    # Ergodic / Mixing Analysis
    # -------------------------------------------------------------------------
    
    def density_series(self) -> List[float]:
        """Time series of bit density (fraction of 1s)."""
        return [s.density for s in self.trajectory]
    
    def entropy_estimate(self) -> float:
        """
        Estimate entropy of the trajectory.
        
        Uses the average bit density as proxy for ergodic measure.
        For ergodic systems, this converges to the invariant measure.
        """
        if not self.trajectory:
            return 0.0
        
        densities = self.density_series()
        avg_density = sum(densities) / len(densities)
        
        # Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
        import math
        if avg_density == 0 or avg_density == 1:
            return 0.0
        return -(avg_density * math.log2(avg_density) + 
                 (1 - avg_density) * math.log2(1 - avg_density))
    
    def mixing_rate(self) -> float:
        """
        Estimate mixing rate from transition costs.
        
        High mixing = transitions change many bits (high Hamming cost).
        Low mixing = transitions are localized.
        """
        if not self.transitions:
            return 0.0
        
        costs = [t.hamming_cost for t in self.transitions]
        return sum(costs) / (len(costs) * self.n_bits)
    
    def recurrence_time(self, threshold: float = 0.1) -> Optional[int]:
        """
        Estimate Poincaré recurrence time.
        
        Finds first time the system returns "close" to initial state.
        threshold: normalized Hamming distance threshold for "close"
        """
        if len(self.trajectory) < 2:
            return None
        
        initial = self.trajectory[0]
        for t, state in enumerate(self.trajectory[1:], 1):
            if initial.normalized_distance(state) < threshold:
                return t
        return None
    
    def lyapunov_estimate(self) -> float:
        """
        Rough estimate of Lyapunov exponent.
        
        For Bernoulli map, λ = log(2) ≈ 0.693.
        We estimate from average Hamming cost growth rate.
        """
        if len(self.transitions) < 2:
            return 0.0
        
        import math
        costs = [max(t.hamming_cost, 1) for t in self.transitions]
        
        # Average log of cost (proxy for expansion rate)
        return sum(math.log(c) for c in costs) / len(costs)
    
    # -------------------------------------------------------------------------
    # De Bruijn Walk Analysis
    # -------------------------------------------------------------------------
    
    def walk_complexity(self) -> Dict:
        """
        Analyze complexity of the De Bruijn walk.
        
        Returns statistics about the trajectory structure.
        """
        if not self.trajectory:
            return {"length": 0}
        
        # Count unique states (vertices visited)
        # Use popcount signature as cheap hash
        signatures = [s.popcount for s in self.trajectory]
        unique_sigs = len(set(signatures))
        
        # Transition pattern analysis
        costs = [t.hamming_cost for t in self.transitions]
        
        return {
            "length": len(self.trajectory),
            "unique_signatures": unique_sigs,
            "avg_transition_cost": sum(costs) / len(costs) if costs else 0,
            "max_transition_cost": max(costs) if costs else 0,
            "min_transition_cost": min(costs) if costs else 0,
            "total_bits_changed": sum(costs),
            "entropy_estimate": self.entropy_estimate(),
            "mixing_rate": self.mixing_rate(),
        }
    
    def summary(self) -> str:
        """Human-readable summary of analysis."""
        wc = self.walk_complexity()
        recur = self.recurrence_time()
        lyap = self.lyapunov_estimate()
        
        lines = [
            "=== Bernoulli / Symbolic Dynamics Analysis ===",
            f"Trajectory length: {wc['length']}",
            f"Unique signatures: {wc['unique_signatures']}",
            f"Entropy estimate: {wc['entropy_estimate']:.3f} bits",
            f"Mixing rate: {wc['mixing_rate']:.3f}",
            f"Lyapunov estimate: {lyap:.3f}",
            f"Recurrence time: {recur if recur else 'not observed'}",
            f"Total bits changed: {wc['total_bits_changed']}",
        ]
        return "\n".join(lines)


# =============================================================================
# Unified Controller (accepts both HLLSet and BitVector)
# =============================================================================

class UnifiedSystemController:
    """
    System controller that works with both HLLSets and BitVectors.
    
    This enables the full Bernoulli map / symbolic dynamics view
    while remaining compatible with the HLLSet algebra layer.
    """
    
    def __init__(self, n_bits: int, mode: SteeringMode = SteeringMode.INCREMENTAL):
        self.n_bits = n_bits
        self.mode = mode
        self.analyzer = BernoulliAnalyzer(n_bits)
        self.target: Optional[BitVectorState] = None
    
    def set_target_hllset(self, hll: HLLSet) -> None:
        """Set target state from HLLSet."""
        self.target = BitVectorState.from_hllset(hll)
    
    def set_target_bitvector(self, bv: BitVector) -> None:
        """Set target state from raw BitVector."""
        self.target = BitVectorState.from_bitvector(bv, self.n_bits)
    
    def observe_hllset(self, hll: HLLSet) -> Dict:
        """Observe HLLSet and return analysis."""
        state = BitVectorState.from_hllset(hll)
        return self._observe(state)
    
    def observe_bitvector(self, bv: BitVector) -> Dict:
        """Observe BitVector and return analysis."""
        state = BitVectorState.from_bitvector(bv, self.n_bits)
        return self._observe(state)
    
    def _observe(self, state: BitVectorState) -> Dict:
        """Internal observation with full analysis."""
        self.analyzer.observe(state)
        
        result = {
            "state_density": state.density,
            "state_popcount": state.popcount,
        }
        
        if len(self.analyzer.transitions) > 0:
            last_trans = self.analyzer.transitions[-1]
            result["last_transition_cost"] = last_trans.hamming_cost
            result["mixing_rate"] = self.analyzer.mixing_rate()
        
        if self.target is not None:
            result["distance_to_target"] = state.normalized_distance(self.target)
            result["at_target"] = result["distance_to_target"] < 0.01
        
        return result
    
    def compute_shift(self, current: BitVectorState) -> Optional[ShiftTransition]:
        """Compute shift transition to move toward target."""
        if self.target is None:
            return None
        return ShiftTransition.from_states(current, self.target)
    
    def analysis_summary(self) -> str:
        """Get full Bernoulli / symbolic dynamics analysis."""
        return self.analyzer.summary()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Observation
    "SystemObservation",
    "SystemMonitor",
    
    # Planning
    "TransitionPlan",
    "PathPlan",
    "plan_transition",
    "find_path",
    
    # Steering
    "SteeringMode",
    "SteeringAction",
    "SystemController",
    
    # Integrated System
    "HLLSetDynamicSystem",
    
    # Bernoulli / Symbolic Dynamics
    "BitVectorState",
    "ShiftTransition",
    "BernoulliAnalyzer",
    "UnifiedSystemController",
]
