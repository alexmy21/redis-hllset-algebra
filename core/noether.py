"""
Noether Steering Law — Conservation monitoring and self-regulation for HLLSet evolution.

Implements the Noether Steering Law from the manuscript (§4):

State transition:
    R(t+1) = [R(t) \\ D(t)] ∪ N(t)

where:
    R(t)  = system state at time t (HLLSet)
    N(t)  = tokens added at time t  (HLLSet)
    D(t)  = tokens deleted at time t (HLLSet)

Net information flux:
    Φ(t) = |N(t)| - |D(t)|

Noether Steering Law (Theorem 4.1):
    If |N(t)| = |D(t)| for all t, then Σᵢ Rᵢ(t) is conserved
    (modulo hash collisions).

Practical implications:
    1. System Health Monitoring: drift in Φ(t) signals imbalance
    2. Self-Regulation: adjust forgetting rate λ and novelty threshold θ
       to maintain Φ(t) ≈ 0
    3. Error Detection: sustained |Φ| > 0 signals bugs or growth phases

The Noether module is a LIBRARY — it provides the monitoring and steering
tools. Applications decide WHEN to call step() and HOW to respond to
diagnostics.

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import time
import numpy as np
import hashlib

from .bss import bss, BSSPair


# =============================================================================
# Data Types
# =============================================================================

class SteeringPhase(Enum):
    """System phase inferred from flux history."""
    BALANCED = "balanced"         # Φ ≈ 0, stable evolution
    GROWTH = "growth"             # Φ > 0 sustained, system expanding
    DECAY = "decay"               # Φ < 0 sustained, system contracting
    VOLATILE = "volatile"         # Φ oscillating, unstable


class FluxRecord(NamedTuple):
    """Record of a single time-step's information flux."""
    timestamp: float
    added_card: float       # |N(t)| — cardinality of additions
    deleted_card: float     # |D(t)| — cardinality of deletions
    flux: float             # Φ(t) = |N(t)| - |D(t)|
    state_card: float       # |R(t+1)| — cardinality after update
    popcount: int           # Total set bits after update (conserved quantity)
    state_id: str           # SHA1 of R(t+1) registers


@dataclass
class SteeringDiagnostics:
    """
    Diagnostics from the Noether steering law at a given time.
    
    Applications use these to decide whether to adjust parameters.
    The module never adjusts anything itself — it only measures.
    """
    step: int                    # Time step number
    timestamp: float             # Wall-clock time
    flux: float                  # Φ(t) = |N(t)| - |D(t)|
    cumulative_flux: float       # Σ Φ(τ) for τ ≤ t
    state_card: float            # |R(t)| estimated cardinality
    popcount: int                # Σᵢ popcount(Rᵢ) — the conserved quantity
    phase: SteeringPhase         # Inferred system phase
    drift_rate: float            # Moving average of Φ
    conservation_error: float    # |popcount(t) - popcount(0) - cumulative_flux|
    state_id: str                # SHA1 of current state

    def is_balanced(self, tolerance: float = 0.05) -> bool:
        """Check if system is approximately balanced."""
        if self.state_card <= 0:
            return True
        return abs(self.drift_rate) / max(1.0, self.state_card) < tolerance

    def __repr__(self) -> str:
        return (
            f"SteeringDiagnostics(step={self.step}, Φ={self.flux:+.1f}, "
            f"ΣΦ={self.cumulative_flux:+.1f}, |R|≈{self.state_card:.1f}, "
            f"phase={self.phase.value})"
        )


# =============================================================================
# Noether Evolution Engine
# =============================================================================

class NoetherEvolution:
    """
    Discrete dynamical system for HLLSet evolution with conservation monitoring.
    
    Implements the state transition:
        R(t+1) = [R(t) \\ D(t)] ∪ N(t)
    
    and monitors the conserved quantity (popcount) and information flux.
    
    Usage:
        from core.hllset import HLLSet
        
        # Initialize with empty state or existing HLLSet
        evolution = NoetherEvolution(p_bits=10)
        
        # Step the system
        additions = HLLSet.from_batch(["new", "tokens"])
        deletions = HLLSet.from_batch(["old", "tokens"])
        diag = evolution.step(additions, deletions)
        
        # Check conservation
        print(f"Flux: {diag.flux}, Phase: {diag.phase}")
        if not diag.is_balanced():
            print("WARNING: system drifting")
    
    The engine is a LIBRARY component — it provides tools, not control flow.
    Applications call step() when they decide to update the system.
    """

    def __init__(self, initial_state=None, p_bits: int = 10):
        """
        Create evolution engine.
        
        Args:
            initial_state: Initial HLLSet state (None = empty)
            p_bits: Precision bits (used if initial_state is None)
        """
        # Lazy import to avoid circular dependency
        from .hllset import HLLSet

        self._p_bits = p_bits

        if initial_state is not None:
            self._state = initial_state
        else:
            self._state = HLLSet(p_bits=p_bits)

        # Conservation tracking
        self._initial_popcount = self._popcount(self._state)
        self._step_count = 0
        self._cumulative_flux = 0.0

        # History (bounded ring buffer)
        self._history: List[FluxRecord] = []
        self._max_history = 10000

        # Moving average window for drift estimation
        self._drift_window = 50

    @property
    def state(self):
        """Current system state (HLLSet). Read-only — use step() to evolve."""
        return self._state

    @property
    def step_count(self) -> int:
        """Number of evolution steps performed."""
        return self._step_count

    @property
    def cumulative_flux(self) -> float:
        """Cumulative net information flux: Σ Φ(τ)."""
        return self._cumulative_flux

    @property
    def history(self) -> List[FluxRecord]:
        """Full flux history (bounded by max_history)."""
        return list(self._history)

    # =========================================================================
    # Core Evolution
    # =========================================================================

    def step(
        self,
        additions=None,
        deletions=None,
        timestamp: Optional[float] = None,
    ) -> SteeringDiagnostics:
        """
        Perform one evolution step: R(t+1) = [R(t) \\ D(t)] ∪ N(t).
        
        This is the fundamental state transition. The method:
        1. Computes the new state
        2. Measures the information flux
        3. Updates conservation tracking
        4. Returns diagnostics (but NEVER adjusts parameters)
        
        Args:
            additions: HLLSet N(t) of tokens to add (None = no additions)
            deletions: HLLSet D(t) of tokens to delete (None = no deletions)
            timestamp: Optional wall-clock timestamp (default: time.time())
            
        Returns:
            SteeringDiagnostics with conservation and flux information
        """
        from .hllset import HLLSet

        ts = timestamp if timestamp is not None else time.time()

        # Measure additions and deletions
        added_card = additions.cardinality() if additions is not None else 0.0
        deleted_card = deletions.cardinality() if deletions is not None else 0.0

        # State transition: R(t+1) = [R(t) \ D(t)] ∪ N(t)
        new_state = self._state

        if deletions is not None and deleted_card > 0:
            new_state = new_state.diff(deletions)

        if additions is not None and added_card > 0:
            new_state = new_state.union(additions)

        # Compute flux
        flux = added_card - deleted_card
        self._cumulative_flux += flux

        # Update state
        self._state = new_state
        self._step_count += 1

        # Conserved quantity: total popcount of registers
        current_popcount = self._popcount(new_state)

        # State identity
        state_id = self._state_id(new_state)

        # Record
        record = FluxRecord(
            timestamp=ts,
            added_card=added_card,
            deleted_card=deleted_card,
            flux=flux,
            state_card=new_state.cardinality(),
            popcount=current_popcount,
            state_id=state_id,
        )
        self._append_history(record)

        # Compute diagnostics
        drift_rate = self._compute_drift_rate()
        phase = self._infer_phase(drift_rate)
        conservation_error = abs(
            current_popcount - self._initial_popcount - self._cumulative_flux
        )

        return SteeringDiagnostics(
            step=self._step_count,
            timestamp=ts,
            flux=flux,
            cumulative_flux=self._cumulative_flux,
            state_card=new_state.cardinality(),
            popcount=current_popcount,
            phase=phase,
            drift_rate=drift_rate,
            conservation_error=conservation_error,
            state_id=state_id,
        )

    def step_with_tokens(
        self,
        add_tokens: Optional[List[str]] = None,
        del_tokens: Optional[List[str]] = None,
        timestamp: Optional[float] = None,
    ) -> SteeringDiagnostics:
        """
        Convenience: step with raw token lists instead of pre-built HLLSets.
        
        Args:
            add_tokens: Tokens to add (N(t))
            del_tokens: Tokens to delete (D(t))
            timestamp: Optional wall-clock timestamp
            
        Returns:
            SteeringDiagnostics
        """
        from .hllset import HLLSet

        additions = None
        deletions = None

        if add_tokens:
            additions = HLLSet.from_batch(add_tokens, p_bits=self._p_bits)
        if del_tokens:
            deletions = HLLSet.from_batch(del_tokens, p_bits=self._p_bits)

        return self.step(additions, deletions, timestamp)

    # =========================================================================
    # BSS Monitoring — Track BSS(R(t), R(t-1))
    # =========================================================================

    def bss_with_previous(self) -> Optional[BSSPair]:
        """
        Compute BSS(R(t) → R(t-1)) — how much of the previous state
        is preserved in the current state.
        
        Returns None if < 2 steps have been performed.
        """
        if len(self._history) < 2:
            return None

        # Reconstruct is expensive; instead track register snapshots.
        # For now, return None — lattice-level tracking is more appropriate.
        return None

    # =========================================================================
    # Conservation Analysis
    # =========================================================================

    def conservation_check(self) -> Dict[str, Any]:
        """
        Comprehensive conservation analysis.
        
        Returns dict with:
            - popcount_initial: Starting popcount
            - popcount_current: Current popcount
            - cumulative_flux: Σ Φ(t)
            - expected_popcount: initial + cumulative_flux
            - conservation_error: |actual - expected|
            - is_conserved: Whether error is within tolerance
            - collision_estimate: Estimated hash collision impact
        """
        current_pop = self._popcount(self._state)
        expected_pop = self._initial_popcount + self._cumulative_flux

        error = abs(current_pop - expected_pop)

        # Hash collision estimate: error grows with density
        state_card = self._state.cardinality()
        total_bits = (1 << self._p_bits) * 32
        density = state_card / total_bits if total_bits > 0 else 0
        collision_estimate = density * state_card  # Rough birthday bound

        return {
            'popcount_initial': self._initial_popcount,
            'popcount_current': current_pop,
            'cumulative_flux': self._cumulative_flux,
            'expected_popcount': expected_pop,
            'conservation_error': error,
            'is_conserved': error <= max(1, collision_estimate * 2),
            'density': density,
            'collision_estimate': collision_estimate,
            'steps': self._step_count,
        }

    def flux_statistics(self) -> Dict[str, float]:
        """
        Statistical summary of the flux history.
        
        Returns:
            Dict with mean, std, min, max, and trend of Φ(t).
        """
        if not self._history:
            return {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'trend': 0.0, 'count': 0,
            }

        fluxes = [r.flux for r in self._history]
        arr = np.array(fluxes)

        # Trend: slope of linear fit
        if len(arr) >= 2:
            x = np.arange(len(arr), dtype=np.float64)
            coeffs = np.polyfit(x, arr, 1)
            trend = coeffs[0]
        else:
            trend = 0.0

        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'trend': trend,
            'count': len(arr),
        }

    # =========================================================================
    # Reset / Snapshot
    # =========================================================================

    def snapshot(self) -> Dict[str, Any]:
        """
        Capture the current state for serialization.
        
        Returns a dict that can be used to restore the evolution state.
        """
        return {
            'state_registers': self._state.dump_numpy().tobytes(),
            'p_bits': self._p_bits,
            'step_count': self._step_count,
            'cumulative_flux': self._cumulative_flux,
            'initial_popcount': self._initial_popcount,
            'history': [r._asdict() for r in self._history[-100:]],  # Last 100
        }

    def reset(self, new_state=None):
        """
        Reset the evolution with a new initial state.
        
        Clears all history and resets conservation tracking.
        """
        from .hllset import HLLSet

        if new_state is not None:
            self._state = new_state
        else:
            self._state = HLLSet(p_bits=self._p_bits)

        self._initial_popcount = self._popcount(self._state)
        self._step_count = 0
        self._cumulative_flux = 0.0
        self._history.clear()

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    @staticmethod
    def _popcount(hllset) -> int:
        """Total set bits in all registers (the conserved quantity)."""
        registers = hllset.dump_numpy()
        return sum(int(r).bit_count() for r in registers)

    @staticmethod
    def _state_id(hllset) -> str:
        """SHA1 of register state (content-addressed identity)."""
        registers = hllset.dump_numpy()
        return hashlib.sha1(registers.tobytes()).hexdigest()

    def _append_history(self, record: FluxRecord):
        """Append to history, maintaining bounded size."""
        self._history.append(record)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def _compute_drift_rate(self) -> float:
        """Moving average of Φ over the drift window."""
        if not self._history:
            return 0.0
        window = self._history[-self._drift_window:]
        return sum(r.flux for r in window) / len(window)

    def _infer_phase(self, drift_rate: float) -> SteeringPhase:
        """Infer system phase from drift rate and variance."""
        if not self._history or len(self._history) < 3:
            return SteeringPhase.BALANCED

        recent = self._history[-self._drift_window:]
        fluxes = [r.flux for r in recent]
        variance = np.var(fluxes) if len(fluxes) > 1 else 0.0
        mean_abs_flux = np.mean(np.abs(fluxes))

        # Thresholds (relative to mean cardinality)
        if mean_abs_flux < 0.5:
            return SteeringPhase.BALANCED
        elif drift_rate > 1.0:
            return SteeringPhase.GROWTH
        elif drift_rate < -1.0:
            return SteeringPhase.DECAY
        elif variance > mean_abs_flux * 2:
            return SteeringPhase.VOLATILE
        else:
            return SteeringPhase.BALANCED

    def __repr__(self) -> str:
        return (
            f"NoetherEvolution(steps={self._step_count}, "
            f"ΣΦ={self._cumulative_flux:+.1f}, "
            f"|R|≈{self._state.cardinality():.1f})"
        )


# =============================================================================
# Standalone Convenience Functions
# =============================================================================

def compute_flux(additions, deletions) -> float:
    """
    Compute information flux Φ = |N| - |D| without performing the transition.
    
    Args:
        additions: HLLSet of tokens added
        deletions: HLLSet of tokens deleted
        
    Returns:
        Φ = |N| - |D|
    """
    n = additions.cardinality() if additions is not None else 0.0
    d = deletions.cardinality() if deletions is not None else 0.0
    return n - d


def apply_transition(state, additions=None, deletions=None):
    """
    Apply the Noether state transition without tracking.
    
    R(t+1) = [R(t) \\ D(t)] ∪ N(t)
    
    Pure function — returns new HLLSet, no side effects.
    
    Args:
        state: Current HLLSet R(t)
        additions: HLLSet N(t) to add
        deletions: HLLSet D(t) to delete
        
    Returns:
        New HLLSet R(t+1)
    """
    result = state
    if deletions is not None:
        result = result.diff(deletions)
    if additions is not None:
        result = result.union(additions)
    return result


def is_balanced(additions, deletions, tolerance: float = 0.05) -> bool:
    """
    Check if an update is approximately balanced: |Φ| / max(|N|, |D|) < tol.
    
    Args:
        additions: HLLSet of tokens to add
        deletions: HLLSet of tokens to delete
        tolerance: Relative tolerance (default 5%)
        
    Returns:
        True if the update is approximately balanced
    """
    n = additions.cardinality() if additions is not None else 0.0
    d = deletions.cardinality() if deletions is not None else 0.0
    denominator = max(n, d, 1.0)
    return abs(n - d) / denominator < tolerance


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'SteeringPhase',
    'FluxRecord',
    'SteeringDiagnostics',
    'NoetherEvolution',
    'compute_flux',
    'apply_transition',
    'is_balanced',
]
