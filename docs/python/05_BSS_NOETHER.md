# BSS and Noether Modules

> Bell State Similarity metrics and conservation-based steering.

**Modules**: `core.bss`, `core.noether`  
**Layer**: L4 — Metrics and Conservation

## Part 1: Bell State Similarity (BSS)

### Overview

BSS provides **directed similarity metrics** for HLLSets:

```
BSS_τ(A → B) = |A ∩ B| / |B|    (inclusion: how much of B is covered by A)
BSS_ρ(A → B) = |A \ B| / |B|    (exclusion: how much of A is extraneous)
```

### Key Properties

1. `0 ≤ τ, ρ ≤ 1`
2. `τ + ρ ≤ 1` (equality when A ⊆ B)
3. `τ(A→A) = 1, ρ(A→A) = 0` (identity morphism)

### BSS is NOT Symmetric

```python
from core import HLLSet
from core.bss import bss

a = HLLSet.from_batch(["x", "y", "z"])      # |A| = 3
b = HLLSet.from_batch(["x", "y"])           # |B| = 2

# A → B: A covers all of B
tau_ab, rho_ab = bss(a, b)  # τ ≈ 1.0, ρ ≈ 0.5 (z is extraneous)

# B → A: B covers 2/3 of A
tau_ba, rho_ba = bss(b, a)  # τ ≈ 0.67, ρ ≈ 0.0 (nothing extraneous)
```

### Core Functions

```python
from core.bss import bss, bss_symmetric, test_morphism, BSSPair

# Basic BSS computation
pair = bss(source, target)  # → BSSPair(tau=0.8, rho=0.1)

# Both directions at once
ab, ba = bss_symmetric(a, b)

# Test if morphism exists
result = test_morphism(
    source=a, 
    target=b, 
    tau_threshold=0.7, 
    rho_threshold=0.3
)
print(result)  # Morphism(→, τ=0.85≥0.7, ρ=0.1≤0.3)
print(result.exists)  # True
```

### MorphismResult

```python
from core.bss import MorphismResult

@dataclass
class MorphismResult:
    exists: bool              # Does morphism exist?
    bss: BSSPair              # The (τ, ρ) pair
    source_card: float        # |A|
    target_card: float        # |B|
    intersection_card: float  # |A ∩ B|
    difference_card: float    # |A \ B|
    tau_threshold: float      # τ threshold used
    rho_threshold: float      # ρ threshold used
    
    @property
    def margin_tau(self) -> float:
        """Distance above τ threshold (positive = passes)."""
        
    @property
    def margin_rho(self) -> float:
        """Distance below ρ threshold (positive = passes)."""
```

### BSS Matrix

```python
from core.bss import bss_matrix

hllsets = [hll1, hll2, hll3, hll4]
labels = ["A", "B", "C", "D"]

# Compute all pairwise BSS values
matrix = bss_matrix(hllsets, labels)
# Returns dict: {("A", "B"): BSSPair, ("A", "C"): BSSPair, ...}

# Access specific pair
ab_bss = matrix[("A", "B")]
print(f"τ(A→B) = {ab_bss.tau:.3f}")
```

### Morphism Graph

```python
from core.bss import morphism_graph

# Build graph of valid morphisms
graph = morphism_graph(
    hllsets, 
    labels, 
    tau_threshold=0.6, 
    rho_threshold=0.4
)

# graph is a dict: {source_label: [(target_label, BSSPair), ...]}
for source, targets in graph.items():
    for target, bss_pair in targets:
        print(f"{source} → {target}: τ={bss_pair.tau:.2f}")
```

### Register-Level BSS

```python
from core.bss import bss_from_registers, bss_summary

# Direct computation from numpy arrays
bss_pair = bss_from_registers(regs_a, regs_b)

# Summary statistics
summary = bss_summary(source, target)
# Returns detailed breakdown including per-register contributions
```

---

## Part 2: Noether Steering Law

### Overview

The Noether module monitors **information conservation** during HLLSet evolution:

```
State Transition:  R(t+1) = [R(t) \ D(t)] ∪ N(t)

where:
  R(t)  = system state at time t
  N(t)  = tokens added (Novel)
  D(t)  = tokens deleted

Net Flux:  Φ(t) = |N(t)| - |D(t)|
```

**Noether's Law**: If `|N(t)| = |D(t)|` for all t, then total popcount is conserved.

### Steering Phases

```python
from core.noether import SteeringPhase

class SteeringPhase(Enum):
    BALANCED = "balanced"   # Φ ≈ 0, stable evolution
    GROWTH = "growth"       # Φ > 0 sustained
    DECAY = "decay"         # Φ < 0 sustained
    VOLATILE = "volatile"   # Φ oscillating
```

### NoetherEvolution Engine

```python
from core import HLLSet
from core.noether import NoetherEvolution

# Initialize evolution tracker
evolution = NoetherEvolution(p_bits=10)

# Or with initial state
initial = HLLSet.from_batch(["existing", "tokens"])
evolution = NoetherEvolution(initial_state=initial)
```

### Stepping the System

```python
# Create additions and deletions
additions = HLLSet.from_batch(["new", "tokens"])
deletions = HLLSet.from_batch(["old", "removed"])

# Step and get diagnostics
diag = evolution.step(additions=additions, deletions=deletions)

print(f"Flux Φ(t): {diag.flux:+.1f}")
print(f"Cumulative: {diag.cumulative_flux:+.1f}")
print(f"Phase: {diag.phase}")
print(f"Balanced? {diag.is_balanced()}")
```

### SteeringDiagnostics

```python
from core.noether import SteeringDiagnostics

@dataclass
class SteeringDiagnostics:
    step: int                    # Time step number
    timestamp: float             # Wall-clock time
    flux: float                  # Φ(t) = |N| - |D|
    cumulative_flux: float       # Σ Φ(τ)
    state_card: float            # |R(t)| estimated
    popcount: int                # Σᵢ popcount(Rᵢ) — conserved
    phase: SteeringPhase         # Current phase
    drift_rate: float            # Moving average of Φ
    conservation_error: float    # Deviation from conservation
    state_id: str                # SHA1 of current state
    
    def is_balanced(self, tolerance: float = 0.05) -> bool:
        """Check if approximately balanced."""
```

### FluxRecord

```python
from core.noether import FluxRecord

# Access flux history
for record in evolution.history:
    print(f"t={record.timestamp:.1f}: Φ={record.flux:+.1f}, |R|={record.state_card:.0f}")
```

### Utility Functions

```python
from core.noether import compute_flux, apply_transition, is_balanced

# Compute flux without stepping
flux = compute_flux(additions, deletions)

# Apply transition manually
new_state = apply_transition(current_state, additions, deletions)

# Quick balance check
balanced = is_balanced(evolution, tolerance=0.1)
```

### Monitoring Workflow

```python
from core.noether import NoetherEvolution

evolution = NoetherEvolution()

# Processing loop with conservation monitoring
for batch_idx, (additions, deletions) in enumerate(data_stream):
    diag = evolution.step(additions, deletions)
    
    # Alert on sustained drift
    if not diag.is_balanced(tolerance=0.1):
        print(f"WARNING: System drifting at step {diag.step}")
        print(f"  Phase: {diag.phase}")
        print(f"  Drift rate: {diag.drift_rate:+.2f}")
        
    # Alert on conservation violation
    if diag.conservation_error > 10:
        print(f"ALERT: Conservation error {diag.conservation_error}")
```

### Self-Regulation Pattern

```python
def self_regulate(evolution: NoetherEvolution, 
                  additions: HLLSet,
                  target_flux: float = 0.0) -> HLLSet:
    """
    Automatically compute deletions to maintain target flux.
    """
    current_state = evolution.current_state
    
    # Compute required deletions to balance
    if target_flux == 0.0:
        # Balance: delete same amount as added
        target_delete_card = additions.cardinality()
    else:
        target_delete_card = additions.cardinality() - target_flux
    
    # Find oldest/least-referenced tokens to delete
    # (application-specific logic here)
    deletions = select_tokens_to_delete(current_state, target_delete_card)
    
    return deletions
```

## Integration: BSS + Noether

BSS and Noether work together for system analysis:

```python
from core.bss import bss
from core.noether import NoetherEvolution

evolution = NoetherEvolution()

# Track evolution with BSS metrics
prev_state = None
for additions, deletions in stream:
    diag = evolution.step(additions, deletions)
    curr_state = evolution.current_state
    
    if prev_state is not None:
        # BSS shows directed relationship
        bss_fwd = bss(prev_state, curr_state)  # How much old→new?
        bss_bwd = bss(curr_state, prev_state)  # How much new→old?
        
        print(f"Step {diag.step}:")
        print(f"  Flux: {diag.flux:+.1f}")
        print(f"  τ(old→new): {bss_fwd.tau:.2f}")
        print(f"  τ(new→old): {bss_bwd.tau:.2f}")
    
    prev_state = curr_state
```

## Related Modules

- [HLLSet](01_HLLSET.md) — Core set operations
- [Lattice](06_LATTICE.md) — Temporal W lattice
- [Dynamics](07_DEBRUIJN.md) — System monitoring and steering
- [Bayesian](08_BAYESIAN.md) — Probabilistic interpretation
