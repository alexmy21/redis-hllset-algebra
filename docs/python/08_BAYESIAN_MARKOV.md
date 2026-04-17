# Bayesian and Markov Modules

> Probabilistic interpretation and Markov constructs on HLLSets.

**Modules**: `core.bayesian`, `core.bayesian_network`, `core.markov_hll`  
**Layer**: L8-L10 — Probabilistic Interpretation

## Part 1: Bayesian Interpretation

### Two Interpretations of HLLSet Evolution

The same transition `R(t+1) = [R(t) \ D(t)] ∪ N(t)` has two interpretations:

| Aspect | Evolution (Noether) | Bayesian |
|--------|---------------------|----------|
| Meaning | Physical change | Belief update |
| Conservation | Popcount, flux | Bayes' theorem |
| Universe | N/A | Context for inference |
| Metric | Φ(t) = \|N\| - \|D\| | P(A\|U) |

### Core Probabilities

```python
from core.bayesian import prior, conditional, joint

# Prior: P(A) = |A| / |U|
result = prior(entity=A, universe=U)
print(f"P(A) = {result.value:.4f}")

# Conditional: P(A|B) = |A∩B| / |B|
result = conditional(A, given=B)
print(f"P(A|B) = {result.value:.4f}")

# Joint: P(A,B) = |A∩B| / |U|
result = joint(A, B, universe=U)
print(f"P(A,B) = {result.value:.4f}")
```

### Bayes' Theorem Verification

```python
from core.bayesian import bayes_theorem

result = bayes_theorem(A, B, universe=U)

print(f"P(A|B) = {result.p_a_given_b:.4f}")
print(f"P(B|A)·P(A)/P(B) = {result.bayes_rhs:.4f}")
print(f"Consistency error: {result.consistency_error:.4f}")
print(f"Consistent? {result.is_consistent()}")
```

### Information-Theoretic Measures

```python
from core.bayesian import surprise, entropy_of_partition, kl_divergence

# Surprise: -log₂(P(A))
s = surprise(entity=A, universe=U)

# Entropy of partition
H = entropy_of_partition([hll1, hll2, hll3], universe=U)

# KL divergence between distributions
kl = kl_divergence(p_distribution, q_distribution)
```

### Temporal Bayesian Analysis

```python
from core.bayesian import (
    bayesian_surprise_temporal,
    temporal_posterior,
    temporal_trajectory,
    TemporalBayesRecord,
)

# Track probability over time
records = []
for t, (state, universe) in enumerate(evolution):
    record = TemporalBayesRecord(
        timestamp=t,
        p_a=prior(entity, universe).value,
        universe_card=universe.cardinality(),
        entity_card=entity.cardinality(),
        surprise=surprise(entity, universe),
    )
    records.append(record)

# Analyze trajectory
trajectory = temporal_trajectory(records)
```

### Interpretation Comparison

```python
from core.bayesian import InterpretationComparison, interpretation_divergence

# Compare evolution vs Bayesian view of same transition
comparison = interpretation_divergence(
    flux=diag.flux,
    popcount_delta=diag.popcount - prev_popcount,
    prior_prob=p_before,
    posterior_prob=p_after,
)

print(comparison)
# Comparison(AGREE: evo=growth(Φ=+10.0), bayes=strengthened(ΔP=+0.05))
# or
# Comparison(DIVERGE: evo=balanced(Φ=+0.0), bayes=weakened(ΔP=-0.15))
```

---

## Part 2: Bayesian Network

### The Love-Hate Triangle

Three structures share the same algebraic root:

```
1. HLLSet Ring     — XOR, AND, OR (ground truth)
2. BSS Lattice     — Morphisms A →(τ,ρ) B (structural)
3. Bayesian Network — P(A|B) dependencies (epistemic)
```

**Key Isomorphism**: `τ(A → B) = |A ∩ B| / |B| = P(A | B)`

### HLLBayesNet

```python
from core.bayesian_network import HLLBayesNet

# Create from HLLSets
hllsets = {"A": hll_a, "B": hll_b, "C": hll_c}
bn = HLLBayesNet(hllsets, universe=U)
```

### Conditional Probability Tables

```python
from core.bayesian_network import CPTEntry

# Get CPT entry
cpt = bn.cpt("A", "B")  # P(A|B)

print(f"P(A|B) = {cpt.probability:.4f}")
print(f"BSS τ(A→B) = {cpt.bss_tau:.4f}")  # Same value!
print(f"BSS ρ(A→B) = {cpt.bss_rho:.4f}")  # No BN analogue
```

### Conditional Independence (d-separation)

```python
from core.bayesian_network import IndependenceResult

# Test: A ⊥ B | C ?
result = bn.test_independence("A", "B", given=["C"])

print(result)
# Independence(A ⊥ B | C, I=0.0012)

if result.independent:
    print("A and B are conditionally independent given C")
```

### Markov Blanket

```python
from core.bayesian_network import MarkovBlanket

blanket = bn.markov_blanket("A")

print(f"Parents: {blanket.parent_ids}")
print(f"Children: {blanket.child_ids}")
print(f"Co-parents: {blanket.coparent_ids}")
```

### Mutual Information

```python
from core.bayesian_network import (
    hllset_mutual_information,
    conditional_mutual_information,
)

# I(A; B)
mi = hllset_mutual_information(A, B, universe=U)

# I(A; B | C)
cmi = conditional_mutual_information(A, B, given=C, universe=U)
```

### Ring → BN Functor

```python
from core.bayesian_network import ring_to_bn_functor

# Convert ring operations to BN structure
bn = ring_to_bn_functor(bitvector_ring, labels, universe)
```

---

## Part 3: Markov Chain and HMM

### The BSS τ-Matrix IS a Transition Kernel

```
τ(A → B) = |A ∩ B| / |B| = P(A | B) = T[A, B]
```

### HLLMarkovChain

```python
from core.markov_hll import HLLMarkovChain

# Create from HLLSets
states = [hll1, hll2, hll3, hll4]
labels = ["S1", "S2", "S3", "S4"]
mc = HLLMarkovChain(states, labels)
```

### Transition Matrix

```python
# Get transition probability
p = mc.transition_prob("S1", "S2")

# Full matrix
T = mc.transition_matrix  # numpy array

# Visualize
mc.print_matrix()
```

### Stationary Distribution

```python
from core.markov_hll import StationaryResult

result = mc.stationary_distribution()

print(f"π = {result.distribution}")
print(f"Dominant state: {result.dominant_state}")
print(f"Entropy: {result.entropy:.2f} bits")
```

### PageRank

```python
from core.markov_hll import hllset_pagerank, PageRankResult

# Quick PageRank computation
result = hllset_pagerank(states, labels, damping=0.85)

for node, score in result.ranked[:5]:
    print(f"{node}: {score:.4f}")
```

### Hitting Time

```python
from core.markov_hll import HittingTimeResult

result = mc.hitting_time(source="S1", target="S4")

print(f"E[T_{{S4}} | start=S1] = {result.expected_time:.2f}")
print(f"Reachable? {result.finite}")
```

### Communicating Classes

```python
classes = mc.communicating_classes()

for cls in classes:
    print(f"States: {cls.states}")
    print(f"  Recurrent: {cls.is_recurrent}")
    print(f"  Absorbing: {cls.is_absorbing}")
    print(f"  Period: {cls.period}")
```

### Random Walk

```python
from core.markov_hll import RandomWalkTrace

trace = mc.random_walk(start="S1", steps=100)

print(f"Path: {' → '.join(trace.states[:10])}...")
print(f"Log probability: {trace.total_log_prob:.2f}")
```

### Entropy Rate

```python
result = mc.entropy_rate()

print(f"H∞ = {result.entropy_rate:.4f} bits/step")
print(f"Efficiency: {result.efficiency:.2%}")
```

### HLLHiddenMarkov (HMM)

```python
from core.markov_hll import HLLHiddenMarkov, ForwardResult, ViterbiResult

# Hidden states are true token-sets
# Observations are HLL estimates
hmm = HLLHiddenMarkov(
    hidden_states=true_hllsets,
    emission_model=hll_noise_model,
)

# Forward algorithm
observations = [obs1, obs2, obs3]
forward_result = hmm.forward(observations)
print(f"Log-likelihood: {forward_result.log_likelihood:.2f}")

# Viterbi decoding
viterbi_result = hmm.viterbi(observations)
print(f"Most likely path: {viterbi_result.path}")
```

### Markov Random Field

```python
from core.markov_hll import MarkovRandomField

# Undirected model using symmetric BSS
mrf = MarkovRandomField(hllsets, labels)

# Gibbs potentials from mutual information
potentials = mrf.pairwise_potentials()

# MAP inference via ICM
map_config = mrf.icm_inference(initial_config)
```

### Causal HLL (do-calculus)

```python
from core.markov_hll import CausalHLL

causal = CausalHLL(bn)

# Interventional query: P(Y | do(X))
result = causal.do_query(target="Y", intervention={"X": hll_x})
```

### Building from Lattice

```python
from core.markov_hll import markov_from_lattice

# Temporal lattice → Markov chain
mc = markov_from_lattice(lattice, time_range=(t0, t1))
```

## Architecture

```
BSS Lattice   ──(τ matrix)──→   Markov Chain   ──(stationary π)──→  BN priors
    ↑                                ↑                                  ↑
  ORDER                           DYNAMICS                           MEASURE
(who ⊆ whom)                  (who follows whom)                (who predicts whom)
```

## Related Modules

- [BSS/Noether](05_BSS_NOETHER.md) — BSS metrics
- [Lattice](06_LATTICE_STORE.md) — Temporal structure
- [Ring Algebra](02_RING_ALGEBRA.md) — Algebraic foundation
