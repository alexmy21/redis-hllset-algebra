# Python Quick Reference

> API cheat sheet for HLLSet Ring Algebra modules.

---

## HLLSet Core

```python
from core import HLLSet

# Create
HLLSet()                              # Empty
HLLSet.from_batch(['a', 'b'])         # From tokens
HLLSet.from_hashes([h1, h2])          # From 64-bit hashes
HLLSet.from_raw_bytes(data)           # Deserialize

# Query
hll.cardinality()                     # Estimated count
hll.popcount()                        # Non-zero registers
hll.sha1_hex                          # Content hash
hll.registers                         # numpy array [1024]

# Operations
A.union(B)                            # A ∪ B (OR)
A.intersection(B)                     # A ∩ B (AND)
A.difference(B)                       # A \ B
A.xor(B)                              # A △ B (XOR)
A.similarity(B)                       # Jaccard

# Serialize
hll.to_raw_bytes()                    # → bytes
```

---

## BitVector Ring

```python
from core import BitVector, BitVectorRing

# BitVector
bv = BitVector(hll)
bv.as_bytes()                         # M×4 bytes
bv.hamming_weight()                   # Popcount

# Ring
ring = BitVectorRing()
ring.add(bv)                          # → (is_new_base, expression)
ring.rank                             # Basis size
ring.express(bv)                      # → [b1, b2, ...] or None
```

---

## BSS Metrics

```python
from core import bss_triple, MorphismDiagnostics

# τ = coverage, ρ = retention
result = bss_triple(A, B)
print(result.tau, result.rho)

# Full diagnostics
diag = MorphismDiagnostics.compute(A, B)
print(diag.flux, diag.steering_vector)
```

---

## HLL Tensor

```python
from core import HLLTensor

tensor = HLLTensor(hll)
tensor[reg, zeros]                    # Access
tensor.project_register(r)            # 1D slice
tensor.density()                      # Fill ratio
```

---

## TokenLUT

```python
from core import TokenLUT

lut = TokenLUT()
lut.add('apple', layer=0)
lut.add(['quick', 'brown'], layer=1)
lut.lookup(reg=42, zeros=5)           # → TokenEntry
lut.lookup_register(42)               # All at register
```

---

## Disambiguation

```python
from core import DisambiguationEngine

engine = DisambiguationEngine(lut)
candidates = engine.candidates_for_hllset(hll)
triangulated = engine.triangulate(hll, target_layer=1)
```

---

## De Bruijn Graph

```python
from core import DeBruijnGraph, HLLSetDeBruijnGraph

# Generic
dbg = DeBruijnGraph(k=3)
dbg.add_sequence('ATCGATCG')
path = dbg.eulerian_path()

# HLLSet-based
hdbg = HLLSetDeBruijnGraph(k=3)
hdbg.add_token_sequence(['a', 'b', 'c', 'd'])
drn = hdbg.get_drn('b')  # DRN triple
```

---

## Lattice & Store

```python
from core import HLLLattice, HLLSetStore

# Lattice
lattice = HLLLattice()
lattice.add(hll, t=1)
lattice.commit(t=1)
diff = lattice.diff(t1=1, t2=2)

# Store
store = HLLSetStore()
sha1 = store.register(hll, source='doc1')
hll = store.get(sha1)
deriv = store.derivation(sha1)
```

---

## Bayesian

```python
from core.bayesian import prior, conditional, joint, bayes_theorem

prior(A, universe=U)                  # P(A)
conditional(A, given=B)               # P(A|B)
joint(A, B, universe=U)               # P(A,B)
bayes_theorem(A, B, U)                # Verify Bayes
```

---

## Bayesian Network

```python
from core.bayesian_network import HLLBayesNet

bn = HLLBayesNet({'A': hllA, 'B': hllB}, universe=U)
bn.cpt('A', 'B')                      # P(A|B)
bn.test_independence('A', 'B', ['C']) # d-separation
bn.markov_blanket('A')
```

---

## Markov Chain

```python
from core.markov_hll import HLLMarkovChain, hllset_pagerank

mc = HLLMarkovChain(states, labels)
mc.transition_prob('S1', 'S2')        # T[S1, S2]
mc.stationary_distribution()
mc.hitting_time('S1', 'S4')
mc.entropy_rate()

hllset_pagerank(states, labels)       # Quick PageRank
```

---

## Noether Conservation

```python
from core import NoetherDiagnostics

diag = NoetherDiagnostics.compute(hll_before, hll_after, delta, novel)
print(diag.flux)                      # Φ = |N| - |D|
print(diag.popcount_conserved)        # Bool
print(diag.steering_type)             # 'growth' | 'decay' | 'balanced'
```

---

## Redis Modules

```python
from core.redis import (
    HLLSetRedis, RedisClientManager,
    HLLSetStoreRedis, HLLSetRingStore,
    TokenLUTRedis, TokenLUTStream, TokenLUTSession,
    HLLSetDisambiguator,
)

# Setup
r = redis.Redis()
RedisClientManager.set_default(r)

# HLLSetRedis
hll = HLLSetRedis.from_batch(['a', 'b'])
hll.union(other).finalize()           # Content-addressed

# TokenLUTRedis
lut = TokenLUTRedis(r)
lut.add_token('apple', layer=0)
lut.lookup(reg=42, zeros=5)

# Ring Store
store = HLLSetRingStore(r)
ring = store.init_ring('session:1')
store.ingest(ring, 'token')
store.decompose(ring, hll)

# Session Streaming
session = TokenLUTSession(r)
session.start()
p = session.create_producer(layer=0)
p.send([('apple', '')])
cp = session.checkpoint()
result = session.commit()

# Disambiguation
disamb = HLLSetDisambiguator(r)
for c in disamb.candidates('hllset:key', 'tokenlut:entry:'):
    print(c.token, c.reg, c.zeros)
```

---

## Constants

```python
from core.hllset import P_BITS, M, BITS_PER_REGISTER, SHARED_SEED

P_BITS = 10              # Precision
M = 1024                 # Registers
BITS_PER_REGISTER = 32   # Width
SHARED_SEED = 42         # Default seed
```

---

## Common Patterns

### Content-Addressable Pipeline

```python
hll = HLLSet.from_batch(tokens)
sha1 = hll.sha1_hex
store.register(hll, source='input')
# Later...
recovered = store.get(sha1)
```

### Ring Decomposition

```python
ring = BitVectorRing()
for hll in hllsets:
    is_new, expr = ring.add(BitVector(hll))
    if is_new:
        print(f"New basis element")
    else:
        print(f"Expressed as XOR of {expr}")
```

### BSS-Guided Steering

```python
diag = MorphismDiagnostics.compute(current, target)
if diag.flux > 0:
    # Growing: add tokens
    pass
elif diag.flux < 0:
    # Decaying: remove tokens
    pass
```

### Temporal Lattice

```python
lattice = HLLLattice()
for t, hll in enumerate(evolution):
    lattice.add(hll, t=t)
    lattice.commit(t=t)
    
diff = lattice.diff(t1=0, t2=10)
print(diff.novel, diff.deleted)
```
