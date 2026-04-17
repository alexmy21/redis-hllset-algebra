# Python Module Documentation

> HLLSet Ring Algebra — Python Implementation

## Quick Start

```python
from core import HLLSet

# Create from tokens
A = HLLSet.from_batch(['hello', 'world', 'hello'])
B = HLLSet.from_batch(['world', 'test'])

# Ring operations
print(A.cardinality())          # ~2
print(A.union(B).cardinality()) # ~3
print(A.similarity(B))          # Jaccard ~0.33
```

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│  L11   Evolution & Noether Conservation                [noether.py]        │
├────────────────────────────────────────────────────────────────────────────┤
│  L10   Bayesian Networks & Markov Chains       [bayesian*.py, markov_hll]  │
├────────────────────────────────────────────────────────────────────────────┤
│  L9    BSS Morphisms & Lattice                 [bss.py, hll_lattice.py]    │
├────────────────────────────────────────────────────────────────────────────┤
│  L8    De Bruijn Graph & DRN                   [debruijn.py, hllset_de*]   │
├────────────────────────────────────────────────────────────────────────────┤
│  L7    Disambiguation Engine                   [disambiguation.py]         │
├────────────────────────────────────────────────────────────────────────────┤
│  L6    Tensor & TokenLUT                       [hll_tensor.py, tokenlut*]  │
├────────────────────────────────────────────────────────────────────────────┤
│  L5    Store & Registry                        [hllset_store.py, *redis*]  │
├────────────────────────────────────────────────────────────────────────────┤
│  L4    Global Registry                         [global_registry.py]        │
├────────────────────────────────────────────────────────────────────────────┤
│  L3    Ring Transactions                       [ring_transaction.py]       │
├────────────────────────────────────────────────────────────────────────────┤
│  L2    BitVector Ring                          [bitvector_ring.py]         │
├────────────────────────────────────────────────────────────────────────────┤
│  L1    HLLSet Algebra                          [hllset.py]                 │
├────────────────────────────────────────────────────────────────────────────┤
│  L0    Cython Core (hll_core.pyx, bitvector_core.pyx)                      │
└────────────────────────────────────────────────────────────────────────────┘
```

## Documentation Index

| # | Document | Modules | Description |
|---|----------|---------|-------------|
| 01 | [HLLSet](01_HLLSET.md) | `hllset.py` | Core HLLSet class, hash config, set operations |
| 02 | [Ring Algebra](02_RING_ALGEBRA.md) | `bitvector_ring.py` | BitVector, XOR ring, Gaussian elimination |
| 03 | [Tensor & LUT](03_TENSOR_LUT.md) | `hll_tensor.py` | 2D tensor view, TokenLUT, position encoding |
| 04 | [Disambiguation](04_DISAMBIGUATION.md) | `disambiguation.py` | Triangulation, De Bruijn recovery |
| 05 | [BSS/Noether](05_BSS_NOETHER.md) | `bss.py`, `noether.py` | BSS metrics, conservation laws, steering |
| 06 | [Lattice/Store](06_LATTICE_STORE.md) | `hll_lattice.py`, `hllset_store.py` | W lattice, derivation LUT |
| 07 | [De Bruijn](07_DEBRUIJN.md) | `debruijn.py`, `hllset_debruijn.py` | De Bruijn graphs, DRN triples |
| 08 | [Bayesian/Markov](08_BAYESIAN_MARKOV.md) | `bayesian*.py`, `markov_hll.py` | Probabilistic interpretation |
| 09 | [Redis Modules](09_REDIS_MODULES.md) | `core.redis.*` | Redis-backed infrastructure |

## Quick Reference

→ See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for API cheat sheet.

## Key Concepts

### Content-Addressable Keys

Every HLLSet is identified by SHA1 of its register contents:

```python
hll = HLLSet.from_batch(['a', 'b', 'c'])
print(hll.sha1_hex)  # "a3f2...c8e1"
```

### Ring Algebra (GF(2))

HLLSets form a ring under XOR with AND as multiplication:

```python
# XOR = symmetric difference (ring addition)
A.xor(B)  # A △ B

# AND = intersection (ring multiplication)
A.intersection(B)  # A ∩ B

# Identity elements
zero = HLLSet()           # Additive identity
# A ⊕ zero = A
```

### Constants

```python
from core.hllset import P_BITS, M, BITS_PER_REGISTER

P_BITS = 10              # Precision bits
M = 2**10 = 1024         # Number of registers
BITS_PER_REGISTER = 32   # Register width
SHARED_SEED = 42         # Default hash seed
```

### Hash Configuration

```python
from core.hllset import HashConfig, HashType

config = HashConfig(
    hash_type=HashType.MURMUR3,
    p_bits=10,
    seed=42
)

hll = HLLSet.from_batch(tokens, config=config)
```

## Package Structure

```
core/
├── __init__.py              # Main exports
├── hllset.py                # L1: HLLSet class
├── bitvector_ring.py        # L2: BitVector ring
├── ring_transaction.py      # L3: Transactions
├── global_registry.py       # L4: Registry
├── hllset_store.py          # L5: In-memory store
├── hll_tensor.py            # L6: Tensor view
├── disambiguation.py        # L7: Triangulation
├── debruijn.py              # L8: De Bruijn graphs
├── hllset_debruijn.py       # L8: HLLSet De Bruijn
├── bss.py                   # L9: BSS metrics
├── hll_lattice.py           # L9: W lattice
├── bayesian.py              # L10: Bayesian
├── bayesian_network.py      # L10: BN
├── markov_hll.py            # L10: Markov chains
├── noether.py               # L11: Conservation
├── evolution.py             # Evolution dynamics
├── hllset_dynamics.py       # State tracking
├── hllset_transformer.py    # Transformations
│
└── redis/                   # Redis-backed modules
    ├── __init__.py
    ├── hllset_redis.py      # HLLSetRedis
    ├── hllset_store_redis.py
    ├── hllset_ring_store.py
    ├── tokenlut_redis.py
    ├── tokenlut_stream.py
    ├── tokenlut_session.py
    └── hllset_disambiguate.py
```

## Installation

```bash
# Core dependencies
pip install numpy mmh3

# Redis support
pip install redis redis-py[hiredis]

# Build Cython extensions
python build_ext.py
```

## Related Documentation

- [Redis Module Docs](../redis_hllset/module/docs/) — Rust module commands
- [Tutorials](../tutorials/) — Jupyter notebook tutorials
- [Redis Tutorials](../redis-tutorials/) — Redis-specific tutorials
