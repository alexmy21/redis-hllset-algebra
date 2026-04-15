# redis-hllset-algebra

HLLSet Algebra — Probabilistic Set Operations with Bitmap Registers

## Overview

HLLSet extends HyperLogLog by storing **bitmap registers** instead of max values.
This enables full set algebra (∪, ∩, \, ⊕) while maintaining probabilistic cardinality estimation.

## Key Features

- **Set Algebra**: Union, Intersection, Difference, Symmetric Difference
- **Bitmap Registers**: Each register is a 32-bit bitmap of observed trailing-zero counts
- **Horvitz-Thompson Estimator**: Statistically correct cardinality for bitmap representation
- **Roaring Bitmap Storage**: Efficient sparse storage in Redis
- **Content Addressing**: SHA1-based deterministic naming

## Cardinality Estimation

HLLSet uses the **Horvitz-Thompson estimator** for cardinality:

```
f̂_s = -n × ln(1 - c_s/n)
```

Where:
- `c_s` = number of registers with bit `s` set  
- `n` = total registers (default: 1024)
- Total cardinality = Σ f̂_s

This is derived from occupancy modeling and provides unbiased estimates under the Poisson sampling model.

### Accuracy

| Cardinality | Error |
|-------------|-------|
| ≤ 1,000     | ~0%   |
| ≤ 10,000    | ~2%   |
| ≤ 100,000   | ~4%   |

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from core.hllset import HLLSet

# Create from tokens
hll = HLLSet.from_batch(["hello", "world", "foo", "bar"])
print(f"Cardinality: {hll.cardinality()}")

# Set operations
a = HLLSet.from_batch(["a", "b", "c"])
b = HLLSet.from_batch(["b", "c", "d"])

union = a.union(b)        # a ∪ b
inter = a.intersect(b)    # a ∩ b  
diff = a.diff(b)          # a \ b
xor = a.xor(b)            # a ⊕ b
```

## Redis Module

The `redis_hllset` module provides native Redis commands:

```
HLLSET.CREATE token1 token2 ...   # Create HLLSet, returns content-addressed key
HLLSET.CARD key                   # Get cardinality (Horvitz-Thompson)
HLLSET.UNION key1 key2            # Union of two sets
HLLSET.INTER key1 key2            # Intersection
HLLSET.DIFF key1 key2             # Difference
HLLSET.XOR key1 key2              # Symmetric difference
```

## Tutorials

See `tutorials/` for detailed examples:

- `01_hllset.ipynb` — Core HLLSet operations
- `02_hll_tensor.ipynb` — 2D tensor view
- `03_hll_lattice.ipynb` — Temporal lattice structures
- `04_hllset_store.ipynb` — Persistent storage

## References

- Horvitz-Thompson estimator for presence/absence data
- Chao1 estimator for species richness
- HyperLogLog (Flajolet et al., 2007)
- Roaring Bitmaps (Lemire et al., 2016)
