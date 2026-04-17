# HLLSet Core Module

> Immutable HyperLogLog with full set algebra operations.

**Module**: `core.hllset`  
**Layer**: L2 — Core Anti-Set

## Overview

HLLSet is the foundational data structure of the entire system. It provides:

- **Immutable HyperLogLog** with uint32 bitmap registers
- **Full set algebra**: union, intersection, difference, XOR
- **Content-addressable** storage via SHA1 hashing
- **Batch processing** as the primary ingestion mode
- **C/Cython acceleration** with pure-Python fallback

## Design Principles

1. **Immutability**: All operations return new HLLSet instances
2. **Content-Addressable**: Same tokens → same SHA1 → same identity
3. **Batch-First**: `from_batch()` is the primary creation method
4. **Single Source of Truth**: Hash configuration lives here

## Constants

```python
P_BITS = 10                    # Precision bits (default)
M = 1024                       # Registers = 2^P_BITS
BITS_PER_REGISTER = 32         # Each register is a uint32 bitmap
REGISTER_DTYPE = np.uint32     # NumPy dtype for registers
SHARED_SEED = 42               # Default hash seed
```

## Hash Configuration

HLLSet is the **single source of truth** for hash settings across the entire system.

```python
from core.hllset import HashConfig, HashType, DEFAULT_HASH_CONFIG

# Default configuration
config = DEFAULT_HASH_CONFIG
# HashConfig(hash_type=HashType.MURMUR3, p_bits=10, seed=42, h_bits=64)

# Custom configuration
config = HashConfig(
    hash_type=HashType.SHA256,
    p_bits=12,      # 4096 registers
    seed=12345,
)

# Hash a token
h = config.hash("hello")              # → 64-bit integer
reg, zeros = config.hash_to_reg_zeros("hello")  # → (register, trailing_zeros)
```

### Hash Types

| Type | Bits | Speed | Use Case |
|------|------|-------|----------|
| `MURMUR3` | 64 | Fast | Default, matches C backend |
| `SHA1` | 160 (32 used) | Medium | Cryptographic applications |
| `SHA256` | 256 (32 used) | Slow | High-entropy requirements |

## Core Class: HLLSet

### Creation

```python
from core import HLLSet

# From tokens (primary method)
hll = HLLSet.from_batch(["apple", "banana", "cherry"])

# From pre-computed hashes
hll = HLLSet.from_hashes([12345678, 87654321])

# Empty HLLSet
hll = HLLSet(p_bits=10)

# Multi-batch with parallel processing
batches = [batch1, batch2, batch3]
hll = HLLSet.from_batches(batches, parallel=True)
```

### Properties

```python
# Cardinality (estimated count)
count = hll.cardinality()  # → float (e.g., 2.98 for 3 items)

# Content hash (SHA1 of registers)
sha1 = hll.sha1  # → "a1b2c3d4..."

# Register access
regs = hll.registers  # → numpy array of uint32
regs = hll.dump_numpy()  # → copy of registers

# Metadata
bits = hll.p_bits      # → 10
n_regs = hll.num_registers  # → 1024
pop = hll.popcount()   # → total set bits
```

### Set Operations

All operations return **new** HLLSet instances (immutable).

```python
a = HLLSet.from_batch(["x", "y", "z"])
b = HLLSet.from_batch(["y", "z", "w"])

# Union (OR) — A ∪ B
union = a.union(b)       # Contains: x, y, z, w
union = a | b            # Operator syntax

# Intersection (AND) — A ∩ B
inter = a.intersect(b)   # Contains: y, z
inter = a & b            # Operator syntax

# Difference (AND-NOT) — A \ B
diff = a.diff(b)         # Contains: x
diff = a - b             # Operator syntax

# Symmetric Difference (XOR) — A △ B
xor = a.xor(b)           # Contains: x, w
xor = a ^ b              # Operator syntax
```

### Similarity

```python
# Jaccard similarity
sim = a.similarity(b)    # → float in [0, 1]

# Overlap coefficient
overlap = a.overlap(b)   # → |A∩B| / min(|A|, |B|)
```

## Register Format

HLLSet uses **bitmap registers** (not max-value registers):

```
Register i = uint32 bitmap
  Bit k is SET when an element with k trailing zeros was observed
  
Position = register * 32 + trailing_zeros
  register ∈ [0, 1023]
  trailing_zeros ∈ [0, 31]
  position ∈ [0, 32767]
```

This format enables:

- **True set operations**: Union = OR, Intersection = AND
- **Efficient difference**: A \ B = A AND (NOT B)
- **XOR ring algebra**: A ⊕ B = A XOR B

## C Backend Acceleration

```python
from core import C_BACKEND_AVAILABLE

if C_BACKEND_AVAILABLE:
    print("Using Cython-accelerated HLL core")
else:
    print("Using pure Python fallback")
```

The C backend (`hll_core.pyx`) provides:

- 10-100x faster batch processing
- Thread-safe parallel ingestion
- Identical hash computation (MurmurHash64A)

## Utility Functions

```python
from core.hllset import compute_sha1, murmur_hash64a

# Compute SHA1 of registers
sha1 = compute_sha1(registers_array)

# Direct MurmurHash64A
h = murmur_hash64a(b"hello", seed=42)  # → 64-bit int
```

## Usage Patterns

### Pattern 1: Document Fingerprinting

```python
def fingerprint_document(text: str) -> HLLSet:
    """Create HLLSet fingerprint from document text."""
    tokens = text.lower().split()
    return HLLSet.from_batch(tokens)

doc1 = fingerprint_document("the quick brown fox")
doc2 = fingerprint_document("the lazy brown dog")

similarity = doc1.similarity(doc2)
print(f"Document similarity: {similarity:.2%}")
```

### Pattern 2: Incremental Building

```python
# Build HLLSet incrementally through union
result = HLLSet(p_bits=10)  # Empty

for batch in data_stream:
    batch_hll = HLLSet.from_batch(batch)
    result = result.union(batch_hll)  # Immutable merge
```

### Pattern 3: Set Analysis

```python
# Analyze overlap between sets
def analyze_sets(a: HLLSet, b: HLLSet) -> dict:
    union = a.union(b)
    inter = a.intersect(b)
    
    return {
        "a_only": a.diff(b).cardinality(),
        "b_only": b.diff(a).cardinality(),
        "shared": inter.cardinality(),
        "total": union.cardinality(),
        "jaccard": a.similarity(b),
    }
```

## Error Characteristics

| Cardinality Range | Typical Error |
|-------------------|---------------|
| 0-100 | ±2-5% |
| 100-10,000 | ±1-3% |
| 10,000-1M | ±1-2% |
| 1M+ | ±1-2% |

Error is **relative** to cardinality, not absolute.

## Related Modules

- [BitvectorRing](02_RING_ALGEBRA.md) — Ring-algebraic operations
- [HLLTensor](03_TENSOR_LUT.md) — 2D tensor view for disambiguation
- [HLLSetRedis](10_REDIS_MODULES.md) — Redis-backed distributed HLLSet
