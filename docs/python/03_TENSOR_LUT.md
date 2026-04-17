# Tensor and TokenLUT Module

> 2D tensor view of HLLSet for token disambiguation.

**Module**: `core.hll_tensor`  
**Layer**: L1 — Tensor View

## Overview

The HLLTensor module provides the bridge between:

- **Ring layer** (bitvector_ring.py): Pure bitwise operations
- **Semantic layer** (hllset.py): Token inscription, cardinality

The HLLSet fingerprint is viewed as a **2D tensor** of shape `(M, 32)`:

```
     Trailing Zeros →
   ┌──────────────────────────┐
 R │ 0  1  2  3  ... 30  31  │
 e │ ·  ·  ·  ·      ·   ·   │
 g │ ·  ·  ■  ·      ·   ·   │  ← bit at (reg=2, zeros=2)
 i │ ·  ·  ·  ·      ·   ·   │
 s │ ·  ■  ·  ·      ·   ·   │  ← bit at (reg=4, zeros=1)
 t │ ·  ·  ·  ·      ·   ·   │
 e │                          │
 r │         ...              │
   └──────────────────────────┘
     1024 rows × 32 columns
```

## Position Encoding

```
Hash(token) → 64-bit value
           ↓
Register = hash & (M - 1)           # Lower P bits
Zeros = count_trailing_zeros(hash >> P)  # Capped at 31
           ↓
Position = register × 32 + zeros
```

**Example**: token "apple" → hash=0x...3A5C → reg=92, zeros=2 → position=2946

## HLLTensor Class

### Creation

```python
from core import HLLTensor

# Empty tensor
tensor = HLLTensor(p_bits=10)  # 1024 registers

# From existing registers
tensor = HLLTensor.from_registers(hllset.registers, p_bits=10)

# From numpy array (alias)
tensor = HLLTensor.from_numpy(registers_array, p_bits=10)

# From BitVector
tensor = HLLTensor.from_bitvector(bv, p_bits=10)
```

### Properties

```python
tensor.p_bits            # → 10
tensor.num_registers     # → 1024
tensor.bits_per_register # → 32
tensor.total_bits        # → 32768
tensor.registers         # → numpy array of uint32
```

### Inscription (Setting Bits)

```python
# Set single bit
tensor.inscribe(reg=42, zeros=5)

# Set multiple bits
positions = [(10, 3), (20, 7), (30, 1)]
tensor.inscribe_batch(positions)

# Check bit
is_set = tensor.get_bit(reg=42, zeros=5)  # → True

# Clear bit
tensor.clear_bit(reg=42, zeros=5)
```

### Active Positions

```python
# Get all (reg, zeros) pairs where bit is set
positions = tensor.active_positions()
# → [(0, 2), (5, 1), (10, 7), ...]

# Iterate lazily
for reg, zeros in tensor.iter_active_positions():
    print(f"Position: reg={reg}, zeros={zeros}")

# Count active positions
count = tensor.active_count()  # → 150
```

### Ring Operations

```python
# Convert to BitVector for ring operations
bv = tensor.to_bitvector()

# Apply ring operation and convert back
result_bv = bv1 ^ bv2
result_tensor = HLLTensor.from_bitvector(result_bv, p_bits=10)
```

## TokenLUT Class

The TokenLUT (Token Lookup Table) maps positions to candidate tokens for disambiguation.

### Creation

```python
from core.hll_tensor import TokenLUT, TokenEntry

# Create empty LUT
lut = TokenLUT(p_bits=10)
```

### TokenEntry Structure

```python
@dataclass
class TokenEntry:
    token: str          # The token string
    reg: int            # Register index [0, 1023]
    zeros: int          # Trailing zeros [0, 31]
    hash_full: int      # Full 64-bit hash
    layer: int          # N-gram layer (0=uni, 1=bi, 2=tri)
    first_token: str    # First token (for n-gram linking)
```

### Adding Entries

```python
# Add single token
entry = TokenEntry(
    token="apple",
    reg=42,
    zeros=5,
    hash_full=12345678,
    layer=0,
    first_token="apple"
)
lut.add(entry)

# Batch add from tokens
tokens = ["apple", "banana", "cherry"]
lut.add_tokens(tokens, layer=0)

# Add n-grams
bigrams = [("quick", "brown"), ("brown", "fox")]
lut.add_ngrams(bigrams, layer=1)
```

### Lookup

```python
# Lookup by position
candidates = lut.lookup(reg=42, zeros=5)
# → [TokenEntry("apple", ...), TokenEntry("app", ...)]  # hash collisions

# Lookup by register (any zeros)
candidates = lut.lookup_register(reg=42)
# → All tokens at register 42

# Lookup by layer
candidates = lut.lookup_layer(layer=1)
# → All bigrams
```

### Position Statistics

```python
# Entries per position
stats = lut.position_stats()
# → {(42, 5): 2, (10, 3): 1, ...}

# Collision count
collisions = lut.collision_count()
# → Number of positions with >1 token
```

## TensorRingAdapter

Bridges HLLTensor and BitVectorRing for algebraic operations.

```python
from core.hll_tensor import TensorRingAdapter

# Create adapter
adapter = TensorRingAdapter(p_bits=10)

# Convert operations
tensor_result = adapter.xor(tensor1, tensor2)
tensor_result = adapter.intersect(tensor1, tensor2)
tensor_result = adapter.union(tensor1, tensor2)
tensor_result = adapter.diff(tensor1, tensor2)

# Basis operations
is_independent = adapter.add_to_basis(tensor)
coefficients = adapter.decompose(tensor)
```

## Disambiguation Workflow

```python
from core import HLLSet, HLLTensor
from core.hll_tensor import TokenLUT

# 1. Build TokenLUT from corpus
lut = TokenLUT(p_bits=10)
for doc in corpus:
    tokens = tokenize(doc)
    lut.add_tokens(tokens, layer=0)
    
    bigrams = make_ngrams(tokens, n=2)
    lut.add_ngrams(bigrams, layer=1)

# 2. Create HLLSet from query
query_hll = HLLSet.from_batch(["quick", "fox"])

# 3. Get active positions via tensor view
tensor = HLLTensor.from_numpy(query_hll.registers)
positions = tensor.active_positions()

# 4. Lookup candidates at each position
candidates = []
for reg, zeros in positions:
    entries = lut.lookup(reg, zeros)
    candidates.extend(entries)

# 5. Rank/filter candidates
# (See disambiguation.py for advanced methods)
```

## Position Distribution Analysis

```python
def analyze_distribution(tensor: HLLTensor) -> dict:
    """Analyze how bits are distributed across the tensor."""
    positions = tensor.active_positions()
    
    # Count per register
    reg_counts = {}
    for reg, zeros in positions:
        reg_counts[reg] = reg_counts.get(reg, 0) + 1
    
    # Count per zeros value
    zeros_counts = {}
    for reg, zeros in positions:
        zeros_counts[zeros] = zeros_counts.get(zeros, 0) + 1
    
    return {
        "total_active": len(positions),
        "registers_used": len(reg_counts),
        "avg_per_register": len(positions) / len(reg_counts),
        "zeros_distribution": zeros_counts,
    }
```

## Efficiency Notes

| Operation | Complexity |
|-----------|------------|
| `inscribe()` | O(1) |
| `get_bit()` | O(1) |
| `active_positions()` | O(M) |
| `to_bitvector()` | O(M) |
| `lookup()` | O(k) where k = entries at position |

## Related Modules

- [HLLSet](01_HLLSET.md) — Core HLLSet class
- [Ring Algebra](02_RING_ALGEBRA.md) — BitVector operations
- [Disambiguation](04_DISAMBIGUATION.md) — Token recovery engine
