# Ring Algebra Module

> Boolean ring (Z/2Z)^N operations on bit vectors.

**Module**: `core.bitvector_ring`  
**Layer**: L0 — Ring Foundation

## Overview

The bitvector_ring module provides the algebraic foundation for HLLSet operations. It implements:

1. **Boolean Ring** (Z/2Z)^N:
   - Addition: XOR (symmetric difference)
   - Multiplication: AND (intersection)
   - Identities: 0 (zero vector), 1 (all-ones)

2. **Lattice Structure**:
   - Join: OR (union)
   - Meet: AND (intersection)
   - Partial order: subset relation

3. **Gaussian Elimination over F₂**:
   - Basis computation
   - Coefficient vectors
   - Unique XOR decomposition

## Ring Axioms

The Boolean ring satisfies:

| Property | Ring | Lattice |
|----------|------|---------|
| Associative | (a⊕b)⊕c = a⊕(b⊕c) | (a∨b)∨c = a∨(b∨c) |
| Commutative | a⊕b = b⊕a | a∨b = b∨a |
| Identity | a⊕0 = a | a∨∅ = a |
| Inverse | a⊕a = 0 | — |
| Idempotent | a∧a = a | a∨a = a |

**Bridge Law**: `A ∪ B = (A △ B) △ (A ∩ B)`

## BitVector Class

### Creation

```python
from core import BitVector

# From integer value
bv = BitVector(value=0b1010, N=8)  # 4 bits set in 8-bit vector

# Special vectors
zeros = BitVector.zeros(N=32)       # Additive identity
ones = BitVector.ones(N=32)         # Multiplicative identity

# From bit positions
bv = BitVector.from_bits([0, 2, 5], N=8)  # bits 0, 2, 5 set

# From numpy array (HLLSet registers)
bv = BitVector.from_numpy(hllset.registers)
```

### Properties

```python
bv = BitVector(value=0b10110, N=8)

bv.N              # → 8 (bit width)
bv.value          # → 22 (integer value)
bv.mask           # → 255 (0xFF for N=8)
bv.popcount()     # → 3 (number of set bits)
bv.is_zero()      # → False
bv.bit(2)         # → True (bit 2 is set)
bv.leading_zeros()   # → 5
bv.trailing_zeros()  # → 1
```

### Ring Operations

```python
a = BitVector.from_bits([0, 1, 2], N=8)  # 0b00000111
b = BitVector.from_bits([1, 2, 3], N=8)  # 0b00001110

# Ring addition (XOR) — symmetric difference
c = a ^ b        # 0b00001001 = bits {0, 3}
c = a.xor(b)     # Same

# Ring multiplication (AND) — intersection
c = a & b        # 0b00000110 = bits {1, 2}
c = a.and_(b)    # Same

# Complement (via XOR with ones)
c = ~a           # 0b11111000
c = a.complement()

# Lattice join (OR) — union
c = a | b        # 0b00001111 = bits {0, 1, 2, 3}
c = a.or_(b)     # Same

# Difference (AND-NOT)
c = a.andnot(b)  # 0b00000001 = bits {0}
```

### Conversions

```python
# To/from numpy
arr = bv.to_numpy(dtype=np.uint32)
bv = BitVector.from_numpy(arr)

# To list of set bit positions
bits = bv.to_bits()  # → [1, 2, 4] for value=0b10110

# To hex string
s = bv.to_hex()  # → "16" for value=22
```

## BitVectorRing Class

The `BitVectorRing` manages collections of bit vectors and provides:

- Gaussian elimination over F₂
- Basis computation
- XOR decomposition

### Basis Computation

```python
from core import BitVectorRing

# Create ring with fixed bit width
ring = BitVectorRing(N=32768)  # 1024 registers × 32 bits

# Add vectors (returns True if linearly independent)
is_new = ring.add(bv1)  # True: new basis element
is_new = ring.add(bv2)  # True: independent of bv1
is_new = ring.add(bv3)  # False: expressible as XOR of bv1, bv2

# Current basis
basis = ring.basis  # List of independent BitVectors
rank = ring.rank    # Number of basis elements
```

### XOR Decomposition

```python
# Decompose vector into XOR of basis elements
coeffs = ring.decompose(target)
# coeffs[i] = 1 if basis[i] is in the XOR expression

# Reconstruct from coefficients
reconstructed = ring.reconstruct(coeffs)
assert reconstructed == target
```

### Gaussian Elimination

```python
# Full Gaussian elimination on a set of vectors
vectors = [bv1, bv2, bv3, bv4]
basis, transform = ring.gaussian_elimination(vectors)

# basis: linearly independent vectors
# transform: how each original vector decomposes
```

## Mathematical Properties

### Characteristic 2

```python
a = BitVector.from_bits([1, 3, 5], N=8)

# Self-XOR is always zero
assert (a ^ a).is_zero()  # a ⊕ a = 0

# Equivalent: a is its own additive inverse
assert a == -a  # In characteristic 2
```

### Ring vs Lattice

```python
a = BitVector.from_bits([0, 1], N=4)
b = BitVector.from_bits([1, 2], N=4)

# Ring: XOR loses information
ring_result = a ^ b  # {0, 2} — lost that both had bit 1

# Lattice: OR preserves all
lattice_result = a | b  # {0, 1, 2} — keeps everything

# Bridge: Union from XOR and AND
assert (a | b) == (a ^ b) ^ (a & b)
```

### Idempotence

```python
a = BitVector.from_bits([0, 2, 4], N=8)

# AND is idempotent
assert (a & a) == a  # a ∧ a = a

# OR is idempotent  
assert (a | a) == a  # a ∨ a = a

# XOR is NOT idempotent
assert (a ^ a) != a  # a ⊕ a = 0 ≠ a
```

## Use Cases

### 1. HLLSet Compression

```python
# Store only basis HLLSets, reconstruct others via XOR
ring = BitVectorRing(N=32768)
bases = []

for hll in hllset_collection:
    bv = BitVector.from_numpy(hll.registers)
    if ring.add(bv):
        bases.append(hll)  # Store this one
    else:
        # Record XOR expression instead
        coeffs = ring.decompose(bv)
        # Store coeffs (much smaller than full HLLSet)
```

### 2. Change Detection

```python
# XOR reveals differences
old_bv = BitVector.from_numpy(old_state.registers)
new_bv = BitVector.from_numpy(new_state.registers)

diff = old_bv ^ new_bv
changed_positions = diff.to_bits()
print(f"Changed positions: {changed_positions}")
```

### 3. Dependency Analysis

```python
# Check if C is linearly dependent on A and B
ring = BitVectorRing(N=32768)
ring.add(BitVector.from_numpy(a.registers))
ring.add(BitVector.from_numpy(b.registers))

c_bv = BitVector.from_numpy(c.registers)
is_independent = ring.add(c_bv)

if not is_independent:
    coeffs = ring.decompose(c_bv)
    print(f"C = {'A' if coeffs[0] else ''} ⊕ {'B' if coeffs[1] else ''}")
```

## Performance Notes

- **Popcount**: O(N/64) using hardware popcount
- **XOR/AND/OR**: O(N/64) — bitwise on 64-bit words
- **Gaussian elimination**: O(r² × N) where r = rank
- **Decomposition**: O(r × N) per vector

For N = 32768 (1024 × 32), typical operations complete in microseconds.

## Related Modules

- [HLLSet](01_HLLSET.md) — Main HLLSet class
- [HLLTensor](03_TENSOR_LUT.md) — 2D tensor view bridging ring and semantic layers
- [Lattice](06_LATTICE.md) — Temporal W lattice using ring operations
