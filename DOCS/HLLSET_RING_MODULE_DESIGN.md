# HLLSET.RING Rust Module Design

This document specifies the Rust module extension for server-side HLLSet ring operations,
eliminating round-trips between Python and Redis.

## Overview

The HLLSET.RING module implements XOR ring algebra inside Redis, enabling:
- Server-side Gaussian elimination (no round-trips)
- Atomic ingest-decompose-store operations
- Efficient basis maintenance
- W lattice commit management

## Commands

### HLLSET.RING.INIT

Initialize a new ring for decomposition.

```
HLLSET.RING.INIT <ring_key> [PBITS <p>]
```

**Arguments:**
- `ring_key`: Key for the ring state
- `PBITS`: Precision bits (default: 10)

**Returns:** OK

**Example:**
```redis
HLLSET.RING.INIT session:ring1 PBITS 10
> OK
```

**Storage:**
```
hllring:ring:session:ring1 → {
    ring_id: "session:ring1",
    basis_sha1s: "[]",
    p_bits: 10,
    rank: 0,
    created_at: 1713200000.0,
    updated_at: 1713200000.0,
    // Internal: matrix stored as compressed bitvector
}
```

---

### HLLSET.RING.INGEST

Create HLLSet from token, decompose into ring, and store.

```
HLLSET.RING.INGEST <ring_key> <token> [SOURCE <source>] [TAGS <tag1,tag2,...>]
```

**Arguments:**
- `ring_key`: Ring to use for decomposition
- `token`: Token string to ingest
- `SOURCE`: Optional source identifier
- `TAGS`: Optional comma-separated tags

**Returns:** Array with decomposition result:
```
[sha1, is_new_base (0/1), expression_count, base1, base2, ...]
```

**Example:**
```redis
HLLSET.RING.INGEST session:ring1 "hello" SOURCE doc1
> ["abc123...", 1, 1, "abc123..."]  # New base

HLLSET.RING.INGEST session:ring1 "world" SOURCE doc1  
> ["def456...", 1, 1, "def456..."]  # New base

HLLSET.RING.INGEST session:ring1 "hello world" SOURCE doc1
> ["xyz789...", 0, 2, "abc123...", "def456..."]  # Compound: hello ⊕ world
```

**Internal Operations:**
1. Hash token → compute (reg, zeros) positions
2. Create HLLSet bitvector
3. Gaussian eliminate against current basis
4. If independent: add to basis, store base bytes
5. If dependent: store XOR expression only
6. Update metadata and refcounts

---

### HLLSET.RING.DECOMPOSE

Decompose an existing HLLSet into XOR of bases.

```
HLLSET.RING.DECOMPOSE <ring_key> <hllset_key> [SOURCE <source>] [TAGS <tag1,tag2,...>]
```

**Arguments:**
- `ring_key`: Ring to use for decomposition
- `hllset_key`: Key of existing HLLSet to decompose
- `SOURCE`: Optional source identifier
- `TAGS`: Optional tags

**Returns:** Same format as INGEST

**Example:**
```redis
# Assuming hllset:doc1 exists with positions from "hello world"
HLLSET.RING.DECOMPOSE session:ring1 hllset:doc1 SOURCE doc1
> ["xyz789...", 0, 2, "abc123...", "def456..."]
```

---

### HLLSET.RING.EXPRESS

Express an HLLSet (by SHA1) as XOR of basis elements.

```
HLLSET.RING.EXPRESS <ring_key> <sha1>
```

**Arguments:**
- `ring_key`: Ring to use
- `sha1`: SHA1 of HLLSet to express

**Returns:** Array of base SHA1s that XOR to this HLLSet

**Example:**
```redis
HLLSET.RING.EXPRESS session:ring1 xyz789...
> ["abc123...", "def456..."]  # xyz789 = abc123 ⊕ def456
```

---

### HLLSET.RING.BASIS

Get current basis SHA1s.

```
HLLSET.RING.BASIS <ring_key>
```

**Returns:** Array of base SHA1s

**Example:**
```redis
HLLSET.RING.BASIS session:ring1
> ["abc123...", "def456...", "ghi789..."]
```

---

### HLLSET.RING.RANK

Get current rank (number of independent bases).

```
HLLSET.RING.RANK <ring_key>
```

**Returns:** Integer

**Example:**
```redis
HLLSET.RING.RANK session:ring1
> 3
```

---

### HLLSET.RECONSTRUCT

Reconstruct HLLSet from XOR of bases (used by GET operations).

```
HLLSET.RECONSTRUCT <prefix> <sha1> [CACHE <ttl_seconds>]
```

**Arguments:**
- `prefix`: Key prefix for the store
- `sha1`: SHA1 to reconstruct
- `CACHE`: Optional TTL for caching reconstructed result

**Returns:** HLLSet bytes (or reference to cached key)

**Internal:**
1. Look up derivation: `hllring:lut:<sha1>`
2. If BASE: return `hllring:base:<sha1>`
3. If XOR: load all bases, XOR together, return result
4. If CACHE: store result temporarily

---

### HLLSET.W.COMMIT

Create W lattice commit (snapshot of ring state).

```
HLLSET.W.COMMIT <ring_key> [TIME <t>] [META <json>]
```

**Arguments:**
- `ring_key`: Ring to snapshot
- `TIME`: Optional time index (auto-increments if not provided)
- `META`: Optional JSON metadata

**Returns:** Time index of commit

**Example:**
```redis
HLLSET.W.COMMIT session:ring1
> 0

HLLSET.W.COMMIT session:ring1 META '{"event": "batch_complete"}'
> 1
```

---

### HLLSET.W.DIFF

Compute difference between two W commits.

```
HLLSET.W.DIFF <ring_key> <t1> <t2>
```

**Returns:** Array with [added_count, removed_count, shared_count, added..., removed..., shared...]

**Example:**
```redis
HLLSET.W.DIFF session:ring1 0 1
> [2, 0, 3, "new1...", "new2...", "shared1...", "shared2...", "shared3..."]
```

---

### HLLSET.EVICT

Evict bases according to policy.

```
HLLSET.EVICT <prefix> [POLICY lru|refcount|age] [KEEP <n>] [DRYRUN]
```

**Arguments:**
- `prefix`: Store prefix
- `POLICY`: Eviction policy
  - `refcount`: Evict bases with refcount=0
  - `age`: Evict oldest bases
  - `lru`: Evict least recently used (requires tracking)
- `KEEP`: Minimum number of bases to keep
- `DRYRUN`: Just return candidates without evicting

**Returns:** Array of evicted (or candidate) SHA1s

---

## Data Structures

### Ring State (In-Memory, inside Rust module)

```rust
struct RingState {
    ring_id: String,
    p_bits: u8,
    num_registers: usize,  // 2^p_bits
    
    // Basis matrix in reduced row echelon form
    // Shape: (rank, num_registers * 32)
    matrix: BitMatrix,
    
    // Pivot columns for each row
    pivots: Vec<usize>,
    
    // SHA1s of basis elements (in same order as rows)
    basis_sha1s: Vec<String>,
    
    // Timestamps
    created_at: f64,
    updated_at: f64,
}

struct BitMatrix {
    rows: Vec<BitVec>,  // Each row is a bitvector
    num_cols: usize,
}
```

### Gaussian Elimination (Core Algorithm)

```rust
impl RingState {
    /// Decompose an HLLSet into XOR of basis elements.
    /// Returns (expression, residual, is_new_base)
    fn decompose(&mut self, hllset: &HLLSet) -> DecomposeResult {
        let bitvec = hllset.to_bitvector();
        let mut residual = bitvec.clone();
        let mut expression = Vec::new();
        
        // Gaussian elimination
        for (row_idx, &pivot) in self.pivots.iter().enumerate() {
            if residual[pivot] == 1 {
                // XOR with this basis row
                residual ^= &self.matrix.rows[row_idx];
                expression.push(row_idx);
            }
        }
        
        if residual.any() {
            // New independent element
            let pivot = residual.first_one().unwrap();
            self.matrix.rows.push(residual);
            self.pivots.push(pivot);
            // Note: SHA1 added separately after storing
            
            DecomposeResult {
                is_new_base: true,
                expression: vec![],  // Will be filled with new SHA1
                residual: Some(residual),
            }
        } else {
            // Dependent - express as XOR of existing bases
            let sha1s: Vec<_> = expression.iter()
                .map(|&i| self.basis_sha1s[i].clone())
                .collect();
            
            DecomposeResult {
                is_new_base: false,
                expression: sha1s,
                residual: None,
            }
        }
    }
}
```

---

## Storage Layout

```
Redis Key                          Type      Description
─────────────────────────────────────────────────────────────────────
hllring:ring:<ring_id>             HASH      Ring state (persisted)
hllring:base:<sha1>                STRING    Base HLLSet bytes
hllring:lut:<sha1>                 HASH      Derivation {op, bases}
hllring:meta:<sha1>                HASH      Metadata
hllring:ref:<sha1>                 STRING    Reference count
hllring:W:<ring_id>:<t>            HASH      W commit at time t
hllring:idx                        INDEX     RediSearch index
```

---

## Integration with Existing Module

The HLLSET.RING commands will be added to the existing `hllset_rust` module:

```rust
// In lib.rs
redis_module! {
    name: "hllset",
    version: 1,
    data_types: [HLLSET_TYPE],
    commands: [
        // Existing commands
        ["HLLSET.CREATE", hllset_create, "write", 1, 1, 1],
        ["HLLSET.ADD", hllset_add, "write", 1, 1, 1],
        // ... other existing commands ...
        
        // New RING commands
        ["HLLSET.RING.INIT", ring_init, "write", 1, 1, 1],
        ["HLLSET.RING.INGEST", ring_ingest, "write", 1, 1, 1],
        ["HLLSET.RING.DECOMPOSE", ring_decompose, "write", 1, 2, 1],
        ["HLLSET.RING.EXPRESS", ring_express, "readonly", 1, 1, 1],
        ["HLLSET.RING.BASIS", ring_basis, "readonly", 1, 1, 1],
        ["HLLSET.RING.RANK", ring_rank, "readonly", 1, 1, 1],
        ["HLLSET.RECONSTRUCT", reconstruct, "readonly", 1, 1, 1],
        ["HLLSET.W.COMMIT", w_commit, "write", 1, 1, 1],
        ["HLLSET.W.DIFF", w_diff, "readonly", 1, 1, 1],
        ["HLLSET.EVICT", evict, "write", 1, 1, 1],
    ],
}
```

---

## Performance Considerations

1. **Gaussian Elimination Complexity**: O(rank × num_bits) per decomposition
   - For p=10: num_bits = 1024 × 32 = 32,768
   - Practical limit: rank < 10,000 before slowdown

2. **Memory for Ring State**: ~4KB per 1000 rank (compressed bitvectors)

3. **Batch Operations**: INGEST with multiple tokens can reuse matrix state

4. **Caching**: Reconstructed compounds cached with TTL to avoid repeated XOR

---

## Migration Path

1. **Phase 1 (Current)**: Python `HLLSetRingStore` with round-trips
2. **Phase 2**: Rust module with RING commands (this design)
3. **Phase 3**: Python wrapper calls Rust module (drop-in replacement)

The Python API remains the same:
```python
store = HLLSetRingStore(redis_client)
result = store.ingest(ring_id, token)  # Calls HLLSET.RING.INGEST internally
```
