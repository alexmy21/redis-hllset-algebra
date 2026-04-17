# Ring Algebra and XOR Decomposition

This document covers the XOR ring algebra commands for decomposing HLLSets into linearly independent basis elements.

## Concept Overview

### What is XOR Ring Algebra?

XOR Ring Algebra treats HLLSets as vectors over GF(2) (binary field). Any HLLSet can be expressed as:

```text
H = B₁ ⊕ B₂ ⊕ ... ⊕ Bₖ
```

Where B₁...Bₖ are linearly independent **basis** HLLSets.

### Key Benefits

1. **Compression**: Store only k basis elements instead of n total HLLSets
2. **Deduplication**: Identical HLLSets decompose to same expression
3. **Provenance**: Track which bases contribute to any HLLSet
4. **Lattice Operations**: W commits enable temporal tracking

### How It Works

```text
┌─────────────────────────────────────────────────────────────┐
│                 Gaussian Elimination over GF(2)             │
│                                                             │
│  Input: HLLSet H (as bitvector of M*32 bits)                │
│                                                             │
│  Process:                                                   │
│  1. Reduce H against current basis rows                     │
│  2. For each basis row with matching pivot:                 │
│     H := H ⊕ BasisRow                                      │
│     Record basis_sha1 in expression                         │
│                                                             │
│  Result:                                                    │
│  - If residual ≠ 0: H is NEW BASE (add to matrix)           │
│  - If residual = 0: H = ⊕{used bases}                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Ring Commands

### HLLSET.RING.INIT

Initializes a new decomposition ring.

**Syntax**:

```redis
HLLSET.RING.INIT ring_key [PBITS p]
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| ring_key | String | Unique identifier for the ring |
| PBITS | Integer | Optional: Precision bits (default: 10) |

**Returns**: Simple string "OK"

**Stored Data** (Redis Hash at `hllring:ring:<ring_key>`):

| Field | Description |
| ----- | ----------- |
| ring_id | Ring identifier |
| p_bits | Precision bits |
| rank | Current rank (0 initially) |
| basis_sha1s | JSON array of basis SHA1s |
| created_at | Unix timestamp |
| updated_at | Unix timestamp |
| matrix_data | Base64-encoded bit matrix |
| pivots | JSON array of pivot columns |

**Example**:

```redis
redis> HLLSET.RING.INIT myring
OK

redis> HLLSET.RING.INIT myring12 PBITS 12
OK
```

**Time Complexity**: O(1)

---

### HLLSET.RING.INGEST

Ingests a token, creates an HLLSet, and decomposes it.

**Syntax**:

```redis
HLLSET.RING.INGEST ring_key token [SOURCE source] [TAGS tag1,tag2,...]
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| ring_key | String | Ring identifier |
| token | String | Token to ingest |
| SOURCE | String | Optional: Source identifier |
| TAGS | String | Optional: Comma-separated tags |

**Returns**: Array

| Index | Value |
| ----- | ----- |
| 0 | SHA1 of the HLLSet |
| 1 | is_new_base (1 or 0) |
| 2 | num_bases in expression |
| 3+ | Base SHA1s |

**Storage**:

1. If new base: Stored at `hllring:base:<sha1>`
2. Derivation: `hllring:lut:<sha1>` → {op, bases}
3. Metadata: `hllring:meta:<sha1>` → {sha1, source, cardinality, is_base, created_at, tags}

**Example**:

```redis
# First token - becomes new base
redis> HLLSET.RING.INGEST myring "hello" SOURCE doc1 TAGS corpus,english
1) "hllset:abc123def456..."
2) (integer) 1
3) (integer) 1
4) "hllset:abc123def456..."

# Similar token - might express as XOR of bases
redis> HLLSET.RING.INGEST myring "hello world" SOURCE doc2
1) "hllset:789xyz..."
2) (integer) 0           # Not a new base
3) (integer) 2           # Expressed as XOR of 2 bases
4) "hllset:abc123..."    # Base 1
5) "hllset:def456..."    # Base 2
```

**Time Complexity**: O(rank² × M) for Gaussian elimination

---

### HLLSET.RING.DECOMPOSE

Decomposes an existing HLLSet into the ring.

**Syntax**:

```redis
HLLSET.RING.DECOMPOSE ring_key hllset_key [SOURCE source] [TAGS tags]
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| ring_key | String | Ring identifier |
| hllset_key | String | Existing HLLSet key |
| SOURCE | String | Optional: Source identifier |
| TAGS | String | Optional: Comma-separated tags |

**Returns**: Same format as RING.INGEST

**Example**:

```redis
# Decompose an existing HLLSet
redis> HLLSET.RING.DECOMPOSE myring hllset:existing123 SOURCE import
1) "hllset:existing123"
2) (integer) 1
3) (integer) 1
4) "hllset:existing123"
```

**Time Complexity**: O(rank² × M)

---

### HLLSET.RING.BASIS

Returns the SHA1s of all basis elements in the ring.

**Syntax**:

```redis
HLLSET.RING.BASIS ring_key
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| ring_key | String | Ring identifier |

**Returns**: Array of SHA1 strings

**Example**:

```redis
redis> HLLSET.RING.BASIS myring
1) "hllset:abc123..."
2) "hllset:def456..."
3) "hllset:789xyz..."
```

**Time Complexity**: O(rank)

---

### HLLSET.RING.RANK

Returns the current rank (number of linearly independent bases).

**Syntax**:

```redis
HLLSET.RING.RANK ring_key
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| ring_key | String | Ring identifier |

**Returns**: Integer - Current rank

**Example**:

```redis
redis> HLLSET.RING.RANK myring
(integer) 42
```

**Maximum Rank**: M × 32 = 32,768 (for P=10)

**Time Complexity**: O(1)

---

## W Lattice Commands

W commits capture snapshots of ring state for temporal tracking.

### HLLSET.W.COMMIT

Creates a W lattice commit (snapshot of current ring state).

**Syntax**:

```redis
HLLSET.W.COMMIT ring_key [META json]
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| ring_key | String | Ring identifier |
| META | String | Optional: JSON metadata |

**Returns**: Integer - Time index of the commit

**Stored Data** (Redis Hash at `hllring:W:<ring_key>:<time_index>`):

| Field | Description |
| ----- | ----------- |
| time_index | Commit sequence number |
| ring_id | Parent ring ID |
| basis_sha1s | JSON array of basis SHA1s at commit time |
| rank | Rank at commit time |
| timestamp | Unix timestamp |
| metadata | Optional JSON metadata |

**Example**:

```redis
redis> HLLSET.W.COMMIT myring META '{"version":"1.0","author":"alice"}'
(integer) 0

redis> HLLSET.W.COMMIT myring
(integer) 1

redis> HLLSET.W.COMMIT myring META '{"milestone":"release"}'
(integer) 2
```

**Time Complexity**: O(rank) for copying basis list

---

### HLLSET.W.DIFF

Computes the difference between two W commits.

**Syntax**:

```redis
HLLSET.W.DIFF ring_key t1 t2
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| ring_key | String | Ring identifier |
| t1 | Integer | First time index |
| t2 | Integer | Second time index |

**Returns**: Array

| Index | Value |
| ----- | ----- |
| 0 | added_count |
| 1 | removed_count |
| 2 | shared_count |
| 3 | delta_rank |
| 4..4+added | Added SHA1s |
| ...+removed | Removed SHA1s |
| ...+shared | Shared SHA1s |

**Example**:

```redis
redis> HLLSET.W.DIFF myring 0 2
1) (integer) 5      # 5 bases added
2) (integer) 1      # 1 base removed
3) (integer) 10     # 10 bases shared
4) (integer) 4      # delta_rank = 5 - 1 = 4
5) "hllset:new1..."
6) "hllset:new2..."
7) "hllset:new3..."
8) "hllset:new4..."
9) "hllset:new5..."
10) "hllset:removed1..."
11) "hllset:shared1..."
... (10 more shared)
```

**Time Complexity**: O(rank₁ + rank₂)

---

### HLLSET.RECONSTRUCT

Reconstructs an HLLSet from its XOR expression.

**Syntax**:

```redis
HLLSET.RECONSTRUCT sha1
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| sha1 | String | SHA1 of the HLLSet to reconstruct |

**Returns**: String - Key of reconstructed HLLSet (temporary, 60s TTL)

**Process**:

1. Look up derivation at `hllring:lut:<sha1>`
2. If base: Return directly from `hllring:base:<sha1>`
3. If XOR: Load all bases, compute XOR, store at `hllring:temp:<sha1>`

**Example**:

```redis
# Reconstruct a derived HLLSet
redis> HLLSET.RECONSTRUCT hllset:abc123
"hllring:temp:abc123"

# Use the reconstructed HLLSet
redis> HLLSET.CARD hllring:temp:abc123
(float) 42.0

# Note: temp key expires in 60 seconds
```

**Time Complexity**: O(k × M) where k = number of bases in expression

---

## Data Model

### Key Hierarchy

```text
hllring:ring:<ring_id>          # Ring state (Hash)
hllring:base:<sha1>             # Basis HLLSets (HLLSet type)
hllring:lut:<sha1>              # Derivation lookup (Hash)
hllring:meta:<sha1>             # Metadata (Hash)
hllring:W:<ring_id>:<t>         # W commits (Hash)
hllring:temp:<sha1>             # Temporary reconstructions (HLLSet)
```

### Derivation Lookup Schema

```redis
HGETALL hllring:lut:abc123
1) "op"
2) "xor"          # or "base"
3) "bases"
4) "[\"sha1_1\",\"sha1_2\",\"sha1_3\"]"
```

### Metadata Schema

```redis
HGETALL hllring:meta:abc123
 1) "sha1"
 2) "abc123..."
 3) "source"
 4) "document_corpus"
 5) "cardinality"
 6) "42.5"
 7) "is_base"
 8) "0"
 9) "created_at"
10) "1713258000.123"
11) "tags"
12) "[\"corpus\",\"v1\"]"
```

---

## Workflow Examples

### Basic Ingestion

```redis
# Initialize ring
HLLSET.RING.INIT corpus_ring

# Ingest documents
HLLSET.RING.INGEST corpus_ring "document one content" SOURCE doc1
HLLSET.RING.INGEST corpus_ring "document two content" SOURCE doc2
HLLSET.RING.INGEST corpus_ring "document three" SOURCE doc3

# Check rank
HLLSET.RING.RANK corpus_ring
# (integer) 3  -- if all linearly independent
```

### Temporal Tracking

```redis
# Initialize and ingest batch 1
HLLSET.RING.INIT timeline_ring
HLLSET.RING.INGEST timeline_ring "batch1_item1"
HLLSET.RING.INGEST timeline_ring "batch1_item2"
HLLSET.W.COMMIT timeline_ring META '{"batch":1}'

# Ingest batch 2
HLLSET.RING.INGEST timeline_ring "batch2_item1"
HLLSET.RING.INGEST timeline_ring "batch2_item2"
HLLSET.W.COMMIT timeline_ring META '{"batch":2}'

# Compare batches
HLLSET.W.DIFF timeline_ring 0 1
```

### Reconstruction Pipeline

```redis
# Original ingestion
HLLSET.RING.INGEST myring "important document"
# Returns sha1 and expression

# Later: reconstruct for processing
HLLSET.RECONSTRUCT hllset:important123

# Process the reconstructed HLLSet
HLLSET.CARD hllring:temp:important123
HLLSET.POSITIONS hllring:temp:important123
```

---

## Performance Characteristics

### Space Complexity

- Basis storage: O(rank × M) bits
- LUT entry: O(k) where k = bases in expression
- W commit: O(rank) for basis list copy

### Time Complexity

| Operation | Complexity |
| --------- | ---------- |
| RING.INIT | O(1) |
| RING.INGEST | O(rank² × M) |
| RING.DECOMPOSE | O(rank² × M) |
| RING.BASIS | O(rank) |
| RING.RANK | O(1) |
| W.COMMIT | O(rank) |
| W.DIFF | O(rank₁ + rank₂) |
| RECONSTRUCT | O(k × M) |

### Scalability

- **Maximum rank**: M × 32 = 32,768 (theoretical)
- **Practical rank**: Typically stabilizes at 100-1000
- **Memory per ring**: ~rank × 4KB (for P=10)

---

## Best Practices

### Ring Management

1. **Use separate rings** for different corpora/domains
2. **Commit regularly** to enable temporal queries
3. **Monitor rank growth** - plateauing indicates saturation

### Performance Tips

1. **Batch ingestion** during low-traffic periods
2. **Use DECOMPOSE** for pre-existing HLLSets
3. **Cache reconstructions** if accessed frequently

### Metadata Usage

1. **Always set SOURCE** for traceability
2. **Use TAGS** for filtering/grouping
3. **Store version in W.COMMIT META**
