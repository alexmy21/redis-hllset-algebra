# Data Structures Reference

This document describes the core data structures used by the Redis HLLSet module and how they can be customized.

## HLLSet Structure

### Tensor Model

HLLSet represents a **3D probabilistic tensor**:

```text
                     Trailing Zeros (0-31)
                     ─────────────────────────────────►
                   0   1   2   3  ...  30  31
                 ┌───┬───┬───┬───┬───┬───┬───┐
Register 0       │ 0 │ 1 │ 0 │ 1 │...│ 0 │ 0 │
                 ├───┼───┼───┼───┼───┼───┼───┤
Register 1       │ 1 │ 0 │ 1 │ 0 │...│ 0 │ 0 │
                 ├───┼───┼───┼───┼───┼───┼───┤
    ...          │   │   │   │   │...│   │   │
                 ├───┼───┼───┼───┼───┼───┼───┤
Register 1023    │ 0 │ 0 │ 1 │ 0 │...│ 0 │ 0 │
                 └───┴───┴───┴───┴───┴───┴───┘
                 │
                 ▼
             Registers
             (0-1023)
```

### Internal Storage

**Roaring Bitmap Encoding**:

```rust
// Bit position = (register_index * 32) + trailing_zeros
// Example: Register 5, trailing zeros 3 → position 163

position = register * BITS_PER_REG + trailing_zeros
         = 5 * 32 + 3
         = 163
```

**Memory Layout**:

```text
┌────────────────────────────────────────────────────────────┐
│                    Roaring Bitmap                          │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Container 0: [positions 0-65535]                     │  │
│  │   - Array/Bitmap depending on cardinality            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  Total possible positions: 1024 * 32 = 32,768              │
│  Typical cardinality: 100-5000 bits                        │
│  Compression ratio: 10-100x vs dense                       │
└────────────────────────────────────────────────────────────┘
```

### Configuration Constants

| Constant | Value | Description | Modifiable |
| ---------- | ------- | ------------- | ------------ |
| `P` | 10 | Precision bits | Compile-time only |
| `M` | 1024 | Number of registers (2^P) | Derived from P |
| `BITS_PER_REG` | 32 | Bits per register | Fixed (64-bit hash) |
| `TOTAL_BITS` | 32,768 | Total tensor size | Derived |
| `ALPHA_M` | 0.7213... | Bias correction | Fixed for M=1024 |

**Modifying Precision (P)**:

To change precision, you must recompile the module:

```rust
// In hllset.rs
pub const P: u32 = 10;  // Change to 12 for 4096 registers
pub const M: usize = 1 << P;  // Automatically updates
```

**Trade-offs**:

| P | Registers | Memory | Error Rate |
| --- | ----------- | -------- | ------------ |
| 8 | 256 | ~0.5KB | ~6.5% |
| 10 | 1024 | ~2KB | ~3.25% |
| 12 | 4096 | ~8KB | ~1.6% |
| 14 | 16384 | ~32KB | ~0.8% |

### Dense Format

For cardinality estimation, HLLSet can be "inflated" to dense format:

```rust
// Dense: Vec<u32> with M elements
// Each u32 is a bitmap of observed trailing zeros
let dense: Vec<u32> = hllset.to_dense();

// Example register value: 0b0000_0101 = bits 0 and 2 are set
// Means trailing zeros of 0 and 2 were observed for this register
```

---

## Content-Addressable Keys

### Key Generation Algorithm

```
Input: tokens = ["banana", "apple", "cherry", "apple"]

Step 1: Deduplicate
        → ["banana", "apple", "cherry"]

Step 2: Sort
        → ["apple", "banana", "cherry"]

Step 3: Join with null byte
        → "apple\0banana\0cherry"

Step 4: SHA-1 hash
        → 7ac66c0f148de9519b8bd264312c4d64f0c2d6b0

Step 5: Format key
        → "hllset:7ac66c0f148de9519b8bd264312c4d64f0c2d6b0"
```

### Key Prefixes

| Prefix | Purpose | Example |
|--------|---------|---------|
| `hllset:` | Content-addressable HLLSet | `hllset:7ac66c0f...` |
| `hllset:union:` | Union result | `hllset:union:7ac66c:1a2b3c` |
| `hllset:inter:` | Intersection result | `hllset:inter:7ac66c:1a2b3c` |
| `hllset:diff:` | Difference result | `hllset:diff:7ac66c:1a2b3c` |
| `hllset:xor:` | Symmetric diff result | `hllset:xor:7ac66c:1a2b3c` |

**Customizing Key Prefixes**:

Key prefixes are hardcoded but can be modified in `commands.rs`:

```rust
// In extract_key_hash() and set operation functions
let key_str = format!("hllset:union:{}:{}", k1_short, k2_short);
// Change "hllset" to your preferred prefix
```

---

## TokenLUT Entry Structure

TokenLUT entries are Redis Hashes with the following schema:

### Hash Fields

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `reg` | Integer | Register index [0-1023] | Yes |
| `zeros` | Integer | Trailing zeros count [0-31] | Yes |
| `hash_full` | String | Full 64-bit hash value | Yes |
| `layer` | Integer | N-gram layer (0=unigram, 1=bigram, ...) | Yes |
| `first_tokens` | JSON Array | Array of first tokens (for collision) | Yes |
| `tokens` | JSON Array | Array of token arrays (n-grams) | For bigrams+ |
| `first_tokens_tag` | String | Comma-joined first tokens (for search) | Yes |
| `collision_count` | Integer | Number of distinct first_tokens | Yes |
| `tf` | Integer | Term frequency (accumulated) | Yes |

### Example Entry

**Unigram**:
```json
{
  "reg": "42",
  "zeros": "3",
  "hash_full": "12345678901234567",
  "layer": "0",
  "first_tokens": "[\"hello\"]",
  "tokens": "[]",
  "first_tokens_tag": "hello",
  "collision_count": "1",
  "tf": "15"
}
```

**Bigram**:
```json
{
  "reg": "100",
  "zeros": "5",
  "hash_full": "98765432109876543",
  "layer": "1",
  "first_tokens": "[\"quick\"]",
  "tokens": "[[\"quick\", \"brown\"]]",
  "first_tokens_tag": "quick",
  "collision_count": "1",
  "tf": "8"
}
```

**With Collision**:
```json
{
  "reg": "200",
  "zeros": "7",
  "hash_full": "55555555555555555",
  "layer": "0",
  "first_tokens": "[\"bank\", \"bank\"]",
  "tokens": "[]",
  "first_tokens_tag": "bank,bank",
  "collision_count": "2",
  "tf": "42"
}
```

### Key Naming Convention

```
<prefix><hash_full>

Examples:
- tokenlut:entry:session123:12345678901234567
- tokenlut:entry:corpus_v1:98765432109876543
```

---

## Ring State Structure

The Ring maintains Gaussian elimination state for XOR decomposition.

### Redis Hash Schema

| Field | Type | Description |
|-------|------|-------------|
| `ring_id` | String | Ring identifier |
| `p_bits` | Integer | Precision bits (default: 10) |
| `rank` | Integer | Current rank (number of bases) |
| `basis_sha1s` | JSON Array | SHA1s of basis HLLSets |
| `created_at` | Float | Unix timestamp |
| `updated_at` | Float | Unix timestamp |
| `matrix_data` | Base64 | Serialized bit matrix |
| `pivots` | JSON Array | Pivot columns for each row |

### Key Prefixes

| Prefix | Purpose |
|--------|---------|
| `hllring:ring:` | Ring state hash |
| `hllring:base:` | Basis HLLSet storage |
| `hllring:lut:` | Derivation lookup table |
| `hllring:meta:` | Metadata for each SHA1 |
| `hllring:W:` | W lattice commits |

### Derivation LUT Entry

Stored as Redis Hash:

| Field | Value |
|-------|-------|
| `op` | `"base"` or `"xor"` |
| `bases` | JSON array of SHA1s (e.g., `["sha1_1", "sha1_2"]`) |

---

## W Commit Structure

W commits capture ring state snapshots for lattice operations.

### Hash Fields

| Field | Type | Description |
|-------|------|-------------|
| `time_index` | Integer | Commit sequence number |
| `ring_id` | String | Parent ring ID |
| `basis_sha1s` | JSON Array | Basis SHA1s at commit time |
| `rank` | Integer | Rank at commit time |
| `timestamp` | Float | Unix timestamp |
| `metadata` | JSON | Optional user metadata |

### W Diff Result

Returned as array:
```
[added_count, removed_count, shared_count, delta_rank, 
 added_sha1_1, ..., removed_sha1_1, ..., shared_sha1_1, ...]
```

---

## Position Index Structure

Created by `HLLSET.POSINDEX` as Redis Sorted Set:

```
Key: <index_key>
Type: Sorted Set

Members:
  - "reg:zeros" format (e.g., "42:3")
  
Scores:
  - Linearized position: reg * 32 + zeros
  
Example:
  ZADD myindex 163 "5:3"   # Register 5, zeros 3
  ZADD myindex 200 "6:8"   # Register 6, zeros 8
```

**Use Cases**:
- Range queries by position
- Finding neighboring positions
- Efficient position-based lookups

---

## Serialization Formats

### RDB Persistence

HLLSets are persisted using Roaring Bitmap's native serialization:

```rust
// Serialize
let bytes = hllset.to_bytes();  // Roaring format

// Deserialize  
let hllset = HLLSet::from_bytes(&bytes);
```

### Network Transfer

For bulk operations or export:

```rust
// Get raw bytes (Roaring format)
let bytes = hllset.to_bytes();

// Or get dense format for compatibility
let dense: Vec<u32> = hllset.to_dense();
```

---

## Extending Data Structures

### Adding Custom Fields to TokenLUT

1. Modify `tokenlut_add()` in `tokenlut.rs`
2. Add field to HSET call
3. Update `tokenlut_get()` if needed

```rust
// Example: Add "source" field
ctx.call("HSET", &[
    key.as_str(),
    // ... existing fields ...
    "source", source_str.as_str(),  // New field
])?;
```

### Adding New Key Prefixes

1. Define constant in relevant module
2. Update key generation functions
3. Document in this file

```rust
const KEY_CUSTOM: &str = "hllring:custom:";
```

### Custom Cardinality Estimators

The module uses Horvitz-Thompson estimator. To add alternatives:

1. Add new method to `HLLSet` struct in `hllset.rs`
2. Optionally expose via new command

```rust
impl HLLSet {
    pub fn cardinality_custom(&self) -> f64 {
        // Your custom estimator
    }
}
```
