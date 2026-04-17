# HLLSet Core Commands Reference

This document provides detailed documentation for all HLLSet core commands.

## Creation Commands

### HLLSET.CREATE

Creates an HLLSet from one or more string tokens.

**Syntax**:

```redis
HLLSET.CREATE token1 [token2 ...]
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| token1, token2, ... | String | Tokens to add to the HLLSet |

**Returns**: String - Content-addressable key (e.g., `hllset:7ac66c0f...`)

**Behavior**:

1. Tokens are sorted and deduplicated
2. SHA-1 hash computed from joined tokens
3. If key already exists, returns existing key (idempotent)
4. Each token is hashed (MurmurHash3) and registered in the bitmap

**Example**:

```redis
redis> HLLSET.CREATE apple banana cherry
"hllset:7ac66c0f148de9519b8bd264312c4d64f0c2d6b0"

# Same content, different order → same key
redis> HLLSET.CREATE cherry apple banana
"hllset:7ac66c0f148de9519b8bd264312c4d64f0c2d6b0"

# With duplicates → same key
redis> HLLSET.CREATE apple banana cherry apple
"hllset:7ac66c0f148de9519b8bd264312c4d64f0c2d6b0"
```

**Time Complexity**: O(n log n) where n = number of tokens (for sorting)

---

### HLLSET.CREATEHASH

Creates an HLLSet from pre-computed 64-bit hashes.

**Syntax**:

```redis
HLLSET.CREATEHASH hash1 [hash2 ...]
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| hash1, hash2, ... | Integer | 64-bit hash values |

**Returns**: String - Content-addressable key

**Use Cases**:

- Pre-hashed data from external systems
- Avoiding string hashing overhead
- Cross-system compatibility

**Example**:

```redis
redis> HLLSET.CREATEHASH 12345678901234567 98765432109876543
"hllset:a1b2c3d4e5f6789012345678901234567890abcd"
```

**Time Complexity**: O(n log n) where n = number of hashes

---

## Cardinality Commands

### HLLSET.CARD

Returns the estimated cardinality of an HLLSet.

**Syntax**:

```redis
HLLSET.CARD key
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key | String | HLLSet key |

**Returns**: Float - Estimated cardinality (0.0 if key doesn't exist)

**Algorithm**:
Uses Horvitz-Thompson estimator for bitmap registers:

```
For each bit position s:
  c_s = count of registers with bit s set
  f̂_s = -n * ln(1 - c_s/n)  (estimated frequency)
  
Total cardinality = Σ f̂_s
```

**Example**:

```redis
redis> HLLSET.CREATE a b c d e
"hllset:abc123..."

redis> HLLSET.CARD hllset:abc123
(float) 5.0

# Non-existent key
redis> HLLSET.CARD nonexistent
(float) 0.0
```

**Error Rate**: ~2-3% standard error for typical cardinalities

**Time Complexity**: O(M) where M = 1024 registers

---

## Set Operations

All set operations create a new key with the result.

### HLLSET.UNION

Computes the union of two HLLSets (A ∪ B).

**Syntax**:

```redis
HLLSET.UNION key1 key2
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key1 | String | First HLLSet key |
| key2 | String | Second HLLSet key |

**Returns**: String - Key of the result HLLSet (`hllset:union:...`)

**Operation**: Roaring bitmap OR - keeps ALL bits from both sets

**Example**:

```redis
redis> HLLSET.CREATE a b c
"hllset:key1..."

redis> HLLSET.CREATE c d e
"hllset:key2..."

redis> HLLSET.UNION hllset:key1 hllset:key2
"hllset:union:key1:key2"

redis> HLLSET.CARD hllset:union:key1:key2
(float) 5.0
```

**Time Complexity**: O(M)

---

### HLLSET.INTER

Computes the intersection of two HLLSets (A ∩ B).

**Syntax**:

```redis
HLLSET.INTER key1 key2
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key1 | String | First HLLSet key |
| key2 | String | Second HLLSet key |

**Returns**: String - Key of the result HLLSet (`hllset:inter:...`)

**Operation**: Roaring bitmap AND - keeps only bits present in BOTH sets

**Example**:

```redis
redis> HLLSET.CREATE a b c
"hllset:key1..."

redis> HLLSET.CREATE b c d
"hllset:key2..."

redis> HLLSET.INTER hllset:key1 hllset:key2
"hllset:inter:key1:key2"

redis> HLLSET.CARD hllset:inter:key1:key2
(float) 2.0
```

**Note**: Intersection cardinality is approximate due to probabilistic nature.

**Time Complexity**: O(M)

---

### HLLSET.DIFF

Computes the difference of two HLLSets (A \ B).

**Syntax**:

```redis
HLLSET.DIFF key1 key2
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key1 | String | First HLLSet key (minuend) |
| key2 | String | Second HLLSet key (subtrahend) |

**Returns**: String - Key of the result HLLSet (`hllset:diff:...`)

**Operation**: Roaring bitmap AND-NOT - bits in A that are NOT in B

**Example**:

```redis
redis> HLLSET.CREATE a b c d
"hllset:key1..."

redis> HLLSET.CREATE c d e
"hllset:key2..."

redis> HLLSET.DIFF hllset:key1 hllset:key2
"hllset:diff:key1:key2"

redis> HLLSET.CARD hllset:diff:key1:key2
(float) 2.0  # {a, b}
```

**Time Complexity**: O(M)

---

### HLLSET.XOR

Computes the symmetric difference of two HLLSets (A ⊕ B).

**Syntax**:

```redis
HLLSET.XOR key1 key2
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key1 | String | First HLLSet key |
| key2 | String | Second HLLSet key |

**Returns**: String - Key of the result HLLSet (`hllset:xor:...`)

**Operation**: Roaring bitmap XOR - bits in A or B but not both

**Example**:

```redis
redis> HLLSET.CREATE a b c
"hllset:key1..."

redis> HLLSET.CREATE b c d
"hllset:key2..."

redis> HLLSET.XOR hllset:key1 hllset:key2
"hllset:xor:key1:key2"

redis> HLLSET.CARD hllset:xor:key1:key2
(float) 2.0  # {a, d}
```

**Time Complexity**: O(M)

---

### HLLSET.MERGE

Merges multiple HLLSets into a destination key (in-place union).

**Syntax**:

```redis
HLLSET.MERGE destkey key1 [key2 ...]
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| destkey | String | Destination key (created if not exists) |
| key1, key2, ... | String | Source HLLSet keys |

**Returns**: Simple string "OK"

**Behavior**:

- Creates destkey if it doesn't exist
- Performs in-place OR with each source key
- More efficient than chained UNION for multiple sets

> ⚠️ **Note**: Unlike other set operations (`UNION`, `INTER`, `DIFF`, `XOR`) which are
> **immutable** and create content-addressable keys, `MERGE` is **mutable** and writes
> to a user-specified key. This intentionally breaks the content-addressable pattern
> to support accumulation use cases (similar to Redis's `PFMERGE`).

**Example**:

```redis
redis> HLLSET.CREATE a b
"hllset:key1"

redis> HLLSET.CREATE c d
"hllset:key2"

redis> HLLSET.CREATE e f
"hllset:key3"

redis> HLLSET.MERGE result hllset:key1 hllset:key2 hllset:key3
OK

redis> HLLSET.CARD result
(float) 6.0
```

**Time Complexity**: O(M * k) where k = number of source keys

---

### HLLSET.FINALIZE

Computes content hash and copies HLLSet to canonical `hllset:<sha1>` key.

**Syntax**:

```redis
HLLSET.FINALIZE source_key [DELETE]
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| source_key | String | Source HLLSet key (typically a mutable accumulator) |
| DELETE | Flag | Optional - delete source key after finalization |

**Returns**: String - The new content-addressable key (`hllset:<sha1>`)

**Use Cases**:

- Convert mutable MERGE results to immutable content-addressable keys
- Deduplicate HLLSets by their content
- Verify content integrity after accumulation

**Example**:

```redis
# Build incrementally with MERGE
redis> HLLSET.CREATE a b c
"hllset:abc123..."

redis> HLLSET.MERGE accumulator hllset:abc123...
OK

redis> HLLSET.CREATE d e f
"hllset:def456..."

redis> HLLSET.MERGE accumulator hllset:def456...
OK

# Finalize to content-addressable key
redis> HLLSET.FINALIZE accumulator
"hllset:7a2b3c4d5e6f..."

# Or finalize and cleanup
redis> HLLSET.FINALIZE accumulator DELETE
"hllset:7a2b3c4d5e6f..."

redis> EXISTS accumulator
(integer) 0
```

**Idempotence**: If source is already at its canonical key (`hllset:<sha1>`),
returns that key without modification.

**Time Complexity**: O(M) for SHA1 computation + O(M) for copy

---

## Similarity Commands

### HLLSET.SIM / HLLSET.JACCARD

Computes Jaccard similarity between two HLLSets.

**Syntax**:

```redis
HLLSET.SIM key1 key2
HLLSET.JACCARD key1 key2  # Alias
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key1 | String | First HLLSet key |
| key2 | String | Second HLLSet key |

**Returns**: Float - Jaccard similarity coefficient [0.0, 1.0]

**Formula**:

```text
J(A, B) = |A ∩ B| / |A ∪ B|
```

**Example**:

```redis
redis> HLLSET.CREATE a b c
"hllset:key1"

redis> HLLSET.CREATE b c d
"hllset:key2"

redis> HLLSET.SIM hllset:key1 hllset:key2
(float) 0.5  # 2 shared / 4 total

# Identical sets
redis> HLLSET.SIM hllset:key1 hllset:key1
(float) 1.0

# Disjoint sets (approximately)
redis> HLLSET.CREATE x y z
"hllset:key3"

redis> HLLSET.SIM hllset:key1 hllset:key3
(float) 0.0
```

**Time Complexity**: O(M)

---

## Information Commands

### HLLSET.INFO

Returns metadata about an HLLSet.

**Syntax**:

```redis
HLLSET.INFO key
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key | String | HLLSet key |

**Returns**: Array of key-value pairs

| Field | Type | Description |
| ----- | ---- | ----------- |
| key | String | The key name |
| cardinality | Float | Estimated cardinality |
| registers | Integer | Total registers (1024) |
| non_zero_registers | Integer | Registers with at least one bit set |
| precision_bits | Integer | Precision (10) |
| memory_bytes | Integer | Serialized size in bytes |

**Example**:

```redis
redis> HLLSET.CREATE apple banana cherry date
"hllset:abc123"

redis> HLLSET.INFO hllset:abc123
 1) "key"
 2) "hllset:abc123"
 3) "cardinality"
 4) (float) 4.0
 5) "registers"
 6) (integer) 1024
 7) "non_zero_registers"
 8) (integer) 4
 9) "precision_bits"
10) (integer) 10
11) "memory_bytes"
12) (integer) 42
```

**Time Complexity**: O(M)

---

### HLLSET.DUMP

Dumps all non-zero register positions for debugging.

**Syntax**:

```redis
HLLSET.DUMP key
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key | String | HLLSet key |

**Returns**: Array of [bucket, highest_bit] pairs

**Example**:

```redis
redis> HLLSET.DUMP hllset:abc123
1) 1) (integer) 42
   2) (integer) 3
2) 1) (integer) 100
   2) (integer) 5
3) 1) (integer) 512
   2) (integer) 2
```

**Time Complexity**: O(M)

---

## Management Commands

### HLLSET.EXISTS

Checks if an HLLSet key exists.

**Syntax**:

```redis
HLLSET.EXISTS key
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key | String | HLLSet key |

**Returns**: Integer - 1 if exists, 0 otherwise

**Example**:

```redis
redis> HLLSET.EXISTS hllset:abc123
(integer) 1

redis> HLLSET.EXISTS nonexistent
(integer) 0
```

**Time Complexity**: O(1)

---

### HLLSET.DEL

Deletes an HLLSet.

**Syntax**:

```redis
HLLSET.DEL key
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key | String | HLLSet key to delete |

**Returns**: Integer - 1 if deleted, 0 if key didn't exist

**Example**:

```redis
redis> HLLSET.DEL hllset:abc123
(integer) 1

redis> HLLSET.DEL nonexistent
(integer) 0
```

**Time Complexity**: O(1)

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
| ----- | ----- | -------- |
| `ERR wrong number of arguments` | Missing required parameters | Check command syntax |
| `ERR Key does not exist` | Key not found (for INFO, DUMP) | Verify key exists |
| `WRONGTYPE` | Key exists but wrong type | Use correct key |
| `ERR No valid hashes provided` | CREATEHASH with unparseable values | Provide valid integers |

### Best Practices

1. **Use content-addressable keys**: Let CREATE generate keys for deduplication
2. **Check existence**: Use EXISTS before INFO/DUMP to avoid errors
3. **Handle empty results**: CARD returns 0.0 for non-existent keys
4. **Clean up derived keys**: Set operations create new keys that may need cleanup
