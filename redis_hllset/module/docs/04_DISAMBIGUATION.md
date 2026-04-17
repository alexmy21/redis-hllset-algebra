# Tensor and Disambiguation Commands

This document covers commands for accessing HLLSet tensor positions and performing disambiguation against TokenLUT entries.

## Tensor Position Commands

These commands expose the internal 2D tensor structure of HLLSets for advanced operations like disambiguation.

### HLLSET.POSITIONS

Returns all active (register, trailing_zeros) positions as a flat array.

**Syntax**:

```redis
HLLSET.POSITIONS key
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key | String | HLLSet key |

**Returns**: Array - Flat array of [reg1, zeros1, reg2, zeros2, ...]

**Use Cases**:

- Token disambiguation lookup
- Finding potential token candidates
- Sparse tensor operations

**Example**:

```redis
redis> HLLSET.CREATE hello world
"hllset:abc123"

redis> HLLSET.POSITIONS hllset:abc123
1) (integer) 42
2) (integer) 3
3) (integer) 789
4) (integer) 7

# Interpretation:
# Position 1: register=42, trailing_zeros=3
# Position 2: register=789, trailing_zeros=7
```

**Time Complexity**: O(popcount) where popcount = number of set bits

---

### HLLSET.POPCOUNT

Returns the total number of set bits in the bitmap.

**Syntax**:

```redis
HLLSET.POPCOUNT key
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key | String | HLLSet key |

**Returns**: Integer - Total count of set bits

**Example**:

```redis
redis> HLLSET.POPCOUNT hllset:abc123
(integer) 42

# Empty set
redis> HLLSET.POPCOUNT nonexistent
(integer) 0
```

**Time Complexity**: O(1) - Roaring bitmap maintains popcount

---

### HLLSET.BITCOUNTS

Returns the count of registers with each bit position set (c_s values).

**Syntax**:

```redis
HLLSET.BITCOUNTS key
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key | String | HLLSet key |

**Returns**: Array - 32 integers, one for each bit position [0-31]

**Purpose**: Used for Horvitz-Thompson cardinality estimation.

**Example**:

```redis
redis> HLLSET.BITCOUNTS hllset:abc123
 1) (integer) 100   # c_0: registers with bit 0 set
 2) (integer) 85    # c_1: registers with bit 1 set
 3) (integer) 50    # c_2
 4) (integer) 30    # c_3
 ...
32) (integer) 0     # c_31
```

**Interpretation**: Each c_s is the count of registers where trailing zeros = s was observed.

**Time Complexity**: O(popcount)

---

### HLLSET.REGISTER

Returns the 32-bit bitmap value for a specific register.

**Syntax**:

```redis
HLLSET.REGISTER key reg
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key | String | HLLSet key |
| reg | Integer | Register index [0-1023] |

**Returns**: Integer - 32-bit bitmap value

**Example**:

```redis
redis> HLLSET.REGISTER hllset:abc123 42
(integer) 13  # Binary: 0b1101 = bits 0, 2, 3 are set

# Out of range
redis> HLLSET.REGISTER hllset:abc123 9999
ERR register index out of range (0-1023)
```

**Time Complexity**: O(BITS_PER_REG) = O(32)

---

### HLLSET.HASBIT

Checks if a specific (register, trailing_zeros) position is set.

**Syntax**:

```redis
HLLSET.HASBIT key reg zeros
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key | String | HLLSet key |
| reg | Integer | Register index [0-1023] |
| zeros | Integer | Trailing zeros count [0-31] |

**Returns**: Integer - 1 if bit is set, 0 otherwise

**Example**:

```redis
redis> HLLSET.HASBIT hllset:abc123 42 3
(integer) 1  # Bit is set

redis> HLLSET.HASBIT hllset:abc123 42 5
(integer) 0  # Bit is not set

# Range check
redis> HLLSET.HASBIT hllset:abc123 42 99
ERR position out of range
```

**Time Complexity**: O(1)

---

## Disambiguation Commands

These commands enable matching HLLSet positions against TokenLUT entries.

### HLLSET.CANDIDATES

Finds TokenLUT entries matching positions in an HLLSet.

**Syntax**:

```redis
HLLSET.CANDIDATES hllset_key lut_prefix [STREAM stream_key] [LAYER n] [LIMIT n]
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| hllset_key | String | HLLSet to get positions from |
| lut_prefix | String | Prefix for TokenLUT keys |
| STREAM | String | Optional: Stream key to write results |
| LAYER | Integer | Optional: Filter by layer (0=unigram, 1=bigram) |
| LIMIT | Integer | Optional: Maximum entries to return |

**Returns**:

- Without STREAM: Array of [key, token, layer, first_token, ...]
- With STREAM: Integer count of entries streamed

**Example**:

```redis
# Direct return
redis> HLLSET.CANDIDATES hllset:query tokenlut:entry:sess1: LAYER 0 LIMIT 10
1) "tokenlut:entry:sess1:12345"
2) "hello"
3) (integer) 0
4) "hello"
5) "tokenlut:entry:sess1:67890"
6) "world"
7) (integer) 0
8) "world"

# Streaming output
redis> HLLSET.CANDIDATES hllset:query tokenlut:entry:sess1: STREAM candidates:out
(integer) 42  # Number of matches streamed
```

**Streaming Output Schema**:

```redis
redis> XREAD STREAMS candidates:out 0
1) 1) "candidates:out"
   2) 1) 1) "1234567890123-0"
         2) 1) "key"
            2) "tokenlut:entry:sess1:12345"
            3) "reg"
            4) "42"
            5) "zeros"
            6) "3"
            7) "layer"
            8) "0"
            9) "token"
           10) "hello"
           11) "first_token"
           12) "hello"
```

**Time Complexity**: O(n) where n = number of LUT entries scanned

---

### HLLSET.SCANMATCH

Full cursor-based scan with streaming output - more efficient for large datasets.

**Syntax**:

```redis
HLLSET.SCANMATCH hllset_key lut_prefix stream_key [LAYER n] [BATCH n]
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| hllset_key | String | HLLSet to get positions from |
| lut_prefix | String | Prefix for TokenLUT keys |
| stream_key | String | Stream key for output (required) |
| LAYER | Integer | Optional: Filter by layer |
| BATCH | Integer | Optional: SCAN count per iteration (default: 1000) |

**Returns**: Integer - Total count of matched entries

**Behavior**:

1. Gets all positions from HLLSet
2. Iterates through all keys matching lut_prefix using SCAN
3. For each match, streams result immediately
4. Continues until SCAN cursor returns 0

**Example**:

```redis
redis> HLLSET.SCANMATCH hllset:query tokenlut:entry: results:stream BATCH 500
(integer) 1234  # Total matches

# Read results from stream
redis> XLEN results:stream
(integer) 1234
```

**Use Cases**:

- Large-scale disambiguation
- Background processing
- Streaming to consumers

**Time Complexity**: O(N) where N = total keys matching prefix

---

### HLLSET.POSINDEX

Creates a sorted set index of positions for fast range queries.

**Syntax**:

```redis
HLLSET.POSINDEX hllset_key index_key
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| hllset_key | String | HLLSet to index |
| index_key | String | Destination sorted set key |

**Returns**: Integer - Number of positions indexed

**Index Format**:

- Member: `"reg:zeros"` (e.g., "42:3")
- Score: `reg * 32 + zeros` (linearized position)

**Example**:

```redis
redis> HLLSET.POSINDEX hllset:abc123 posidx:abc123
(integer) 42

# Query by position range
redis> ZRANGEBYSCORE posidx:abc123 0 1000
1) "5:3"     # Position 163
2) "10:15"   # Position 335
3) "31:0"    # Position 992

# Get specific register's positions
redis> ZRANGEBYSCORE posidx:abc123 320 351
# Returns positions for register 10 (10*32=320 to 10*32+31=351)
```

**Use Cases**:

- Fast position lookups without loading full HLLSet
- Range queries by register
- Position-based joins

**Time Complexity**: O(popcount) for creation, O(log n) for queries

---

## TokenLUT Entry Schema

TokenLUT entries are Redis Hashes with this structure:

```redis
HGETALL tokenlut:entry:sess1:12345678901234567
 1) "reg"
 2) "42"
 3) "zeros"
 4) "3"
 5) "hash_full"
 6) "12345678901234567"
 7) "layer"
 8) "0"
 9) "first_tokens"
10) "[\"hello\"]"
11) "tokens"
12) "[]"
13) "first_tokens_tag"
14) "hello"
15) "collision_count"
16) "1"
17) "tf"
18) "15"
```

### Field Descriptions

| Field | Description |
| ----- | ----------- |
| reg | Register index (0-1023) |
| zeros | Trailing zeros count (0-31) |
| hash_full | Full 64-bit hash value |
| layer | N-gram layer (0=unigram, 1=bigram, 2=trigram) |
| first_tokens | JSON array of first tokens |
| tokens | JSON array of token arrays (for n-grams) |
| first_tokens_tag | Comma-separated first tokens (for RediSearch) |
| collision_count | Number of distinct tokens at this position |
| tf | Term frequency (accumulated) |

---

## Disambiguation Workflow

### Basic Flow

```text
1. Create HLLSet from query
   HLLSET.CREATE "user" "query" "tokens"
   → hllset:query123

2. Get candidate matches
   HLLSET.CANDIDATES hllset:query123 tokenlut:entry:corpus: LAYER 0
   → Returns matching unigram entries

3. Triangulate bigrams (client-side)
   For each bigram candidate:
     Check if both unigrams are present

4. Score and rank candidates
```

### Streaming Flow

```text
1. Create query HLLSet
   HLLSET.CREATE "user" "query"
   → hllset:query123

2. Stream all matches to a stream
   HLLSET.SCANMATCH hllset:query123 tokenlut:entry: matches:stream

3. Consume stream with workers
   XREADGROUP GROUP workers consumer1 STREAMS matches:stream >

4. Process matches asynchronously
```

### Position Index Flow

```text
1. Index corpus HLLSets
   For each document:
     HLLSET.POSINDEX hllset:doc123 posidx:doc123

2. Query by position
   ZRANGEBYSCORE posidx:doc123 160 165
   → Positions near register 5

3. Fast intersection check
   ZINTER 2 posidx:query posidx:doc123
   → Common positions
```

---

## Performance Considerations

### Memory

- Positions array: ~8 bytes per position (reg + zeros)
- Position index: ~32 bytes per member (sorted set overhead)

### Latency

| Operation | Typical Latency |
| --------- | --------------- |
| POSITIONS | < 1ms |
| HASBIT | < 0.1ms |
| CANDIDATES (100 matches) | 10-50ms |
| SCANMATCH (1M keys) | 1-10s |

### Optimization Tips

1. **Use LAYER filter** to reduce candidate set
2. **Use LIMIT** for interactive queries
3. **Use SCANMATCH streaming** for batch processing
4. **Create POSINDEX** for frequently queried HLLSets
5. **Use BATCH** parameter to tune SCAN behavior
