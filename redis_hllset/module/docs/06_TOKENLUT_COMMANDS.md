# TokenLUT Commands Reference

This document covers commands for managing TokenLUT entries - the lookup table for token disambiguation.

## Overview

TokenLUT stores token information indexed by their HLLSet position (register, trailing_zeros). This enables:

- Fast position-based lookups
- Term frequency (TF) tracking
- N-gram layer organization
- Collision tracking for shared positions

## Commands

### TOKENLUT.ADD

Adds or updates a TokenLUT entry with automatic TF increment.

**Syntax**:

```redis
TOKENLUT.ADD prefix hash_full reg zeros layer first_token [TOKEN t1 t2 ...] [TF n]
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| prefix | String | Key prefix (e.g., "tokenlut:entry:session:") |
| hash_full | String | Full 64-bit hash value |
| reg | Integer | Register index [0-1023] |
| zeros | Integer | Trailing zeros count [0-31] |
| layer | Integer | N-gram layer (0=unigram, 1=bigram, ...) |
| first_token | String | First/main token |
| TOKEN | String... | Optional: Additional tokens for n-grams |
| TF | Integer | Optional: TF increment (default: 1) |

**Returns**: Array of [collision_count, tf]

**Behavior**:

1. **New entry**: Creates hash with all fields, collision_count=1
2. **Existing entry**: Merges tokens, increments collision_count and TF

**Example**:

```redis
# Add unigram
redis> TOKENLUT.ADD tokenlut:entry:sess1: 12345678 42 3 0 hello
1) (integer) 1    # collision_count
2) (integer) 1    # tf

# Add same position again (collision)
redis> TOKENLUT.ADD tokenlut:entry:sess1: 12345678 42 3 0 world
1) (integer) 2    # collision_count increased
2) (integer) 2    # tf increased

# Add bigram with TF boost
redis> TOKENLUT.ADD tokenlut:entry:sess1: 98765432 100 5 1 quick TOKEN quick brown TF 5
1) (integer) 1
2) (integer) 5    # TF was set to 5
```

**Created Hash Structure**:

```redis
HGETALL tokenlut:entry:sess1:12345678
 1) "reg"
 2) "42"
 3) "zeros"
 4) "3"
 5) "hash_full"
 6) "12345678"
 7) "layer"
 8) "0"
 9) "first_tokens"
10) "[\"hello\",\"world\"]"
11) "tokens"
12) "[]"
13) "first_tokens_tag"
14) "hello,world"
15) "collision_count"
16) "2"
17) "tf"
18) "2"
```

**Time Complexity**: O(1) for new, O(n) for updates where n = existing tokens

---

### TOKENLUT.INCR

Increments the TF (term frequency) for an existing entry.

**Syntax**:

```redis
TOKENLUT.INCR key [BY n]
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key | String | Full key of the entry |
| BY | Integer | Optional: Increment amount (default: 1) |

**Returns**: Integer - New TF value, or -1 if key doesn't exist

**Example**:

```redis
# Increment by 1
redis> TOKENLUT.INCR tokenlut:entry:sess1:12345678
(integer) 3

# Increment by 10
redis> TOKENLUT.INCR tokenlut:entry:sess1:12345678 BY 10
(integer) 13

# Non-existent key
redis> TOKENLUT.INCR tokenlut:entry:sess1:nonexistent
(integer) -1
```

**Time Complexity**: O(1)

---

### TOKENLUT.GET

Retrieves a TokenLUT entry by its full key.

**Syntax**:

```redis
TOKENLUT.GET key
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key | String | Full key of the entry |

**Returns**: Array - All hash fields and values

**Example**:

```redis
redis> TOKENLUT.GET tokenlut:entry:sess1:12345678
 1) "reg"
 2) "42"
 3) "zeros"
 4) "3"
 5) "hash_full"
 6) "12345678"
 7) "layer"
 8) "0"
 9) "first_tokens"
10) "[\"hello\",\"world\"]"
11) "tokens"
12) "[]"
13) "first_tokens_tag"
14) "hello,world"
15) "collision_count"
16) "2"
17) "tf"
18) "13"
```

**Time Complexity**: O(n) where n = number of fields

---

### TOKENLUT.MGET

Batch retrieves multiple TokenLUT entries.

**Syntax**:

```redis
TOKENLUT.MGET key1 [key2 ...]
```

**Parameters**:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| key1, key2, ... | String | Full keys to retrieve |

**Returns**: Array of arrays - Fields/values for each key

**Example**:

```redis
redis> TOKENLUT.MGET tokenlut:entry:sess1:12345678 tokenlut:entry:sess1:98765432
1) 1) "reg"
   2) "42"
   3) "zeros"
   4) "3"
   ... (all fields for first key)
2) 1) "reg"
   2) "100"
   3) "zeros"
   4) "5"
   ... (all fields for second key)
```

**Time Complexity**: O(n × m) where n = keys, m = fields per key

---

## Entry Schema

### Hash Fields

| Field | Type | Description | Example |
| ----- | ---- | ----------- | ------- |
| reg | String | Register index | "42" |
| zeros | String | Trailing zeros | "3" |
| hash_full | String | Full 64-bit hash | "12345678901234567" |
| layer | String | N-gram layer | "0" |
| first_tokens | JSON | Array of first tokens | `["hello","world"]` |
| tokens | JSON | Array of token arrays | `[["quick","brown"]]` |
| first_tokens_tag | String | Comma-joined tokens | "hello,world" |
| collision_count | String | Distinct token count | "2" |
| tf | String | Term frequency | "15" |

### Layer Convention

| Layer | Meaning | tokens Field |
| ----- | ------- | ------------ |
| 0 | Unigram | Empty `[]` |
| 1 | Bigram | `[["word1","word2"]]` |
| 2 | Trigram | `[["w1","w2","w3"]]` |
| n | N-gram | Array of n-token arrays |

---

## Key Naming

### Recommended Format

```text
<namespace>:<type>:<context>:<hash_full>

Examples:
tokenlut:entry:session123:12345678901234567
tokenlut:entry:corpus_v1:98765432109876543
tokenlut:entry:user_alice:55555555555555555
```

### Prefix Guidelines

| Component | Purpose | Example |
| --------- | ------- | ------- |
| namespace | System identifier | `tokenlut` |
| type | Entry type | `entry` |
| context | Session/corpus ID | `session123` |
| hash_full | Unique identifier | `12345678...` |

---

## Usage Patterns

### Building TokenLUT from Corpus

```redis
# For each token in corpus
TOKENLUT.ADD tokenlut:entry:corpus_v1: <hash> <reg> <zeros> 0 <token>

# For each bigram
TOKENLUT.ADD tokenlut:entry:corpus_v1: <hash> <reg> <zeros> 1 <first> TOKEN <first> <second>
```

### Session-Based LUT

```redis
# Create session-scoped entries
TOKENLUT.ADD tokenlut:entry:sess_abc: <hash> <reg> <zeros> 0 <token>

# Query session
HLLSET.CANDIDATES hllset:query tokenlut:entry:sess_abc:

# Cleanup session
DEL tokenlut:entry:sess_abc:*
```

### TF-IDF Preparation

```redis
# During indexing: accumulate TF
TOKENLUT.ADD tokenlut:entry:doc123: <hash> <reg> <zeros> 0 <token> TF 1

# Or increment existing
TOKENLUT.INCR tokenlut:entry:doc123:<hash>

# Read TF for scoring
TOKENLUT.GET tokenlut:entry:doc123:<hash>
```

---

## Integration with Disambiguation

### Workflow

```text
1. Build TokenLUT during corpus processing
   For each document:
     For each token:
       Compute hash, reg, zeros
       TOKENLUT.ADD ... layer=0 token

2. Create query HLLSet
   HLLSET.CREATE query tokens

3. Find candidates
   HLLSET.CANDIDATES hllset:query tokenlut:entry:corpus:

4. Retrieve full entries for scoring
   TOKENLUT.MGET matched_key1 matched_key2 ...

5. Score using TF and collision_count
```

### RediSearch Integration

The `first_tokens_tag` field is designed for RediSearch indexing:

```redis
# Create index
FT.CREATE tokenlut_idx ON HASH PREFIX 1 tokenlut:entry:
    SCHEMA
        reg NUMERIC SORTABLE
        zeros NUMERIC SORTABLE
        layer NUMERIC SORTABLE
        first_tokens_tag TAG SEPARATOR ,
        tf NUMERIC SORTABLE

# Search by token
FT.SEARCH tokenlut_idx "@first_tokens_tag:{hello}"

# Search by layer
FT.SEARCH tokenlut_idx "@layer:[0 0]"

# Search by position
FT.SEARCH tokenlut_idx "@reg:[42 42] @zeros:[3 3]"
```

---

## Performance Considerations

### Memory

- Each entry: ~200-500 bytes (depending on token lengths)
- Per million entries: ~200-500 MB

### Throughput

| Operation | Typical Rate |
| --------- | ------------ |
| ADD (new) | 50,000/sec |
| ADD (update) | 30,000/sec |
| GET | 100,000/sec |
| MGET (10 keys) | 50,000/sec |

### Optimization Tips

1. **Batch operations** with MGET for bulk retrieval
2. **Use TF increment** instead of repeated ADD
3. **Partition by context** for cleanup efficiency
4. **Index with RediSearch** for complex queries

---

## Error Handling

| Error | Cause | Solution |
| ----- | ----- | -------- |
| ERR wrong number of arguments | Missing required params | Check syntax |
| ERR TF requires a number | Invalid TF value | Provide integer |
| ERR unknown argument | Typo in TOKEN/TF | Check spelling |
| -1 from INCR | Key doesn't exist | Use ADD first |

---

## Best Practices

### Key Design

1. **Include context** in prefix for multi-tenancy
2. **Use consistent hash format** (decimal or hex)
3. **Plan for cleanup** with prefix-based deletion

### Data Quality

1. **Normalize tokens** before hashing
2. **Track collision_count** for ambiguity detection
3. **Monitor TF distribution** for corpus health

### Scalability

1. **Shard by prefix** across Redis clusters
2. **Archive old sessions** periodically
3. **Use SCAN** for large-scale iteration
