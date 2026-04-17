# Redis HLLSet Module - Quick Reference

> Internal reference for fast lookup of commands, parameters, and data structures.

## Constants

```
P = 10                    # Precision bits
M = 1024                  # Registers (2^P)
BITS_PER_REG = 32         # Bits per register
TOTAL_BITS = 32768        # M * 32
ALPHA_M = 0.7213/(1+1.079/M)  # Bias correction
```

## Key Prefixes

| Prefix | Type | Purpose |
|--------|------|---------|
| `hllset:` | HLLSet | Content-addressable HLLSet |
| `hllset:union:` | HLLSet | Union result |
| `hllset:inter:` | HLLSet | Intersection result |
| `hllset:diff:` | HLLSet | Difference result |
| `hllset:xor:` | HLLSet | Symmetric diff result |
| `hllring:ring:` | Hash | Ring state |
| `hllring:base:` | HLLSet | Basis element |
| `hllring:lut:` | Hash | Derivation lookup |
| `hllring:meta:` | Hash | Metadata |
| `hllring:W:` | Hash | W commit |
| `hllring:temp:` | HLLSet | Temp reconstruction (60s TTL) |
| `tokenlut:entry:` | Hash | TokenLUT entry |

## Position Encoding

```
position = register * 32 + trailing_zeros
register = position / 32
trailing_zeros = position % 32

# Example: reg=5, tz=3 → pos=163
```

## Hash Algorithm

```
MurmurHash3 64-bit → hash
register = hash & (M-1)           # Lower P bits
trailing_zeros = (hash >> P).trailing_zeros().min(31)
```

## Content Key Generation

```
tokens → sort → dedup → join("\0") → SHA1 → "hllset:" + hex
```

---

## HLLSET CORE COMMANDS

### HLLSET.CREATE

```
HLLSET.CREATE token1 [token2 ...]
→ String: "hllset:<sha1>"
```

- Idempotent (same content = same key)
- Tokens sorted, deduped before hashing
- O(n log n)

### HLLSET.CREATEHASH

```
HLLSET.CREATEHASH hash1 [hash2 ...]
→ String: "hllset:<sha1>"
```

- Hashes are u64 integers
- O(n log n)

### HLLSET.CARD

```
HLLSET.CARD key
→ Float: estimated cardinality
→ 0.0 if key missing
```

- Uses Horvitz-Thompson estimator
- O(M)

### HLLSET.UNION

```
HLLSET.UNION key1 key2
→ String: "hllset:union:<k1_short>:<k2_short>"
```

- Creates new key
- Bitmap OR
- O(M)

### HLLSET.INTER

```
HLLSET.INTER key1 key2
→ String: "hllset:inter:<k1_short>:<k2_short>"
```

- Creates new key
- Bitmap AND
- O(M)

### HLLSET.DIFF

```
HLLSET.DIFF key1 key2
→ String: "hllset:diff:<k1_short>:<k2_short>"
```

- Creates new key
- Bitmap AND-NOT (A \ B)
- O(M)

### HLLSET.XOR

```
HLLSET.XOR key1 key2
→ String: "hllset:xor:<k1_short>:<k2_short>"
```

- Creates new key
- Bitmap XOR (A ⊕ B)
- O(M)

### HLLSET.MERGE

```
HLLSET.MERGE destkey key1 [key2 ...]
→ "OK"
```

- **MUTABLE** - breaks content-addressable pattern (by design)
- In-place union into destkey
- Creates destkey if not exists
- Use for accumulation (like PFMERGE)
- O(M * k)

### HLLSET.FINALIZE

```
HLLSET.FINALIZE source_key [DELETE]
→ String: "hllset:<sha1>"
```

- Computes content hash, copies to canonical key
- DELETE flag removes source after copy
- Idempotent (returns same key if already canonical)
- Use after MERGE to get content-addressable key
- O(M)

### HLLSET.SIM / HLLSET.JACCARD

```
HLLSET.SIM key1 key2
→ Float: |A∩B| / |A∪B| in [0.0, 1.0]
```

- Alias: HLLSET.JACCARD
- O(M)

### HLLSET.INFO

```
HLLSET.INFO key
→ Array: [key, <key>, cardinality, <float>, registers, 1024,
          non_zero_registers, <int>, precision_bits, 10, memory_bytes, <int>]
```

- ERR if key missing
- O(M)

### HLLSET.DUMP

```
HLLSET.DUMP key
→ Array of [bucket, highest_bit] pairs
```

- ERR if key missing
- O(M)

### HLLSET.EXISTS

```
HLLSET.EXISTS key
→ Integer: 1 or 0
```

- O(1)

### HLLSET.DEL

```
HLLSET.DEL key
→ Integer: 1 (deleted) or 0 (not found)
```

- O(1)

---

## TENSOR/POSITION COMMANDS

### HLLSET.POSITIONS

```
HLLSET.POSITIONS key
→ Array: [reg1, zeros1, reg2, zeros2, ...]
```

- Flat array of (reg, zeros) pairs
- Empty array if key missing
- O(popcount)

### HLLSET.POPCOUNT

```
HLLSET.POPCOUNT key
→ Integer: total set bits
```

- 0 if key missing
- O(1)

### HLLSET.BITCOUNTS

```
HLLSET.BITCOUNTS key
→ Array: [c_0, c_1, ..., c_31] (32 integers)
```

- c_s = registers with bit s set
- Used for HT estimator
- O(popcount)

### HLLSET.REGISTER

```
HLLSET.REGISTER key reg
→ Integer: 32-bit bitmap value
```

- reg in [0, 1023]
- ERR if reg out of range
- O(32)

### HLLSET.HASBIT

```
HLLSET.HASBIT key reg zeros
→ Integer: 1 or 0
```

- reg in [0, 1023], zeros in [0, 31]
- ERR if out of range
- O(1)

---

## DISAMBIGUATION COMMANDS

### HLLSET.CANDIDATES

```
HLLSET.CANDIDATES hllset_key lut_prefix [STREAM stream_key] [LAYER n] [LIMIT n]
→ Without STREAM: Array [key, token, layer, first_token, ...]
→ With STREAM: Integer (count streamed)
```

- Scans keys matching lut_prefix*
- Matches against HLLSet positions
- LAYER filters by n-gram layer (0=unigram, 1=bigram)
- O(n) where n = scanned keys

### HLLSET.SCANMATCH

```
HLLSET.SCANMATCH hllset_key lut_prefix stream_key [LAYER n] [BATCH n]
→ Integer: total matches
```

- Full cursor iteration with SCAN
- BATCH controls SCAN COUNT (default: 1000)
- Results streamed to stream_key
- O(N) where N = all matching keys

### HLLSET.POSINDEX

```
HLLSET.POSINDEX hllset_key index_key
→ Integer: positions indexed
```

- Creates sorted set at index_key
- Member: "reg:zeros", Score: reg*32+zeros
- Deletes existing index first
- O(popcount)

---

## RING COMMANDS

### HLLSET.RING.INIT

```
HLLSET.RING.INIT ring_key [PBITS p]
→ "OK"
```

- Creates ring state hash at hllring:ring:<ring_key>
- PBITS default: 10
- O(1)

### HLLSET.RING.INGEST

```
HLLSET.RING.INGEST ring_key token [SOURCE src] [TAGS t1,t2]
→ Array: [sha1, is_new_base (0|1), num_bases, base1, base2, ...]
```

- Creates HLLSet from single token
- Decomposes into ring
- Stores base/derivation/metadata
- O(rank² × M)

### HLLSET.RING.DECOMPOSE

```
HLLSET.RING.DECOMPOSE ring_key hllset_key [SOURCE src] [TAGS t1,t2]
→ Array: [sha1, is_new_base (0|1), num_bases, base1, base2, ...]
```

- Decomposes existing HLLSet
- O(rank² × M)

### HLLSET.RING.BASIS

```
HLLSET.RING.BASIS ring_key
→ Array of SHA1 strings
```

- O(rank)

### HLLSET.RING.RANK

```
HLLSET.RING.RANK ring_key
→ Integer: current rank
```

- Max theoretical: M*32 = 32768
- O(1)

### HLLSET.W.COMMIT

```
HLLSET.W.COMMIT ring_key [META json]
→ Integer: time_index
```

- Snapshots current ring state
- Stored at hllring:W:<ring_key>:<time_index>
- O(rank)

### HLLSET.W.DIFF

```
HLLSET.W.DIFF ring_key t1 t2
→ Array: [added_count, removed_count, shared_count, delta_rank,
          added_sha1s..., removed_sha1s..., shared_sha1s...]
```

- Compares two W commits
- O(rank₁ + rank₂)

### HLLSET.RECONSTRUCT

```
HLLSET.RECONSTRUCT sha1
→ String: key of reconstructed HLLSet
```

- If base: returns hllring:base:<sha1>
- If XOR: creates hllring:temp:<sha1> with 60s TTL
- ERR if derivation not found
- O(k × M) where k = bases in expression

---

## TOKENLUT COMMANDS

### TOKENLUT.ADD

```
TOKENLUT.ADD prefix hash_full reg zeros layer first_token [TOKEN t1 t2...] [TF n]
→ Array: [collision_count, tf]
```

- Creates/updates hash at <prefix><hash_full>
- TOKEN: additional tokens for n-grams
- TF: increment amount (default: 1)
- Merges first_tokens/tokens arrays on collision
- O(1) new, O(n) update

### TOKENLUT.INCR

```
TOKENLUT.INCR key [BY n]
→ Integer: new TF value
→ -1 if key doesn't exist
```

- BY default: 1
- O(1)

### TOKENLUT.GET

```
TOKENLUT.GET key
→ Array: all hash fields/values
```

- Equivalent to HGETALL
- O(n) where n = fields

### TOKENLUT.MGET

```
TOKENLUT.MGET key1 [key2 ...]
→ Array of arrays (each is HGETALL result)
```

- O(n × m)

---

## TOKENLUT ENTRY SCHEMA

```
{
  reg: "42",                          # Register [0-1023]
  zeros: "3",                         # Trailing zeros [0-31]
  hash_full: "12345678901234567",     # Full u64 hash
  layer: "0",                         # 0=unigram, 1=bigram, ...
  first_tokens: "[\"hello\"]",        # JSON array
  tokens: "[]",                       # JSON array of arrays (n-grams)
  first_tokens_tag: "hello",          # Comma-joined (for search)
  collision_count: "1",               # Distinct first_tokens
  tf: "15"                            # Term frequency
}
```

---

## RING STATE SCHEMA

```
hllring:ring:<ring_id>
{
  ring_id: "myring",
  p_bits: "10",
  rank: "42",
  basis_sha1s: "[\"sha1_1\",\"sha1_2\"]",  # JSON
  created_at: "1713258000.123",
  updated_at: "1713259000.456",
  matrix_data: "<base64>",
  pivots: "[0,5,12,...]"               # JSON
}
```

---

## DERIVATION LUT SCHEMA

```
hllring:lut:<sha1>
{
  op: "base" | "xor",
  bases: "[\"sha1_1\",\"sha1_2\"]"     # JSON (empty for base)
}
```

---

## W COMMIT SCHEMA

```
hllring:W:<ring_id>:<time_index>
{
  time_index: "0",
  ring_id: "myring",
  basis_sha1s: "[...]",                # JSON
  rank: "42",
  timestamp: "1713258000.123",
  metadata: "{...}"                    # Optional JSON
}
```

---

## STREAMING OUTPUT SCHEMA

```
XADD <stream_key> *
  key <lut_key>
  reg <reg>
  zeros <zeros>
  layer <layer>
  token <token>
  first_token <first_token>
```

---

## ERROR MESSAGES

| Error | Trigger |
|-------|---------|
| `ERR wrong number of arguments` | Missing params |
| `ERR Key does not exist` | INFO/DUMP on missing key |
| `ERR register index out of range (0-1023)` | REGISTER/HASBIT invalid reg |
| `ERR position out of range` | HASBIT invalid zeros |
| `ERR No valid hashes provided` | CREATEHASH with bad input |
| `ERR Ring not initialized` | INGEST before INIT |
| `ERR Ring not found` | Bad ring_key |
| `ERR Derivation not found` | RECONSTRUCT unknown sha1 |
| `ERR Base not found during reconstruction` | Missing base HLLSet |
| `ERR unknown argument` | Invalid option |

---

## COMPLEXITY SUMMARY

| Operation | Time | Space |
|-----------|------|-------|
| CREATE | O(n log n) | O(M) |
| CARD | O(M) | - |
| UNION/INTER/DIFF/XOR | O(M) | O(M) |
| MERGE | O(M × k) | - |
| SIM | O(M) | - |
| POSITIONS | O(popcount) | O(popcount) |
| POPCOUNT | O(1) | - |
| CANDIDATES | O(scanned) | O(matches) |
| SCANMATCH | O(all keys) | O(1) |
| RING.INGEST | O(rank² × M) | O(rank) |
| RECONSTRUCT | O(k × M) | O(M) |

Where: n=tokens, M=1024, k=number of sources/bases, popcount=set bits
