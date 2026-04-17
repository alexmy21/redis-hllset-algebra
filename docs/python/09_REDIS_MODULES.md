# Redis Modules

> Redis-backed HLLSet infrastructure for persistence, streaming, and disambiguation.

**Package**: `core.redis`  
**Dependencies**: `redis`, `redis-py[hiredis]`, custom `hllset` Rust module

## Module Overview

| Module | Purpose |
|--------|---------|
| `hllset_redis` | HLLSet backed by Redis native module |
| `hllset_store_redis` | Registry with RediSearch indexing |
| `hllset_ring_store` | XOR ring algebra with base-only storage |
| `tokenlut_redis` | RediSearch-indexed token lookup table |
| `tokenlut_stream` | Streaming batch ingestion |
| `tokenlut_session` | Multi-producer sessions with checkpoints |
| `hllset_disambiguate` | Zero-copy position matching |

---

## 1. HLLSetRedis

Redis-native HLLSet using the Rust `hllset` module.

### Setup

```python
import redis
from core.redis import HLLSetRedis, RedisClientManager

r = redis.Redis(host='localhost', port=6379, decode_responses=False)
RedisClientManager.set_default(r)

# Verify module loaded
assert RedisClientManager.ensure_module_loaded(r)
```

### Creating HLLSets

```python
# From tokens
hll = HLLSetRedis.from_batch(['hello', 'world', 'hello'])

# From hashes
hll = HLLSetRedis.from_hashes([0x12345678ABCDEF00, 0xDEADBEEFCAFE0001])

# Access key
print(hll.key)  # hllset:a3f2...c8e1
```

### Set Operations

```python
A = HLLSetRedis.from_batch(['a', 'b', 'c'])
B = HLLSetRedis.from_batch(['b', 'c', 'd'])

# Union (OR) — mutable result
union = A.union(B)  # {a, b, c, d}

# Intersection (AND) — mutable result
inter = A.intersection(B)  # {b, c}

# Difference (A \ B)
diff = A.difference(B)  # {a}

# XOR (symmetric difference)
xor = A.xor(B)  # {a, d}

# Finalize mutable result → content-addressable key
finalized = union.finalize()
print(finalized.key)  # hllset:sha1 of content
```

### Queries

```python
print(hll.cardinality())     # Estimated count
print(A.similarity(B))       # Jaccard similarity
print(A.contains_token('a')) # Membership test
```

### Roaring Bitmap Access

```python
# Get as raw bytes
roaring_bytes = hll.to_roaring_bytes()

# Iterate positions (reg, zeros)
for reg, zeros in hll.iter_positions():
    print(f"reg={reg}, zeros={zeros}")
```

---

## 2. HLLSetStoreRedis

Registry with RediSearch indexing for metadata queries.

### Schema

```
hllstore:entry:<sha1>  HASH  {sha1, source, cardinality, created_at, layer, is_base, tags, metadata}
hllstore:lut:<sha1>    HASH  {operation, operands, timestamp}
hllstore:idx           INDEX RediSearch
```

### Setup

```python
from core.redis import HLLSetStoreRedis

store = HLLSetStoreRedis(r)
store.create_index()  # One-time setup
```

### Register Base HLLSets

```python
sha1 = store.register_base(
    hllset,
    source="document.txt",
    layer=1,
    tags=["corpus", "v1"]
)
```

### Compound Operations (Derivation Tracked)

```python
sha3 = store.union(sha1, sha2)
sha4 = store.intersect(sha1, sha2)
sha5 = store.xor(sha1, sha2)
```

### Query by Metadata

```python
# By time
recent = store.query_by_time(since=time.time() - 3600)

# By source
docs = store.query_by_source("document.txt")

# By tag
corpus = store.query_by_tag("corpus")

# Full RediSearch
results = store.search("@layer:[0 2] @tags:{corpus}")
```

### Trace Derivation

```python
deriv = store.get_derivation(sha5)
print(deriv.operation)  # Operation.XOR
print(deriv.operands)   # [sha1, sha2]

# Get all bases
bases = store.get_bases(sha5)  # Recursive expansion
```

---

## 3. HLLSetRingStore

XOR ring algebra with base-only storage — compounds are reconstructed.

### Key Principle

Every HLLSet `H` can be expressed as XOR of basis elements:
```
H = B₁ ⊕ B₂ ⊕ ... ⊕ Bₖ
```

Only bases are stored; compounds are reconstructed on demand.

### Schema

```
hllring:base:<sha1>       STRING  Raw bytes (ONLY for bases)
hllring:lut:<sha1>        HASH    Derivation: {op, bases:[...]}
hllring:ring:<ring_id>    HASH    Ring state (basis SHA1s, rank)
hllring:W:<ring_id>:<t>   HASH    W commit snapshot
```

### Setup

```python
from core.redis import HLLSetRingStore

store = HLLSetRingStore(r)
ring = store.init_ring("session:ring1")
```

### Ingest and Decompose

```python
# Ingest token (auto-decompose)
result = store.ingest(ring, "hello")
print(result.is_new_base)   # True if independent
print(result.expression)    # [sha1] or [b1, b2, ...]

# Decompose HLLSet
result = store.decompose(ring, hllset, source="doc1")
print(f"Rank: {result.rank_before} → {result.rank_after}")
```

### Retrieve (Reconstructs from XOR)

```python
hll = store.get(sha1)  # Reconstructs if compound
```

### W Lattice Commits

```python
store.commit_W(ring, t=1)
store.commit_W(ring, t=2)

diff = store.diff_W(ring, t1=1, t2=2)
print(diff.added_sha1s)
print(diff.removed_sha1s)
```

---

## 4. TokenLUTRedis

RediSearch-indexed token lookup table for disambiguation.

### Schema (Collision-Aware)

```
tokenlut:entry:<hash_full>  HASH  {reg, zeros, hash_full, layer, first_tokens, tokens, tf}
tokenlut:idx                INDEX RediSearch
```

Collisions stored as JSON arrays:
- `first_tokens`: `["apple", "app"]`
- `tokens`: `[["quick","brown"], ["quest","ion"]]`

### Setup

```python
from core.redis import TokenLUTRedis

lut = TokenLUTRedis(r)
lut.create_index()
```

### Add Entries

```python
# Single token (handles collision merging)
lut.add_token("apple", layer=0)

# N-gram
lut.add_ngram(["quick", "brown"], layer=1, first_token="quick")

# Batch (Lua atomic merge)
lut.add_tokens_batch(["apple", "banana", "cherry"], layer=0)
```

### Query

```python
# By position (exact match)
entries = lut.lookup(reg=42, zeros=5)

# By register (any zeros)
entries = lut.lookup_register(reg=42)

# By layer
unigrams = lut.lookup_layer(layer=0)

# First tokens for triangulation
first_tokens = lut.first_tokens_at_register(reg=42, layer=1)
```

### TokenEntry Dataclass

```python
@dataclass
class TokenEntry:
    reg: int
    zeros: int
    hash_full: int
    layer: int
    first_tokens: List[str]
    tokens: List[List[str]]
    tf: int  # Term frequency
```

---

## 5. TokenLUTStream

Streaming batch ingestion with Redis Streams.

### Architecture

```
Client → XADD → Stream → Consumer → LUT Index
                tokenlut:stream       (RediSearch)
```

### Setup

```python
from core.redis import TokenLUTStream

stream = TokenLUTStream(r, index_name='vocab:lut')
```

### Batch Ingestion

```python
# Unigrams
tokens = ['apple', 'banana', 'cherry']
stream.ingest_tokens(tokens, layer=0)

# N-grams with first_token
bigrams = [('quick brown', 'quick'), ('brown fox', 'brown')]
stream.ingest_ngrams(bigrams, layer=1)
```

### Process Stream (Consumer Side)

```python
stream.process_pending()
# or
stream.consume_forever()  # Blocking consumer
```

### Statistics

```python
stats = stream.stats
print(f"Processed: {stats.tokens_processed}")
print(f"Rate: {stats.tokens_per_second:.1f} tok/s")
```

---

## 6. TokenLUTSession

Multi-producer sessions with checkpoints.

### Architecture

```
Producer 0 (unigrams)  ─┐
Producer 1 (bigrams)   ─┼→ Stream → Consumer → LUT
Producer 2 (trigrams)  ─┘      │
                               │
                           session_id
                               │
                               ├→ Checkpoint 1 → temp HLLSet
                               ├→ Checkpoint 2 → temp HLLSet
                               └→ Commit → UNION → final SHA1
```

### Workflow

```python
from core.redis import TokenLUTSession

session = TokenLUTSession(r)
session.start()

# Create producers
p0 = session.create_producer(layer=0)
p1 = session.create_producer(layer=1)

# Send tokens
p0.send([('apple', ''), ('banana', '')])
p1.send([('apple pie', 'apple'), ('banana split', 'banana')])

# Checkpoint — creates temp HLLSet
cp1 = session.checkpoint()
print(cp1.hllset_key)     # temp key
print(cp1.cardinality)    # 4

# More data...
p0.send([('cherry', '')])

# Commit — UNION all checkpoints
result = session.commit()
print(result.hllset_key)  # hllset:<sha1>
print(result.total_tokens)
```

### CheckpointResult

```python
@dataclass
class CheckpointResult:
    checkpoint_id: str
    hllset_key: str
    cardinality: float
    tokens_in_checkpoint: int
    total_tokens: int
```

### CommitResult

```python
@dataclass
class CommitResult:
    session_id: str
    hllset_key: str
    cardinality: float
    total_tokens: int
    tokens_by_layer: Dict[int, int]
    checkpoint_keys: List[str]
    elapsed_seconds: float
    lut_entries: int
```

---

## 7. HLLSetDisambiguator

Zero-copy position matching against TokenLUT.

### Setup

```python
from core.redis import HLLSetDisambiguator

disamb = HLLSetDisambiguator(r)
```

### Find Candidates

```python
candidates = list(disamb.candidates(
    hllset_key="hllset:abc123",
    lut_prefix="tokenlut:entry:",
    layer=0,      # Optional: filter by layer
    limit=1000    # Optional: limit results
))

for c in candidates:
    print(f"{c.token} at ({c.reg}, {c.zeros})")
    if c.has_collision:
        print(f"  Collisions: {c.all_tokens}")
```

### Candidate Dataclass

```python
@dataclass
class Candidate:
    key: str
    reg: int
    zeros: int
    layer: int
    collision_count: int
    first_tokens: List[str]
    tokens: List[List[str]]
    
    # Properties
    token: str          # First token
    has_collision: bool # collision_count > 1
    all_tokens: List[str]
```

### Streaming to Redis Stream

```python
# Output candidates to stream for async processing
stream_key = disamb.candidates_to_stream(
    hllset_key="hllset:abc123",
    lut_prefix="tokenlut:entry:",
    output_stream="disamb:results"
)
```

---

## Import Patterns

### All from `core.redis`

```python
from core.redis import (
    # Main classes
    HLLSetRedis,
    HLLSetStoreRedis,
    HLLSetRingStore,
    TokenLUTRedis,
    TokenLUTStream,
    TokenLUTSession,
    HLLSetDisambiguator,
    
    # Data classes
    TokenEntry,
    Candidate,
    DecomposeResult,
    RingState,
    CheckpointResult,
    CommitResult,
    
    # Utilities
    RedisClientManager,
)
```

### From Parent Package

```python
from core import (
    HLLSetRedis,
    TokenLUTRedis,
    # ... etc (re-exported)
)
```

## Related Modules

- [HLLSet Core](01_HLLSET.md) — In-memory implementation
- [Disambiguation](04_DISAMBIGUATION.md) — Triangulation algorithms
- [Lattice/Store](06_LATTICE_STORE.md) — W lattice concepts
