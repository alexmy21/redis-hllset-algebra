# TokenLUT Best Practices

Best practices for working with the TokenLUT disambiguation system.

## Schema Overview

```text
Hash Key: tokenlut:entry:<hash_full>
Fields:
  reg              NUMERIC   Register index [0, 1023]
  zeros            NUMERIC   Trailing zeros count [0, 31]
  hash_full        NUMERIC   Full 64-bit hash (also used as key suffix)
  layer            NUMERIC   N-gram layer (0=unigram, 1=bigram, 2=trigram)
  collision_count  NUMERIC   Number of tokens sharing this hash (observability)
  first_tokens_tag TAG       Comma-separated first tokens (efficient queries)
  first_tokens     TEXT      JSON array of first tokens
  tokens           TEXT      JSON array of token arrays
```

### Collision Handling

When multiple tokens produce the same 64-bit hash, they're stored together:

```python
# Single token (no collision)
{"first_tokens": ["apple"], "tokens": [], "collision_count": 1}

# Collision (two unigrams share hash)
{"first_tokens": ["apple", "app"], "tokens": [], "collision_count": 2}

# Bigram collision
{"first_tokens": ["quick", "quest"], 
 "tokens": [["quick","brown"], ["quest","ion"]], 
 "collision_count": 2}
```

---

## 1. Atomic Merge with Lua Script

**Problem**: Adding tokens with collision handling requires read-before-write,
causing two round-trips and potential race conditions.

**Solution**: Use Lua script for atomic merge in a single round-trip.

```python
# ✅ Good: Atomic merge via Lua script
entry = TokenEntry(reg=42, zeros=3, hash_full=123456, layer=0, 
                   first_tokens=["apple"])
lut.add_entry(entry)  # Uses MERGE_ENTRY_SCRIPT internally

# ❌ Avoid: Manual read-before-write pattern
# existing = lut.get_by_hash(hash_full)
# if existing: existing.add_collision(...)
# lut.set(...)
```

The Lua script handles:

- New entry creation
- Collision merging (deduplication of first_tokens and tokens arrays)
- Updating `collision_count` and `first_tokens_tag`

---

## 2. collision_count for Observability

**Problem**: Need to identify high-collision entries without parsing JSON arrays.

**Solution**: Stored `collision_count` field enables direct queries.

```python
# Find entries with 3+ collisions (hash quality issues)
high_collision = lut.lookup_high_collision(min_count=3)

for entry in high_collision:
    print(f"Hash {entry.hash_full} has {entry.collision_count} collisions:")
    print(f"  Tokens: {entry.first_tokens}")
```

RediSearch query:

```text
@collision_count:[3 +inf]
```

---

## 3. TagField for Efficient first_token Queries

**Problem**: TEXT field search on JSON arrays is inefficient for exact matching.

**Solution**: Use `first_tokens_tag` TagField for O(1) exact matching.

```python
# ✅ Good: TagField query (efficient)
entries = lut.lookup_by_first_token_tag(["quick", "brown"], layer=1)

# RediSearch: @first_tokens_tag:{quick|brown} @layer:[1 1]

# ❌ Slower: TEXT field search
entries = lut.lookup_by_first_token("quick", layer=1)

# RediSearch: @first_tokens:quick @layer:[1 1]
```

Use cases:

- Triangulation: Find all bigrams starting with a known unigram
- Forward chaining: Find n-grams by their first token

---

## 4. Batch Collision Check with MGET

**Problem**: Adding many entries individually causes N round-trips.

**Solution**: Use `add_batch()` which:

1. Groups entries by hash (handles in-batch collisions)
2. Fetches existing data with pipelined HGETALL
3. Merges collisions in Python
4. Writes back with pipelining

```python
# ✅ Good: Batch add with collision handling
entries = [TokenEntry(...) for token in tokens]
count = lut.add_batch(entries, pipeline_size=1000)

# ❌ Avoid: Individual adds for bulk operations
# for entry in entries:
#     lut.add_entry(entry)  # N round-trips
```

Performance characteristics:

- Single batch: ~2 round-trips (fetch existing + write merged)
- Individual adds: N round-trips

---

## 5. Direct get_by_hash Lookup

**Problem**: RediSearch queries for known hashes are slower than needed.

**Solution**: Use `get_by_hash()` for O(1) direct access when you know the hash.

```python
# ✅ Good: Direct lookup by hash (fastest)
entry = lut.get_by_hash(known_hash)

# ✅ Good: Batch direct lookup
entries = lut.get_by_hash_batch([hash1, hash2, hash3])

# ✅ Good: Existence check only
if lut.exists_by_hash(hash_value):
    ...

# ❌ Slower: RediSearch for known hash
# query = Query(f"@hash_full:[{hash} {hash}]")
```

Use cases:

- Token validation: Check if a hash exists
- Triangulation: Directly access linked entries
- Debugging: Inspect specific entries

---

## 6. Collision-Aware Candidate Dataclass

**Problem**: Old `Candidate` dataclass assumes single token per entry.

**Solution**: Updated `Candidate` with arrays and legacy compatibility.

```python
from core.hllset_disambiguate import Candidate

# New collision-aware access
for c in candidates:
    if c.has_collision:
        print(f"Collision at hash! Possible tokens: {c.first_tokens}")
        # Use context to disambiguate
    else:
        print(f"Unique token: {c.token}")
    
    # Get all tokens as strings
    all_tokens = c.all_tokens  # ["quick brown", "quest ion"]

# Legacy access still works
token = c.token        # First token (backwards compatible)
first = c.first_token  # First of first_tokens
```

---

## 7. Query Patterns

### Primary Disambiguation Query

```python
# Find all candidates at a specific HLL position
candidates = lut.lookup(reg=42, zeros=3)
```

### Triangulation Support

```python
# Find bigrams that could follow a known unigram
unigram = "quick"
bigrams = lut.lookup_by_first_token_tag([unigram], layer=1)

# Get all unigrams at a register (for constraint checking)
unigrams = lut.unigrams_at_register(reg=42)
```

### Layer-Based Queries

```python
# Get all entries at a layer
unigrams = lut.lookup_layer(0)
bigrams = lut.lookup_layer(1)

# Get all entries at register+layer
entries = lut.lookup_register(reg=42, layer=1)
```

### Collision Analysis

```python
# Find problematic entries
high_collision = lut.lookup_high_collision(min_count=5)

# Get stats
stats = lut.stats()
print(f"Total entries: {stats['total_entries']}")
print(f"Entries with collisions: {stats.get('collided_entries', 'N/A')}")
```

---

## Performance Guidelines

| Operation | Round-trips | Use When |
| ----------- | ------------- | ---------- |
| `add_entry()` | 1 | Single entry, collision-aware |
| `add_entry_simple()` | 1 | Bulk insert, no existing collisions |
| `add_batch()` | 2 | Many entries, collision-aware |
| `get_by_hash()` | 1 | Known hash, fastest |
| `lookup(reg, zeros)` | 1 | RediSearch query |
| `lookup_by_first_token_tag()` | 1 | Triangulation (TagField) |

---

## Index Setup

```python
lut = TokenLUTRedis(redis_client)

# Create index (one-time setup)
lut.create_index(drop_existing=False)

# Verify index
if lut.index_exists():
    print("Index ready")
```

Fields indexed:

- `reg` - Sortable, for position queries
- `zeros` - Sortable, for position queries  
- `hash_full` - For direct lookups (though get_by_hash is faster)
- `layer` - Sortable, for layer filtering
- `collision_count` - Sortable, for observability queries
- `first_tokens_tag` - TagField, for efficient exact matching
- `first_tokens` - TextField, for text search
- `tokens` - TextField, for text search

---

## Migration from Old Schema

If migrating from single-token schema:

```python
# Old schema
{"token": "apple", "first_token": "apple"}

# New schema (auto-converted)
{"first_tokens": ["apple"], "tokens": [], "collision_count": 1}
```

The `TokenEntry.from_dict()` handles legacy format gracefully.
