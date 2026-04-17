# Best Practices and Usage Patterns

This document provides guidance on effectively using the Redis HLLSet module in production environments.

## General Principles

### Content-Addressable Design

Leverage the content-addressable nature of HLLSet keys:

```redis
# Same content always produces same key
redis> HLLSET.CREATE apple banana cherry
"hllset:abc123..."

redis> HLLSET.CREATE cherry apple banana  # Different order
"hllset:abc123..."                         # Same key!

# This enables natural deduplication
```

**Benefits**:

- Automatic deduplication of identical sets
- Idempotent operations
- Natural caching

**Pattern**: Always use `HLLSET.CREATE` return value as the authoritative key.

### Probabilistic Nature

Remember that HLLSet is probabilistic:

- Cardinality estimates have ~2-3% error
- Set operations are approximate
- Intersection accuracy decreases with set divergence

**Pattern**: Use for approximate counting and similarity, not exact membership.

---

## Memory Management

### Key Lifecycle

Set operations create new keys that accumulate:

```redis
# This creates a new key each time
HLLSET.UNION key1 key2  → hllset:union:xxx:yyy
HLLSET.INTER key1 key2  → hllset:inter:xxx:yyy
```

**Pattern**: Clean up derived keys after use:

```redis
# Process result
SET result_key (HLLSET.UNION key1 key2)
HLLSET.CARD $result_key
# Then cleanup
HLLSET.DEL $result_key
```

### Memory Estimation

Typical HLLSet memory usage:

| Cardinality | Typical Size |
| ----------- | ------------ |
| 1-100 | 50-200 bytes |
| 100-1000 | 200-500 bytes |
| 1000-10000 | 500-2000 bytes |
| 10000+ | 1-4 KB |

**Pattern**: For capacity planning, assume ~2KB per HLLSet average.

### TTL for Temporary Keys

```redis
# Set TTL on derived keys
SET result_key (HLLSET.UNION key1 key2)
EXPIRE $result_key 3600  # 1 hour TTL

# Ring temp keys auto-expire in 60s
HLLSET.RECONSTRUCT sha1  → hllring:temp:sha1 (60s TTL)
```

---

## Performance Optimization

### Batch Operations

Use `HLLSET.MERGE` instead of chained unions:

```redis
# Inefficient
result1 = HLLSET.UNION key1 key2
result2 = HLLSET.UNION result1 key3
result3 = HLLSET.UNION result2 key4

# Efficient
HLLSET.MERGE result key1 key2 key3 key4
```

### Pipelining

Combine multiple reads in a pipeline:

```python
# Python example
pipe = r.pipeline()
for key in hllset_keys:
    pipe.execute_command('HLLSET.CARD', key)
cardinalities = pipe.execute()
```

### Caching Strategies

Cache frequently accessed results:

```redis
# Check cache first
GET similarity:key1:key2
# If miss, compute and cache
SET similarity:key1:key2 (HLLSET.SIM key1 key2) EX 3600
```

---

## Disambiguation Patterns

### Layered Disambiguation

Process by layer for better accuracy:

```redis
# 1. Get unigram candidates
HLLSET.CANDIDATES hllset:query tokenlut:entry:corpus: LAYER 0 LIMIT 100

# 2. Get bigram candidates
HLLSET.CANDIDATES hllset:query tokenlut:entry:corpus: LAYER 1 LIMIT 50

# 3. Triangulate: verify bigrams have both unigrams present
```

### Streaming for Large Scale

Use streaming for batch processing:

```redis
# Create stream
HLLSET.SCANMATCH hllset:query tokenlut:entry: candidates:stream BATCH 500

# Process with consumer group
XGROUP CREATE candidates:stream workers $ MKSTREAM
XREADGROUP GROUP workers worker1 STREAMS candidates:stream >
```

### Position Index Pattern

Pre-index frequently queried HLLSets:

```redis
# Index once
HLLSET.POSINDEX hllset:corpus_query posidx:corpus_query

# Query many times
ZRANGEBYSCORE posidx:corpus_query 160 170
ZINTER 2 posidx:query posidx:document WITHSCORES
```

---

## Ring Algebra Patterns

### Multi-Ring Architecture

Separate rings for different domains:

```redis
# Separate by content type
HLLSET.RING.INIT ring:news
HLLSET.RING.INIT ring:social
HLLSET.RING.INIT ring:scientific

# Or by time window
HLLSET.RING.INIT ring:2024Q1
HLLSET.RING.INIT ring:2024Q2
```

### Checkpoint Strategy

Regular commits for temporal tracking:

```redis
# Hourly commits
HLLSET.W.COMMIT ring:main META '{"type":"hourly","hour":"2024-04-16T10"}'

# Daily checkpoints
HLLSET.W.COMMIT ring:main META '{"type":"daily","date":"2024-04-16"}'

# Compare across time
HLLSET.W.DIFF ring:main 100 200  # Hourly checkpoints 100 to 200
```

### Efficient Reconstruction

Cache reconstructed HLLSets if reused:

```redis
# Reconstruct
temp_key = HLLSET.RECONSTRUCT sha1

# If needed longer, copy and set TTL
COPY $temp_key reconstructed:$sha1
EXPIRE reconstructed:$sha1 86400
```

---

## Error Handling

### Graceful Degradation

Handle missing keys gracefully:

```python
def safe_card(key):
    card = r.execute_command('HLLSET.CARD', key)
    return card if card else 0.0

def safe_union(key1, key2):
    try:
        return r.execute_command('HLLSET.UNION', key1, key2)
    except ResponseError:
        # One key might not exist
        return None
```

### Validation Pattern

Validate before operations:

```redis
# Check existence before expensive operations
IF (HLLSET.EXISTS key1) AND (HLLSET.EXISTS key2) THEN
    HLLSET.SIM key1 key2
ELSE
    RETURN 0.0
```

### Retry Logic

Handle transient failures:

```python
@retry(tries=3, delay=0.1)
def create_hllset(tokens):
    return r.execute_command('HLLSET.CREATE', *tokens)
```

---

## Monitoring

### Key Metrics

Track these metrics:

```redis
# Memory usage per key
HLLSET.INFO key → memory_bytes

# Ring health
HLLSET.RING.RANK ring_id  # Should stabilize

# Position density
HLLSET.POPCOUNT key  # Higher = more data
```

### Alert Thresholds

| Metric | Warning | Critical |
| ------ | ------- | -------- |
| Ring rank growth | > 100/hour | > 1000/hour |
| Memory per key | > 10KB | > 50KB |
| SCANMATCH duration | > 5s | > 30s |

### Logging Pattern

Log important operations:

```python
import logging

def log_ring_ingest(ring_id, token, result):
    is_new_base = result[1] == 1
    logging.info(f"Ring ingest: ring={ring_id}, new_base={is_new_base}, "
                 f"rank_after={len(result) - 3}")
```

---

## Security Considerations

### Key Namespace Isolation

Isolate by tenant/user:

```redis
# Per-tenant prefixes
tokenlut:entry:tenant_abc:...
hllset:tenant_abc:...
hllring:ring:tenant_abc:...

# Access control at prefix level
ACL SETUSER tenant_abc ~tenant_abc:* +@all
```

### Input Validation

Validate before processing:

```python
def validate_tokens(tokens):
    if not tokens:
        raise ValueError("Empty token list")
    if len(tokens) > 10000:
        raise ValueError("Too many tokens")
    for t in tokens:
        if len(t) > 1000:
            raise ValueError("Token too long")
    return tokens
```

---

## Testing Strategies

### Unit Testing

Test individual commands:

```python
def test_hllset_create():
    key = r.execute_command('HLLSET.CREATE', 'a', 'b', 'c')
    assert key.startswith('hllset:')
    card = r.execute_command('HLLSET.CARD', key)
    assert 2.5 <= card <= 3.5  # Allow for probabilistic error
```

### Integration Testing

Test workflows:

```python
def test_disambiguation_flow():
    # Setup corpus
    r.execute_command('TOKENLUT.ADD', 'test:', '123', '42', '3', '0', 'hello')
    
    # Create query
    query_key = r.execute_command('HLLSET.CREATE', 'hello')
    
    # Find candidates
    candidates = r.execute_command('HLLSET.CANDIDATES', query_key, 'test:')
    
    assert len(candidates) > 0
```

### Load Testing

Benchmark critical paths:

```python
import time

def benchmark_create(n=10000):
    start = time.time()
    for i in range(n):
        r.execute_command('HLLSET.CREATE', f'token_{i}')
    elapsed = time.time() - start
    print(f"Create: {n/elapsed:.0f} ops/sec")
```

---

## Migration Guide

### From Standard HLL

If migrating from `PFADD/PFCOUNT`:

```python
# Old way
r.pfadd('hll:key', 'a', 'b', 'c')
r.pfcount('hll:key')

# New way (more features)
key = r.execute_command('HLLSET.CREATE', 'a', 'b', 'c')
r.execute_command('HLLSET.CARD', key)
```

### Version Upgrades

When upgrading the module:

1. **Backup RDB** before upgrade
2. Test with non-production data
3. Monitor memory and performance
4. Keep old module available for rollback

---

## Common Pitfalls

### Pitfall: Ignoring Probabilistic Nature

```redis
# DON'T expect exact membership
HLLSET.INTER key1 key2
# Intersection cardinality is approximate!

# DO use for approximate analytics
HLLSET.SIM key1 key2  # Good for similarity
```

### Pitfall: Key Accumulation

```redis
# DON'T create keys without cleanup plan
LOOP:
    HLLSET.UNION key1 key2  # Creates new key each time!

# DO clean up or use MERGE
HLLSET.MERGE result key1 key2
```

### Pitfall: Large Token Lists

```redis
# DON'T create HLLSet from millions of tokens in one call
HLLSET.CREATE token1 token2 ... token_million

# DO batch and merge
for batch in token_batches:
    temp_key = HLLSET.CREATE *batch
    HLLSET.MERGE result temp_key
    HLLSET.DEL temp_key
```

### Pitfall: Synchronous SCANMATCH

```redis
# DON'T block on large scans in request path
HLLSET.SCANMATCH hllset:query tokenlut: results BATCH 10000

# DO use background processing
# Or limit scope with specific prefix/layer
HLLSET.CANDIDATES hllset:query tokenlut:entry:sess1: LAYER 0 LIMIT 100
```
