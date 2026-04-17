# Redis Tutorials

This folder contains tutorials for the Redis-native HLLSet Algebra implementation.

## Tutorial Structure

| # | Tutorial | Description | Status |
| --- | ---------- | ------------- | -------- |
| 01 | [HLLSet Redis](01_hllset_redis.ipynb) | Basic HLLSet operations with Redis backend | ✓ POC |
| 02 | [TokenLUT Redis](02_tokenlut_redis.ipynb) | Token lookup table with Redis | ✓ POC |
| 03 | [Disambiguation](03_disambiguation.ipynb) | Token recovery with HLLSet | ✓ POC |
| 04 | [Ring Store](04_hllset_ring_store.ipynb) | IICA protocol with Rust RING commands | ✓ POC |
| 05 | [De Bruijn RedisGraph](05_debruijn_redisgraph.ipynb) | Sequence reconstruction with RedisGraph | ✓ POC |
| 06 | Global Registry Redis | Persistent universe sets G₁, G₂, G₃ | Planned |
| 07 | HLL Tensor Redis | 2D tensor view operations | Planned |
| 08 | HLL Lattice Redis | Content-addressed DAG with RedisGraph | Planned |
| 09 | BSS Redis | Bell State Similarity metrics | Planned |
| 10 | Bayesian Redis | Probabilistic inference | Planned |
| 11 | Markov Redis | Markov chains on HLLSets | Planned |

## Prerequisites

1. Redis server running with modules:
   - `hllset` (HLLSet native operations)
   - `redis-roaring` (Roaring bitmaps)
   - `redisgraph` (graph queries for De Bruijn)
   - `redisearch` (full-text search, future)

2. HLLSet Lua functions loaded:

   ```bash
   cd redis/scripts
   ./start_redis.sh
   ./load_functions.sh
   ```

3. Python packages:

   ```bash
   pip install redis
   ```

## Reference Tutorials

The `tutorials/` folder contains the original Python/Cython-based tutorials.
These serve as reference for the Redis reimplementation:

- API design patterns
- Expected behavior and test cases
- Mathematical foundations

## Migration Guide

To migrate code from Python HLLSet to Redis:

```python
# Before (Python backend)
from core import HLLSet
hll = HLLSet.from_batch(tokens)

# After (Redis backend)
from core import HLLSetRedis, RedisClientManager
import redis

RedisClientManager.set_default(redis.Redis())
hll = HLLSetRedis.from_batch(tokens)

# Same API works!
print(hll.cardinality())
union = hll.union(other)
```
