# Redis Tutorials

This folder contains tutorials for the Redis-native HLLSet Algebra implementation.

## Tutorial Structure

| # | Tutorial | Description | Status |
| --- | ---------- | ------------- | -------- |
| 01 | [HLLSet Redis](01_hllset_redis.ipynb) | Basic HLLSet operations with Redis backend | ✓ POC |
| 02 | HLL Tensor Redis | 2D tensor view operations | Planned |
| 03 | HLL Lattice Redis | Content-addressed DAG with RedisGraph | Planned |
| 04 | Global Registry Redis | Persistent universe sets G₁, G₂, G₃ | Planned |
| 05 | BSS Redis | Bell State Similarity metrics | Planned |
| 06 | Ring Transaction Redis | IICA protocol with Redis transactions | Planned |
| 07 | Disambiguation Redis | Token recovery with RedisSearch | Planned |
| 08 | De Bruijn Redis | Sequence reconstruction | Planned |
| 09 | Bayesian Redis | Probabilistic inference | Planned |
| 10 | Markov Redis | Markov chains on HLLSets | Planned |

## Prerequisites

1. Redis server running with modules:
   - `redis-roaring` (Roaring bitmaps)
   - `redisearch` (full-text search, future)
   - `redisgraph` (graph queries, future)

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
