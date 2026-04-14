# HLLSet Algebra Redis Refactoring Roadmap

## Project Vision

Transform the HLLSet Algebra library from a Python/Cython-based implementation to a Redis-native implementation using:

1. **Roaring Bitmaps** for efficient set operations (via `redis-roaring` module)
2. **Redis Functions** (Lua scripting) for POC/prototyping
3. **Native Redis Module** (C/Rust) for production
4. **Content-addressed storage** using SHA1-based keys
5. **RedisSearch** for metadata indexing (future)
6. **RedisGraph** for relationship modeling (future)

## Core Philosophy

### Redis as Shared Memory

Redis is not just storage — it's a **sophisticated shared memory manager** for multiple processes and applications:

- **IPC with rich data structures** — Atomic operations on complex types
- **Global namespace** — Content-addressed keys visible to all clients
- **Zero serialization** — Data stays in Redis format, no pickling
- **Language agnostic** — Python, Rust, Go, Julia can share the same HLLSets

### Why NOT Native Redis HLL

Redis `PFADD`/`PFCOUNT` uses standard HyperLogLog with dense 6-bit registers optimized for cardinality-only estimation. Our **bitmap register model** is fundamentally different:

| Aspect | Redis Native HLL | HLLSet Algebra |
|--------|------------------|----------------|
| Register | 6-bit max(zeros+1) | 32-bit bitmap |
| Storage | Dense array | Roaring bitmap |
| Union | ✓ (merge max) | ✓ (bitwise OR) |
| Intersection | ✗ (impossible) | ✓ (bitwise AND) |
| Difference | ✗ (impossible) | ✓ (bitwise ANDNOT) |
| XOR | ✗ (impossible) | ✓ (bitwise XOR) |

The bitmap model preserves full observation history, enabling TRUE set algebra.

## Implementation Phases

### Phase 0: POC (Current) — Lua Functions

- Validate API design and data model
- Test Redis pipelining (million-size batches ✓)
- Hash roundtrip Python→Redis acceptable for prototyping
- Focus on correctness over performance

### Phase 1: Production — C/Rust Redis Module

Native module provides:
- **Zero-copy** Roaring bitmap access
- **Native hashing** — MurmurHash64A in module, no roundtrip
- **SIMD cardinality** — Vectorized HLL++ estimation
- **Single command** — `HLLSET.CREATE key token1 token2 ...`
- **Atomic operations** — Thread-safe batch ingestion

## Architectural Rationale

### Why Redis-Centric?

1. **Natural data model fit** — HLLSet registers are bitmaps; Redis Roaring provides native bitwise ops (OR/AND/XOR/ANDNOT) mapping directly to union/intersect/xor/diff.

2. **Content-addressing is first-class** — Redis keys are strings, SHA1 is a string. `hllset:{sha1}` gives automatic deduplication and global namespace.

3. **Lattice/DAG structures** — Content-addressed nodes with parent relationships map naturally to Redis data structures.

4. **Distributed by design** — Multiple processes/notebooks share the same HLLSet universe without serialization overhead.

5. **Persistence without ceremony** — RDB/AOF durability; global registries survive restarts.

### Architecture Pattern

```
┌─────────────────────────────────────────────────────────────┐
│  Python Client (thin orchestration layer)                   │
│  - HLLSetRedis: API-compatible wrapper                      │
│  - High-level workflows, analysis                           │
│  - (POC: hash computation; Prod: delegated to module)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Redis Server (single source of truth)                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  HLLSet Module (C/Rust) — Production                │   │
│  │  - HLLSET.CREATE / HLLSET.UNION / HLLSET.CARD      │   │
│  │  - Native MurmurHash64A                             │   │
│  │  - Direct Roaring bitmap manipulation               │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Lua Functions — POC/Fallback                       │   │
│  │  - hllset.create / hllset.union / hllset.card      │   │
│  └─────────────────────────────────────────────────────┘   │
│  - Roaring Bitmaps: HLLSet storage                          │
│  - Hash: metadata, provenance                               │
│  - Graph: lattice relationships (future)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Container Runtime

**Using Podman on Fedora** (rootless containers):

```bash
# Start Redis
./redis/scripts/start_redis.sh

# Or with podman-compose
cd redis && podman-compose up -d

# Load functions
./redis/scripts/load_functions.sh
```

---

## Phase 1: HLLSet Core (Current Focus)

### 1.1 Data Model Translation

| Python Concept | Redis Implementation |
|----------------|---------------------|
| `HLLSet.registers` (uint32[m]) | Roaring Bitmap (bit positions) |
| `HLLSet.name` (SHA1 hash) | Redis key: `hllset:{sha1}` |
| `HLLSet.p_bits` | Stored in Hash: `hllset:{sha1}:meta` |
| `HLLSet.seed` | Stored in Hash: `hllset:{sha1}:meta` |

### 1.2 Register Encoding

The bitmap register model uses positions:
```
position = register_index * 32 + bit_index
```

For `p_bits=10` (1024 registers × 32 bits):
- Total addressable positions: 0 to 32,767
- Roaring bitmap compresses sparse sets efficiently

### 1.3 Operations to Implement

| Operation | Python | Redis/Lua |
|-----------|--------|-----------|
| Create from tokens | `HLLSet.from_batch()` | `hllset.create(tokens)` |
| Union | `hll.union(other)` | `R.BITOR(bm1, bm2)` |
| Intersection | `hll.intersect(other)` | `R.BITAND(bm1, bm2)` |
| Difference | `hll.diff(other)` | `R.BITANDNOT(bm1, bm2)` |
| XOR | `hll.xor(other)` | `R.BITXOR(bm1, bm2)` |
| Cardinality | `hll.cardinality()` | Custom Lua (highest_set_bit) |
| Jaccard | `hll.similarity()` | Lua: `|A∩B| / |A∪B|` |
| Content ID | SHA1 of registers | SHA1 in Lua |

---

## Phase 2: Supporting Modules

### 2.1 HLLTensor (`hll_tensor.py`)
- 2D view of registers (2^p × 32)
- Redis: Store as JSON or nested Hash
- Operations: inscription, active positions

### 2.2 HLLLattice (`hll_lattice.py`)
- Content-addressed DAG
- Redis: Graph structure using RedisGraph
- Nodes: HLLSet keys
- Edges: subset/superset relationships

### 2.3 Global Registry (`global_registry.py`)
- Universe sets G₁, G₂, G₃
- Redis: Persistent HLLSets with special keys
- `hllset:registry:G1`, `hllset:registry:G2`, etc.

### 2.4 BSS (Bell State Similarity)
- Directed similarity metric
- Redis: Computed on-demand via Lua

---

## Phase 3: Advanced Features

### 3.1 Disambiguation Engine
- Token recovery from fingerprints
- Redis: Inverted index via RedisSearch

### 3.2 De Bruijn Graphs
- Sequence reconstruction
- Redis: RedisGraph for edge storage

### 3.3 Markov/Bayesian
- Probability distributions over HLLSets
- Redis: Hash structures for CPTs

---

## Implementation Strategy

### Step 1: Create Lua Function Library

Location: `redis/functions/hllset.lua`

```lua
#!lua name=hllset

-- Constants
local P_BITS = 10
local M = 2^P_BITS  -- 1024 registers
local SEED = 42

-- MurmurHash64A implementation
local function murmur_hash64(data, seed)
    -- Implementation matching Python/Cython
end

-- Create HLLSet from tokens
local function create(keys, args)
    -- Hash tokens, set bits in roaring bitmap
end

-- Register functions
redis.register_function('hllset_create', create)
redis.register_function('hllset_union', union)
redis.register_function('hllset_intersect', intersect)
-- etc.
```

### Step 2: Python Redis Client Wrapper

Location: `core/hllset_redis.py`

```python
class HLLSetRedis:
    """Redis-backed HLLSet with same API as HLLSet."""
    
    def __init__(self, redis_client, key=None):
        self.redis = redis_client
        self.key = key
    
    @classmethod
    def from_batch(cls, tokens, redis_client):
        # Call Lua function, get key back
        key = redis_client.fcall('hllset_create', 0, *tokens)
        return cls(redis_client, key)
```

### Step 3: Gradual Migration

1. Run both implementations in parallel
2. Compare results (shadow mode)
3. Switch to Redis as primary
4. Deprecate Python implementation

---

## Key Files Structure

```
redis/
├── Dockerfile              # Redis with modules
├── redis.conf              # Configuration
├── functions/
│   ├── hllset.lua         # Core HLLSet operations
│   ├── hll_tensor.lua     # Tensor view operations
│   ├── bss.lua            # Bell State Similarity
│   └── registry.lua       # Global registry
├── scripts/
│   └── load_functions.sh  # Deploy functions to Redis
└── tests/
    └── test_hllset.lua    # Lua test suite

core/
├── hllset.py              # Original (to be deprecated)
├── hllset_redis.py        # Redis-backed implementation
└── ...
```

---

## Testing Strategy

1. **Unit Tests**: Test each Lua function independently
2. **Parity Tests**: Compare Redis vs Python results
3. **Performance Tests**: Benchmark operations
4. **Integration Tests**: Full workflow testing

---

## Success Criteria

- [ ] All HLLSet operations work in Redis
- [ ] SHA1-based content addressing works
- [ ] Cardinality estimation matches Python (±5%)
- [ ] Set operations are mathematically correct
- [ ] Performance is comparable or better
- [ ] Existing tutorials work with minimal changes

---

## Timeline Estimate

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1.1 | 1 week | Lua library for HLLSet |
| Phase 1.2 | 1 week | Python Redis wrapper |
| Phase 1.3 | 1 week | Testing & validation |
| Phase 2 | 2-3 weeks | Supporting modules |
| Phase 3 | 2-3 weeks | Advanced features |

---

## Notes

- Redis Roaring Bitmap module provides: `R.SETBIT`, `R.GETBIT`, `R.BITOR`, `R.BITAND`, `R.BITANDNOT`, `R.BITXOR`, `R.BITCOUNT`
- SHA1 is available via `redis.sha1hex()` in Lua
- MurmurHash64A must be implemented in Lua (matching Python exactly)
- Cardinality estimation requires HLL++ algorithm in Lua
