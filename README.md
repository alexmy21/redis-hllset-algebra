# redis-hllset-algebra

**HLLSet Algebra implemented as Lua Libraries in Redis**

Set-algebra operations (union, intersection, difference, symmetric difference) for Redis [HyperLogLog](https://redis.io/docs/data-types/probabilistic/hyperloglogs/) keys.  
All heavy computation runs atomically inside Redis as Lua scripts — only one round-trip per operation.

> **Note**: HyperLogLog is a *probabilistic* data structure.  
> Results are cardinality **estimates** with a typical error ≤ 0.81 %.

---

## Installation

```bash
pip install redis-hllset-algebra          # from PyPI
# or directly from source
pip install .
```

**Dependencies**: `redis>=4.0`  
**Optional (tests)**: `fakeredis[lua]>=2.0`, `pytest>=7`

---

## Quick start

```python
import redis
from redis_hllset_algebra import HLLSetAlgebra

r   = redis.Redis()
hll = HLLSetAlgebra(r)

# Populate two HLL keys
hll.add("users:2024", "alice", "bob", "carol")
hll.add("users:2025", "bob", "carol", "dave")

# Union – stores result in "users:all", returns cardinality ≈ 4
hll.union("users:all", "users:2024", "users:2025")

# Union cardinality without storing
hll.union_card("users:2024", "users:2025")        # ≈ 4

# Intersection  users:2024 ∩ users:2025  ≈ 2  (bob, carol)
hll.intersect_card("users:2024", "users:2025")

# Difference  users:2025 \ users:2024  ≈ 1  (dave)
hll.diff_card("users:2025", "users:2024")

# Symmetric difference  users:2024 △ users:2025  ≈ 2  (alice, dave)
hll.symmdiff_card("users:2024", "users:2025")

# Subset test (probabilistic)
hll.is_subset("users:2024", "users:all")          # True
```

---

## API reference

### `HLLSetAlgebra(redis_client, tmp_key_prefix="__hllset_tmp__:")`

Creates a new instance bound to *redis_client* (any `redis.Redis`-compatible object).  
`tmp_key_prefix` controls the namespace used for intermediate keys that are created and immediately deleted inside the Lua scripts.

---

| Method | Description |
|--------|-------------|
| `add(key, *elements)` | Add elements to the HLL at *key* (`PFADD`). Returns 1 if the internal representation changed, 0 otherwise. |
| `count(*keys)` | Return approximate cardinality of the union of *keys* (`PFCOUNT`). |
| `union(dest, *sources)` | Merge *sources* into *dest* (`PFMERGE`) and return the cardinality. |
| `union_card(*keys)` | Return cardinality of the union without storing anything. |
| `intersect_card(*keys)` | Return estimated cardinality of the intersection using the inclusion-exclusion principle. |
| `diff_card(key_a, key_b)` | Return estimated cardinality of `A \ B` using `\|A ∪ B\| − \|B\|`. |
| `symmdiff_card(key_a, key_b)` | Return estimated cardinality of `A △ B` using `2·\|A ∪ B\| − \|A\| − \|B\|`. |
| `is_subset(key_a, key_b)` | Return `True` if `diff_card(key_a, key_b) == 0` (probabilistic). |

---

## Mathematical foundations

| Operation | Formula used |
|-----------|-------------|
| Union cardinality | `PFCOUNT` natively |
| Intersection cardinality | Inclusion-exclusion: `Σ (-1)^(\|S\|+1) · \|⋃_{i∈S} Aᵢ\|` |
| Difference cardinality | `\|A ∪ B\| − \|B\|` |
| Symmetric difference cardinality | `2·\|A ∪ B\| − \|A\| − \|B\|` |

The inclusion-exclusion computation for *n* keys requires 2ⁿ − 1 union evaluations.  
For practical performance, keep *n* ≤ 4 when calling `intersect_card`.

---

## Lua scripts

The scripts live in `redis_hllset_algebra/scripts/` and can also be loaded and called directly via `EVAL`/`EVALSHA`:

| Script | KEYS | ARGV | Returns |
|--------|------|------|---------|
| `hll_union.lua` | `[dest, src1, ...]` | — | cardinality of the union |
| `hll_intersect_card.lua` | `[key1, key2, ...]` | `[tmp_key_prefix]` | estimated intersection cardinality |
| `hll_diff_card.lua` | `[key_a, key_b]` | `[tmp_key]` | estimated `A \ B` cardinality |
| `hll_symmdiff_card.lua` | `[key_a, key_b]` | `[tmp_key]` | estimated `A △ B` cardinality |

All scripts clean up any temporary keys they create before returning.

---

## Running the tests

```bash
pip install ".[test]"
pytest
```

---

## License

MIT — see [LICENSE](LICENSE).
