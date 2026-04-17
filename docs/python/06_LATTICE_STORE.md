# Lattice and Store Modules

> Temporal W lattice and HLLSet storage with derivation tracking.

**Modules**: `core.hll_lattice`, `core.hllset_store`  
**Layer**: L4/L6 — Structure and Persistence

## Part 1: HLL Lattice (W Lattice)

### Overview

The W Lattice provides a **temporal structure** for HLLSet observations:

- **Partial order**: A ≤ B ⟺ A ⊆ B (bitwise inclusion)
- **Distributive lattice**: meet = AND, join = OR
- **Content-addressed nodes**: SHA1 of merged registers
- **Temporal ordering**: causally linked snapshots

### Lattice Partial Order

```
A ≤ B  ⟺  (R_A AND NOT R_B) = 0

This means: every bit set in A is also set in B.
```

### LatticeNode

```python
from core.hll_lattice import LatticeNode

@dataclass(frozen=True)
class LatticeNode:
    node_id: str              # SHA1 content hash
    timestamp: float          # Creation time
    merged: HLLSet            # Union of components
    cardinality: float        # |merged|
    popcount: int             # Total set bits
    component_ids: Tuple[str, ...]  # Contributing HLLSet IDs
    parent_ids: Tuple[str, ...]     # Causal predecessors
    metadata: Dict[str, Any]        # Application data
    
    def is_subset_of(self, other: 'LatticeNode') -> bool:
        """Check if self ⊆ other in lattice order."""
```

### HLLLattice

```python
from core.hll_lattice import HLLLattice, InMemoryStorage

# Create lattice with default in-memory storage
lattice = HLLLattice()

# Or with custom storage
storage = InMemoryStorage()
lattice = HLLLattice(storage=storage)
```

### Adding Observations

```python
from core import HLLSet

# Single HLLSet observation
hll1 = HLLSet.from_batch(["a", "b", "c"])
node1 = lattice.observe(hll1, timestamp=1.0)

# Multiple HLLSets merged into one node
hll2 = HLLSet.from_batch(["d", "e"])
hll3 = HLLSet.from_batch(["f", "g"])
node2 = lattice.observe_multiple([hll2, hll3], timestamp=2.0)

# With parent linkage (causal chain)
node3 = lattice.observe(
    hll4, 
    timestamp=3.0, 
    parents=[node1.node_id, node2.node_id]
)
```

### Querying the Lattice

```python
# Get node by ID
node = lattice.get(node_id)

# Get all nodes
all_nodes = lattice.all_nodes()

# Get root nodes (no parents)
roots = lattice.roots()

# Get leaf nodes (no children)
leaves = lattice.leaves()

# Find ancestors/descendants
ancestors = lattice.ancestors(node_id)
descendants = lattice.descendants(node_id)
```

### Lattice Operations

```python
# Meet (greatest lower bound) = intersection
meet = lattice.meet(node_a, node_b)

# Join (least upper bound) = union  
join = lattice.join(node_a, node_b)

# Check ordering
if lattice.is_below(node_a, node_b):
    print(f"{node_a.node_id} ⊆ {node_b.node_id}")
```

### Temporal Queries

```python
# Nodes in time range
recent = lattice.nodes_in_range(start_time=100.0, end_time=200.0)

# Latest node
latest = lattice.latest_node()

# Temporal chain (path through time)
chain = lattice.temporal_chain(start_node_id, end_node_id)
```

### Storage Protocol

```python
from core.hll_lattice import LatticeStorage

class LatticeStorage(Protocol):
    def store_node(self, node: LatticeNode) -> None: ...
    def load_node(self, node_id: str) -> Optional[LatticeNode]: ...
    def list_node_ids(self) -> List[str]: ...
    def node_count(self) -> int: ...
```

Custom storage backends (SQLite, Redis, etc.) implement this protocol.

---

## Part 2: HLLSet Store

### Overview

HLLSetStore provides **persistent storage** with:

- **Base HLLSets**: Stored as serialized bitmaps
- **Compound HLLSets**: Generated on-the-fly via derivation LUT
- **Derivation tracking**: Full operation history

### Design Principle

> "Store only base HLLSets; reconstruct compounds from LUT."

```
H₄ = H₁ ∪ H₂  →  LUT["H₄"] = ("union", ["H₁", "H₂"])
H₅ = H₁ ⊕ H₃  →  LUT["H₅"] = ("xor", ["H₁", "H₃"])
```

### HLLSetStore

```python
from core.hllset_store import HLLSetStore, InMemoryBackend

# Create with in-memory backend
store = HLLSetStore()

# Or with custom backend
backend = InMemoryBackend()
store = HLLSetStore(backend=backend)
```

### Registering Base HLLSets

```python
from core import HLLSet

hll1 = HLLSet.from_batch(["apple", "banana"])
hll2 = HLLSet.from_batch(["cherry", "date"])

# Register bases
id1 = store.register_base(hll1, source="doc1", tags=["fruit"])
id2 = store.register_base(hll2, source="doc2", tags=["fruit"])

print(f"Registered: {id1[:8]}...")  # SHA1 ID
```

### Compound Operations

```python
# Operations create LUT entries (not stored HLLSets)
id3 = store.union(id1, id2)       # LUT: id3 → ("union", [id1, id2])
id4 = store.intersect(id1, id2)   # LUT: id4 → ("intersect", [id1, id2])
id5 = store.diff(id1, id2)        # LUT: id5 → ("diff", [id1, id2])
id6 = store.xor(id1, id2)         # LUT: id6 → ("xor", [id1, id2])
```

### Retrieval

```python
# Get any HLLSet (base or compound)
hll = store.get(id3)  # Reconstructs on-the-fly if compound

# Check if exists
exists = store.exists(id1)

# Get derivation info
deriv = store.get_derivation(id3)
print(deriv.operation)  # Operation.UNION
print(deriv.operands)   # (id1, id2)
```

### Derivation Class

```python
from core.hllset_store import Derivation, Operation

@dataclass
class Derivation:
    operation: Operation        # BASE, UNION, INTERSECT, DIFF, XOR
    operands: Tuple[str, ...]  # Input HLLSet IDs
    timestamp: float           # When created
    metadata: Dict[str, Any]   # Extra info
    
    def is_base(self) -> bool:
        """True if this is a base HLLSet."""
```

### Querying

```python
# List all IDs
all_ids = store.list_ids()

# List only bases
base_ids = store.list_base_ids()

# List only compounds
compound_ids = store.list_compound_ids()

# Query by metadata
results = store.query(source="doc1")
results = store.query(tags=["fruit"])
```

### EphemeralLattice

For complex multi-step operations within a transaction:

```python
# Use ephemeral lattice for complex derivations
with store.lattice() as lat:
    # Operations recorded to LUT on exit
    temp1 = lat.union(id1, id2)
    temp2 = lat.intersect(temp1, id3)
    final = lat.xor(temp2, id4)
    
# final is now tracked in LUT
```

### HLLSetLUT

Direct access to the derivation lookup table:

```python
from core.hllset_store import HLLSetLUT

lut = store.lut

# Add derivation manually
lut.add(target_id, Derivation(
    operation=Operation.UNION,
    operands=(id1, id2),
))

# Get derivation
deriv = lut.get(target_id)

# Check if compound
is_compound = lut.is_compound(target_id)

# Get full derivation tree
tree = lut.derivation_tree(target_id)
```

### Storage Backend Protocol

```python
from core.hllset_store import StorageBackend

class StorageBackend(Protocol):
    def get(self, key: str) -> Optional[bytes]: ...
    def put(self, key: str, value: bytes) -> None: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...
    def keys(self, prefix: str = "") -> Iterator[str]: ...
```

Implement this for custom backends (RocksDB, S3, etc.).

## Workflow: Corpus Storage

```python
from core import HLLSet
from core.hllset_store import HLLSetStore

store = HLLSetStore()

# 1. Ingest documents as base HLLSets
doc_ids = {}
for doc_name, doc_text in documents.items():
    hll = HLLSet.from_batch(doc_text.split())
    doc_id = store.register_base(hll, source=doc_name)
    doc_ids[doc_name] = doc_id

# 2. Compute corpus-wide union
corpus_id = doc_ids["doc1"]
for doc_name, doc_id in list(doc_ids.items())[1:]:
    corpus_id = store.union(corpus_id, doc_id)

# 3. Query: what's unique to doc1?
unique_to_doc1 = store.diff(doc_ids["doc1"], corpus_id)

# 4. Retrieve and analyze
unique_hll = store.get(unique_to_doc1)
print(f"Unique tokens in doc1: ~{unique_hll.cardinality():.0f}")
```

## Related Modules

- [HLLSet](01_HLLSET.md) — Core HLLSet class
- [Ring Algebra](02_RING_ALGEBRA.md) — XOR decomposition
- [Redis Store](10_REDIS_MODULES.md) — Redis-backed storage
- [BSS/Noether](05_BSS_NOETHER.md) — Conservation monitoring
