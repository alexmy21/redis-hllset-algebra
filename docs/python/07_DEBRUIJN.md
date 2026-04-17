# De Bruijn Graph Modules

> Sequence reconstruction from overlapping k-mers and HLLSet evolution.

**Modules**: `core.debruijn`, `core.hllset_debruijn`  
**Layer**: L3/L4 — Sequence Recovery

## Part 1: Generic De Bruijn Graph

### Overview

A De Bruijn graph represents overlapping k-mers as a directed graph:

- **Nodes**: (k-1)-mers (prefixes/suffixes)
- **Edges**: k-mers, connecting prefix → suffix
- **Multiplicities**: How many times each k-mer appears

### Applications

- Token order restoration from n-gram tensors
- Genome assembly from DNA reads
- Text reconstruction from overlapping fragments

### DeBruijnGraph Class

```python
from core.debruijn import DeBruijnGraph, Edge, PathResult

# Create graph with k-mer size
graph = DeBruijnGraph(k=3)  # Trigrams: nodes are bigrams
```

### Adding K-mers

```python
# Add trigram: ("quick", "brown", "fox")
# Creates edge: ("quick", "brown") → ("brown", "fox"), label="fox"
graph.add_kmer(("quick", "brown", "fox"))

# Add with multiplicity (for repeated patterns)
graph.add_kmer(("the", "quick", "the"), count=2)  # Appears twice

# Batch add
kmers = [
    ("START", "the", "quick"),
    ("the", "quick", "brown"),
    ("quick", "brown", "fox"),
    ("brown", "fox", "END"),
]
graph.add_kmers(kmers)
```

### Edge Structure

```python
from core.debruijn import Edge

@dataclass
class Edge:
    source: Tuple[str, ...]    # (k-1)-mer: ("quick", "brown")
    target: Tuple[str, ...]    # (k-1)-mer: ("brown", "fox")
    label: str                 # k-th element: "fox"
    kmer: Tuple[str, ...]      # Full k-mer: ("quick", "brown", "fox")
    multiplicity: int          # Edge weight (default 1)
```

### Graph Properties

```python
graph.k               # → 3 (k-mer size)
graph.nodes           # → set of (k-1)-mer tuples
graph.num_nodes       # → count
graph.num_edges       # → count (unique k-mers)
graph.total_edges     # → sum of multiplicities
```

### Finding Eulerian Paths

An **Eulerian path** visits each edge exactly as many times as its multiplicity.

```python
# Find path from specific start node
path_result = graph.find_eulerian_path(start_prefix=("START", "the"))

# Or auto-detect start (node with out_degree > in_degree)
path_result = graph.find_eulerian_path()

# PathResult structure
print(path_result.path)       # List of nodes visited
print(path_result.sequence)   # Reconstructed sequence
print(path_result.is_eulerian)  # True if all edges used
print(path_result.edges_used)   # Edges traversed
```

### Sequence Reconstruction

```python
# Full workflow: k-mers → sequence
graph = DeBruijnGraph(k=3)
trigrams = [
    ("<S>", "the", "quick"),
    ("the", "quick", "brown"),
    ("quick", "brown", "fox"),
    ("brown", "fox", "</S>"),
]
graph.add_kmers(trigrams)

result = graph.find_eulerian_path()
sequence = result.sequence
# → ["<S>", "the", "quick", "brown", "fox", "</S>"]

# Remove markers
clean = [t for t in sequence if t not in ("<S>", "</S>")]
# → ["the", "quick", "brown", "fox"]
```

### Handling Branches

When multiple paths exist:

```python
# Get all possible continuations from a node
continuations = graph.successors(("the", "quick"))
# → [("quick", "brown"), ("quick", "jump")]

# Find all paths (for small graphs)
all_paths = graph.find_all_paths(
    start=("START", "a"),
    end=("z", "END"),
    max_paths=100
)
```

### Utility Functions

```python
from core.debruijn import (
    build_debruijn_from_sequence,
    build_debruijn_from_kmers,
    restore_sequence_debruijn,
)

# From raw token sequence
graph = build_debruijn_from_sequence(
    ["the", "quick", "brown", "fox"],
    k=3,
    add_markers=True
)

# From k-mer list
graph = build_debruijn_from_kmers(trigram_list, k=3)

# Quick reconstruction
sequence = restore_sequence_debruijn(trigram_list)
```

---

## Part 2: HLLSet De Bruijn Graph

### Overview

Extends De Bruijn graphs to **HLLSet evolution**:

- **Nodes**: HLLSet states
- **Edges**: (D, R, N) transformation triples
- **Path**: Evolution order reconstruction

```
Token-Level:   nodes = bigrams,    edges = trigrams
HLLSet-Level:  nodes = HLLSets,    edges = (D, R, N)
```

### DRN Triple

The (D, R, N) triple decomposes state transitions:

```python
from core.hllset_debruijn import DRNTriple

# For transition: hll_1 → hll_2
# Decomposition: hll_2 = (hll_1 \ D) ∪ R ∪ N

@dataclass
class DRNTriple(NamedTuple):
    deleted: HLLSet   # D = hll_1 \ hll_2 (removed)
    retained: HLLSet  # R = hll_1 ∩ hll_2 (kept)
    novel: HLLSet     # N = hll_2 \ hll_1 (added)
```

### Computing DRN

```python
from core.hllset_debruijn import decompose_transition

hll_before = HLLSet.from_batch(["a", "b", "c"])
hll_after = HLLSet.from_batch(["b", "c", "d"])

drn = decompose_transition(hll_before, hll_after)

print(f"Deleted: ~{drn.deleted_card:.0f}")   # a
print(f"Retained: ~{drn.retained_card:.0f}") # b, c
print(f"Novel: ~{drn.novel_card:.0f}")       # d
print(f"Net change: {drn.net_change():.0f}") # 0
```

### Evolution Phases

```python
from core.hllset_debruijn import EvolutionPhase, classify_transition

class EvolutionPhase(Enum):
    GROWTH = "growth"       # |N| > |D|
    DECAY = "decay"         # |D| > |N|
    STABLE = "stable"       # |N| ≈ |D|
    REPLACEMENT = "replace" # High |D| and |N|, low |R|

phase = classify_transition(drn)
print(f"Phase: {phase}")
```

### HLLSetEdge

```python
from core.hllset_debruijn import HLLSetEdge

@dataclass
class HLLSetEdge:
    source_idx: int       # Index in HLLSet list
    target_idx: int       # Index in HLLSet list
    drn: DRNTriple        # Transformation triple
    bss: BSSPair          # BSS metrics for transition
    phase: EvolutionPhase
```

### HLLSetDeBruijnGraph

```python
from core.hllset_debruijn import HLLSetDeBruijnGraph

# Build from HLLSet collection
hllsets = [hll1, hll2, hll3, hll4]
graph = HLLSetDeBruijnGraph(hllsets)

# Or build incrementally
graph = HLLSetDeBruijnGraph()
graph.add_state(hll1)
graph.add_state(hll2)
```

### Finding Evolution Path

```python
# Build graph with BSS-based adjacency
graph = build_hllset_debruijn(hllsets, tau_threshold=0.5)

# Find evolution order
path = find_evolution_path(graph, start_idx=0)

for edge in path:
    print(f"State {edge.source_idx} → {edge.target_idx}")
    print(f"  Phase: {edge.phase}")
    print(f"  Net change: {edge.drn.net_change():.0f}")
```

### Full Token Recovery

```python
from core.hllset_debruijn import recover_tokens_from_drn

# Recover tokens from each DRN component
for edge in evolution_path:
    deleted_tokens = recover_tokens_from_drn(edge.drn.deleted, engine)
    novel_tokens = recover_tokens_from_drn(edge.drn.novel, engine)
    
    print(f"Step {edge.source_idx} → {edge.target_idx}:")
    print(f"  Deleted: {deleted_tokens}")
    print(f"  Added: {novel_tokens}")
```

### FullDRNTriple

Complete invertible transformation with token recovery:

```python
from core.hllset_debruijn import FullDRNTriple, full_decompose_transition

full_drn = full_decompose_transition(
    hll_before, 
    hll_after,
    disambiguation_engine
)

print(f"Deleted tokens: {full_drn.deleted_tokens}")
print(f"Novel tokens: {full_drn.novel_tokens}")

# Round-trip verification
assert verify_reconstruction(hll_before, hll_after, full_drn)
```

### Evolution Analysis

```python
from core.hllset_debruijn import analyze_evolution, EvolutionSummary

summary = analyze_evolution(hllsets)

print(f"Total steps: {summary.num_steps}")
print(f"Net growth: {summary.total_growth:.0f}")
print(f"Phases: {summary.phase_counts}")
# → {"growth": 3, "decay": 1, "stable": 2}
```

## Architecture Diagram

```
    HLLSet Collection (unordered)
              │
              ▼ BSS adjacency graph
    ┌─────────────────────────┐
    │  HLLSet De Bruijn Graph │
    │    Nodes = HLLSet states│
    │    Edges = (D, R, N)    │
    └─────────────────────────┘
              │
              ▼ Find evolution path
    ┌─────────────────────────┐
    │  Ordered Lattice        │
    │    H₀ → H₁ → H₂ → ...   │
    └─────────────────────────┘
              │
              ▼ For each edge (D, R, N)
    ┌─────────────────────────┐
    │  Token-Level De Bruijn  │
    │    D → deleted tokens   │
    │    N → novel tokens     │
    └─────────────────────────┘
```

## Related Modules

- [Disambiguation](04_DISAMBIGUATION.md) — Token recovery
- [BSS/Noether](05_BSS_NOETHER.md) — Transition metrics
- [Lattice](06_LATTICE_STORE.md) — Temporal structure
- [Dynamics](08_DYNAMICS.md) — System monitoring
