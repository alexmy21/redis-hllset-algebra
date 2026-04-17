# Disambiguation Module

> Recover tokens from HLLSet fingerprints via triangulation.

**Module**: `core.disambiguation`  
**Layer**: L3 — Token Recovery

## Overview

The disambiguation module recovers original tokens from HLLSet fingerprints using:

1. **Multi-layer n-gram storage** (unigrams, bigrams, trigrams)
2. **Triangulation** across layers for confidence
3. **De Bruijn graphs** for sequence reconstruction
4. **Parallel register disambiguation**

## Architecture

```
Input Text → tokenize → n-grams → hash → (reg, zeros, layer)
                                              ↓
                                    TriangulationTensor
                                              ↓
HLLSet → active positions → lookup → candidates → triangulate → tokens
                                              ↓
                                      De Bruijn Graph
                                              ↓
                                  Eulerian Path → Sequence
```

## Constants

```python
from core.disambiguation import NUM_LAYERS, START_TOKEN, END_TOKEN

NUM_LAYERS = 3        # unigram (0), bigram (1), trigram (2)
START_TOKEN = "<S>"   # Sequence start marker
END_TOKEN = "</S>"    # Sequence end marker
```

## TokenEntry Class

```python
from core.disambiguation import TokenEntry

# Create from n-gram tuple
entry = TokenEntry.from_ntoken(("quick", "brown"), p_bits=10, seed=42)

# Properties
entry.token        # → ("quick", "brown")
entry.token_str    # → "quick brown"
entry.hash_full    # → 64-bit hash
entry.reg          # → register index
entry.zeros        # → trailing zeros count
entry.layer        # → 1 (bigram)
entry.first_token  # → "quick" (for triangulation)
entry.position     # → (reg, zeros) tuple
```

## DisambiguationEngine

The main API for token ingestion and recovery.

### Creation

```python
from core.disambiguation import DisambiguationEngine

engine = DisambiguationEngine(p_bits=10, seed=42)
```

### Ingestion

```python
# Ingest raw text
engine.ingest_text("The quick brown fox jumps over the lazy dog")

# Ingest tokens directly
tokens = ["quick", "brown", "fox"]
engine.ingest_tokens(tokens)

# Ingest with explicit layers
engine.ingest_unigrams(["quick", "brown", "fox"])
engine.ingest_bigrams([("quick", "brown"), ("brown", "fox")])
engine.ingest_trigrams([("quick", "brown", "fox")])
```

### Disambiguation

```python
from core import HLLSet

# Create query HLLSet
query = HLLSet.from_batch(["quick", "brown"])

# Get candidates at all active positions
results = engine.disambiguate(query)

for result in results:
    print(f"Position ({result.reg}, {result.zeros}):")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Method: {result.method}")
    for candidate in result.candidates:
        print(f"    - {candidate.token_str}")
```

### DisambiguationResult

```python
@dataclass
class DisambiguationResult:
    reg: int                      # Register index
    zeros: int                    # Trailing zeros
    candidates: List[TokenEntry]  # Matching tokens
    confidence: float             # Confidence score [0, 1]
    method: str                   # "exact", "triangulated", "best_guess"
```

## Triangulation

Triangulation uses multiple n-gram layers to resolve ambiguity:

```python
# Position (42, 5) has candidates: ["apple", "app", "application"]
# - "apple" appears in layer 0 (unigram) AND layer 1 (bigram "apple pie")
# - "app" appears only in layer 0
# - "application" appears in layers 0, 1, AND 2

# Triangulation picks "application" (appears in most layers)
```

### Manual Triangulation

```python
# Get layer-specific candidates
layer0 = engine.lookup_position(reg=42, zeros=5, layer=0)  # unigrams
layer1 = engine.lookup_position(reg=42, zeros=5, layer=1)  # bigrams
layer2 = engine.lookup_position(reg=42, zeros=5, layer=2)  # trigrams

# Find common first_tokens
first_tokens_0 = {e.first_token for e in layer0}
first_tokens_1 = {e.first_token for e in layer1}
first_tokens_2 = {e.first_token for e in layer2}

common = first_tokens_0 & first_tokens_1 & first_tokens_2
print(f"Triangulated tokens: {common}")
```

## ParallelDisambiguator

Register-parallel disambiguation with De Bruijn reconstruction.

```python
from core.disambiguation import ParallelDisambiguator

# Create with existing engine
parallel = ParallelDisambiguator(engine, n_workers=4)

# Disambiguate in parallel
results = parallel.disambiguate(query_hllset)

# Get reconstructed sequence
sequence = parallel.reconstruct_sequence(results)
print(f"Recovered: {' '.join(sequence)}")
```

### RegisterDisambiguationResult

```python
@dataclass
class RegisterDisambiguationResult:
    register: int
    positions: List[DisambiguationResult]  # All zeros values
    best_sequence: List[str]               # Reconstructed tokens
    debruijn_used: bool                    # Was De Bruijn needed?
```

## De Bruijn Reconstruction

When multiple tokens share positions, De Bruijn graphs restore order:

```python
from core.disambiguation import restore_sequence_debruijn

# Trigram candidates at a register
trigrams = [
    ("quick", "brown", "fox"),
    ("brown", "fox", "jumps"),
    ("fox", "jumps", "over"),
]

# Build De Bruijn graph and find Eulerian path
sequence = restore_sequence_debruijn(trigrams)
# → ["quick", "brown", "fox", "jumps", "over"]
```

## Layer Filtering

```python
# Filter by layer during disambiguation
results = engine.disambiguate(query, layer_filter=[0, 1])  # Only uni/bigrams

# Get layer statistics
stats = engine.layer_statistics()
# → {0: 1500, 1: 2800, 2: 3500}  # entries per layer
```

## Global Layer HLLSets

For fast pre-filtering, maintain HLLSets per layer:

```python
# Build global layer sets during ingestion
engine.build_global_sets()

# Check if position COULD have layer-2 entries
if engine.position_in_layer(reg=42, zeros=5, layer=2):
    # Worth doing expensive lookup
    results = engine.lookup_position(reg=42, zeros=5, layer=2)
```

## Collision Handling

Hash collisions are handled via candidate lists:

```python
# Multiple tokens at same position
candidates = engine.lookup_position(reg=42, zeros=5)
if len(candidates) > 1:
    print(f"Collision: {[c.token_str for c in candidates]}")
    
    # Use triangulation to resolve
    resolved = engine.triangulate_candidates(candidates)
```

## Workflow: Full Text Recovery

```python
from core import HLLSet
from core.disambiguation import DisambiguationEngine

# 1. Build corpus index
engine = DisambiguationEngine()
for doc in corpus:
    engine.ingest_text(doc)
engine.build_global_sets()

# 2. Create query fingerprint
query_text = "The quick brown fox"
query_hll = HLLSet.from_batch(query_text.split())

# 3. Disambiguate
results = engine.disambiguate(query_hll)

# 4. Collect high-confidence tokens
recovered = []
for r in results:
    if r.confidence > 0.8 and r.candidates:
        recovered.append(r.candidates[0].token_str)

# 5. Reconstruct sequence using De Bruijn
final_sequence = engine.reconstruct_sequence(recovered)
print(f"Recovered: {' '.join(final_sequence)}")
```

## Performance

| Operation | Complexity |
|-----------|------------|
| `ingest_text()` | O(n) — n = tokens |
| `lookup_position()` | O(k) — k = entries |
| `disambiguate()` | O(p × k) — p = positions |
| `triangulate()` | O(k × L) — L = layers |
| De Bruijn path | O(e) — e = edges |

## Related Modules

- [HLLTensor](03_TENSOR_LUT.md) — Position encoding
- [De Bruijn](07_DEBRUIJN.md) — Sequence reconstruction
- [Redis Disambiguation](10_REDIS_MODULES.md) — Distributed disambiguation
