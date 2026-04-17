# De Bruijn Graph Design Document

## Overview

This document describes the design for a **dual-mode De Bruijn graph system** that:

1. Runs server-side in Rust (no Python round-trip)
2. Can be invoked from Python for flexible processing
3. Integrates with HLLSet disambiguation for full text restoration
4. Supports corpus-level "Micro-LLM" / Markov / Bayesian interpretation

## 1. Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  De Bruijn System Components                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ Rust Module  │    │ RedisGraph   │    │  Python API  │                   │
│  │ (Commands)   │◄──►│ (Storage)    │◄──►│  (Client)    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                    │                          │
│         ▼                   ▼                    ▼                          │
│  HLLSET.DEBRUIJN.*   Graph Cypher         DeBruijnRedis                     │
│                      Queries              class                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Data Storage Schema

### 2.1 Per-Document Storage

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  Document Keys                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  {ns}:doc:{hash}:hll      │  HLLSet     │  All n-grams (for similarity)     │
│  {ns}:doc:{hash}:tg       │  List       │  Ordered trigrams with counts     │
│  {ns}:doc:{hash}:meta     │  Hash       │  Metadata                         │
│                           │             │    - original_length              │
│                           │             │    - token_count                  │
│                           │             │    - created_at                   │
│                           │             │    - source_id                    │
│                                                                             │
│  Example trigram list entry: "the|cat|sat:2" (count=2 occurrences)          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Global Universe Sets (G_n)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  Global N-gram HLLSets                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  {ns}:global:G1           │  HLLSet     │  All unigrams (vocabulary)        │
│  {ns}:global:G2           │  HLLSet     │  All bigrams                      │
│  {ns}:global:G3           │  HLLSet     │  All trigrams                     │
│                                                                             │
│  {ns}:global:G1:count     │  Hash       │  Unigram → corpus frequency       │
│  {ns}:global:G2:count     │  Hash       │  Bigram → corpus frequency        │
│  {ns}:global:G3:count     │  Hash       │  Trigram → corpus frequency       │
│                                                                             │
│  {ns}:global:doc_count    │  String     │  Total document count             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 TokenLUT Enhancement (TF only)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  Enhanced TokenLUT Schema                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Existing fields:                                                           │
│    reg, zeros, hash_full, layer, collision_count                            │
│    first_tokens, first_tokens_tag, tokens                                   │
│                                                                             │
│  NEW field:                                                                 │
│    tf            │  NumericField  │  Term frequency (total occurrences)     │
│                                                                             │
│  Note: DF (document frequency) is NOT tracked at TokenLUT level             │
│  - Computationally expensive to maintain                                    │
│  - TF alone sufficient for De Bruijn edge weights                           │
│  - IDF can be computed on-demand if needed from corpus stats                │
│                                                                             │
│  FT.CREATE tokenlut:idx ON HASH PREFIX 1 tokenlut:entry:                    │
│    SCHEMA                                                                   │
│      reg NUMERIC SORTABLE                                                   │
│      zeros NUMERIC SORTABLE                                                 │
│      hash_full NUMERIC                                                      │
│      layer NUMERIC SORTABLE                                                 │
│      collision_count NUMERIC SORTABLE                                       │
│      first_tokens_tag TAG SEPARATOR ","                                     │
│      first_tokens TEXT                                                      │
│      tokens TEXT                                                            │
│      tf NUMERIC SORTABLE          # Term frequency                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Corpus De Bruijn Graph (RedisGraph)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  RedisGraph Schema                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Graph Key: {ns}:debruijn:corpus OR {ns}:debruijn:doc:{hash}                │
│                                                                             │
│  Node: :Bigram                                                              │
│  ──────────────                                                             │
│    id           │  STRING   │  "word1|word2" normalized                     │
│    w1           │  STRING   │  First word                                   │
│    w2           │  STRING   │  Second word                                  │
│    total_count  │  INT      │  Total occurrences across corpus              │
│    doc_count    │  INT      │  Number of documents containing               │
│    is_start     │  BOOL     │  Contains START marker                        │
│    is_end       │  BOOL     │  Contains END marker                          │
│                                                                             │
│  Edge: :TRIGRAM                                                             │
│  ───────────────                                                            │
│    label        │  STRING   │  Third word (edge label)                      │
│    count        │  INT      │  Total occurrences                            │
│    doc_count    │  INT      │  Document frequency                           │
│    probability  │  FLOAT    │  P(label | source_bigram) = count/out_total   │
│    log_prob     │  FLOAT    │  log(probability) for path scoring            │
│                                                                             │
│  Indexes:                                                                   │
│    CREATE INDEX ON :Bigram(id)                                              │
│    CREATE INDEX ON :Bigram(is_start)                                        │
│    CREATE INDEX ON :Bigram(is_end)                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. Rust Module Commands

### 3.1 Global G_n Management

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  HLLSET.GLOBAL.* Commands                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HLLSET.GLOBAL.ADD <namespace> <n> <ngram> [COUNT <count>]                  │
│    Add n-gram to global G_n HLLSet, optionally with count                   │
│    Example: HLLSET.GLOBAL.ADD myns 3 "the|cat|sat" COUNT 5                  │
│                                                                             │
│  HLLSET.GLOBAL.ADDMANY <namespace> <n> <ngram1> [<ngram2> ...]              │
│    Bulk add n-grams to G_n                                                  │
│                                                                             │
│  HLLSET.GLOBAL.CARD <namespace> <n>                                         │
│    Get cardinality of G_n                                                   │
│    Example: HLLSET.GLOBAL.CARD myns 3  →  12345                             │
│                                                                             │
│  HLLSET.GLOBAL.FREQ <namespace> <n> <ngram>                                 │
│    Get corpus frequency of n-gram                                           │
│    Example: HLLSET.GLOBAL.FREQ myns 3 "the|cat|sat"  →  42                  │
│                                                                             │
│  HLLSET.GLOBAL.STATS <namespace>                                            │
│    Return stats for all G_n                                                 │
│    Returns: {G1: {card, unique}, G2: {card, unique}, G3: {card, unique}}    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Enhanced INGEST (Auto G_n Update)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  Enhanced HLLSET.RING.INGEST                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HLLSET.RING.INGEST <session> <text>                                        │
│    [NGRAMS <n1> <n2> ...]     # Which n-grams to generate (default: 1 2 3)  │
│    [GLOBAL <namespace>]       # Update global G_n (default: session ns)     │
│    [TRIGRAMS <list_key>]      # Store ordered trigrams in list              │
│    [TF]                       # Update TokenLUT term frequencies            │
│    [MARKERS <start> <end>]    # Custom START/END markers                    │
│                                                                             │
│  Flow:                                                                      │
│    1. Tokenize text with START/END markers                                  │
│    2. Generate n-grams for each specified n                                 │
│    3. Create document HLLSet with all n-grams                               │
│    4. IF GLOBAL: Update G_1, G_2, G_3 HLLSets and counts                    │
│    5. IF TRIGRAMS: Store ordered trigrams with counts in list               │
│    6. IF TF: Increment term frequencies in TokenLUT entries                 │
│    7. Return document key and stats                                         │
│                                                                             │
│  Returns:                                                                   │
│    {                                                                        │
│      key: "session:basis:doc:abc123",                                       │
│      cardinality: 45,                                                       │
│      tokens: 12,                                                            │
│      ngrams: {1: 12, 2: 11, 3: 10},                                         │
│      global_updated: true                                                   │
│    }                                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 De Bruijn Commands

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  HLLSET.DEBRUIJN.* Commands                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│  GRAPH BUILDING                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  HLLSET.DEBRUIJN.BUILD <graph_key> <trigrams_key>                           │
│    [WEIGHTED]                 # Include probability weights                 │
│    [REPLACE]                  # Replace existing graph                      │
│    [MERGE]                    # Merge into existing (for corpus building)   │
│                                                                             │
│    Build De Bruijn graph from trigram list.                                 │
│    Trigram list format: ["a|b|c:count", ...] or ["a|b|c", ...]              │
│                                                                             │
│    Returns: {nodes: 15, edges: 20, graph: "debruijn:xyz"}                   │
│                                                                             │
│  HLLSET.DEBRUIJN.BUILDCORPUS <graph_key> <doc_key1> [<doc_key2> ...]        │
│    Build corpus-level graph from multiple document trigram lists            │
│    Aggregates counts, computes probabilities                                │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│  PATH FINDING                                                               │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  HLLSET.DEBRUIJN.PATH <graph_key>                                           │
│    [START <bigram>]           # Explicit start (default: find START marker) │
│    [END <bigram>]             # Explicit end (default: find END marker)     │
│    [MODE EULERIAN|GREEDY|VITERBI]  # Path algorithm                         │
│    [WEIGHTED]                 # Use probability weights (Viterbi)           │
│                                                                             │
│    Find path through De Bruijn graph.                                       │
│                                                                             │
│    MODE EULERIAN: Visit every edge exactly once (exact reconstruction)      │
│    MODE GREEDY: Follow highest probability edges                            │
│    MODE VITERBI: Most likely path using log probabilities                   │
│                                                                             │
│    Returns: ["START", "the", "cat", "sat", "END"]                           │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│  FULL RESTORATION                                                           │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  HLLSET.DEBRUIJN.RESTORE <trigrams_key>                                     │
│    [GRAPH <graph_key>]        # Reuse existing graph                        │
│    [EPHEMERAL]                # Delete graph after use (default)            │
│    [KEEP]                     # Keep graph for inspection                   │
│    [MODE EULERIAN|GREEDY|VITERBI]                                           │
│                                                                             │
│    Full pipeline: build graph → find path → return sequence                 │
│                                                                             │
│    Returns: {                                                               │
│      sequence: ["the", "cat", "sat", ...],                                  │
│      path_length: 12,                                                       │
│      is_eulerian: true,                                                     │
│      graph_key: "debruijn:temp:xyz" (if KEEP)                               │
│    }                                                                        │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│  GENERATION (Micro-LLM)                                                     │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  HLLSET.DEBRUIJN.GENERATE <graph_key>                                       │
│    [START <bigram>]           # Starting context                            │
│    [MAX <n>]                  # Maximum tokens to generate                  │
│    [TEMPERATURE <t>]          # Sampling temperature (0=greedy, 1=random)   │
│    [STOP <token>]             # Stop token (default: END marker)            │
│                                                                             │
│    Generate text using Markov random walk on corpus graph.                  │
│                                                                             │
│    Returns: ["the", "cat", "sat", "on", "the", "mat", "END"]                │
│                                                                             │
│  HLLSET.DEBRUIJN.PROBABILITY <graph_key> <sequence>                         │
│    Compute P(sequence) = ∏ P(w_i | w_{i-2}, w_{i-1})                        │
│                                                                             │
│    Returns: {probability: 0.00234, log_prob: -6.058, perplexity: 42.5}      │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│  MANAGEMENT                                                                 │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  HLLSET.DEBRUIJN.INFO <graph_key>                                           │
│    Returns graph statistics                                                 │
│                                                                             │
│  HLLSET.DEBRUIJN.DELETE <graph_key>                                         │
│    Delete De Bruijn graph                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 4. Python API

### 4.1 DeBruijnRedis Class

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import redis

@dataclass
class PathResult:
    """Result of De Bruijn path finding."""
    sequence: List[str]
    path_length: int
    is_eulerian: bool
    unused_edges: int
    log_probability: Optional[float] = None

@dataclass  
class GenerateResult:
    """Result of text generation."""
    tokens: List[str]
    text: str
    probability: float
    log_prob: float

class DeBruijnRedis:
    """
    De Bruijn graph operations with Redis backend.
    
    Supports both server-side (Rust) and client-side (Python) execution.
    """
    
    def __init__(self, client: redis.Redis, namespace: str = "default"):
        self.client = client
        self.namespace = namespace
        self._rust_available = self._check_rust()
    
    def _check_rust(self) -> bool:
        """Check if Rust DEBRUIJN commands are available."""
        try:
            self.client.execute_command('HLLSET.DEBRUIJN.INFO', '__test__')
            return True
        except redis.ResponseError as e:
            if "unknown command" in str(e).lower():
                return False
            return True  # Command exists, just no graph
    
    # === Building ===
    
    def build(self, trigrams_key: str, graph_key: Optional[str] = None,
              weighted: bool = True, replace: bool = True) -> Dict[str, Any]:
        """
        Build De Bruijn graph from trigram list.
        
        Uses Rust if available, falls back to Python + RedisGraph.
        """
        if self._rust_available:
            return self._build_rust(trigrams_key, graph_key, weighted, replace)
        return self._build_python(trigrams_key, graph_key, weighted, replace)
    
    def build_corpus(self, doc_keys: List[str], graph_key: str) -> Dict[str, Any]:
        """Build corpus-level graph from multiple documents."""
        if self._rust_available:
            return self._build_corpus_rust(doc_keys, graph_key)
        return self._build_corpus_python(doc_keys, graph_key)
    
    # === Path Finding ===
    
    def restore(self, trigrams_key: str, 
                mode: str = "eulerian",
                keep_graph: bool = False) -> PathResult:
        """
        Restore sequence from trigrams via De Bruijn path.
        
        Args:
            trigrams_key: Redis key containing trigram list
            mode: "eulerian", "greedy", or "viterbi"
            keep_graph: Keep graph after restoration
            
        Returns:
            PathResult with reconstructed sequence
        """
        if self._rust_available:
            return self._restore_rust(trigrams_key, mode, keep_graph)
        return self._restore_python(trigrams_key, mode, keep_graph)
    
    def find_path(self, graph_key: str,
                  start: Optional[str] = None,
                  end: Optional[str] = None,
                  mode: str = "eulerian") -> PathResult:
        """Find path in existing graph."""
        if self._rust_available:
            return self._path_rust(graph_key, start, end, mode)
        return self._path_python(graph_key, start, end, mode)
    
    # === Generation (Micro-LLM) ===
    
    def generate(self, graph_key: str,
                 start: Optional[str] = None,
                 max_tokens: int = 50,
                 temperature: float = 0.8) -> GenerateResult:
        """
        Generate text via Markov random walk on corpus graph.
        
        Args:
            graph_key: Corpus De Bruijn graph
            start: Starting bigram (default: random START node)
            max_tokens: Maximum tokens to generate
            temperature: 0=greedy, 1=random sampling
            
        Returns:
            GenerateResult with generated text
        """
        if self._rust_available:
            return self._generate_rust(graph_key, start, max_tokens, temperature)
        return self._generate_python(graph_key, start, max_tokens, temperature)
    
    def probability(self, graph_key: str, sequence: List[str]) -> Dict[str, float]:
        """Compute probability of sequence under the model."""
        if self._rust_available:
            return self._probability_rust(graph_key, sequence)
        return self._probability_python(graph_key, sequence)
    
    # === Global G_n Management ===
    
    def update_global(self, tokens: List[str], namespace: Optional[str] = None):
        """
        Update global G_1, G_2, G_3 from tokens.
        
        Called automatically by ingest() if GLOBAL option set.
        """
        ns = namespace or self.namespace
        # ... implementation
```

### 4.2 Integration with HLLSetRingStore

```python
class HLLSetRingStore:
    """Enhanced with De Bruijn integration."""
    
    def ingest(self, text: str, 
               update_global: bool = True,
               store_trigrams: bool = True,
               update_tf: bool = True) -> IngestResult:
        """
        Ingest text with full n-gram processing.
        
        Args:
            text: Input text
            update_global: Update G_1, G_2, G_3 HLLSets
            store_trigrams: Store ordered trigrams for reconstruction
            update_tf: Update term frequencies in TokenLUT
        """
        if self.using_rust:
            # Single Rust command handles everything
            args = ['HLLSET.RING.INGEST', self.session, text, 'NGRAMS', 1, 2, 3]
            if update_global:
                args.extend(['GLOBAL', self.namespace])
            if store_trigrams:
                args.extend(['TRIGRAMS', f'{self.namespace}:doc:{doc_hash}:tg'])
            if update_tf:
                args.append('TF')
            return self._parse_ingest_result(self.client.execute_command(*args))
        else:
            return self._ingest_python(text, update_global, store_trigrams, update_tf)
    
    def restore_text(self, doc_key: str) -> str:
        """
        Full text restoration pipeline.
        
        1. Disambiguate HLLSet → candidate tokens
        2. Extract trigrams
        3. De Bruijn path → ordered sequence
        """
        debruijn = DeBruijnRedis(self.client, self.namespace)
        
        # Get trigrams key
        tg_key = f"{doc_key}:tg"
        
        # Restore via De Bruijn
        result = debruijn.restore(tg_key, mode="eulerian")
        
        # Remove START/END markers
        tokens = [t for t in result.sequence if t not in (START, END)]
        
        return " ".join(tokens)
```

## 5. Full Restoration Pipeline

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  Full Text Restoration Pipeline (All Server-Side)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Client sends ONE command:                                                  │
│                                                                             │
│    HLLSET.RESTORE.FULL <doc_key>                                            │
│                                                                             │
│  Server (Rust) executes:                                                    │
│                                                                             │
│    1. HLLSET.CANDIDATES doc:xyz:hll tokenlut:entry:                         │
│       → Get candidate tokens via disambiguation                             │
│                                                                             │
│    2. LRANGE doc:xyz:tg 0 -1                                                │
│       → Get ordered trigrams                                                │
│                                                                             │
│    3. HLLSET.DEBRUIJN.RESTORE doc:xyz:tg EPHEMERAL                          │
│       → Build graph, find path, return sequence                             │
│                                                                             │
│    4. Return reconstructed text                                             │
│                                                                             │
│  Returns:                                                                   │
│    {                                                                        │
│      text: "the cat sat on the mat",                                        │
│      tokens: ["the", "cat", "sat", "on", "the", "mat"],                     │
│      candidates: 6,                                                         │
│      is_exact: true,                                                        │
│      confidence: 0.95                                                       │
│    }                                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 6. Micro-LLM / Bayesian / Markov Use Cases

### 6.1 Text Completion

```text
HLLSET.DEBRUIJN.GENERATE corpus:graph START "the|cat" MAX 20 TEMPERATURE 0.7
→ ["sat", "on", "the", "mat", "END"]
```

### 6.2 Sequence Scoring

```text
HLLSET.DEBRUIJN.PROBABILITY corpus:graph "the cat sat on the mat"
→ {probability: 0.0023, perplexity: 12.5}

HLLSET.DEBRUIJN.PROBABILITY corpus:graph "cat the on sat mat the"
→ {probability: 0.000001, perplexity: 1000.0}  # Ungrammatical
```

### 6.3 Next Word Prediction

```cypher
// Cypher query for top-k next words given context
MATCH (b:Bigram {id: 'the|cat'})-[e:TRIGRAM]->(next:Bigram)
RETURN e.label, e.probability
ORDER BY e.probability DESC
LIMIT 5

// Returns: [("sat", 0.4), ("ran", 0.25), ("ate", 0.15), ...]
```

### 6.4 Bayesian Inference

```text
// P(word | context, evidence)
// Use graph structure + additional constraints

HLLSET.DEBRUIJN.INFER corpus:graph 
  CONTEXT "the|cat"
  EVIDENCE "contains:sat" "topic:animals"
→ {next: "sat", probability: 0.72}
```

## 7. Implementation Priority

### Phase 1: Core (Immediate)

1. ✅ Tutorial 05 with RedisGraph (done)
2. 🔲 Enhance HLLSET.RING.INGEST for G_n updates
3. 🔲 Add TF/DF fields to TokenLUT schema
4. 🔲 Basic HLLSET.DEBRUIJN.BUILD command
5. 🔲 Basic HLLSET.DEBRUIJN.PATH command

### Phase 2: Full Pipeline

6. 🔲 HLLSET.DEBRUIJN.RESTORE (combined command)
7. 🔲 Python DeBruijnRedis class
8. 🔲 Integration with HLLSetRingStore.restore_text()

### Phase 3: Micro-LLM

9. 🔲 HLLSET.DEBRUIJN.BUILDCORPUS
10. 🔲 HLLSET.DEBRUIJN.GENERATE
11. 🔲 HLLSET.DEBRUIJN.PROBABILITY
12. 🔲 Weighted path algorithms (Viterbi)

### Phase 4: Advanced

13. 🔲 Bayesian inference integration
14. 🔲 Online learning (incremental graph updates)
15. 🔲 Pruning and optimization for large corpora

## 8. Open Questions

1. **Graph Size Management**: For large corpora, should we:
   - Prune low-frequency edges?
   - Use approximate probabilities?
   - Shard by first character?

2. **Temperature Sampling**: For generation, implement:
   - Nucleus sampling (top-p)?
   - Top-k sampling?
   - Both?

3. **Trigram Storage Format**:

   - "a|b|c:count" (current proposal)
   - Separate list + hash for counts?
   - RedisJSON for structured storage?

4. **Integration with existing Markov/Bayesian modules**:
   - Merge into debruijn.rs?
   - Keep separate, share graph?
