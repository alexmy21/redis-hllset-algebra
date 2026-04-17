# De Bruijn Implementation Roadmap

## Current State

The current `HLLSET.RING.INGEST` command:
- Takes a **single token** as input
- Creates HLLSet from that token
- Decomposes against ring basis
- Does NOT handle n-grams or G_n updates

## Required Enhancements

### Phase 1: Enhanced Ingest with N-grams and G_n

#### 1.1 Modify `ring_ingest` in `ring_commands.rs`

```rust
/// HLLSET.RING.INGEST <ring_key> <text>
///   [NGRAMS <n1> <n2> ...]      -- Generate n-grams (default: 1 2 3)
///   [GLOBAL <namespace>]        -- Update global G_n HLLSets
///   [TRIGRAMS <list_key>]       -- Store ordered trigrams
///   [TF]                        -- Update term frequencies
///   [SOURCE <source>]
///   [TAGS <tags>]
///   [MARKERS <start> <end>]     -- Custom markers (default: ⟨START⟩ ⟨END⟩)
pub fn ring_ingest(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    // ... parse arguments
    
    // 1. Tokenize with markers
    let tokens = tokenize_with_markers(&text, &start_marker, &end_marker);
    
    // 2. Generate n-grams
    let mut all_ngrams = Vec::new();
    let mut ngram_counts: HashMap<String, usize> = HashMap::new();
    
    for n in &ngram_sizes {
        let ngrams = generate_ngrams(&tokens, *n);
        for ng in &ngrams {
            let ng_str = ngram_to_string(ng);
            *ngram_counts.entry(ng_str.clone()).or_insert(0) += 1;
            all_ngrams.push(ng_str);
        }
    }
    
    // 3. Create document HLLSet from all n-grams
    let hll = HLLSet::from_tokens(&all_ngrams);
    
    // 4. Update global G_n if requested
    if let Some(ns) = &global_namespace {
        for n in &ngram_sizes {
            let g_key = format!("{}:global:G{}", ns, n);
            let count_key = format!("{}:global:G{}:count", ns, n);
            
            for ng in generate_ngrams(&tokens, *n) {
                let ng_str = ngram_to_string(&ng);
                // Add to HLLSet
                ctx.call("HLLSET.ADD", &[&g_key, &ng_str])?;
                // Increment count
                ctx.call("HINCRBY", &[&count_key, &ng_str, "1"])?;
            }
        }
        // Increment doc count
        ctx.call("INCR", &[&format!("{}:global:doc_count", ns)])?;
    }
    
    // 5. Store trigrams if requested
    if let Some(tg_key) = &trigrams_key {
        ctx.call("DEL", &[tg_key])?;
        for ng in generate_ngrams(&tokens, 3) {
            let ng_str = ngram_to_string(&ng);
            let count = ngram_counts.get(&ng_str).unwrap_or(&1);
            // Store as "a|b|c:count"
            ctx.call("RPUSH", &[tg_key, &format!("{}:{}", ng_str, count)])?;
        }
    }
    
    // 6. Update TF in TokenLUT if requested
    if update_tf {
        for (ng_str, count) in &ngram_counts {
            // Get TokenLUT entry key from hash
            let hash = compute_hash(ng_str);
            let entry_key = format!("tokenlut:entry:{}", hash);
            ctx.call("HINCRBY", &[&entry_key, "tf", &count.to_string()])?;
        }
    }
    
    // 7. Decompose against ring (existing logic)
    // ...
}
```

#### 1.2 Helper Functions

```rust
/// Tokenize text with START/END markers
fn tokenize_with_markers(text: &str, start: &str, end: &str) -> Vec<String> {
    let mut tokens = vec![start.to_string()];
    tokens.extend(text.split_whitespace().map(|s| s.to_lowercase()));
    tokens.push(end.to_string());
    tokens.push(end.to_string()); // 2 END markers for 3-gram alignment
    tokens
}

/// Generate n-grams from token list
fn generate_ngrams(tokens: &[String], n: usize) -> Vec<Vec<String>> {
    if tokens.len() < n {
        return Vec::new();
    }
    (0..=tokens.len() - n)
        .map(|i| tokens[i..i + n].to_vec())
        .collect()
}

/// Convert n-gram to pipe-separated string
fn ngram_to_string(ngram: &[String]) -> String {
    ngram.join("|")
}
```

### Phase 2: De Bruijn Commands

#### 2.1 New file: `debruijn_commands.rs`

```rust
//! De Bruijn Graph Commands
//!
//! Server-side De Bruijn graph operations using RedisGraph.

/// HLLSET.DEBRUIJN.BUILD <graph_key> <trigrams_key> [WEIGHTED] [REPLACE|MERGE]
pub fn debruijn_build(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    // 1. Read trigrams from list
    let trigrams = read_trigrams(ctx, &trigrams_key)?;
    
    // 2. Build graph via Cypher
    if replace {
        ctx.call("GRAPH.DELETE", &[&graph_key])?;
    }
    
    // 3. Create nodes and edges
    for (trigram, count) in &trigrams {
        let (a, b, c) = parse_trigram(trigram);
        
        // Create/merge source node
        let cypher = format!(
            "MERGE (src:Bigram {{id: '{}|{}', w1: '{}', w2: '{}'}}) \
             ON CREATE SET src.total_count = 0, src.doc_count = 0 \
             SET src.total_count = src.total_count + {}",
            a, b, a, b, count
        );
        ctx.call("GRAPH.QUERY", &[&graph_key, &cypher])?;
        
        // Create/merge target node
        // ...
        
        // Create edge with count
        let cypher = format!(
            "MATCH (src:Bigram {{id: '{}|{}'}}), (tgt:Bigram {{id: '{}|{}'}}) \
             CREATE (src)-[:TRIGRAM {{label: '{}', count: {}}}]->(tgt)",
            a, b, b, c, c, count
        );
        ctx.call("GRAPH.QUERY", &[&graph_key, &cypher])?;
    }
    
    // 4. If weighted, compute probabilities
    if weighted {
        compute_edge_probabilities(ctx, &graph_key)?;
    }
    
    Ok(...)
}

/// HLLSET.DEBRUIJN.PATH <graph_key> [MODE EULERIAN|GREEDY|VITERBI]
pub fn debruijn_path(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    match mode {
        Mode::Eulerian => find_eulerian_path(ctx, &graph_key),
        Mode::Greedy => find_greedy_path(ctx, &graph_key),
        Mode::Viterbi => find_viterbi_path(ctx, &graph_key),
    }
}

/// Find Eulerian path using Hierholzer's algorithm
fn find_eulerian_path(ctx: &Context, graph_key: &str) -> RedisResult {
    // 1. Find start node (has START marker or out_degree > in_degree)
    let start_query = "MATCH (n:Bigram) WHERE n.is_start = true RETURN n.id LIMIT 1";
    let start_node = ctx.call("GRAPH.QUERY", &[graph_key, start_query])?;
    
    // 2. Iteratively build path using edge traversal
    // Track visited edges (by internal ID)
    let mut visited_edges: HashSet<i64> = HashSet::new();
    let mut path: Vec<String> = Vec::new();
    let mut current = start_node;
    
    loop {
        // Find unvisited outgoing edge
        let edge_query = format!(
            "MATCH (src:Bigram {{id: '{}'}})-[e:TRIGRAM]->(tgt:Bigram) \
             WHERE NOT id(e) IN [{}] \
             RETURN id(e), e.label, tgt.id, tgt.w2 \
             LIMIT 1",
            current,
            visited_edges.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(",")
        );
        
        let result = ctx.call("GRAPH.QUERY", &[graph_key, &edge_query])?;
        
        // Parse result and continue or break
        // ...
    }
    
    Ok(RedisValue::Array(path.into_iter().map(RedisValue::BulkString).collect()))
}

/// HLLSET.DEBRUIJN.RESTORE <trigrams_key> [EPHEMERAL|KEEP]
pub fn debruijn_restore(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    // 1. Generate temp graph key
    let graph_key = format!("debruijn:temp:{}", uuid::Uuid::new_v4());
    
    // 2. Build graph
    debruijn_build_internal(ctx, &graph_key, &trigrams_key, true)?;
    
    // 3. Find path
    let path = find_eulerian_path(ctx, &graph_key)?;
    
    // 4. Clean up if ephemeral
    if ephemeral {
        ctx.call("GRAPH.DELETE", &[&graph_key])?;
    }
    
    Ok(path)
}
```

### Phase 3: Corpus-Level (Micro-LLM)

```rust
/// HLLSET.DEBRUIJN.BUILDCORPUS <graph_key> <doc_key1> [<doc_key2> ...]
pub fn debruijn_build_corpus(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    // Merge trigrams from all documents with count aggregation
}

/// HLLSET.DEBRUIJN.GENERATE <graph_key> [START <bigram>] [MAX <n>] [TEMP <t>]
pub fn debruijn_generate(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    // Markov random walk with temperature-controlled sampling
}

/// HLLSET.DEBRUIJN.PROBABILITY <graph_key> <sequence>
pub fn debruijn_probability(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    // Compute P(sequence) under the model
}
```

### Phase 4: TokenLUT Schema Update

Add migration for new TF/DF fields:

```python
# In tokenlut_redis.py

def migrate_schema_v2(self):
    """Add TF/DF fields to existing TokenLUT entries."""
    
    # Drop and recreate index with new schema
    try:
        self.client.ft(self.index_name).dropindex(delete_documents=False)
    except:
        pass
    
    schema = (
        NumericField("reg", sortable=True),
        NumericField("zeros", sortable=True),
        NumericField("hash_full"),
        NumericField("layer", sortable=True),
        NumericField("collision_count", sortable=True),
        TagField("first_tokens_tag", separator=","),
        TextField("first_tokens"),
        TextField("tokens"),
        # NEW fields
        NumericField("tf", sortable=True),           # Term frequency
        NumericField("df", sortable=True),           # Document frequency
        NumericField("tf_log", sortable=True),       # log(1 + tf)
        NumericField("idf", sortable=True),          # log(N / df)
        NumericField("last_seen", sortable=True),    # Unix timestamp
    )
    
    definition = IndexDefinition(
        prefix=[self.prefix],
        index_type=IndexType.HASH
    )
    
    self.client.ft(self.index_name).create_index(schema, definition=definition)
    
    # Set defaults for existing entries
    for key in self.client.scan_iter(f"{self.prefix}*"):
        self.client.hset(key, mapping={
            "tf": 1,
            "df": 1,
            "tf_log": 0.0,
            "idf": 0.0,
            "last_seen": int(time.time()),
        })
```

## File Changes Summary

1. **`redis_hllset/module/src/ring_commands.rs`**
   - Enhance `ring_ingest` for n-grams
   - Add helper functions for tokenization

2. **`redis_hllset/module/src/debruijn_commands.rs`** (NEW)
   - `debruijn_build`
   - `debruijn_path`
   - `debruijn_restore`
   - `debruijn_generate`
   - `debruijn_probability`

3. **`redis_hllset/module/src/lib.rs`**
   - Register new DEBRUIJN commands

4. **`core/tokenlut_redis.py`**
   - Add TF/DF fields to schema
   - Add migration function

5. **`core/debruijn_redis.py`** (NEW)
   - Python `DeBruijnRedis` class
   - Client-side fallback implementation

6. **`redis-tutorials/05_debruijn_redisgraph.ipynb`**
   - Update to use new commands when available

## Testing Plan

1. Unit tests for n-gram generation
2. Integration tests for G_n updates
3. End-to-end test: ingest → restore
4. Corpus building and generation tests
5. Performance benchmarks for large graphs
