# Redis HLLSet Module

A native Redis module implementing HLLSet Algebra - HyperLogLog with full set operations.

## Features

- **Native Redis Type**: HLLSet is a first-class Redis data type with RDB persistence
- **Content-Addressable**: Keys are automatically generated from content (SHA-1 of sorted tokens)
- **Full Set Algebra**: Union (∪), Intersection (∩), Difference (\), Symmetric Difference (⊕)
- **O(1) Space**: Probabilistic structure with ~2% error rate
- **High Performance**: Written in Rust with optimized Roaring Bitmaps

## Building

### Prerequisites

- Rust 1.75+ (install from https://rustup.rs)
- Redis 7.0+

### Build

```bash
# Build release version
./build.sh --release

# Build and run tests
./build.sh --test

# Build Docker image
./build.sh --docker
```

### Output

The build produces `libredis_hllset.so` which can be loaded into Redis.

## Commands

### Creation

```redis
# Create HLLSet from tokens (returns content-addressable key)
HLLSET.CREATE token1 token2 token3
# Returns: "hllset:abc123..."

# Create from pre-computed hashes
HLLSET.CREATEHASH 12345 67890 11111
```

### Cardinality

```redis
# Get estimated cardinality
HLLSET.CARD hllset:abc123
# Returns: (float) 3.0
```

### Set Operations

All operations create a new key with the result:

```redis
# Union: A ∪ B
HLLSET.UNION key1 key2

# Intersection: A ∩ B  
HLLSET.INTER key1 key2

# Difference: A \ B
HLLSET.DIFF key1 key2

# Symmetric Difference: A ⊕ B
HLLSET.XOR key1 key2

# In-place merge (union into destination)
HLLSET.MERGE destkey key1 key2 key3
```

### Similarity

```redis
# Jaccard similarity: |A ∩ B| / |A ∪ B|
HLLSET.SIM key1 key2
# Returns: (float) 0.5

# Alias
HLLSET.JACCARD key1 key2
```

### Info & Debug

```redis
# Get metadata
HLLSET.INFO key
# Returns: key, cardinality, registers, non_zero_registers, precision_bits, memory_bytes

# Dump register positions
HLLSET.DUMP key

# Check existence
HLLSET.EXISTS key

# Delete
HLLSET.DEL key
```

## Usage Example

```redis
# Create two sets
redis> HLLSET.CREATE apple banana cherry
"hllset:1a2b3c..."

redis> HLLSET.CREATE banana cherry date
"hllset:4d5e6f..."

# Get cardinalities
redis> HLLSET.CARD hllset:1a2b3c
(float) 3.0

# Union (should be ~4)
redis> HLLSET.UNION hllset:1a2b3c hllset:4d5e6f
"hllset:union:1a2b3c:4d5e6f"

redis> HLLSET.CARD hllset:union:1a2b3c:4d5e6f
(float) 4.0

# Similarity
redis> HLLSET.SIM hllset:1a2b3c hllset:4d5e6f
(float) 0.5
```

## Configuration

Load the module in `redis.conf`:

```text
loadmodule /path/to/libredis_hllset.so
```

Or via command line:

```bash
redis-server --loadmodule ./libredis_hllset.so
```

## Algorithm Details

### Register Model

HLLSet uses M=1024 registers (10-bit precision). Each register stores the maximum
leading zero count observed for hashes mapping to that bucket. The value is stored
as a position in a Roaring Bitmap: `position = bucket * 32 + value`.

### Cardinality Estimation

Uses the standard HyperLogLog harmonic mean estimator with:

- α correction factor for M=1024
- Linear counting correction for small cardinalities
- Bias correction for edge cases

### Set Operations

- **Union**: Max of register values (standard HLL merge)
- **Intersection**: Min of register values (inclusion-exclusion approximation)
- **Difference**: Saturating subtraction of register values
- **Symmetric Difference**: Absolute difference of register values

### Content-Addressable Keys

Keys are SHA-1 hashes of sorted, deduplicated tokens joined with null bytes.
This ensures:

- Same content → Same key (idempotent creation)
- Different order → Same key
- Duplicate tokens → Collapsed

## Performance

| Operation | Time Complexity | Space |
| ----------- | ---------------- | ------- |
| CREATE | O(n) | O(M) |
| CARD | O(M) | - |
| UNION | O(M) | O(M) |
| INTER | O(M) | O(M) |
| SIMILARITY | O(M) | - |

Where n = number of tokens, M = 1024 registers.

## Testing

```bash
# Run Rust unit tests
cargo test

# Run integration tests (requires Redis running with module)
./test.sh

# Run benchmarks
cargo bench
```

## License

MIT License
