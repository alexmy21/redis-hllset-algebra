# Redis HLLSet Module

A native Redis module implementing **HLLSet Algebra** - HyperLogLog with full set operations, XOR ring decomposition, and high-performance token disambiguation.

## Features

- **Native Redis Type**: HLLSet is a first-class Redis data type with RDB persistence
- **Content-Addressable**: Keys are automatically generated from content (SHA-1 of sorted tokens)
- **Full Set Algebra**: Union (∪), Intersection (∩), Difference (\), Symmetric Difference (⊕)
- **O(1) Space**: Probabilistic structure with ~2% error rate
- **High Performance**: Written in Rust with optimized Roaring Bitmaps
- **XOR Ring Decomposition**: Express any HLLSet as XOR of linearly independent bases
- **Token Disambiguation**: Position-based lookup with streaming support
- **W Lattice Commits**: Temporal tracking of ring state

## Documentation

📚 **[Full Documentation](docs/README.md)** - Comprehensive guides and reference

- [Overview](docs/01_OVERVIEW.md) - Architecture and quick start
- [Data Structures](docs/02_DATA_STRUCTURES.md) - Internal formats and schemas
- [HLLSet Commands](docs/03_HLLSET_COMMANDS.md) - Core operations reference
- [Disambiguation](docs/04_DISAMBIGUATION.md) - Tensor positions and TokenLUT
- [Ring Algebra](docs/05_RING_ALGEBRA.md) - XOR decomposition commands
- [TokenLUT Commands](docs/06_TOKENLUT_COMMANDS.md) - Lookup table management
- [Best Practices](docs/07_BEST_PRACTICES.md) - Usage patterns and optimization

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

## Quick Start

### Loading the Module

```bash
redis-server --loadmodule ./libredis_hllset.so
```

### Basic Usage

```redis
# Create HLLSet from tokens (returns content-addressable key)
redis> HLLSET.CREATE apple banana cherry
"hllset:7ac66c0f148de9519b8bd264312c4d64f0c2d6b0"

# Get estimated cardinality
redis> HLLSET.CARD hllset:7ac66c0f148de9519b8bd264312c4d64f0c2d6b0
(float) 3.0

# Set operations
redis> HLLSET.UNION key1 key2
redis> HLLSET.INTER key1 key2
redis> HLLSET.DIFF key1 key2
redis> HLLSET.XOR key1 key2

# Similarity
redis> HLLSET.SIM key1 key2
(float) 0.5
```

## Command Summary

### HLLSet Core

| Command | Description |
|---------|-------------|
| `HLLSET.CREATE token [...]` | Create HLLSet from tokens |
| `HLLSET.CREATEHASH hash [...]` | Create from pre-computed hashes |
| `HLLSET.CARD key` | Get estimated cardinality |
| `HLLSET.UNION key1 key2` | Union (A ∪ B) |
| `HLLSET.INTER key1 key2` | Intersection (A ∩ B) |
| `HLLSET.DIFF key1 key2` | Difference (A \ B) |
| `HLLSET.XOR key1 key2` | Symmetric difference (A ⊕ B) |
| `HLLSET.SIM key1 key2` | Jaccard similarity |
| `HLLSET.MERGE dest key [...]` | In-place union |
| `HLLSET.INFO key` | Get metadata |
| `HLLSET.EXISTS key` | Check existence |
| `HLLSET.DEL key` | Delete HLLSet |

### Tensor/Position Commands

| Command | Description |
|---------|-------------|
| `HLLSET.POSITIONS key` | Get all (reg, zeros) pairs |
| `HLLSET.POPCOUNT key` | Total set bits count |
| `HLLSET.BITCOUNTS key` | Per-bit position counts |
| `HLLSET.REGISTER key reg` | Get register bitmap |
| `HLLSET.HASBIT key reg zeros` | Check if position is set |

### Disambiguation Commands

| Command | Description |
|---------|-------------|
| `HLLSET.CANDIDATES key prefix [opts]` | Find LUT matches |
| `HLLSET.SCANMATCH key prefix stream [opts]` | Stream all matches |
| `HLLSET.POSINDEX key index_key` | Create position index |

### Ring Algebra Commands

| Command | Description |
|---------|-------------|
| `HLLSET.RING.INIT ring [PBITS p]` | Initialize decomposition ring |
| `HLLSET.RING.INGEST ring token [opts]` | Ingest and decompose |
| `HLLSET.RING.BASIS ring` | Get basis SHA1s |
| `HLLSET.RING.RANK ring` | Get current rank |
| `HLLSET.W.COMMIT ring [META json]` | Create W commit |
| `HLLSET.W.DIFF ring t1 t2` | Compare commits |
| `HLLSET.RECONSTRUCT sha1` | Rebuild from XOR expression |

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
