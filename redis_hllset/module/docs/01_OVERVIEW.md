# Redis HLLSet Module - Overview

## Introduction

The Redis HLLSet Module is a native Redis module implementing **HLLSet Algebra** - a probabilistic data structure that combines HyperLogLog cardinality estimation with full set algebra operations. Written in Rust for high performance, it provides:

- **Native Redis Type**: HLLSet is a first-class Redis data type with RDB persistence
- **Content-Addressable Storage**: Keys are automatically generated from content (SHA-1 of sorted tokens)
- **Full Set Algebra**: Union (∪), Intersection (∩), Difference (\), Symmetric Difference (⊕)
- **O(1) Space Complexity**: Probabilistic structure with ~2% error rate
- **XOR Ring Decomposition**: Express any HLLSet as XOR of linearly independent basis elements
- **TokenLUT Integration**: High-performance token disambiguation with streaming support

## Architecture

### Storage Model

HLLSet uses a **3D tensor model** stored as a compressed Roaring Bitmap:

```text
┌─────────────────────────────────────────────────────────────┐
│                      HLLSet Tensor                          │
│                                                             │
│  Dimensions:                                                │
│  - Registers (M):    1024 (2^10)                            │
│  - Bits per reg:     32 (trailing zeros count)              │
│  - Total positions:  32,768 bits                            │
│                                                             │
│  Storage: Roaring Bitmap (compressed)                       │
│  Position encoding: (register * 32) + trailing_zeros        │
└─────────────────────────────────────────────────────────────┘
```

### Key Concepts

1. **Register Model**: Each token is hashed (MurmurHash3 64-bit), with:
   - Lower 10 bits → register index [0, 1023]
   - Remaining bits → trailing zeros count [0, 31]

2. **Bit Position**: Unlike standard HLL (max-only), HLLSet stores ALL observed states as bits in a bitmap, enabling true set algebra.

3. **Content-Addressable Keys**: Keys are SHA-1 hashes of sorted, deduplicated tokens, ensuring:
   - Same content → Same key (idempotent creation)
   - Different order → Same key
   - Duplicate tokens → Collapsed

### Module Components

| Component | Purpose |
| ----------- | --------- |
| **HLLSet Core** | Basic HLLSet operations (create, card, set algebra) |
| **Ring Decomposition** | XOR ring algebra for basis compression |
| **TokenLUT** | Token lookup table management with TF tracking |
| **Disambiguation** | Position-based candidate matching and streaming |

## Command Summary

### HLLSet Core Commands

| Command | Description |
| --------- | ------------- |
| `HLLSET.CREATE` | Create HLLSet from tokens |
| `HLLSET.CREATEHASH` | Create from pre-computed hashes |
| `HLLSET.CARD` | Get estimated cardinality |
| `HLLSET.UNION` | Union of two sets (A ∪ B) |
| `HLLSET.INTER` | Intersection (A ∩ B) |
| `HLLSET.DIFF` | Difference (A \ B) |
| `HLLSET.XOR` | Symmetric difference (A ⊕ B) |
| `HLLSET.SIM` | Jaccard similarity |
| `HLLSET.MERGE` | In-place union |
| `HLLSET.INFO` | Get metadata |
| `HLLSET.DUMP` | Dump register positions |
| `HLLSET.EXISTS` | Check existence |
| `HLLSET.DEL` | Delete HLLSet |

### Tensor/Position Commands

| Command | Description |
|---------|-------------|
| `HLLSET.POSITIONS` | Get all active (reg, zeros) positions |
| `HLLSET.POPCOUNT` | Total set bits count |
| `HLLSET.BITCOUNTS` | Count per bit position (for HT estimator) |
| `HLLSET.REGISTER` | Get bitmap value for specific register |
| `HLLSET.HASBIT` | Check if specific position is set |

### Disambiguation Commands

| Command | Description |
|---------|-------------|
| `HLLSET.CANDIDATES` | Find LUT entries matching positions |
| `HLLSET.SCANMATCH` | Full scan with streaming output |
| `HLLSET.POSINDEX` | Create sorted set index of positions |

### Ring Commands (Planned)

| Command | Description |
|---------|-------------|
| `HLLSET.RING.INIT` | Initialize a decomposition ring |
| `HLLSET.RING.INGEST` | Ingest token and decompose |
| `HLLSET.RING.DECOMPOSE` | Decompose existing HLLSet |
| `HLLSET.RING.BASIS` | Get current basis SHA1s |
| `HLLSET.RING.RANK` | Get current rank |
| `HLLSET.W.COMMIT` | Create W lattice commit |
| `HLLSET.W.DIFF` | Compute diff between commits |
| `HLLSET.RECONSTRUCT` | Reconstruct from XOR expression |

### TokenLUT Commands (Planned)

| Command | Description |
|---------|-------------|
| `TOKENLUT.ADD` | Add/update token entry |
| `TOKENLUT.INCR` | Increment TF for entry |
| `TOKENLUT.GET` | Get entry by key |
| `TOKENLUT.MGET` | Batch get entries |

## Performance Characteristics

| Operation | Time Complexity | Space |
|-----------|-----------------|-------|
| CREATE | O(n) | O(M) |
| CARD | O(M) | - |
| UNION | O(M) | O(M) |
| INTER | O(M) | O(M) |
| DIFF | O(M) | O(M) |
| XOR | O(M) | O(M) |
| SIMILARITY | O(M) | - |
| POSITIONS | O(popcount) | O(popcount) |

Where:
- n = number of tokens
- M = 1024 registers
- popcount = number of set bits (typically << M*32)

## Quick Start

### Loading the Module

```bash
# Command line
redis-server --loadmodule /path/to/libredis_hllset.so

# Or in redis.conf
loadmodule /path/to/libredis_hllset.so
```

### Basic Usage

```redis
# Create two sets
redis> HLLSET.CREATE apple banana cherry
"hllset:1a2b3c4d5e6f..."

redis> HLLSET.CREATE banana cherry date
"hllset:9a8b7c6d5e4f..."

# Get cardinalities
redis> HLLSET.CARD hllset:1a2b3c4d5e6f
(float) 3.0

# Union
redis> HLLSET.UNION hllset:1a2b3c4d5e6f hllset:9a8b7c6d5e4f
"hllset:union:1a2b3c4d:9a8b7c6d"

# Similarity
redis> HLLSET.SIM hllset:1a2b3c4d5e6f hllset:9a8b7c6d5e4f
(float) 0.5
```

## Next Steps

- [Data Structures](02_DATA_STRUCTURES.md) - Detailed format specifications
- [HLLSet Commands](03_HLLSET_COMMANDS.md) - Complete command reference
- [Disambiguation](04_DISAMBIGUATION.md) - TokenLUT and matching
- [Ring Algebra](05_RING_ALGEBRA.md) - XOR decomposition
- [Best Practices](06_BEST_PRACTICES.md) - Usage patterns and optimization
