# Redis HLLSet Module Documentation

Welcome to the Redis HLLSet Module documentation. This module provides HyperLogLog with full set algebra operations, XOR ring decomposition, and token disambiguation capabilities.

## Documentation Index

### Getting Started

1. [Overview](01_OVERVIEW.md) - Introduction, architecture, and quick start guide

### Reference

2. [Data Structures](02_DATA_STRUCTURES.md) - Internal formats, schemas, and customization options
3. [HLLSet Commands](03_HLLSET_COMMANDS.md) - Core HLLSet operations (create, card, union, etc.)
4. [Disambiguation](04_DISAMBIGUATION.md) - Tensor positions and TokenLUT matching
5. [Ring Algebra](05_RING_ALGEBRA.md) - XOR decomposition and W lattice commands
6. [TokenLUT Commands](06_TOKENLUT_COMMANDS.md) - Token lookup table management

### Best Practices

7. [Best Practices](07_BEST_PRACTICES.md) - Usage patterns, optimization, and common pitfalls

---

## Quick Command Reference

### HLLSet Core

| Command | Purpose |
| ------- | ------- |
| `HLLSET.CREATE token [...]` | Create from tokens |
| `HLLSET.CREATEHASH hash [...]` | Create from hashes |
| `HLLSET.CARD key` | Get cardinality |
| `HLLSET.UNION key1 key2` | Union (A ∪ B) |
| `HLLSET.INTER key1 key2` | Intersection (A ∩ B) |
| `HLLSET.DIFF key1 key2` | Difference (A \ B) |
| `HLLSET.XOR key1 key2` | Symmetric diff (A ⊕ B) |
| `HLLSET.SIM key1 key2` | Jaccard similarity |
| `HLLSET.MERGE dest key [...]` | In-place union |
| `HLLSET.FINALIZE key [DELETE]` | Copy to content-addressable key |
| `HLLSET.INFO key` | Get metadata |
| `HLLSET.DUMP key` | Debug positions |
| `HLLSET.EXISTS key` | Check existence |
| `HLLSET.DEL key` | Delete |

### Tensor/Position

| Command | Purpose |
| ------- | ------- |
| `HLLSET.POSITIONS key` | Get (reg, zeros) pairs |
| `HLLSET.POPCOUNT key` | Total set bits |
| `HLLSET.BITCOUNTS key` | Per-bit counts |
| `HLLSET.REGISTER key reg` | Register bitmap |
| `HLLSET.HASBIT key reg zeros` | Check position |

### Disambiguation

| Command | Purpose |
| ------- | ------- |
| `HLLSET.CANDIDATES key prefix [opts]` | Find LUT matches |
| `HLLSET.SCANMATCH key prefix stream [opts]` | Stream all matches |
| `HLLSET.POSINDEX key index_key` | Create position index |

### Ring Algebra

| Command | Purpose |
| ------- | ------- |
| `HLLSET.RING.INIT ring [PBITS p]` | Initialize ring |
| `HLLSET.RING.INGEST ring token [opts]` | Ingest and decompose |
| `HLLSET.RING.DECOMPOSE ring key [opts]` | Decompose existing |
| `HLLSET.RING.BASIS ring` | Get basis SHA1s |
| `HLLSET.RING.RANK ring` | Get current rank |
| `HLLSET.W.COMMIT ring [META json]` | Create W commit |
| `HLLSET.W.DIFF ring t1 t2` | Compare commits |
| `HLLSET.RECONSTRUCT sha1` | Rebuild from bases |

### TokenLUT

| Command | Purpose |
| ------- | ------- |
| `TOKENLUT.ADD prefix hash reg zeros layer token [opts]` | Add entry |
| `TOKENLUT.INCR key [BY n]` | Increment TF |
| `TOKENLUT.GET key` | Get entry |
| `TOKENLUT.MGET key [...]` | Batch get |

---

## Version

- Module Version: 0.2.0
- Documentation Version: 1.0.0
- Last Updated: April 2026

## Support

For issues and feature requests, please refer to the project repository.
