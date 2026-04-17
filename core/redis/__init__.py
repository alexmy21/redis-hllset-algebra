"""
Redis Backend Modules for HLLSet Algebra

This subpackage contains all Redis-dependent modules:
- hllset_redis: Redis-backed HLLSet using native hllset module
- hllset_store_redis: Redis-native HLLSet Registry with RediSearch
- hllset_ring_store: XOR Ring Algebra with Base-Only Storage
- tokenlut_redis: Token Lookup Table using RediSearch
- tokenlut_session: Multi-Producer HLLSet Building with Checkpoints
- tokenlut_stream: Batch Token Ingestion with Redis Streams
- hllset_disambiguate: Python interface for HLLSet disambiguation

All modules require redis-py and a running Redis server with:
- RediSearch module (for indexing)
- hllset module (for native HLLSet operations)
"""

from __future__ import annotations

# === HLLSet Redis (L2r) ===
from .hllset_redis import (
    HLLSetRedis,
    RedisClientManager,
    load_functions as load_redis_functions,
    check_redis_modules,
)

# === HLLSet Store Redis (L6r) ===
from .hllset_store_redis import (
    HLLSetStoreRedis,
    HLLSetEntry,
    Operation as StoreOperation,
    Derivation as StoreDerivation,
)

# === HLLSet Ring Store (L6x) ===
from .hllset_ring_store import (
    HLLSetRingStore,
    RingState,
    DecomposeResult,
    WCommit,
    WDiff,
    HLLSetMeta,
    Derivation as RingDerivation,
    Operation as RingOperation,
    LRUCache,
)

# === TokenLUT Redis ===
from .tokenlut_redis import (
    TokenLUTRedis,
    TokenEntry,
)

# === TokenLUT Stream ===
from .tokenlut_stream import (
    TokenLUTStream,
    StreamConfig,
    IngestStats,
    HashMode,
    create_stream_lut,
    ingest_file,
)

# === TokenLUT Session ===
from .tokenlut_session import (
    TokenLUTSession,
    SessionProducer,
    SessionConfig,
    CheckpointResult,
    CommitResult,
    SessionStats,
    create_session,
    quick_ingest,
)

# === HLLSet Disambiguate ===
from .hllset_disambiguate import (
    Candidate,
    HLLSetDisambiguator,
    DisambiguationPipeline,
    disambiguate,
    disambiguate_stream,
)

__all__ = [
    # HLLSet Redis
    'HLLSetRedis',
    'RedisClientManager',
    'load_redis_functions',
    'check_redis_modules',
    # HLLSet Store Redis
    'HLLSetStoreRedis',
    'HLLSetEntry',
    'StoreOperation',
    'StoreDerivation',
    # HLLSet Ring Store
    'HLLSetRingStore',
    'RingState',
    'DecomposeResult',
    'WCommit',
    'WDiff',
    'HLLSetMeta',
    'RingDerivation',
    'RingOperation',
    'LRUCache',
    # TokenLUT Redis
    'TokenLUTRedis',
    'TokenEntry',
    # TokenLUT Stream
    'TokenLUTStream',
    'StreamConfig',
    'IngestStats',
    'HashMode',
    'create_stream_lut',
    'ingest_file',
    # TokenLUT Session
    'TokenLUTSession',
    'SessionProducer',
    'SessionConfig',
    'CheckpointResult',
    'CommitResult',
    'SessionStats',
    'create_session',
    'quick_ingest',
    # HLLSet Disambiguate
    'Candidate',
    'HLLSetDisambiguator',
    'DisambiguationPipeline',
    'disambiguate',
    'disambiguate_stream',
]
