"""
TokenLUT Session Streaming — Multi-Producer HLLSet Building with Checkpoints

This module extends TokenLUT streaming with:
- Multi-producer sessions (different n-gram layers from parallel producers)
- Checkpoints — intermediate Roaring bitmaps during streaming
- Commit — UNION all checkpoint bitmaps → final content-addressed HLLSet
- Access to Roaring bitmap via SHA1 key

Architecture:
    Producer 1 (layer 0)  ─┐
    Producer 2 (layer 1)  ─┼─→ Stream ─→ Consumer ─→ LUT entries
    Producer 3 (layer 2)  ─┘      │
                                  │
                              session_id
                                  │
                                  ├─→ Checkpoint 1 → temp HLLSet (Roaring)
                                  ├─→ Checkpoint 2 → temp HLLSet (Roaring)
                                  └─→ Commit → UNION all checkpoints → SHA1

Workflow (from Tutorial 01):
    1. Checkpoint → HLLSET.CREATEHASH with batch hashes → temp HLLSet key
    2. Keep checkpoint HLLSet as Roaring bitmap
    3. Commit → HLLSET.UNION all checkpoint keys → final SHA1 key

Usage:
    from core.tokenlut_session import TokenLUTSession, SessionProducer
    
    session = TokenLUTSession(redis_client)
    session.start()
    
    # Create producers for different layers
    p0 = session.create_producer(layer=0)
    p1 = session.create_producer(layer=1)
    
    # Send tokens (unified API)
    p0.send([('apple', ''), ('banana', '')])  # unigrams
    p1.send([('apple pie', 'apple'), ('banana split', 'banana')])  # bigrams
    
    # Checkpoint — creates temp HLLSet
    cp1 = session.checkpoint()
    
    # More data...
    p0.send([('cherry', '')])
    
    # Commit — UNION all checkpoints → final SHA1
    result = session.commit()
    print(result.hllset_key)  # hllset:<sha1>
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
import redis
from redis.commands.search.field import TextField, NumericField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SessionConfig:
    """Configuration for streaming sessions."""
    stream_prefix: str = "tokenlut:session"
    consumer_group: str = "tokenlut:consumers"
    batch_size: int = 1000
    block_ms: int = 1000
    checkpoint_interval: int = 0  # Auto-checkpoint every N tokens (0 = disabled)
    trim_after_commit: bool = True
    delete_checkpoints_after_commit: bool = False  # Keep checkpoint HLLSets


@dataclass
class CheckpointResult:
    """Result of a checkpoint operation."""
    checkpoint_id: str
    hllset_key: str  # temp HLLSet key (Roaring bitmap)
    cardinality: float
    tokens_in_checkpoint: int  # tokens in this checkpoint batch
    total_tokens: int  # cumulative tokens so far
    timestamp: float = field(default_factory=time.time)


@dataclass
class CommitResult:
    """Result of a commit operation."""
    session_id: str
    hllset_key: str  # final SHA1 key from UNION
    cardinality: float
    total_tokens: int
    tokens_by_layer: Dict[int, int]  # layer -> count
    checkpoint_keys: List[str]  # keys that were unioned
    elapsed_seconds: float
    lut_entries: int


@dataclass
class SessionStats:
    """Statistics for a streaming session."""
    session_id: str
    tokens_by_layer: Dict[int, int] = field(default_factory=dict)
    tokens_pending: int = 0
    tokens_processed: int = 0
    checkpoint_hashes: List[int] = field(default_factory=list)  # hashes since last checkpoint
    start_time: float = field(default_factory=time.time)
    
    def add_token(self, layer: int, hash_val: int):
        self.tokens_by_layer[layer] = self.tokens_by_layer.get(layer, 0) + 1
        self.checkpoint_hashes.append(hash_val)
        self.tokens_pending += 1


class MessageType(Enum):
    """Types of messages in the session stream."""
    TOKEN = "token"
    CHECKPOINT = "checkpoint"
    COMMIT = "commit"


# =============================================================================
# Session Producer — Unified API for sending tokens
# =============================================================================

class SessionProducer:
    """
    Producer for a streaming session.
    
    Multiple producers can send to the same session concurrently,
    each handling different n-gram layers.
    
    Args:
        client: Redis client
        session_id: Session identifier (from TokenLUTSession.start())
        layer: N-gram layer this producer handles (0=unigram, 1=bigram, etc.)
        producer_id: Optional producer identifier
    """
    
    def __init__(
        self,
        client: redis.Redis,
        session_id: str,
        layer: int = 0,
        producer_id: Optional[str] = None,
        config: Optional[SessionConfig] = None
    ):
        self.client = client
        self.session_id = session_id
        self.layer = layer
        self.producer_id = producer_id or f"producer-{layer}-{uuid.uuid4().hex[:8]}"
        self.config = config or SessionConfig()
        self.stream_key = f"{self.config.stream_prefix}:{session_id}"
        
        # Hash function
        from ..hllset import DEFAULT_HASH_CONFIG
        self._hash_config = DEFAULT_HASH_CONFIG
        
        # Stats
        self.tokens_sent = 0
    
    def _hash_token(self, token: str) -> Tuple[int, int, int]:
        """Compute (reg, zeros, hash_full) for a token."""
        hash_full = self._hash_config.hash(token)
        reg, zeros = self._hash_config.hash_to_reg_zeros(token)
        return reg, zeros, hash_full
    
    def send(self, tokens: List[Tuple[str, str]]) -> int:
        """
        Send tokens to the session stream (unified API).
        
        Args:
            tokens: List of (token_string, first_token) tuples.
                    For unigrams, use ('word', '') or ('word', 'word').
                    For bigrams, use ('word1 word2', 'word1').
        
        Returns:
            Number of tokens sent
        """
        pipe = self.client.pipeline()
        
        for token, first_token in tokens:
            reg, zeros, hash_full = self._hash_token(token)
            pipe.xadd(
                self.stream_key,
                {
                    'type': MessageType.TOKEN.value,
                    'token': token,
                    'layer': self.layer,
                    'first_token': first_token,
                    'reg': reg,
                    'zeros': zeros,
                    'hash': hash_full,
                    'producer': self.producer_id,
                }
            )
        
        results = pipe.execute()
        count = sum(1 for r in results if r)
        self.tokens_sent += count
        return count
    
    # Convenience methods
    
    def send_unigrams(self, words: List[str]) -> int:
        """Send unigrams (first_token = empty)."""
        return self.send([(w, '') for w in words])
    
    def send_ngrams(self, ngrams: List[Tuple[str, str]]) -> int:
        """Send n-grams with first_token for triangulation."""
        return self.send(ngrams)


# =============================================================================
# TokenLUT Session — Main Session Controller
# =============================================================================

class TokenLUTSession:
    """
    Streaming session for building HLLSet with checkpoints.
    
    Workflow:
        1. start() — create session and stream
        2. create_producer() — get producers for different layers
        3. checkpoint() — create intermediate HLLSet (Roaring bitmap)
        4. commit() — UNION all checkpoints → final SHA1 key
    
    Args:
        client: Redis client
        index_name: RediSearch index for TokenLUT
        prefix: Key prefix for LUT entries
        config: SessionConfig for tuning
    """
    
    # RediSearch schema
    SCHEMA = (
        TextField("token", sortable=True),
        NumericField("reg", sortable=True),
        NumericField("zeros", sortable=True),
        NumericField("hash_full"),
        NumericField("layer", sortable=True),
        TextField("first_token"),
        TextField("session"),
    )
    
    def __init__(
        self,
        client: redis.Redis,
        index_name: str = "tokenlut:idx",
        prefix: str = "tokenlut:entry:",
        config: Optional[SessionConfig] = None
    ):
        self.client = client
        self.index_name = index_name
        self.prefix = prefix
        self.config = config or SessionConfig()
        
        # Session state
        self.session_id: Optional[str] = None
        self.stream_key: Optional[str] = None
        self.stats: Optional[SessionStats] = None
        self._checkpoint_keys: List[str] = []  # HLLSet keys from checkpoints
        self._checkpoint_results: List[CheckpointResult] = []
    
    # =========================================================================
    # Index Management
    # =========================================================================
    
    def create_index(self, drop_existing: bool = False) -> bool:
        """Create RediSearch index for TokenLUT."""
        if drop_existing:
            try:
                self.client.ft(self.index_name).dropindex(delete_documents=True)
            except redis.ResponseError:
                pass
        
        try:
            definition = IndexDefinition(
                prefix=[self.prefix],
                index_type=IndexType.HASH
            )
            self.client.ft(self.index_name).create_index(
                self.SCHEMA,
                definition=definition
            )
            return True
        except redis.ResponseError as e:
            if "Index already exists" in str(e):
                return False
            raise
    
    # =========================================================================
    # Session Lifecycle
    # =========================================================================
    
    def start(self, session_id: Optional[str] = None) -> str:
        """
        Start a new streaming session.
        
        Args:
            session_id: Optional custom session ID (auto-generated if None)
            
        Returns:
            Session ID for use with producers
        """
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:12]}"
        self.stream_key = f"{self.config.stream_prefix}:{self.session_id}"
        self.stats = SessionStats(session_id=self.session_id)
        self._checkpoint_keys = []
        self._checkpoint_results = []
        
        # Create consumer group
        try:
            self.client.xgroup_create(
                self.stream_key,
                self.config.consumer_group,
                id='0',
                mkstream=True
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        
        # Store session metadata
        self.client.hset(
            f"{self.config.stream_prefix}:meta:{self.session_id}",
            mapping={
                'session_id': self.session_id,
                'start_time': time.time(),
                'status': 'active',
                'index_name': self.index_name,
            }
        )
        
        return self.session_id
    
    def create_producer(self, layer: int = 0) -> SessionProducer:
        """
        Create a producer for this session.
        
        Args:
            layer: N-gram layer for this producer
            
        Returns:
            SessionProducer instance
        """
        if not self.session_id:
            raise RuntimeError("Session not started. Call start() first.")
        
        return SessionProducer(
            client=self.client,
            session_id=self.session_id,
            layer=layer,
            config=self.config
        )
    
    # =========================================================================
    # Stream Processing
    # =========================================================================
    
    def process_pending(self) -> int:
        """
        Process all pending messages in the session stream.
        
        Returns:
            Number of messages processed
        """
        if not self.session_id:
            raise RuntimeError("Session not started")
        
        processed = 0
        
        while True:
            messages = self.client.xreadgroup(
                self.config.consumer_group,
                "consumer-main",
                {self.stream_key: '>'},
                count=self.config.batch_size,
                block=100
            )
            
            if not messages:
                break
            
            for stream_name, entries in messages:
                for msg_id, data in entries:
                    self._process_message(msg_id, data)
                    self.client.xack(
                        self.stream_key,
                        self.config.consumer_group,
                        msg_id
                    )
                    processed += 1
        
        return processed
    
    def _process_message(self, msg_id: str, data: Dict):
        """Process a single stream message."""
        def decode(v):
            return v.decode() if isinstance(v, bytes) else v
        
        msg_type = decode(data.get('type', data.get(b'type', '')))
        
        if msg_type == MessageType.TOKEN.value:
            token = decode(data.get('token', data.get(b'token', '')))
            layer = int(decode(data.get('layer', data.get(b'layer', 0))))
            first_token = decode(data.get('first_token', data.get(b'first_token', '')))
            reg = int(decode(data.get('reg', data.get(b'reg', 0))))
            zeros = int(decode(data.get('zeros', data.get(b'zeros', 0))))
            hash_full = int(decode(data.get('hash', data.get(b'hash', 0))))
            
            # Store in LUT (RediSearch)
            entry_key = f"{self.prefix}{self.session_id}:{hash_full}"
            self.client.hset(entry_key, mapping={
                'token': token,
                'reg': reg,
                'zeros': zeros,
                'hash_full': hash_full,
                'layer': layer,
                'first_token': first_token,
                'session': self.session_id,
            })
            
            # Update stats
            self.stats.add_token(layer, hash_full)
            self.stats.tokens_processed += 1
            self.stats.tokens_pending -= 1
    
    def _decode_value(self, v):
        """Decode bytes to string."""
        return v.decode() if isinstance(v, bytes) else v
    
    def _get_cardinality(self, hllset_key: str) -> float:
        """Get cardinality from HLLSet."""
        card = self.client.execute_command('HLLSET.CARD', hllset_key)
        if isinstance(card, bytes):
            return float(card.decode())
        return float(card)
    
    # =========================================================================
    # Checkpoints — create intermediate HLLSets
    # =========================================================================
    
    def checkpoint(self, checkpoint_id: Optional[str] = None) -> CheckpointResult:
        """
        Create an intermediate HLLSet checkpoint.
        
        Workflow:
            1. Process all pending messages
            2. Create HLLSet from hashes collected since last checkpoint
            3. Store HLLSet key for later UNION at commit
        
        Returns:
            CheckpointResult with temp HLLSet key
        """
        if not self.session_id:
            raise RuntimeError("Session not started")
        
        # Process any pending messages
        self.process_pending()
        
        # Create checkpoint ID
        checkpoint_id = checkpoint_id or f"cp-{len(self._checkpoint_keys)}"
        
        # Get hashes since last checkpoint
        hashes = self.stats.checkpoint_hashes
        tokens_in_checkpoint = len(hashes)
        
        if hashes:
            # Create HLLSet from checkpoint hashes
            hllset_key = self.client.execute_command('HLLSET.CREATEHASH', *hashes)
            hllset_key = self._decode_value(hllset_key)
            cardinality = self._get_cardinality(hllset_key)
            
            # Store checkpoint key for UNION at commit
            self._checkpoint_keys.append(hllset_key)
            
            # Clear checkpoint hashes for next batch
            self.stats.checkpoint_hashes = []
        else:
            hllset_key = None
            cardinality = 0
        
        result = CheckpointResult(
            checkpoint_id=checkpoint_id,
            hllset_key=hllset_key,
            cardinality=cardinality,
            tokens_in_checkpoint=tokens_in_checkpoint,
            total_tokens=self.stats.tokens_processed,
        )
        
        self._checkpoint_results.append(result)
        
        # Store checkpoint metadata
        if hllset_key:
            self.client.hset(
                f"{self.config.stream_prefix}:checkpoint:{self.session_id}:{checkpoint_id}",
                mapping={
                    'checkpoint_id': checkpoint_id,
                    'hllset_key': hllset_key,
                    'cardinality': cardinality,
                    'tokens_in_checkpoint': tokens_in_checkpoint,
                    'total_tokens': self.stats.tokens_processed,
                    'timestamp': time.time(),
                }
            )
        
        return result
    
    # =========================================================================
    # Commit — UNION all checkpoints → final SHA1
    # =========================================================================
    
    def commit(self) -> CommitResult:
        """
        Commit the session — UNION all checkpoint HLLSets.
        
        Workflow:
            1. Process remaining messages
            2. Create final checkpoint if there are pending hashes
            3. UNION all checkpoint HLLSets → final SHA1 key
            4. Cleanup stream
        
        Returns:
            CommitResult with final HLLSet key
        """
        if not self.session_id:
            raise RuntimeError("Session not started")
        
        # Process remaining messages
        self.process_pending()
        
        # Create final checkpoint for any remaining hashes
        if self.stats.checkpoint_hashes:
            self.checkpoint(checkpoint_id=f"cp-final")
        
        # UNION all checkpoint HLLSets
        if len(self._checkpoint_keys) == 0:
            hllset_key = None
            cardinality = 0
        elif len(self._checkpoint_keys) == 1:
            # Only one checkpoint — use it directly
            hllset_key = self._checkpoint_keys[0]
            cardinality = self._get_cardinality(hllset_key)
        else:
            # Multiple checkpoints — UNION them
            # HLLSET.UNION takes two keys at a time, so we chain
            hllset_key = self._checkpoint_keys[0]
            for cp_key in self._checkpoint_keys[1:]:
                union_key = self.client.execute_command('HLLSET.UNION', hllset_key, cp_key)
                hllset_key = self._decode_value(union_key)
            cardinality = self._get_cardinality(hllset_key)
        
        elapsed = time.time() - self.stats.start_time
        
        # Count LUT entries for this session
        try:
            query = Query(f"@session:{{{self.session_id}}}")
            results = self.client.ft(self.index_name).search(query)
            lut_entries = results.total
        except:
            lut_entries = self.stats.tokens_processed
        
        result = CommitResult(
            session_id=self.session_id,
            hllset_key=hllset_key,
            cardinality=cardinality,
            total_tokens=self.stats.tokens_processed,
            tokens_by_layer=dict(self.stats.tokens_by_layer),
            checkpoint_keys=list(self._checkpoint_keys),
            elapsed_seconds=elapsed,
            lut_entries=lut_entries,
        )
        
        # Update session metadata
        self.client.hset(
            f"{self.config.stream_prefix}:meta:{self.session_id}",
            mapping={
                'status': 'committed',
                'hllset_key': hllset_key or '',
                'cardinality': cardinality,
                'total_tokens': self.stats.tokens_processed,
                'checkpoint_count': len(self._checkpoint_keys),
                'commit_time': time.time(),
            }
        )
        
        # Optionally delete intermediate checkpoints
        if self.config.delete_checkpoints_after_commit and len(self._checkpoint_keys) > 1:
            for cp_key in self._checkpoint_keys[:-1]:  # Keep final key
                try:
                    self.client.execute_command('HLLSET.DEL', cp_key)
                except:
                    pass
        
        # Optionally trim stream
        if self.config.trim_after_commit:
            self.client.xtrim(self.stream_key, maxlen=0)
        
        return result
    
    # =========================================================================
    # Query API
    # =========================================================================
    
    def get_hllset_info(self, key: str) -> Dict:
        """Get HLLSet info by key."""
        info = self.client.execute_command('HLLSET.INFO', key)
        result = {}
        for i in range(0, len(info), 2):
            k = info[i].decode() if isinstance(info[i], bytes) else info[i]
            v = info[i+1].decode() if isinstance(info[i+1], bytes) else info[i+1]
            result[k] = v
        return result
    
    def get_positions(self, key: str) -> List[Tuple[int, int]]:
        """Get active (reg, zeros) positions from HLLSet."""
        positions_flat = self.client.execute_command('HLLSET.POSITIONS', key)
        return [(positions_flat[i], positions_flat[i+1]) 
                for i in range(0, len(positions_flat), 2)]
    
    def lookup(self, reg: int, zeros: int) -> List[Dict]:
        """Look up tokens at a specific position."""
        query = Query(f"@reg:[{reg} {reg}] @zeros:[{zeros} {zeros}]")
        results = self.client.ft(self.index_name).search(query)
        
        entries = []
        for doc in results.docs:
            entries.append({
                'token': doc.token,
                'reg': int(doc.reg),
                'zeros': int(doc.zeros),
                'layer': int(doc.layer),
                'first_token': getattr(doc, 'first_token', ''),
            })
        return entries
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def cleanup(self, delete_hllsets: bool = False):
        """
        Clean up session resources.
        
        Args:
            delete_hllsets: Also delete created HLLSets
        """
        if not self.session_id:
            return
        
        # Delete stream
        self.client.delete(self.stream_key)
        
        # Delete session metadata
        self.client.delete(f"{self.config.stream_prefix}:meta:{self.session_id}")
        
        # Delete checkpoint metadata
        for cp in self._checkpoint_results:
            self.client.delete(
                f"{self.config.stream_prefix}:checkpoint:{self.session_id}:{cp.checkpoint_id}"
            )
            if delete_hllsets and cp.hllset_key:
                try:
                    self.client.execute_command('HLLSET.DEL', cp.hllset_key)
                except:
                    pass
        
        self.session_id = None
        self.stream_key = None
        self.stats = None
        self._checkpoint_keys = []
        self._checkpoint_results = []


# =============================================================================
# Convenience Functions
# =============================================================================

def create_session(
    client: redis.Redis,
    index_name: str = "tokenlut:idx",
    session_id: Optional[str] = None
) -> TokenLUTSession:
    """Create and start a streaming session."""
    session = TokenLUTSession(client, index_name=index_name)
    session.create_index(drop_existing=False)
    session.start(session_id=session_id)
    return session


def quick_ingest(
    client: redis.Redis,
    unigrams: List[str],
    bigrams: Optional[List[Tuple[str, str]]] = None,
    trigrams: Optional[List[Tuple[str, str]]] = None,
    index_name: str = "tokenlut:idx"
) -> CommitResult:
    """
    Quick ingestion — single-shot token ingestion with checkpoints.
    
    Creates session, ingests all tokens with checkpoints, commits.
    """
    session = create_session(client, index_name=index_name)
    
    # Layer 0: unigrams
    p0 = session.create_producer(layer=0)
    p0.send_unigrams(unigrams)
    session.checkpoint("unigrams")
    
    # Layer 1: bigrams
    if bigrams:
        p1 = session.create_producer(layer=1)
        p1.send_ngrams(bigrams)
        session.checkpoint("bigrams")
    
    # Layer 2: trigrams
    if trigrams:
        p2 = session.create_producer(layer=2)
        p2.send_ngrams(trigrams)
        session.checkpoint("trigrams")
    
    return session.commit()
