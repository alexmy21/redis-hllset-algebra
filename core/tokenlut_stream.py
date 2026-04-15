"""
TokenLUT Streaming Module — Batch Token Ingestion with Redis Streams

This module provides a streaming interface for populating TokenLUT indexes,
eliminating round-trip overhead and ensuring hash consistency.

Architecture:
    ┌─────────────┐    XADD     ┌─────────────┐    consumer     ┌─────────────┐
    │   Client    │ ─────────▶ │   Stream    │ ─────────────▶ │  LUT Index  │
    │  (tokens)   │             │ tokenlut:in │                 │ (RediSearch)│
    └─────────────┘             └─────────────┘                 └─────────────┘

Usage:
    from core.tokenlut_stream import TokenLUTStream
    
    stream = TokenLUTStream(redis_client, index_name='vocab:lut')
    
    # Batch ingestion
    tokens = ['apple', 'banana', 'cherry']
    stream.ingest_tokens(tokens, layer=0)
    
    # N-gram ingestion with first_token
    bigrams = [('quick brown', 'quick'), ('brown fox', 'brown')]
    stream.ingest_ngrams(bigrams, layer=1)
    
    # Process stream (consumer side)
    stream.process_pending()

Hash Modes:
    - PYTHON: Use Python's MurmurHash64A (requires HLLSET.CREATEHASH)
    - RUST: Use Rust's murmur3_x64_128 via HLLSET.LUT.ADD (future)
    
When RUST mode is available, the streaming consumer runs server-side,
eliminating all hash computation from the client.
"""

from typing import List, Dict, Tuple, Optional, Iterator, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import redis
from redis.commands.search.field import TextField, NumericField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query


class HashMode(Enum):
    """Hash computation mode."""
    PYTHON = "python"  # Client-side Python hash (current)
    RUST = "rust"      # Server-side Rust hash (future HLLSET.LUT.ADD)


@dataclass
class StreamConfig:
    """Configuration for TokenLUT streaming."""
    stream_key: str = "tokenlut:stream"
    consumer_group: str = "tokenlut:consumers"
    consumer_name: str = "worker-1"
    batch_size: int = 1000
    block_ms: int = 5000
    max_retries: int = 3
    trim_threshold: int = 10000  # MAXLEN for stream trimming


@dataclass 
class IngestStats:
    """Statistics for ingestion operations."""
    tokens_submitted: int = 0
    tokens_processed: int = 0
    tokens_failed: int = 0
    batches_submitted: int = 0
    batches_processed: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time
    
    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds > 0:
            return self.tokens_processed / self.elapsed_seconds
        return 0.0
    
    def to_dict(self) -> Dict:
        return {
            'tokens_submitted': self.tokens_submitted,
            'tokens_processed': self.tokens_processed,
            'tokens_failed': self.tokens_failed,
            'batches_submitted': self.batches_submitted,
            'batches_processed': self.batches_processed,
            'elapsed_seconds': round(self.elapsed_seconds, 2),
            'tokens_per_second': round(self.tokens_per_second, 1)
        }


class TokenLUTStream:
    """
    Streaming interface for TokenLUT population.
    
    Supports batch token ingestion via Redis Streams, with automatic
    hash computation and RediSearch index population.
    
    Args:
        client: Redis client instance
        index_name: RediSearch index name for the LUT
        prefix: Key prefix for LUT entries
        hash_mode: PYTHON (client-side) or RUST (server-side, future)
        config: StreamConfig for tuning behavior
        hash_func: Optional custom hash function (token -> (reg, zeros, hash_full))
    """
    
    # RediSearch schema for TokenLUT (collision-aware with JSON arrays)
    SCHEMA = (
        NumericField("reg", sortable=True),
        NumericField("zeros", sortable=True),
        NumericField("hash_full"),
        NumericField("layer", sortable=True),
        NumericField("collision_count", sortable=True),  # For observability
        TagField("first_tokens_tag", separator=","),     # For efficient queries
        TextField("first_tokens"),  # JSON array of first tokens
        TextField("tokens"),        # JSON array of token arrays
    )
    
    def __init__(
        self,
        client: redis.Redis,
        index_name: str = "tokenlut:idx",
        prefix: str = "tokenlut:entry:",
        hash_mode: HashMode = HashMode.PYTHON,
        config: Optional[StreamConfig] = None,
        hash_func: Optional[Callable[[str], Tuple[int, int, int]]] = None
    ):
        self.client = client
        self.index_name = index_name
        self.prefix = prefix
        self.hash_mode = hash_mode
        self.config = config or StreamConfig()
        self.stats = IngestStats()
        
        # Set up hash function
        if hash_func:
            self._hash_func = hash_func
        else:
            # Import default hash config
            from .hllset import DEFAULT_HASH_CONFIG
            self._hash_config = DEFAULT_HASH_CONFIG
            self._hash_func = self._default_hash
    
    def _default_hash(self, token: str) -> Tuple[int, int, int]:
        """Default hash using Python's MurmurHash64A."""
        hash_full = self._hash_config.hash(token)
        reg, zeros = self._hash_config.hash_to_reg_zeros(token)
        return reg, zeros, hash_full
    
    # =========================================================================
    # Index Management
    # =========================================================================
    
    def create_index(self, drop_existing: bool = False) -> bool:
        """Create the RediSearch index for TokenLUT."""
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
    
    def drop_index(self, delete_documents: bool = True) -> bool:
        """Drop the RediSearch index."""
        try:
            self.client.ft(self.index_name).dropindex(delete_documents=delete_documents)
            return True
        except redis.ResponseError:
            return False
    
    def ensure_consumer_group(self) -> bool:
        """Ensure the consumer group exists for the stream."""
        try:
            self.client.xgroup_create(
                self.config.stream_key,
                self.config.consumer_group,
                id='0',
                mkstream=True
            )
            return True
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                return False  # Group already exists
            raise
    
    # =========================================================================
    # Producer API — Batch Token Submission
    # =========================================================================
    
    def ingest_tokens(
        self,
        tokens: List[str],
        layer: int = 0,
        batch_size: Optional[int] = None
    ) -> IngestStats:
        """
        Ingest a batch of tokens (unigrams) into the LUT.
        
        Args:
            tokens: List of token strings
            layer: N-gram layer (0 for unigrams)
            batch_size: Override default batch size
            
        Returns:
            IngestStats with submission counts
        """
        batch_size = batch_size or self.config.batch_size
        
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size]
            self._submit_batch(batch, layer=layer)
            self.stats.batches_submitted += 1
        
        return self.stats
    
    def ingest_ngrams(
        self,
        ngrams: List[Tuple[str, str]],
        layer: int = 1,
        batch_size: Optional[int] = None
    ) -> IngestStats:
        """
        Ingest n-grams with their first_token for triangulation.
        
        Args:
            ngrams: List of (ngram_string, first_token) tuples
            layer: N-gram layer (1 for bigrams, 2 for trigrams, etc.)
            batch_size: Override default batch size
            
        Returns:
            IngestStats with submission counts
        """
        batch_size = batch_size or self.config.batch_size
        
        for i in range(0, len(ngrams), batch_size):
            batch = ngrams[i:i + batch_size]
            self._submit_ngram_batch(batch, layer=layer)
            self.stats.batches_submitted += 1
        
        return self.stats
    
    def ingest_vocabulary(
        self,
        vocabulary: Iterator[str],
        layer: int = 0,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> IngestStats:
        """
        Ingest a large vocabulary from an iterator.
        
        Useful for streaming from files or generators without
        loading entire vocabulary into memory.
        
        Args:
            vocabulary: Iterator yielding token strings
            layer: N-gram layer
            progress_callback: Called with count after each batch
            
        Returns:
            IngestStats with final counts
        """
        batch = []
        count = 0
        
        for token in vocabulary:
            batch.append(token)
            count += 1
            
            if len(batch) >= self.config.batch_size:
                self._submit_batch(batch, layer=layer)
                self.stats.batches_submitted += 1
                batch = []
                
                if progress_callback:
                    progress_callback(count)
        
        # Submit remaining
        if batch:
            self._submit_batch(batch, layer=layer)
            self.stats.batches_submitted += 1
        
        return self.stats
    
    def _submit_batch(self, tokens: List[str], layer: int = 0) -> List[str]:
        """Submit a batch of tokens to the stream."""
        message_ids = []
        pipe = self.client.pipeline()
        
        for token in tokens:
            # Add to stream
            pipe.xadd(
                self.config.stream_key,
                {
                    'token': token,
                    'layer': layer,
                    'first_token': ''
                },
                maxlen=self.config.trim_threshold
            )
            self.stats.tokens_submitted += 1
        
        results = pipe.execute()
        message_ids = [r for r in results if r]
        return message_ids
    
    def _submit_ngram_batch(
        self,
        ngrams: List[Tuple[str, str]],
        layer: int = 1
    ) -> List[str]:
        """Submit a batch of n-grams to the stream."""
        message_ids = []
        pipe = self.client.pipeline()
        
        for ngram, first_token in ngrams:
            pipe.xadd(
                self.config.stream_key,
                {
                    'token': ngram,
                    'layer': layer,
                    'first_token': first_token
                },
                maxlen=self.config.trim_threshold
            )
            self.stats.tokens_submitted += 1
        
        results = pipe.execute()
        message_ids = [r for r in results if r]
        return message_ids
    
    # =========================================================================
    # Consumer API — Stream Processing
    # =========================================================================
    
    def process_pending(self, max_messages: Optional[int] = None) -> int:
        """
        Process all pending messages in the stream.
        
        This is the consumer side that reads from the stream,
        computes hashes (in PYTHON mode), and stores in RediSearch.
        
        Args:
            max_messages: Maximum messages to process (None = all)
            
        Returns:
            Number of messages processed
        """
        self.ensure_consumer_group()
        processed = 0
        
        while True:
            # Read from stream
            messages = self.client.xreadgroup(
                self.config.consumer_group,
                self.config.consumer_name,
                {self.config.stream_key: '>'},
                count=self.config.batch_size,
                block=100  # Short block for batch processing
            )
            
            if not messages:
                break
            
            for stream_name, entries in messages:
                for msg_id, data in entries:
                    try:
                        self._process_message(msg_id, data)
                        self.client.xack(
                            self.config.stream_key,
                            self.config.consumer_group,
                            msg_id
                        )
                        processed += 1
                        self.stats.tokens_processed += 1
                        
                        if max_messages and processed >= max_messages:
                            return processed
                            
                    except Exception as e:
                        self.stats.tokens_failed += 1
                        # Log error but continue processing
                        print(f"Error processing {msg_id}: {e}")
        
        self.stats.batches_processed += 1
        return processed
    
    def process_stream(
        self,
        timeout_ms: int = 0,
        callback: Optional[Callable[[str, Dict], None]] = None
    ) -> Iterator[Tuple[str, Dict]]:
        """
        Continuously process the stream (blocking consumer).
        
        Args:
            timeout_ms: Block timeout (0 = forever)
            callback: Optional callback for each processed entry
            
        Yields:
            (message_id, entry_data) tuples
        """
        self.ensure_consumer_group()
        
        while True:
            messages = self.client.xreadgroup(
                self.config.consumer_group,
                self.config.consumer_name,
                {self.config.stream_key: '>'},
                count=self.config.batch_size,
                block=timeout_ms or self.config.block_ms
            )
            
            if not messages:
                if timeout_ms == 0:
                    continue
                else:
                    break
            
            for stream_name, entries in messages:
                for msg_id, data in entries:
                    try:
                        entry = self._process_message(msg_id, data)
                        self.client.xack(
                            self.config.stream_key,
                            self.config.consumer_group,
                            msg_id
                        )
                        self.stats.tokens_processed += 1
                        
                        if callback:
                            callback(msg_id, entry)
                        
                        yield msg_id, entry
                        
                    except Exception as e:
                        self.stats.tokens_failed += 1
                        print(f"Error processing {msg_id}: {e}")
    
    def _process_message(self, msg_id: str, data: Dict) -> Dict:
        """Process a single stream message and store in LUT (collision-aware)."""
        import json
        
        token = data.get('token', data.get(b'token', b'')).decode() \
            if isinstance(data.get('token', data.get(b'token', b'')), bytes) \
            else data.get('token', '')
        layer = int(data.get('layer', data.get(b'layer', 0)))
        first_token_input = data.get('first_token', data.get(b'first_token', b'')).decode() \
            if isinstance(data.get('first_token', data.get(b'first_token', b'')), bytes) \
            else data.get('first_token', '')
        
        # Compute hash (Python mode)
        if self.hash_mode == HashMode.PYTHON:
            reg, zeros, hash_full = self._hash_func(token)
        else:
            # Future: RUST mode would use HLLSET.LUT.ADD
            raise NotImplementedError("RUST hash mode not yet implemented")
        
        # Parse token into parts
        token_parts = token.split() if " " in token else [token]
        first_token = first_token_input or token_parts[0]
        
        # Collision-aware storage
        entry_key = f"{self.prefix}{hash_full}"
        
        # Check for existing entry (collision)
        existing = self.client.hgetall(entry_key)
        if existing:
            # Merge with existing
            existing_first_tokens = json.loads(
                existing.get(b'first_tokens', existing.get('first_tokens', '[]'))
                if isinstance(existing.get(b'first_tokens', existing.get('first_tokens', '[]')), (str, bytes))
                else '[]'
            )
            existing_tokens = json.loads(
                existing.get(b'tokens', existing.get('tokens', '[]'))
                if isinstance(existing.get(b'tokens', existing.get('tokens', '[]')), (str, bytes))
                else '[]'
            )
            
            # Handle bytes
            if isinstance(existing_first_tokens, bytes):
                existing_first_tokens = json.loads(existing_first_tokens.decode())
            if isinstance(existing_tokens, bytes):
                existing_tokens = json.loads(existing_tokens.decode())
            
            # Add new first_token if not present
            if first_token not in existing_first_tokens:
                existing_first_tokens.append(first_token)
            
            # Add new token_parts if n-gram and not present
            if layer > 0 and token_parts not in existing_tokens:
                existing_tokens.append(token_parts)
            
            entry = {
                'reg': reg,
                'zeros': zeros,
                'hash_full': hash_full,
                'layer': layer,
                'first_tokens': json.dumps(existing_first_tokens),
                'tokens': json.dumps(existing_tokens)
            }
        else:
            # New entry
            if layer == 0:
                entry = {
                    'reg': reg,
                    'zeros': zeros,
                    'hash_full': hash_full,
                    'layer': layer,
                    'first_tokens': json.dumps([first_token]),
                    'tokens': json.dumps([])
                }
            else:
                entry = {
                    'reg': reg,
                    'zeros': zeros,
                    'hash_full': hash_full,
                    'layer': layer,
                    'first_tokens': json.dumps([first_token]),
                    'tokens': json.dumps([token_parts])
                }
        
        self.client.hset(entry_key, mapping=entry)
        return entry
    
    # =========================================================================
    # Direct API (bypass stream for simple cases) - collision-aware
    # =========================================================================
    
    def add_token(
        self,
        token: str,
        layer: int = 0,
        first_token: str = ""
    ) -> Dict:
        """
        Add a single token directly (no stream), collision-aware.
        
        For simple cases where streaming overhead isn't needed.
        """
        import json
        
        reg, zeros, hash_full = self._hash_func(token)
        token_parts = token.split() if " " in token else [token]
        first_token = first_token or token_parts[0]
        
        entry_key = f"{self.prefix}{hash_full}"
        
        # Check for collision
        existing = self.client.hgetall(entry_key)
        if existing:
            # Merge
            existing_first_tokens = json.loads(
                existing.get(b'first_tokens', existing.get('first_tokens', '[]'))
            )
            existing_tokens = json.loads(
                existing.get(b'tokens', existing.get('tokens', '[]'))
            )
            if isinstance(existing_first_tokens, bytes):
                existing_first_tokens = json.loads(existing_first_tokens.decode())
            if isinstance(existing_tokens, bytes):
                existing_tokens = json.loads(existing_tokens.decode())
            
            if first_token not in existing_first_tokens:
                existing_first_tokens.append(first_token)
            if layer > 0 and token_parts not in existing_tokens:
                existing_tokens.append(token_parts)
            
            entry = {
                'reg': reg,
                'zeros': zeros,
                'hash_full': hash_full,
                'layer': layer,
                'first_tokens': json.dumps(existing_first_tokens),
                'tokens': json.dumps(existing_tokens)
            }
        else:
            if layer == 0:
                entry = {
                    'reg': reg,
                    'zeros': zeros,
                    'hash_full': hash_full,
                    'layer': layer,
                    'first_tokens': json.dumps([first_token]),
                    'tokens': json.dumps([])
                }
            else:
                entry = {
                    'reg': reg,
                    'zeros': zeros,
                    'hash_full': hash_full,
                    'layer': layer,
                    'first_tokens': json.dumps([first_token]),
                    'tokens': json.dumps([token_parts])
                }
        
        self.client.hset(entry_key, mapping=entry)
        self.stats.tokens_processed += 1
        return entry
    
    def add_tokens_pipeline(self, tokens: List[str], layer: int = 0) -> int:
        """
        Add multiple tokens using Redis pipeline (no stream), collision-aware.
        
        Groups by hash to handle collisions, then uses pipelining.
        """
        import json
        
        # Group by hash to handle collisions
        by_hash: Dict[int, Dict] = {}
        
        for token in tokens:
            reg, zeros, hash_full = self._hash_func(token)
            token_parts = token.split() if " " in token else [token]
            first_token = token_parts[0]
            
            if hash_full in by_hash:
                # Collision - merge
                existing = by_hash[hash_full]
                if first_token not in existing['first_tokens']:
                    existing['first_tokens'].append(first_token)
                if layer > 0 and token_parts not in existing['tokens']:
                    existing['tokens'].append(token_parts)
            else:
                if layer == 0:
                    by_hash[hash_full] = {
                        'reg': reg,
                        'zeros': zeros,
                        'hash_full': hash_full,
                        'layer': layer,
                        'first_tokens': [first_token],
                        'tokens': []
                    }
                else:
                    by_hash[hash_full] = {
                        'reg': reg,
                        'zeros': zeros,
                        'hash_full': hash_full,
                        'layer': layer,
                        'first_tokens': [first_token],
                        'tokens': [token_parts]
                    }
        
        # Write merged entries
        pipe = self.client.pipeline()
        for hash_full, entry in by_hash.items():
            entry_key = f"{self.prefix}{hash_full}"
            # Convert lists to JSON
            entry['first_tokens'] = json.dumps(entry['first_tokens'])
            entry['tokens'] = json.dumps(entry['tokens'])
            pipe.hset(entry_key, mapping=entry)
        
        results = pipe.execute()
        count = len(by_hash)
        self.stats.tokens_processed += count
        return count
        return count
    
    # =========================================================================
    # Query API
    # =========================================================================
    
    def lookup(self, reg: int, zeros: int) -> List[Dict]:
        """Look up tokens at a specific (reg, zeros) position."""
        query = Query(f"@reg:[{reg} {reg}] @zeros:[{zeros} {zeros}]")
        results = self.client.ft(self.index_name).search(query)
        
        entries = []
        for doc in results.docs:
            entries.append({
                'token': doc.token,
                'reg': int(doc.reg),
                'zeros': int(doc.zeros),
                'hash_full': int(doc.hash_full),
                'layer': int(doc.layer),
                'first_token': getattr(doc, 'first_token', '')
            })
        return entries
    
    def lookup_positions(
        self,
        positions: List[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], List[Dict]]:
        """Look up candidates for multiple positions."""
        result = {}
        for reg, zeros in positions:
            result[(reg, zeros)] = self.lookup(reg, zeros)
        return result
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Get current ingestion statistics."""
        return self.stats.to_dict()
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = IngestStats()
    
    def stream_info(self) -> Dict:
        """Get information about the stream."""
        try:
            info = self.client.xinfo_stream(self.config.stream_key)
            return {
                'length': info.get('length', 0),
                'first_entry': info.get('first-entry'),
                'last_entry': info.get('last-entry'),
                'groups': info.get('groups', 0)
            }
        except redis.ResponseError:
            return {'length': 0, 'groups': 0}
    
    def stream_length(self) -> int:
        """Get current stream length."""
        try:
            return self.client.xlen(self.config.stream_key)
        except redis.ResponseError:
            return 0
    
    def trim_stream(self, maxlen: Optional[int] = None) -> int:
        """Trim the stream to a maximum length."""
        maxlen = maxlen or self.config.trim_threshold
        return self.client.xtrim(self.config.stream_key, maxlen=maxlen)
    
    def delete_stream(self) -> bool:
        """Delete the stream entirely."""
        return bool(self.client.delete(self.config.stream_key))
    
    def count(self) -> int:
        """Count total entries in the LUT index."""
        try:
            info = self.client.ft(self.index_name).info()
            return int(info.get('num_docs', 0))
        except redis.ResponseError:
            return 0


# =============================================================================
# Convenience Functions
# =============================================================================

def create_stream_lut(
    client: redis.Redis,
    index_name: str = "tokenlut:idx",
    prefix: str = "tokenlut:entry:",
    stream_key: str = "tokenlut:stream",
    batch_size: int = 1000
) -> TokenLUTStream:
    """
    Create and initialize a TokenLUTStream with index.
    
    Convenience function that creates the index and stream
    consumer group in one call.
    """
    config = StreamConfig(
        stream_key=stream_key,
        batch_size=batch_size
    )
    
    stream = TokenLUTStream(
        client=client,
        index_name=index_name,
        prefix=prefix,
        config=config
    )
    
    stream.create_index(drop_existing=False)
    stream.ensure_consumer_group()
    
    return stream


def ingest_file(
    client: redis.Redis,
    filepath: str,
    index_name: str = "tokenlut:idx",
    layer: int = 0,
    encoding: str = 'utf-8'
) -> IngestStats:
    """
    Ingest tokens from a file (one per line).
    
    Convenience function for vocabulary file ingestion.
    """
    stream = create_stream_lut(client, index_name=index_name)
    
    def token_generator():
        with open(filepath, 'r', encoding=encoding) as f:
            for line in f:
                token = line.strip()
                if token:
                    yield token
    
    # Submit all tokens
    stream.ingest_vocabulary(token_generator(), layer=layer)
    
    # Process immediately
    stream.process_pending()
    
    return stream.stats
