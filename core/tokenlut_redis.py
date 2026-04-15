"""
TokenLUT Redis - Token Lookup Table using RediSearch

This module provides a Redis-native implementation of TokenLUT for disambiguation.
It uses RediSearch for fast indexed queries on (reg, zeros) positions.

Schema (collision-aware, JSON arrays):
    - reg: NUMERIC (register index, 0-1023)
    - zeros: NUMERIC (trailing zeros count, 0-31)
    - hash_full: NUMERIC (full 64-bit hash, used as key)
    - layer: NUMERIC (n-gram layer: 0=unigram, 1=bigram, 2=trigram)
    - first_tokens: TEXT (JSON array of first tokens for triangulation)
    - tokens: TEXT (JSON array of token arrays for collided entries)
    
    Examples:
        Unigram (layer=0):  first_tokens=["apple","app"], tokens=[]
        Bigram (layer=1):   first_tokens=["quick","quest"], tokens=[["quick","brown"],["quest","ion"]]
        Trigram (layer=2):  first_tokens=["the","them"], tokens=[["the","quick","brown"],["them","selves","are"]]

Key Queries:
    1. lookup(reg, zeros) → all tokens at position
    2. lookup_register(reg) → all tokens at register (any zeros)
    3. lookup_layer(layer) → all n-grams at layer
    4. first_tokens_at_register(reg, layer) → triangulation support

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional, Iterator
from dataclasses import dataclass, asdict
import json
import redis
from redis.commands.search.field import TextField, NumericField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query


# Default index name
DEFAULT_INDEX = "tokenlut:idx"
DEFAULT_PREFIX = "tokenlut:entry:"


@dataclass
class TokenEntry:
    """
    Entry in the token lookup table for disambiguation (collision-aware).
    
    For hash collisions, multiple tokens share the same hash_full.
    We store them as JSON arrays to preserve all candidates.
    
    Attributes:
        reg: Register index [0, num_registers)
        zeros: Trailing zeros count [0, 31]
        hash_full: Full 64-bit hash (used as Redis key)
        layer: N-gram layer (0=unigram, 1=bigram, 2=trigram)
        first_tokens: List of first tokens (for triangulation)
        tokens: List of token arrays (the n-grams that collided)
        
    Examples:
        Unigram:  first_tokens=["apple","app"], tokens=[]
        Bigram:   first_tokens=["quick","quest"], tokens=[["quick","brown"],["quest","ion"]]
        Trigram:  first_tokens=["the"], tokens=[["the","quick","brown"]]
    """
    reg: int
    zeros: int
    hash_full: int = 0
    layer: int = 0
    first_tokens: List[str] = None  # JSON array of first tokens
    tokens: List[List[str]] = None  # JSON array of token arrays
    
    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.first_tokens is None:
            self.first_tokens = []
        if self.tokens is None:
            self.tokens = []
    
    @property
    def position(self) -> Tuple[int, int]:
        return (self.reg, self.zeros)
    
    @property
    def is_unigram(self) -> bool:
        return self.layer == 0
    
    @property
    def token(self) -> str:
        """Legacy compatibility: return first token or joined n-gram."""
        if self.layer == 0 and self.first_tokens:
            return self.first_tokens[0]
        elif self.tokens:
            return " ".join(self.tokens[0])
        return ""
    
    @property
    def first_token(self) -> str:
        """Legacy compatibility: return first of first_tokens."""
        return self.first_tokens[0] if self.first_tokens else ""
    
    def add_collision(self, first_token: str, token_parts: List[str] = None):
        """
        Add a colliding token to this entry.
        
        Args:
            first_token: The first token (for unigrams, the token itself)
            token_parts: For n-grams, the list of tokens; None for unigrams
        """
        if first_token not in self.first_tokens:
            self.first_tokens.append(first_token)
        if token_parts and token_parts not in self.tokens:
            self.tokens.append(token_parts)
    
    def to_dict(self) -> Dict:
        """Convert to dict for Redis storage (JSON arrays)."""
        return {
            'reg': self.reg,
            'zeros': self.zeros,
            'hash_full': self.hash_full,
            'layer': self.layer,
            'first_tokens': json.dumps(self.first_tokens),
            'tokens': json.dumps(self.tokens),
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TokenEntry':
        """Create from Redis hash (parse JSON arrays)."""
        first_tokens_raw = d.get('first_tokens', '[]')
        tokens_raw = d.get('tokens', '[]')
        
        # Handle both string (from Redis) and already-parsed (from Python)
        if isinstance(first_tokens_raw, str):
            first_tokens = json.loads(first_tokens_raw)
        else:
            first_tokens = first_tokens_raw or []
            
        if isinstance(tokens_raw, str):
            tokens = json.loads(tokens_raw)
        else:
            tokens = tokens_raw or []
        
        return cls(
            reg=int(d.get('reg', 0)),
            zeros=int(d.get('zeros', 0)),
            hash_full=int(d.get('hash_full', 0)),
            layer=int(d.get('layer', 0)),
            first_tokens=first_tokens,
            tokens=tokens,
        )


# Lua script for atomic merge (avoids read-before-write pattern)
MERGE_ENTRY_SCRIPT = """
local key = KEYS[1]
local new_first_tokens = cjson.decode(ARGV[1])
local new_tokens = cjson.decode(ARGV[2])
local reg, zeros, hash_full, layer = ARGV[3], ARGV[4], ARGV[5], ARGV[6]

-- Check if key exists
local exists = redis.call('EXISTS', key)
if exists == 0 then
    -- New entry
    local collision_count = #new_first_tokens
    local first_tokens_tag = table.concat(new_first_tokens, ',')
    redis.call('HSET', key, 
        'reg', reg, 
        'zeros', zeros, 
        'hash_full', hash_full, 
        'layer', layer,
        'first_tokens', ARGV[1], 
        'tokens', ARGV[2],
        'collision_count', collision_count,
        'first_tokens_tag', first_tokens_tag)
    return collision_count
end

-- Merge with existing
local old_ft_json = redis.call('HGET', key, 'first_tokens') or '[]'
local old_tk_json = redis.call('HGET', key, 'tokens') or '[]'
local old_ft = cjson.decode(old_ft_json)
local old_tk = cjson.decode(old_tk_json)

-- Merge first_tokens (deduplicate)
for _, ft in ipairs(new_first_tokens) do
    local found = false
    for _, v in ipairs(old_ft) do 
        if v == ft then found = true; break end 
    end
    if not found then table.insert(old_ft, ft) end
end

-- Merge tokens arrays (deduplicate)
for _, ta in ipairs(new_tokens) do
    local found = false
    for _, v in ipairs(old_tk) do
        if #v == #ta then
            local match = true
            for i, t in ipairs(ta) do 
                if v[i] ~= t then match = false; break end 
            end
            if match then found = true; break end
        end
    end
    if not found then table.insert(old_tk, ta) end
end

-- Write merged entry
local collision_count = #old_ft
local first_tokens_tag = table.concat(old_ft, ',')
redis.call('HSET', key, 
    'first_tokens', cjson.encode(old_ft), 
    'tokens', cjson.encode(old_tk),
    'collision_count', collision_count,
    'first_tokens_tag', first_tokens_tag)
return collision_count
"""


class TokenLUTRedis:
    """
    Token Lookup Table backed by RediSearch.
    
    Provides fast indexed queries for disambiguation:
    - (reg, zeros) → candidate tokens
    - register → all tokens (for parallel disambiguation)
    - layer filtering (unigrams, bigrams, trigrams)
    - triangulation support via first_token linking
    
    Best Practices Implemented:
    - Atomic merge via Lua script (no read-before-write)
    - collision_count field for observability
    - first_tokens_tag for efficient TagField queries
    - Batch collision checking with MGET
    - Direct get_by_hash for O(1) lookups
    
    Usage:
        lut = TokenLUTRedis(redis_client)
        lut.create_index()  # One-time setup
        
        # Add tokens
        lut.add_token("hello", reg=42, zeros=3, hash_full=12345, layer=0)
        
        # Lookup
        candidates = lut.lookup(42, 3)
        
        # Direct lookup by hash (fast path)
        entry = lut.get_by_hash(12345)
        
        # Find high-collision entries
        high_collision = lut.lookup_high_collision(min_count=3)
    """
    
    def __init__(self, client: redis.Redis, 
                 index_name: str = DEFAULT_INDEX,
                 prefix: str = DEFAULT_PREFIX,
                 p_bits: int = 10):
        """
        Initialize TokenLUT with Redis connection.
        
        Args:
            client: Redis client instance
            index_name: RediSearch index name
            prefix: Key prefix for token entries
            p_bits: Precision bits (for validation)
        """
        self.client = client
        self.index_name = index_name
        self.prefix = prefix
        self.p_bits = p_bits
        self.num_registers = 1 << p_bits
        self._entry_counter = 0
        
        # Register Lua script for atomic merge
        self._merge_script = self.client.register_script(MERGE_ENTRY_SCRIPT)
    
    def create_index(self, drop_existing: bool = False) -> bool:
        """
        Create RediSearch index for TokenLUT.
        
        Args:
            drop_existing: If True, drop existing index first
            
        Returns:
            True if index was created, False if it already exists
        """
        try:
            if drop_existing:
                try:
                    self.client.ft(self.index_name).dropindex(delete_documents=True)
                except redis.ResponseError:
                    pass  # Index didn't exist
            
            # Define schema (collision-aware with JSON arrays + TagField)
            schema = (
                NumericField("reg", sortable=True),
                NumericField("zeros", sortable=True),
                NumericField("hash_full"),
                NumericField("layer", sortable=True),
                NumericField("collision_count", sortable=True),  # For observability
                TagField("first_tokens_tag", separator=","),     # For efficient queries
                TextField("first_tokens"),   # JSON array of first tokens
                TextField("tokens"),         # JSON array of token arrays
            )
            
            # Create index
            definition = IndexDefinition(
                prefix=[self.prefix],
                index_type=IndexType.HASH
            )
            
            self.client.ft(self.index_name).create_index(
                schema,
                definition=definition
            )
            return True
            
        except redis.ResponseError as e:
            if "Index already exists" in str(e):
                return False
            raise
    
    def drop_index(self, delete_documents: bool = True):
        """Drop the RediSearch index."""
        try:
            self.client.ft(self.index_name).dropindex(delete_documents=delete_documents)
        except redis.ResponseError:
            pass  # Index didn't exist
    
    def index_exists(self) -> bool:
        """Check if index exists."""
        try:
            self.client.ft(self.index_name).info()
            return True
        except redis.ResponseError:
            return False
    
    # === Add Entries (collision-aware with Lua script) ===
    
    def add_entry(self, entry: TokenEntry) -> str:
        """
        Add or merge a token entry to the LUT using 64-bit hash as the Redis key.
        
        Uses atomic Lua script to avoid read-before-write pattern.
        Single round-trip, no race conditions.
        
        Args:
            entry: TokenEntry to add
        Returns:
            Redis key of the created/updated entry (tokenlut:entry:<hash_full>)
        """
        key = f"{self.prefix}{entry.hash_full}"
        
        # Use Lua script for atomic merge
        collision_count = self._merge_script(
            keys=[key],
            args=[
                json.dumps(entry.first_tokens),
                json.dumps(entry.tokens),
                str(entry.reg),
                str(entry.zeros),
                str(entry.hash_full),
                str(entry.layer)
            ]
        )
        return key
    
    def add_entry_simple(self, entry: TokenEntry) -> str:
        """
        Add entry without collision handling (for bulk inserts where collisions are pre-merged).
        
        Faster than add_entry when you know there are no collisions.
        """
        key = f"{self.prefix}{entry.hash_full}"
        mapping = entry.to_dict()
        # Add derived fields
        mapping['collision_count'] = len(entry.first_tokens)
        mapping['first_tokens_tag'] = ",".join(entry.first_tokens)
        self.client.hset(key, mapping=mapping)
        return key
    
    def add_token(self, token: str, reg: int, zeros: int,
                  hash_full: int = 0, layer: int = 0, 
                  first_token: str = "") -> str:
        """
        Add a token using 64-bit hash as key (collision-aware).
        
        For hash collisions, the token is appended to the existing entry's arrays.
        
        Args:
            token: Token string (space-separated for n-grams)
            reg: Register index
            zeros: Trailing zeros count
            hash_full: Full 64-bit hash
            layer: N-gram layer (0=unigram, 1=bigram, 2=trigram)
            first_token: First token of n-gram (auto-detected if empty)
        Returns:
            Redis key of the created/updated entry (tokenlut:entry:<hash_full>)
        """
        # Parse token into parts
        token_parts = token.split() if " " in token else [token]
        
        # Determine first_token
        if not first_token:
            first_token = token_parts[0]
        
        # Build entry
        if layer == 0:
            # Unigram: first_tokens = [token], tokens = []
            entry = TokenEntry(
                reg=reg, zeros=zeros, hash_full=hash_full, layer=layer,
                first_tokens=[first_token],
                tokens=[]
            )
        else:
            # N-gram: first_tokens = [first_token], tokens = [[token_parts]]
            entry = TokenEntry(
                reg=reg, zeros=zeros, hash_full=hash_full, layer=layer,
                first_tokens=[first_token],
                tokens=[token_parts]
            )
        return self.add_entry(entry)
    
    def add_batch(self, entries: List[TokenEntry], pipeline_size: int = 1000) -> int:
        """
        Add multiple entries efficiently with batch collision detection.
        
        Uses MGET to fetch existing entries in one round-trip,
        merges collisions in Python, then writes back with pipelining.
        
        This is more efficient than individual add_entry calls when:
        - Adding many entries at once
        - Expecting some hash collisions with existing data
        
        Args:
            entries: List of TokenEntry objects
            pipeline_size: Batch size for pipelining
        Returns:
            Number of unique keys written
        """
        if not entries:
            return 0
        
        # Step 1: Group batch entries by hash_full (in-batch collision handling)
        by_hash: Dict[int, TokenEntry] = {}
        for entry in entries:
            if entry.hash_full in by_hash:
                # Merge collision within batch
                existing = by_hash[entry.hash_full]
                for ft in entry.first_tokens:
                    if ft not in existing.first_tokens:
                        existing.first_tokens.append(ft)
                for ta in entry.tokens:
                    if ta not in existing.tokens:
                        existing.tokens.append(ta)
            else:
                # Clone entry to avoid mutating the original
                by_hash[entry.hash_full] = TokenEntry(
                    reg=entry.reg,
                    zeros=entry.zeros,
                    hash_full=entry.hash_full,
                    layer=entry.layer,
                    first_tokens=list(entry.first_tokens),
                    tokens=[list(t) for t in entry.tokens]
                )
        
        # Step 2: Use MGET to fetch all potentially existing entries
        keys = [f"{self.prefix}{h}" for h in by_hash.keys()]
        hash_full_list = list(by_hash.keys())
        
        # Batch fetch with pipelining (HGETALL for each key)
        pipe = self.client.pipeline()
        for key in keys:
            pipe.hgetall(key)
        existing_data = pipe.execute()
        
        # Step 3: Merge with existing Redis data
        for i, (hash_full, entry) in enumerate(by_hash.items()):
            existing = existing_data[i]
            if existing:  # Key already exists in Redis
                # Parse existing entry
                old_ft_json = existing.get(b'first_tokens') or existing.get('first_tokens') or '[]'
                old_tk_json = existing.get(b'tokens') or existing.get('tokens') or '[]'
                
                if isinstance(old_ft_json, bytes):
                    old_ft_json = old_ft_json.decode('utf-8')
                if isinstance(old_tk_json, bytes):
                    old_tk_json = old_tk_json.decode('utf-8')
                
                old_ft = json.loads(old_ft_json)
                old_tk = json.loads(old_tk_json)
                
                # Merge first_tokens
                for ft in entry.first_tokens:
                    if ft not in old_ft:
                        old_ft.append(ft)
                entry.first_tokens = old_ft
                
                # Merge tokens arrays
                for ta in entry.tokens:
                    if ta not in old_tk:
                        old_tk.append(ta)
                entry.tokens = old_tk
        
        # Step 4: Write all merged entries with pipelining
        count = 0
        pipe = self.client.pipeline()
        for hash_full, entry in by_hash.items():
            key = f"{self.prefix}{hash_full}"
            mapping = entry.to_dict()
            # Add derived fields
            mapping['collision_count'] = len(entry.first_tokens)
            mapping['first_tokens_tag'] = ",".join(entry.first_tokens)
            pipe.hset(key, mapping=mapping)
            count += 1
            if count % pipeline_size == 0:
                pipe.execute()
                pipe = self.client.pipeline()
        if count % pipeline_size != 0:
            pipe.execute()
        return count
    
    # === Direct Lookups (O(1) by hash, no RediSearch) ===
    
    def get_by_hash(self, hash_full: int) -> Optional[TokenEntry]:
        """
        Direct O(1) lookup by 64-bit hash without RediSearch.
        
        This is the fastest way to retrieve a known entry.
        Uses direct Redis HGETALL on the key.
        
        Args:
            hash_full: Full 64-bit hash value
        Returns:
            TokenEntry if found, None otherwise
        """
        key = f"{self.prefix}{hash_full}"
        data = self.client.hgetall(key)
        
        if not data:
            return None
        
        # Handle bytes vs string keys from Redis
        decoded = {}
        for k, v in data.items():
            if isinstance(k, bytes):
                k = k.decode('utf-8')
            if isinstance(v, bytes):
                v = v.decode('utf-8')
            decoded[k] = v
        
        return TokenEntry.from_dict(decoded)
    
    def get_by_hash_batch(self, hash_fulls: List[int]) -> Dict[int, Optional[TokenEntry]]:
        """
        Batch O(1) lookup by multiple 64-bit hashes.
        
        Single round-trip using pipelining.
        
        Args:
            hash_fulls: List of full 64-bit hash values
        Returns:
            Dict mapping hash_full to TokenEntry (or None if not found)
        """
        if not hash_fulls:
            return {}
        
        keys = [f"{self.prefix}{h}" for h in hash_fulls]
        
        pipe = self.client.pipeline()
        for key in keys:
            pipe.hgetall(key)
        results = pipe.execute()
        
        entries = {}
        for i, (hash_full, data) in enumerate(zip(hash_fulls, results)):
            if not data:
                entries[hash_full] = None
            else:
                # Handle bytes vs string keys
                decoded = {}
                for k, v in data.items():
                    if isinstance(k, bytes):
                        k = k.decode('utf-8')
                    if isinstance(v, bytes):
                        v = v.decode('utf-8')
                    decoded[k] = v
                entries[hash_full] = TokenEntry.from_dict(decoded)
        
        return entries
    
    def exists_by_hash(self, hash_full: int) -> bool:
        """
        Check if entry exists by hash (O(1)).
        
        Faster than get_by_hash when you only need existence.
        """
        key = f"{self.prefix}{hash_full}"
        return self.client.exists(key) > 0
    
    # === Lookup Queries ===
    
    def lookup(self, reg: int, zeros: int) -> List[TokenEntry]:
        """
        Get all candidate tokens at (reg, zeros) position.
        
        This is the primary disambiguation query.
        
        Args:
            reg: Register index
            zeros: Trailing zeros count
            
        Returns:
            List of TokenEntry at this position
        """
        query = Query(f"@reg:[{reg} {reg}] @zeros:[{zeros} {zeros}]")
        results = self.client.ft(self.index_name).search(query)
        
        return [TokenEntry.from_dict(doc.__dict__) for doc in results.docs]
    
    def lookup_position(self, position: Tuple[int, int]) -> List[TokenEntry]:
        """Get all candidate tokens at position (alias for lookup)."""
        return self.lookup(position[0], position[1])
    
    def has_candidates(self, reg: int, zeros: int) -> bool:
        """Check if position has any candidates."""
        query = Query(f"@reg:[{reg} {reg}] @zeros:[{zeros} {zeros}]").paging(0, 0)
        results = self.client.ft(self.index_name).search(query)
        return results.total > 0
    
    def lookup_register(self, reg: int, layer: Optional[int] = None) -> List[TokenEntry]:
        """
        Get all entries at a specific register.
        
        This is key for parallel disambiguation: tokens at different 
        registers are mutually exclusive.
        
        Args:
            reg: Register index
            layer: Optional layer filter
            
        Returns:
            List of TokenEntry at this register
        """
        if layer is not None:
            query = Query(f"@reg:[{reg} {reg}] @layer:[{layer} {layer}]")
        else:
            query = Query(f"@reg:[{reg} {reg}]")
        
        # Increase limit for potentially many results
        query = query.paging(0, 10000)
        results = self.client.ft(self.index_name).search(query)
        
        return [TokenEntry.from_dict(doc.__dict__) for doc in results.docs]
    
    def lookup_layer(self, layer: int) -> List[TokenEntry]:
        """
        Get all entries at a specific n-gram layer.
        
        Args:
            layer: N-gram layer (0=unigram, 1=bigram, 2=trigram)
            
        Returns:
            List of TokenEntry at this layer
        """
        query = Query(f"@layer:[{layer} {layer}]").paging(0, 10000)
        results = self.client.ft(self.index_name).search(query)
        
        return [TokenEntry.from_dict(doc.__dict__) for doc in results.docs]
    
    # === Triangulation Support (collision-aware) ===
    
    def first_tokens_at_register(self, reg: int, layer: int) -> Set[str]:
        """
        Get set of all first_tokens for entries at a register and layer.
        
        Used for triangulation constraint checking.
        Handles collisions by flattening all first_tokens arrays.
        
        Args:
            reg: Register index
            layer: N-gram layer (1=bigram, 2=trigram)
            
        Returns:
            Set of first_token strings (all collided tokens)
        """
        entries = self.lookup_register(reg, layer=layer)
        result = set()
        for e in entries:
            result.update(e.first_tokens)
        return result
    
    def unigrams_at_register(self, reg: int) -> Set[str]:
        """
        Get all unigram tokens at a specific register.
        
        Handles collisions by flattening all first_tokens arrays.
        
        Args:
            reg: Register index
            
        Returns:
            Set of unigram token strings (all collided tokens)
        """
        entries = self.lookup_register(reg, layer=0)
        result = set()
        for e in entries:
            result.update(e.first_tokens)
        return result
    
    def ngrams_at_register(self, reg: int, layer: int) -> List[List[str]]:
        """
        Get all n-gram token arrays at a specific register and layer.
        
        Handles collisions by flattening all tokens arrays.
        
        Args:
            reg: Register index
            layer: N-gram layer (1=bigram, 2=trigram)
            
        Returns:
            List of token arrays (all collided n-grams)
        """
        entries = self.lookup_register(reg, layer=layer)
        result = []
        for e in entries:
            result.extend(e.tokens)
        return result
    
    def lookup_by_first_token(self, first_token: str, layer: int) -> List[TokenEntry]:
        """
        Find n-grams that start with a given token (TEXT field search).
        
        Useful for forward triangulation.
        Note: Searches in JSON array, may return entries where
        first_token is one of multiple collided first_tokens.
        
        Args:
            first_token: The starting token
            layer: N-gram layer
            
        Returns:
            List of TokenEntry matching (may contain collisions)
        """
        # Search in JSON array field
        escaped = first_token.replace("-", "\\-").replace("@", "\\@").replace('"', '\\"')
        query = Query(f"@first_tokens:{escaped} @layer:[{layer} {layer}]").paging(0, 10000)
        results = self.client.ft(self.index_name).search(query)
        
        return [TokenEntry.from_dict(doc.__dict__) for doc in results.docs]
    
    def lookup_by_first_token_tag(self, first_tokens: List[str], layer: Optional[int] = None) -> List[TokenEntry]:
        """
        Find entries by first_token using TagField (efficient exact match).
        
        This uses the first_tokens_tag field which is more efficient than
        TEXT search when you need exact token matching.
        
        Args:
            first_tokens: List of tokens to search for (OR query)
            layer: Optional layer filter
            
        Returns:
            List of TokenEntry matching any of the first_tokens
        """
        if not first_tokens:
            return []
        
        # Escape special characters for TagField
        escaped = [t.replace(",", "\\,").replace("-", "\\-") for t in first_tokens]
        tag_query = "|".join(escaped)
        
        if layer is not None:
            query = Query(f"@first_tokens_tag:{{{tag_query}}} @layer:[{layer} {layer}]")
        else:
            query = Query(f"@first_tokens_tag:{{{tag_query}}}")
        
        query = query.paging(0, 10000)
        results = self.client.ft(self.index_name).search(query)
        
        return [TokenEntry.from_dict(doc.__dict__) for doc in results.docs]
    
    def lookup_high_collision(self, min_count: int = 2, limit: int = 100) -> List[TokenEntry]:
        """
        Find entries with high collision counts.
        
        Useful for analyzing hash distribution and debugging.
        
        Args:
            min_count: Minimum collision count to include
            limit: Maximum number of results
            
        Returns:
            List of TokenEntry with collision_count >= min_count,
            sorted by collision_count descending
        """
        query = Query(f"@collision_count:[{min_count} +inf]")
        query = query.sort_by("collision_count", asc=False).paging(0, limit)
        results = self.client.ft(self.index_name).search(query)
        
        return [TokenEntry.from_dict(doc.__dict__) for doc in results.docs]
    
    # === Bulk Operations ===
    
    def lookup_positions(self, positions: List[Tuple[int, int]]) -> Dict[Tuple[int, int], List[TokenEntry]]:
        """
        Lookup multiple positions efficiently.
        
        Args:
            positions: List of (reg, zeros) tuples
            
        Returns:
            Dict mapping positions to their entries
        """
        result = {}
        for pos in positions:
            result[pos] = self.lookup(pos[0], pos[1])
        return result
    
    def active_positions(self) -> Set[Tuple[int, int]]:
        """
        Return all positions that have entries.
        
        Note: This can be expensive for large LUTs.
        """
        query = Query("*").paging(0, 100000).return_fields("reg", "zeros")
        results = self.client.ft(self.index_name).search(query)
        
        positions = set()
        for doc in results.docs:
            reg = int(doc.reg)
            zeros = int(doc.zeros)
            positions.add((reg, zeros))
        
        return positions
    
    def active_registers(self) -> Set[int]:
        """Return set of all registers that have entries."""
        positions = self.active_positions()
        return {reg for reg, zeros in positions}
    
    # === Statistics ===
    
    def stats(self) -> Dict:
        """
        Return LUT statistics.
        
        Returns:
            Dict with statistics about the LUT
        """
        try:
            info = self.client.ft(self.index_name).info()
            num_docs = int(info.get('num_docs', 0))
            
            # Count collisions (positions with multiple tokens)
            positions = self.active_positions()
            collisions = 0
            max_collision = 0
            
            for pos in positions:
                count = len(self.lookup(pos[0], pos[1]))
                if count > 1:
                    collisions += 1
                max_collision = max(max_collision, count)
            
            return {
                'total_tokens': num_docs,
                'unique_positions': len(positions),
                'positions_with_collisions': collisions,
                'max_collision_size': max_collision,
                'index_name': self.index_name,
                'index_size_mb': float(info.get('inverted_sz_mb', 0)),
            }
        except redis.ResponseError:
            return {'error': 'Index not found'}
    
    def count(self) -> int:
        """Return total number of entries."""
        try:
            info = self.client.ft(self.index_name).info()
            return int(info.get('num_docs', 0))
        except redis.ResponseError:
            return 0
    
    # === Cleanup ===
    
    def clear(self):
        """Remove all entries (keeps index)."""
        # Find and delete all keys with our prefix
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor, match=f"{self.prefix}*", count=1000)
            if keys:
                self.client.delete(*keys)
            if cursor == 0:
                break
        self._entry_counter = 0
    
    def __len__(self) -> int:
        return self.count()
    
    def __repr__(self) -> str:
        return f"TokenLUTRedis(index={self.index_name}, entries={self.count()})"


# === Integration with HLLSet ===

def populate_lut_from_tokens(
    lut: TokenLUTRedis,
    tokens: List[str],
    hash_fn,  # Function: token → (reg, zeros, hash_full)
    layer: int = 0
) -> int:
    """
    Populate TokenLUT from a list of tokens (collision-aware).
    
    Args:
        lut: TokenLUTRedis instance
        tokens: List of tokens to add (space-separated for n-grams)
        hash_fn: Hash function returning (reg, zeros, hash_full)
        layer: N-gram layer (0=unigram, 1=bigram, 2=trigram)
        
    Returns:
        Number of unique hash keys written
    """
    entries = []
    for token in tokens:
        reg, zeros, hash_full = hash_fn(token)
        token_parts = token.split() if " " in token else [token]
        first_token = token_parts[0]
        
        if layer == 0:
            # Unigram: first_tokens = [token], tokens = []
            entry = TokenEntry(
                reg=reg, zeros=zeros, hash_full=hash_full, layer=layer,
                first_tokens=[first_token],
                tokens=[]
            )
        else:
            # N-gram: first_tokens = [first_token], tokens = [[token_parts]]
            entry = TokenEntry(
                reg=reg, zeros=zeros, hash_full=hash_full, layer=layer,
                first_tokens=[first_token],
                tokens=[token_parts]
            )
        entries.append(entry)
    
    return lut.add_batch(entries)


def lookup_active_positions(
    lut: TokenLUTRedis,
    hllset_active: List[Tuple[int, int]]
) -> Dict[Tuple[int, int], List[TokenEntry]]:
    """
    Given active positions from an HLLSet, lookup all candidate tokens.
    
    This is the main disambiguation entry point:
    1. Get active positions from HLLSet (via HLLSET.DUMP or tensor.active_positions())
    2. For each position, get candidate tokens from LUT
    3. Apply triangulation to narrow down
    
    Args:
        lut: TokenLUTRedis instance
        hllset_active: List of (reg, zeros) from HLLSet
        
    Returns:
        Dict mapping positions to candidate tokens
    """
    return lut.lookup_positions(hllset_active)
