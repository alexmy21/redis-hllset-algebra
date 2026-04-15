"""
HLLSet Disambiguation Module

Provides Python interface for HLLSet-based disambiguation of TokenLUT entries.
Uses zero-copy position matching in Redis via the Rust module.

Design principle: The Rust module streams matches immediately to release Redis,
while Python orchestrates the pipeline and consumes results.

Collision-aware schema:
    LUT entries with the same 64-bit hash are stored together with JSON arrays:
    - first_tokens: ["apple", "app"]  (all first tokens that collided)
    - tokens: [["quick","brown"], ["quest","ion"]]  (all n-grams that collided)
    
    The Candidate dataclass reflects this collision-aware design.
"""

from typing import Optional, Iterator, Callable, List, Dict, Any
import redis
from dataclasses import dataclass, field
import json


@dataclass
class Candidate:
    """
    A candidate token from disambiguation (collision-aware).
    
    When hash collisions occur, a single LUT entry may contain multiple
    tokens. This is reflected in the first_tokens and tokens arrays.
    
    For disambiguation, you typically need to resolve which of the
    collided tokens is the actual match using context or triangulation.
    
    Attributes:
        key: LUT entry key (tokenlut:entry:<hash_full>)
        reg: Register index (0-1023)
        zeros: Trailing zeros count (0-31)
        layer: Layer: 0=unigram, 1=bigram, 2=trigram, etc.
        collision_count: Number of tokens that collided at this hash
        first_tokens: List of first tokens (for unigrams: the tokens themselves)
        tokens: List of token arrays (for n-grams: [["quick","brown"], ["quest","ion"]])
        
    Legacy attributes (for backwards compatibility):
        token: First token string (or first n-gram joined)
        first_token: First of first_tokens
    """
    key: str                              # LUT entry key
    reg: int                              # Register index (0-1023)
    zeros: int                            # Trailing zeros count (0-31)
    layer: int                            # Layer: 0=unigram, 1=bigram, etc.
    collision_count: int = 1              # Number of collided tokens
    first_tokens: List[str] = field(default_factory=list)   # All first tokens
    tokens: List[List[str]] = field(default_factory=list)   # All n-gram arrays
    
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
    
    @property
    def has_collision(self) -> bool:
        """True if this entry has hash collisions (multiple tokens)."""
        return self.collision_count > 1
    
    @property
    def all_tokens(self) -> List[str]:
        """Get all tokens as flat list of strings."""
        if self.layer == 0:
            return list(self.first_tokens)
        else:
            return [" ".join(t) for t in self.tokens]
    
    @classmethod
    def from_dict(cls, d: Dict, key: str = "") -> 'Candidate':
        """Create Candidate from Redis hash dict."""
        first_tokens_raw = d.get('first_tokens', '[]')
        tokens_raw = d.get('tokens', '[]')
        
        # Handle both bytes and string from Redis
        if isinstance(first_tokens_raw, bytes):
            first_tokens_raw = first_tokens_raw.decode('utf-8')
        if isinstance(tokens_raw, bytes):
            tokens_raw = tokens_raw.decode('utf-8')
        
        # Parse JSON arrays
        if isinstance(first_tokens_raw, str):
            first_tokens = json.loads(first_tokens_raw)
        else:
            first_tokens = first_tokens_raw or []
        
        if isinstance(tokens_raw, str):
            tokens = json.loads(tokens_raw)
        else:
            tokens = tokens_raw or []
        
        # Handle collision_count
        collision_count = d.get('collision_count', len(first_tokens) or 1)
        if isinstance(collision_count, bytes):
            collision_count = int(collision_count.decode('utf-8'))
        elif isinstance(collision_count, str):
            collision_count = int(collision_count)
        
        return cls(
            key=key,
            reg=int(d.get('reg', 0)),
            zeros=int(d.get('zeros', 0)),
            layer=int(d.get('layer', 0)),
            collision_count=int(collision_count),
            first_tokens=first_tokens,
            tokens=tokens,
        )


class HLLSetDisambiguator:
    """
    Disambiguate HLLSet positions against TokenLUT entries.
    
    Uses the Rust module for zero-copy position matching, with optional
    streaming to Redis Streams for non-blocking output.
    
    Example:
        >>> disamb = HLLSetDisambiguator(redis_client)
        >>> candidates = list(disamb.candidates("hllset:abc123", "tokenlut:sess1:"))
        >>> for c in candidates:
        ...     print(f"{c.token} at register {c.reg}")
    """
    
    def __init__(self, r: redis.Redis):
        """Initialize with Redis client."""
        self.r = r
    
    def candidates(
        self,
        hllset_key: str,
        lut_prefix: str,
        layer: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Candidate]:
        """
        Find LUT entries matching HLLSet positions.
        
        This uses the HLLSET.CANDIDATES command which performs
        zero-copy position matching in the Rust module.
        
        Args:
            hllset_key: Key of the HLLSet
            lut_prefix: Prefix for LUT entry keys (e.g., "tokenlut:entry:sess1:")
            layer: Optional layer filter (0=unigram, 1=bigram, etc.)
            limit: Optional maximum number of results
            
        Yields:
            Candidate objects for each matching LUT entry
        """
        args = [hllset_key, lut_prefix]
        
        if layer is not None:
            args.extend(["LAYER", str(layer)])
        
        if limit is not None:
            args.extend(["LIMIT", str(limit)])
        
        result = self.r.execute_command("HLLSET.CANDIDATES", *args)
        
        if not result:
            return
        
        # Result is [key, token, layer, first_token, key, token, layer, first_token, ...]
        # But we don't have reg/zeros in the direct response - need to fetch or use stream
        for i in range(0, len(result), 4):
            key = result[i].decode() if isinstance(result[i], bytes) else result[i]
            token = result[i+1].decode() if isinstance(result[i+1], bytes) else result[i+1]
            layer_val = int(result[i+2])
            first_token = result[i+3].decode() if isinstance(result[i+3], bytes) else (result[i+3] or "")
            
            # Fetch full entry from the hash (collision-aware)
            fields = self.r.hgetall(key)
            
            # Decode bytes if needed
            decoded = {}
            for k, v in fields.items():
                if isinstance(k, bytes):
                    k = k.decode('utf-8')
                if isinstance(v, bytes):
                    v = v.decode('utf-8')
                decoded[k] = v
            
            yield Candidate.from_dict(decoded, key=key)
    
    def stream_candidates(
        self,
        hllset_key: str,
        lut_prefix: str,
        stream_key: str,
        layer: Optional[int] = None,
    ) -> int:
        """
        Stream matching LUT entries to a Redis Stream.
        
        The Rust module writes matches immediately to the stream as they're found,
        releasing Redis for other operations. Use this for large-scale disambiguation.
        
        Args:
            hllset_key: Key of the HLLSet
            lut_prefix: Prefix for LUT entry keys
            stream_key: Key of the output stream
            layer: Optional layer filter
            
        Returns:
            Number of entries streamed
        """
        args = [hllset_key, lut_prefix, "STREAM", stream_key]
        
        if layer is not None:
            args.extend(["LAYER", str(layer)])
        
        return self.r.execute_command("HLLSET.CANDIDATES", *args)
    
    def scan_match(
        self,
        hllset_key: str,
        lut_prefix: str,
        stream_key: str,
        layer: Optional[int] = None,
        batch_size: int = 1000,
    ) -> int:
        """
        Full cursor-based scan with streaming output.
        
        Like stream_candidates but handles cursor iteration properly
        for very large LUT prefixes.
        
        Args:
            hllset_key: Key of the HLLSet
            lut_prefix: Prefix for LUT entry keys
            stream_key: Key of the output stream
            layer: Optional layer filter
            batch_size: Number of keys per SCAN iteration
            
        Returns:
            Total number of entries streamed
        """
        args = [hllset_key, lut_prefix, stream_key]
        
        if layer is not None:
            args.extend(["LAYER", str(layer)])
        
        args.extend(["BATCH", str(batch_size)])
        
        return self.r.execute_command("HLLSET.SCANMATCH", *args)
    
    def consume_stream(
        self,
        stream_key: str,
        block_ms: int = 0,
        count: Optional[int] = None,
    ) -> Iterator[Candidate]:
        """
        Consume candidates from a disambiguation stream.
        
        Args:
            stream_key: Key of the stream
            block_ms: Block timeout in ms (0 = no blocking)
            count: Max entries to read
            
        Yields:
            Candidate objects
        """
        last_id = "0"
        
        while True:
            entries = self.r.xread(
                {stream_key: last_id},
                block=block_ms if block_ms > 0 else None,
                count=count
            )
            
            if not entries:
                break
            
            for stream_name, messages in entries:
                for msg_id, fields in messages:
                    last_id = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                    
                    yield Candidate(
                        key=fields.get(b"key", b"").decode(),
                        token=fields.get(b"token", b"").decode(),
                        reg=int(fields.get(b"reg", 0)),
                        zeros=int(fields.get(b"zeros", 0)),
                        layer=int(fields.get(b"layer", 0)),
                        first_token=fields.get(b"first_token", b"").decode()
                    )
    
    def create_position_index(
        self,
        hllset_key: str,
        index_key: str,
    ) -> int:
        """
        Create a sorted set index of HLLSet positions.
        
        Score = reg * 32 + zeros (linearized position).
        Useful for range queries on positions.
        
        Args:
            hllset_key: Key of the HLLSet
            index_key: Key for the output sorted set
            
        Returns:
            Number of positions indexed
        """
        return self.r.execute_command("HLLSET.POSINDEX", hllset_key, index_key)


class DisambiguationPipeline:
    """
    Multi-stage disambiguation pipeline.
    
    Composes multiple filtering steps to narrow down candidates:
    1. Position matching (HLLSET.CANDIDATES)
    2. Layer filtering (unigrams only, bigrams only, etc.)
    3. Triangulation (verify bigrams against their unigrams)
    4. Custom filters
    
    Example:
        >>> pipe = DisambiguationPipeline(redis_client)
        >>> pipe.add_stage("unigrams", layer=0)
        >>> pipe.add_stage("bigrams", layer=1, triangulate=True)
        >>> results = pipe.run("hllset:abc123", "tokenlut:sess1:")
    """
    
    def __init__(self, r: redis.Redis):
        self.r = r
        self.disamb = HLLSetDisambiguator(r)
        self.stages: List[Dict[str, Any]] = []
    
    def add_stage(
        self,
        name: str,
        layer: Optional[int] = None,
        triangulate: bool = False,
        filter_fn: Optional[Callable[[Candidate], bool]] = None,
    ) -> "DisambiguationPipeline":
        """
        Add a filtering stage to the pipeline.
        
        Args:
            name: Stage identifier
            layer: Layer filter
            triangulate: If True, verify bigrams against their unigram positions
            filter_fn: Custom filter function
            
        Returns:
            self for chaining
        """
        self.stages.append({
            "name": name,
            "layer": layer,
            "triangulate": triangulate,
            "filter_fn": filter_fn,
        })
        return self
    
    def run(
        self,
        hllset_key: str,
        lut_prefix: str,
        stream_output: Optional[str] = None,
    ) -> Dict[str, List[Candidate]]:
        """
        Run the disambiguation pipeline.
        
        Args:
            hllset_key: Key of the HLLSet
            lut_prefix: Prefix for LUT entry keys
            stream_output: Optional stream key for final output
            
        Returns:
            Dict mapping stage name to list of candidates
        """
        results = {}
        
        # Get unigram positions for triangulation
        unigram_positions = set()
        
        for stage in self.stages:
            candidates = list(self.disamb.candidates(
                hllset_key,
                lut_prefix,
                layer=stage["layer"]
            ))
            
            # Apply custom filter
            if stage["filter_fn"]:
                candidates = [c for c in candidates if stage["filter_fn"](c)]
            
            # Triangulation: verify bigrams
            if stage["triangulate"]:
                # First pass: collect unigram positions
                if stage["layer"] == 0:
                    unigram_positions = {(c.reg, c.zeros) for c in candidates}
                else:
                    # For bigrams: the first_token should match an existing unigram
                    # This requires looking up the first_token's position
                    # For now, just pass through - proper triangulation needs more work
                    pass
            
            results[stage["name"]] = candidates
            
            # Stream output
            if stream_output:
                for c in candidates:
                    self.r.xadd(
                        f"{stream_output}:{stage['name']}",
                        {
                            "key": c.key,
                            "token": c.token,
                            "reg": c.reg,
                            "zeros": c.zeros,
                            "layer": c.layer,
                            "first_token": c.first_token,
                        }
                    )
        
        return results


# Convenience functions
def disambiguate(
    r: redis.Redis,
    hllset_key: str,
    lut_prefix: str,
    layer: Optional[int] = None,
) -> List[Candidate]:
    """
    Simple disambiguation - find matching LUT entries.
    
    Args:
        r: Redis client
        hllset_key: Key of the HLLSet
        lut_prefix: Prefix for LUT entry keys
        layer: Optional layer filter
        
    Returns:
        List of Candidate objects
    """
    disamb = HLLSetDisambiguator(r)
    return list(disamb.candidates(hllset_key, lut_prefix, layer=layer))


def disambiguate_stream(
    r: redis.Redis,
    hllset_key: str,
    lut_prefix: str,
    stream_key: str,
    layer: Optional[int] = None,
) -> int:
    """
    Streaming disambiguation - write matches to a Redis Stream.
    
    Args:
        r: Redis client
        hllset_key: Key of the HLLSet
        lut_prefix: Prefix for LUT entry keys
        stream_key: Output stream key
        layer: Optional layer filter
        
    Returns:
        Number of entries streamed
    """
    disamb = HLLSetDisambiguator(r)
    return disamb.stream_candidates(hllset_key, lut_prefix, stream_key, layer=layer)
