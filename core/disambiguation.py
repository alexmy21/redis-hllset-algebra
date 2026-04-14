"""
Disambiguation Engine — Recover tokens from HLLSet fingerprints.

This module provides the core disambiguation functionality:

1. TokenEntry: Metadata for hashed tokens (position, layer, hash)
2. TriangulationTensor: Sparse 4D tensor for candidate storage
3. GlobalLayerHLLSets: HLLSets per n-gram layer for fast filtering
4. DisambiguationEngine: Main API for ingestion and recovery
5. DisambiguationResult: Output with candidates and confidence
6. ParallelDisambiguator: Register-parallel disambiguation with De Bruijn graph

The key insight is **triangulation**: by maintaining multiple n-gram layers
(unigrams, bigrams, trigrams), we can intersect candidates across layers
to disambiguate positions with high confidence.

Token order restoration uses **De Bruijn graphs** where:
- Nodes are (k-1)-mers (bigrams for k=3)
- Edges are k-mers (trigrams)
- Edge multiplicity tracks occurrence counts
- Eulerian path gives optimal reconstruction

Architecture:
    
    Input Text → tokenize → n-grams → hash → (reg, zeros, layer)
                                                    ↓
                                        TriangulationTensor
                                                    ↓
    HLLSet → active positions → lookup → candidates → triangulate → tokens
                                                    ↓
                                          De Bruijn Graph
                                                    ↓
                                        Eulerian Path → Sequence

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set, Iterator, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from core.debruijn import DeBruijnGraph

# Import from our modules
from .hllset import HLLSet, murmur_hash64a, HashConfig, DEFAULT_HASH_CONFIG
from .hll_tensor import HLLTensor, TokenLUT, TokenEntry as TensorTokenEntry


# Constants
NUM_LAYERS = 3  # unigram, bigram, trigram
HASH_MOD_BITS = 16  # For compact hash indexing
HASH_MOD = 1 << HASH_MOD_BITS

# Boundary markers for sequence reconstruction
START_TOKEN = "<S>"   # Marks sequence start
END_TOKEN = "</S>"    # Marks sequence end


@dataclass
class TokenEntry:
    """
    Complete token entry for disambiguation.
    
    Extends the basic TokenEntry with full hash decomposition
    and n-gram metadata.
    
    The first_token field enables efficient triangulation:
    - For unigrams: first_token == token[0]
    - For bigrams: first_token links back to the starting unigram
    - For trigrams: first_token links back to the starting unigram
    
    This allows O(1) constraint checking during parallel disambiguation.
    """
    token: Tuple[str, ...]  # N-gram tuple ("quick",) or ("quick", "brown")
    hash_full: int          # Full 64-bit MurmurHash
    hash_mod: int           # Compressed hash (HASH_MOD_BITS)
    reg: int                # HLL register index
    zeros: int              # Trailing zeros count
    layer: int              # N-gram layer (0=uni, 1=bi, 2=tri)
    ref_hash: int = 0       # Hash of first token (for linking)
    first_token: str = ""   # First token of n-gram (for triangulation)
    
    @classmethod
    def from_ntoken(cls, ntoken: Tuple[str, ...], 
                    p_bits: int = 10, seed: int = 42) -> 'TokenEntry':
        """
        Create TokenEntry from n-gram tuple.
        
        Args:
            ntoken: Tuple of tokens, e.g., ("quick",) or ("quick", "brown")
            p_bits: HLL precision bits
            seed: Hash seed
            
        Returns:
            TokenEntry with computed hash decomposition
        """
        # Join tokens for hashing
        token_str = " ".join(ntoken)
        token_bytes = token_str.encode('utf-8')
        hash_full = murmur_hash64a(token_bytes, seed)
        
        # Decompose hash
        reg = hash_full & ((1 << p_bits) - 1)
        remaining = hash_full >> p_bits
        zeros = _count_trailing_zeros(remaining)
        hash_mod = hash_full % HASH_MOD
        
        # Layer is determined by n-gram length
        layer = min(len(ntoken) - 1, NUM_LAYERS - 1)
        
        # Reference hash (first token)
        ref_hash = murmur_hash64a(ntoken[0].encode('utf-8'), seed) if ntoken else 0
        
        # First token for triangulation linking
        first_token = ntoken[0] if ntoken else ""
        
        return cls(
            token=ntoken,
            hash_full=hash_full,
            hash_mod=hash_mod,
            reg=reg,
            zeros=zeros,
            layer=layer,
            ref_hash=ref_hash,
            first_token=first_token,
        )
    
    @property
    def position(self) -> Tuple[int, int]:
        """HLL position as (reg, zeros) tuple."""
        return (self.reg, self.zeros)
    
    @property
    def token_str(self) -> str:
        """Token as space-joined string."""
        return " ".join(self.token)


@dataclass
class DisambiguationResult:
    """
    Result of disambiguating a single HLL position.
    """
    reg: int
    zeros: int
    candidates: List[TokenEntry]
    confidence: float
    method: str  # "exact", "triangulated", "best_guess"
    
    @property
    def position(self) -> Tuple[int, int]:
        return (self.reg, self.zeros)
    
    @property
    def best_token(self) -> Optional[str]:
        """Return best candidate token string, or None."""
        if self.candidates:
            return self.candidates[0].token_str
        return None


def _count_trailing_zeros(value: int) -> int:
    """Count trailing zeros in 64-bit integer."""
    if value == 0:
        return 64
    count = 0
    while (value & 1) == 0:
        count += 1
        value >>= 1
    return count


class TriangulationTensor:
    """
    Sparse 4D tensor for triangulation-based disambiguation.
    
    Conceptual shape: [NUM_LAYERS, 32, 2^p_bits, HASH_MOD]
    Actual storage: sparse dict indexed by (layer, zeros, reg, hash_mod)
    
    This structure enables:
    - O(1) insertion of token entries
    - O(k) lookup of candidates at position
    - Layer-based triangulation for disambiguation
    """
    
    def __init__(self, p_bits: int = 10):
        self.p_bits = p_bits
        self.num_registers = 1 << p_bits
        
        # Primary index: (layer, zeros, reg) → list of TokenEntry
        self._data: Dict[Tuple[int, int, int], List[TokenEntry]] = defaultdict(list)
        
        # Secondary index: hash_full → TokenEntry (for dedup)
        self._hash_index: Dict[int, TokenEntry] = {}
        
        # Count by layer
        self._layer_counts = [0] * NUM_LAYERS
    
    def add_entry(self, entry: TokenEntry) -> bool:
        """
        Add token entry to tensor.
        
        Returns:
            True if entry was new, False if duplicate
        """
        # Dedup by full hash
        if entry.hash_full in self._hash_index:
            return False
        
        self._hash_index[entry.hash_full] = entry
        key = (entry.layer, entry.zeros, entry.reg)
        self._data[key].append(entry)
        self._layer_counts[entry.layer] += 1
        return True
    
    def lookup(self, layer: int, reg: int, zeros: int) -> List[TokenEntry]:
        """Get candidates at specific layer and position."""
        return self._data.get((layer, zeros, reg), [])
    
    def lookup_position(self, reg: int, zeros: int) -> Dict[int, List[TokenEntry]]:
        """Get candidates at position across all layers."""
        result = {}
        for layer in range(NUM_LAYERS):
            entries = self.lookup(layer, reg, zeros)
            if entries:
                result[layer] = entries
        return result
    
    def lookup_all_layers(self, reg: int, zeros: int) -> List[TokenEntry]:
        """Get all candidates at position, all layers combined."""
        result = []
        for layer in range(NUM_LAYERS):
            result.extend(self.lookup(layer, reg, zeros))
        return result
    
    def positions_at_layer(self, layer: int) -> Set[Tuple[int, int]]:
        """Get all (reg, zeros) positions that have entries at given layer."""
        positions = set()
        for (l, zeros, reg), entries in self._data.items():
            if l == layer and entries:
                positions.add((reg, zeros))
        return positions
    
    def stats(self) -> Dict[str, Any]:
        """Return tensor statistics."""
        total_entries = sum(len(v) for v in self._data.values())
        unique_positions = len(self._data)
        
        # Collision stats
        collision_counts = [len(v) for v in self._data.values() if len(v) > 1]
        
        return {
            'total_entries': total_entries,
            'unique_positions': unique_positions,
            'entries_by_layer': list(self._layer_counts),
            'positions_with_collisions': len(collision_counts),
            'max_collision_depth': max(collision_counts) if collision_counts else 0,
            'avg_collision_depth': np.mean(collision_counts) if collision_counts else 0,
        }
    
    def __len__(self) -> int:
        return len(self._hash_index)
    
    def add_ngrams(self, tokens: List[str], max_n: int = 3) -> int:
        """
        Add all n-grams (n=1..max_n) from token list to tensor.
        
        Args:
            tokens: List of tokens
            max_n: Maximum n-gram length (default 3)
            
        Returns:
            Number of new entries added
        """
        added = 0
        for n in range(1, min(max_n + 1, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n])
                entry = TokenEntry.from_ntoken(ngram, p_bits=self.p_bits)
                if self.add_entry(entry):
                    added += 1
        return added
    
    def get_candidates(self, reg: int, layer: int = 0) -> List[TokenEntry]:
        """
        Get candidates at register across all zero counts.
        
        This is useful when you know the register but not the exact zeros.
        
        Args:
            reg: Register index
            layer: Layer to search (default 0 = unigrams)
            
        Returns:
            List of candidate TokenEntry objects
        """
        result = []
        for zeros in range(32):  # Max possible zeros
            result.extend(self.lookup(layer, reg, zeros))
        return result

class GlobalLayerHLLSets:
    """
    Maintains one HLLSet per n-gram layer for fast filtering.
    
    When disambiguating, we first check if a position is "hot" in a layer
    before doing expensive triangulation lookup.
    """
    
    def __init__(self, p_bits: int = 10):
        self.p_bits = p_bits
        self.layers: List[HLLSet] = [
            HLLSet(p_bits=p_bits) for _ in range(NUM_LAYERS)
        ]
    
    def add_entry(self, entry: TokenEntry):
        """Add entry's position to appropriate layer HLLSet."""
        if 0 <= entry.layer < NUM_LAYERS:
            # Create a synthetic "token" from position to inscribe
            position_token = f"_pos_{entry.reg}_{entry.zeros}"
            # Use absorb_and_track to add single token
            self.layers[entry.layer].absorb_and_track([position_token])
    
    def check_layer(self, layer: int, reg: int, zeros: int) -> bool:
        """
        Check if position might have entries at layer.
        
        Note: This is probabilistic (HLL), may have false positives.
        """
        # We can't directly query (reg, zeros) in HLLSet
        # This is a simplified check - in practice you'd use the tensor
        return True  # Always check tensor for accuracy
    
    def layer_cardinalities(self) -> List[float]:
        """Return estimated cardinality per layer."""
        return [layer.cardinality() for layer in self.layers]


# Deprecated, use ParallelDisambiguator
#=============================================================================
class DisambiguationEngine:
    """
    Main API for token disambiguation.
    
    Usage:
        engine = DisambiguationEngine()
        
        # Ingest documents
        engine.ingest_tokens(["the", "quick", "brown", "fox"], max_n=3)
        
        # Create HLLSet to disambiguate
        hll = HLLSet.from_batch(["quick", "fox"])
        
        # Recover tokens
        results = engine.disambiguate_hllset(hll)
        for r in results:
            print(f"Position {r.position}: {r.best_token} ({r.confidence:.2f})")
    """
    
    def __init__(self, p_bits: int = 10, seed: int = 42):
        self.p_bits = p_bits
        self.seed = seed
        
        # Core data structures
        self.tensor = TriangulationTensor(p_bits=p_bits)
        self.global_layers = GlobalLayerHLLSets(p_bits=p_bits)
        
        # Token LUT for fast lookup (from hll_tensor.py)
        self.token_lut = TokenLUT(p_bits=p_bits)
        
        # Hash index for direct lookup
        self._hash_index: Dict[int, TokenEntry] = {}
        
        # Occurrence counts for n-grams (hash → count)
        # This tracks how many times each n-gram appears in input
        self._occurrence_counts: Dict[int, int] = defaultdict(int)
    
    def _increment_occurrence(self, hash_full: int):
        """Increment occurrence count for an n-gram."""
        self._occurrence_counts[hash_full] += 1
    
    def get_occurrence_count(self, hash_full: int) -> int:
        """Get occurrence count for an n-gram hash."""
        return self._occurrence_counts.get(hash_full, 0)
    
    def get_trigram_counts(self) -> Dict[Tuple[str, str, str], int]:
        """
        Get occurrence counts for all trigrams.
        
        Returns:
            Dict mapping trigram tuple → occurrence count
        """
        result = {}
        for (layer, zeros, reg), entries in self.tensor._data.items():
            if layer == 2:  # trigrams
                for entry in entries:
                    count = self._occurrence_counts.get(entry.hash_full, 1)
                    result[entry.token] = count
        return result
    
    def ingest_token(self, token: str, max_n: int = 3) -> List[TokenEntry]:
        """
        Ingest a single token (adds unigram only).
        
        For n-grams, use ingest_tokens() with context.
        """
        entry = TokenEntry.from_ntoken((token,), self.p_bits, self.seed)
        self._add_entry(entry)
        self._increment_occurrence(entry.hash_full)
        return [entry]
    
    def ingest_tokens(self, tokens: List[str], max_n: int = 3, 
                       add_boundaries: bool = True,
                       end_padding: Optional[int] = None) -> int:
        """
        Ingest a sequence of tokens, creating all n-grams up to max_n.
        
        Args:
            tokens: List of tokens
            max_n: Maximum n-gram size (1=unigrams, 2=+bigrams, 3=+trigrams)
            add_boundaries: If True, prepend START_TOKEN and append END_TOKEN
                           to anchor the sequence for De Bruijn reconstruction
            end_padding: Number of END markers to add (asymmetric padding).
                           - None (default): auto = max_n - 1 (bulletproof)
                           - 1: single </S> (may lose last token coverage)
                           - 2: double </S> (bulletproof for trigrams)
                           
                           Asymmetric design rationale:
                           - START: Single <S> suffices to anchor De Bruijn path
                           - END: Multiple </S> needed for last-token n-gram coverage
                           
                           For forward reconstruction, only the ending tokens need
                           extra padding. Two </S> ensures the last token appears
                           in (max_n) trigrams for full disambiguation support.
            
        Returns:
            Number of entries added
        """
        # Add boundary markers for sequence reconstruction
        if add_boundaries:
            # Asymmetric: single start, multiple end for last-token coverage
            pad = end_padding if end_padding is not None else (max_n - 1)
            start_markers = [START_TOKEN]  # Single start anchor
            end_markers = [END_TOKEN] * pad  # Multiple end for coverage
            tokens = start_markers + list(tokens) + end_markers
        
        count = 0
        
        for n in range(1, min(max_n + 1, NUM_LAYERS + 1)):
            for i in range(len(tokens) - n + 1):
                ntoken = tuple(tokens[i:i+n])
                entry = TokenEntry.from_ntoken(ntoken, self.p_bits, self.seed)
                if self._add_entry(entry):
                    count += 1
                # Track occurrence count (including duplicates)
                self._increment_occurrence(entry.hash_full)
        
        return count
    
    def ingest_document(self, text: str, max_n: int = 3) -> int:
        """
        Ingest a document (splits on whitespace).
        
        For more control over tokenization, use ingest_tokens().
        """
        tokens = text.split()
        return self.ingest_tokens(tokens, max_n)
    
    def _add_entry(self, entry: TokenEntry) -> bool:
        """Add entry to all indexes."""
        # Check for duplicate
        if entry.hash_full in self._hash_index:
            return False
        
        self._hash_index[entry.hash_full] = entry
        self.tensor.add_entry(entry)
        self.global_layers.add_entry(entry)
        
        # Also add to TokenLUT for compatibility
        self.token_lut.add_token(
            token=entry.token_str,
            reg=entry.reg,
            zeros=entry.zeros,
            hash_full=entry.hash_full,
            layer=entry.layer,
        )
        
        return True
    
    def disambiguate_hllset(self, hll: HLLSet, 
                           min_confidence: float = 0.0) -> List[DisambiguationResult]:
        """
        Disambiguate all active positions in an HLLSet.
        
        Args:
            hll: HLLSet to disambiguate
            min_confidence: Minimum confidence threshold (0-1)
            
        Returns:
            List of DisambiguationResult for each active position
        """
        results = []
        
        # Get active positions from HLLSet via tensor view
        tensor = HLLTensor.from_numpy(hll.dump_numpy(), self.p_bits)
        active_positions = tensor.active_positions()
        
        for reg, zeros in active_positions:
            result = self._disambiguate_position(reg, zeros)
            if result.confidence >= min_confidence:
                results.append(result)
        
        return results
    
    def _disambiguate_position(self, reg: int, zeros: int) -> DisambiguationResult:
        """
        Disambiguate a single (reg, zeros) position using triangulation.
        """
        # Get candidates at each layer
        layer_candidates = self.tensor.lookup_position(reg, zeros)
        
        if not layer_candidates:
            return DisambiguationResult(
                reg=reg, zeros=zeros,
                candidates=[], confidence=0.0, method="no_candidates"
            )
        
        # If only one candidate total, high confidence
        all_candidates = []
        for entries in layer_candidates.values():
            all_candidates.extend(entries)
        
        if len(all_candidates) == 1:
            return DisambiguationResult(
                reg=reg, zeros=zeros,
                candidates=all_candidates,
                confidence=1.0, method="exact"
            )
        
        # Triangulation: intersect across layers
        triangulated = self._triangulate(layer_candidates)
        
        if triangulated:
            # Confidence based on how many layers agree
            layers_with_match = sum(1 for l in layer_candidates if l in [e.layer for e in triangulated])
            confidence = layers_with_match / NUM_LAYERS
            
            return DisambiguationResult(
                reg=reg, zeros=zeros,
                candidates=triangulated,
                confidence=confidence, method="triangulated"
            )
        
        # Fallback: return all candidates with low confidence
        # Sort by layer (prefer unigrams)
        all_candidates.sort(key=lambda e: e.layer)
        confidence = 1.0 / len(all_candidates)
        
        return DisambiguationResult(
            reg=reg, zeros=zeros,
            candidates=all_candidates,
            confidence=confidence, method="best_guess"
        )
    
    def _triangulate(self, layer_candidates: Dict[int, List[TokenEntry]]) -> List[TokenEntry]:
        """
        Triangulate across layers to find consistent candidates.
        
        Strategy:
        1. Build set of unigram tokens from each layer's candidates
        2. Find unigrams that appear consistently across layers
        3. Return corresponding entries
        """
        if not layer_candidates:
            return []
        
        # Extract unigrams from each layer's candidates
        layer_unigrams: Dict[int, Set[str]] = {}
        
        for layer, entries in layer_candidates.items():
            unigrams = set()
            for entry in entries:
                # For unigrams, add the token directly
                if entry.layer == 0:
                    unigrams.add(entry.token[0])
                else:
                    # For n-grams, add component unigrams
                    unigrams.update(entry.token)
            layer_unigrams[layer] = unigrams
        
        # Find intersection across all layers
        if not layer_unigrams:
            return []
        
        common_unigrams = set.intersection(*layer_unigrams.values())
        
        if not common_unigrams:
            # No perfect match - try majority voting
            all_unigrams = [u for unigrams in layer_unigrams.values() for u in unigrams]
            from collections import Counter
            counts = Counter(all_unigrams)
            # Get unigrams appearing in majority of layers
            threshold = len(layer_candidates) / 2
            common_unigrams = {u for u, c in counts.items() if c >= threshold}
        
        if not common_unigrams:
            return []
        
        # Return entries matching common unigrams
        result = []
        for entries in layer_candidates.values():
            for entry in entries:
                if entry.layer == 0 and entry.token[0] in common_unigrams:
                    result.append(entry)
                elif any(t in common_unigrams for t in entry.token):
                    result.append(entry)
        
        # Deduplicate by hash
        seen = set()
        deduped = []
        for entry in result:
            if entry.hash_full not in seen:
                seen.add(entry.hash_full)
                deduped.append(entry)
        
        return deduped
    
    def lookup_token(self, token: str) -> Optional[TokenEntry]:
        """Look up a token by string (unigram only)."""
        entry = TokenEntry.from_ntoken((token,), self.p_bits, self.seed)
        return self._hash_index.get(entry.hash_full)
    
    def lookup_position(self, reg: int, zeros: int) -> List[TokenEntry]:
        """Get all candidates at a position."""
        return self.tensor.lookup_all_layers(reg, zeros)
    
    def stats(self) -> Dict[str, Any]:
        """Return engine statistics."""
        tensor_stats = self.tensor.stats()
        return {
            **tensor_stats,
            'p_bits': self.p_bits,
            'layer_cardinalities': self.global_layers.layer_cardinalities(),
        }
    
    def __len__(self) -> int:
        return len(self._hash_index)
    
    def __repr__(self) -> str:
        return f"DisambiguationEngine(entries={len(self)}, p_bits={self.p_bits})"
    
    def train(self, documents: List[str], max_n: int = 3) -> int:
        """
        Train the engine on a list of documents.
        
        Args:
            documents: List of text documents
            max_n: Maximum n-gram size
            
        Returns:
            Total number of entries added
        """
        total = 0
        for doc in documents:
            total += self.ingest_document(doc, max_n)
        return total
    
    def disambiguate(self, hll_or_registers: Union[HLLSet, np.ndarray], 
                      max_candidates: int = 10) -> DisambiguationResult:
        """
        Disambiguate tokens from HLLSet or register array.
        
        Args:
            hll_or_registers: HLLSet instance or numpy array from dump_numpy()
            max_candidates: Maximum candidates to return
            
        Returns:
            DisambiguationResult with aggregated candidates
        """
        # Get registers
        if isinstance(hll_or_registers, HLLSet):
            registers = hll_or_registers.dump_numpy()
        else:
            registers = hll_or_registers
        
        # Convert to HLLTensor for position extraction
        tensor = HLLTensor.from_numpy(registers, self.p_bits)
        active_positions = tensor.active_positions()
        
        all_candidates = []
        methods = []
        
        for reg, zeros in active_positions:
            result = self._disambiguate_position(reg, zeros)
            all_candidates.extend(result.candidates)
            methods.append(result.method)
        
        # Deduplicate and sort by frequency
        from collections import Counter
        candidate_counts = Counter(e.token_str for e in all_candidates)
        
        # Get unique entries sorted by count
        seen = set()
        unique_candidates = []
        for entry in all_candidates:
            if entry.token_str not in seen:
                seen.add(entry.token_str)
                unique_candidates.append(entry)
        
        # Sort by count (descending)
        unique_candidates.sort(key=lambda e: candidate_counts[e.token_str], reverse=True)
        unique_candidates = unique_candidates[:max_candidates]
        
        # Compute aggregate confidence
        if unique_candidates:
            # Higher confidence if fewer candidates and consistent methods
            method_counts = Counter(methods)
            confidence = 1.0 / (1 + len(unique_candidates) * 0.1)
            if "exact" in method_counts:
                confidence = min(1.0, confidence + 0.3)
        else:
            confidence = 0.0
        
        primary_method = max(set(methods), key=methods.count) if methods else "no_data"
        
        return DisambiguationResult(
            reg=-1,  # Aggregate result
            zeros=-1,
            candidates=unique_candidates,
            confidence=confidence,
            method=primary_method
        )

    def restore_token_order(self, 
                            surviving_unigrams: Optional[Set[str]] = None,
                            random_tiebreaker: bool = True,
                            ) -> List[str]:
        """
        Convenience method to restore original token order using De Bruijn graph.
        
        This is a wrapper that creates a ParallelDisambiguator internally.
        For more control, use ParallelDisambiguator.restore_token_order() directly.
        
        Args:
            surviving_unigrams: Set of tokens that passed disambiguation
                               If None, use all unigrams from tensor
            random_tiebreaker: If True, randomize edge selection for variety
            
        Returns:
            Ordered list of tokens (without START/END markers)
        """
        # Create parallel disambiguator and delegate
        parallel = ParallelDisambiguator.from_engine(self)
        return parallel.restore_token_order(
            surviving_unigrams=surviving_unigrams,
            start_marker=START_TOKEN,
            end_marker=END_TOKEN,
            random_tiebreaker=random_tiebreaker,
        )

    def build_debruijn_graph(self, surviving_unigrams: Optional[Set[str]] = None):
        """
        Convenience method to build De Bruijn graph from tensor trigrams.
        
        For analysis, visualization, or custom path finding.
        
        Args:
            surviving_unigrams: Optional filter set; if None, use all trigrams
            
        Returns:
            DeBruijnGraph instance
        """
        parallel = ParallelDisambiguator.from_engine(self)
        return parallel.build_debruijn_graph(
            surviving_unigrams=surviving_unigrams,
            start_marker=START_TOKEN,
            end_marker=END_TOKEN,
        )


# =============================================================================
# Parallel Register-Based Disambiguation
# =============================================================================

@dataclass
class RegisterDisambiguationResult:
    """
    Result of disambiguating a single register.
    
    Contains all surviving unigram candidates after layer-based filtering.
    """
    reg: int
    surviving_unigrams: Set[str]
    initial_count: int  # Unigrams before filtering
    layer_reductions: List[int]  # Count after each layer filter
    confidence: float
    
    @property
    def reduction_ratio(self) -> float:
        """Ratio of eliminated candidates."""
        if self.initial_count == 0:
            return 0.0
        return 1.0 - len(self.surviving_unigrams) / self.initial_count


class ParallelDisambiguator:
    """
    Register-parallel disambiguation engine.
    
    Exploits mutual exclusivity: tokens at different registers
    can be disambiguated independently (embarrassingly parallel up to 2^P workers).
    
    Algorithm for each register r_i:
    1. Load all unigram candidates at r_i into buffer
    2. Get bigrams at r_i → extract first_tokens → filter buffer
    3. Get trigrams at r_i → extract first_tokens → filter buffer
    4. Return surviving unigrams
    
    The key insight: a unigram survives only if it STARTS an n-gram
    at the same register for each layer.
    
    Usage:
        engine = DisambiguationEngine()
        engine.ingest_tokens(["the", "quick", "brown", "fox"], max_n=3)
        
        parallel = ParallelDisambiguator.from_engine(engine)
        results = parallel.disambiguate_all()
        
        # Or disambiguate specific registers in parallel
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(parallel.disambiguate_register, r): r 
                       for r in active_registers}
    """
    
    def __init__(self, tensor: TriangulationTensor, p_bits: int = 10,
                 occurrence_counts: Optional[Dict[int, int]] = None):
        """
        Create ParallelDisambiguator from TriangulationTensor.
        
        Args:
            tensor: TriangulationTensor with ingested n-grams
            p_bits: Precision bits (must match tensor)
            occurrence_counts: Optional dict mapping hash → occurrence count
        """
        self.tensor = tensor
        self.p_bits = p_bits
        self.num_registers = 1 << p_bits
        self.num_layers = NUM_LAYERS
        self._occurrence_counts = occurrence_counts or {}
    
    @classmethod
    def from_engine(cls, engine: DisambiguationEngine) -> 'ParallelDisambiguator':
        """Create from an existing DisambiguationEngine."""
        return cls(engine.tensor, engine.p_bits, engine._occurrence_counts)
    
    def get_trigram_counts(self) -> Dict[Tuple[str, str, str], int]:
        """
        Get occurrence counts for all trigrams.
        
        Returns:
            Dict mapping trigram tuple → occurrence count
        """
        result = {}
        for (layer, zeros, reg), entries in self.tensor._data.items():
            if layer == 2:  # trigrams
                for entry in entries:
                    count = self._occurrence_counts.get(entry.hash_full, 1)
                    result[entry.token] = count
        return result
    
    def disambiguate_register(self, reg: int) -> RegisterDisambiguationResult:
        """
        Disambiguate all positions in a single register using layer filtering.
        
        Algorithm:
        1. buffer = {unigrams at reg}
        2. for layer in [1, 2, ...]:
               first_tokens = {e.first_token for e in entries at (reg, layer)}
               buffer = buffer ∩ first_tokens
        3. return buffer
        
        Args:
            reg: Register index [0, 2^p_bits)
            
        Returns:
            RegisterDisambiguationResult with surviving unigrams
        """
        # Step 1: Load all unigram candidates at this register (all zeros values)
        unigrams: Set[str] = set()
        for zeros in range(32):  # Include zeros=0
            entries = self.tensor.lookup(layer=0, reg=reg, zeros=zeros)
            for e in entries:
                unigrams.add(e.first_token)  # For unigrams, first_token == token[0]
        
        initial_count = len(unigrams)
        layer_reductions = [initial_count]
        
        if not unigrams:
            return RegisterDisambiguationResult(
                reg=reg,
                surviving_unigrams=set(),
                initial_count=0,
                layer_reductions=[0],
                confidence=0.0,
            )
        
        # Step 2-3: Filter by higher layers
        for layer in range(1, self.num_layers):
            # Get first_tokens from n-grams at this register and layer
            first_tokens: Set[str] = set()
            for zeros in range(32):  # Include zeros=0
                entries = self.tensor.lookup(layer=layer, reg=reg, zeros=zeros)
                for e in entries:
                    first_tokens.add(e.first_token)
            
            # If no n-grams at this layer, skip filtering
            if first_tokens:
                unigrams = unigrams & first_tokens
            
            layer_reductions.append(len(unigrams))
        
        # Compute confidence based on reduction
        if initial_count > 0:
            # High confidence if we narrowed down significantly
            final_count = len(unigrams)
            if final_count == 0:
                confidence = 0.0
            elif final_count == 1:
                confidence = 1.0
            else:
                # Confidence decreases with more candidates
                confidence = 1.0 / final_count
        else:
            confidence = 0.0
        
        return RegisterDisambiguationResult(
            reg=reg,
            surviving_unigrams=unigrams,
            initial_count=initial_count,
            layer_reductions=layer_reductions,
            confidence=confidence,
        )
    
    def disambiguate_all(self, 
                         active_registers: Optional[List[int]] = None,
                         ) -> Dict[int, RegisterDisambiguationResult]:
        """
        Disambiguate all (or specified) registers sequentially.
        
        For parallel execution, use ThreadPoolExecutor or ProcessPoolExecutor
        with disambiguate_register().
        
        Args:
            active_registers: Optional list of registers to process.
                             If None, processes all non-empty registers.
            
        Returns:
            Dict mapping register index → RegisterDisambiguationResult
        """
        if active_registers is None:
            # Find registers with any entries
            active_registers = self._find_active_registers()
        
        results = {}
        for reg in active_registers:
            results[reg] = self.disambiguate_register(reg)
        
        return results
    
    def _find_active_registers(self) -> List[int]:
        """Find all registers that have entries at layer 0."""
        active = set()
        for (layer, zeros, reg), entries in self.tensor._data.items():
            if layer == 0 and entries:
                active.add(reg)
        return sorted(active)
    
    def disambiguate_parallel(self, 
                              active_registers: Optional[List[int]] = None,
                              max_workers: int = 4,
                              ) -> Dict[int, RegisterDisambiguationResult]:
        """
        Disambiguate registers in parallel using ThreadPoolExecutor.
        
        Args:
            active_registers: Optional list of registers to process.
            max_workers: Number of parallel workers (default 4).
            
        Returns:
            Dict mapping register index → RegisterDisambiguationResult
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if active_registers is None:
            active_registers = self._find_active_registers()
        
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_reg = {
                executor.submit(self.disambiguate_register, reg): reg 
                for reg in active_registers
            }
            
            for future in as_completed(future_to_reg):
                reg = future_to_reg[future]
                results[reg] = future.result()
        
        return results
    
    def collect_surviving_tokens(self, 
                                 results: Dict[int, RegisterDisambiguationResult]
                                 ) -> Set[str]:
        """
        Collect all surviving unigram tokens from disambiguation results.
        
        Args:
            results: Dict from disambiguate_all() or disambiguate_parallel()
            
        Returns:
            Set of all surviving unigram tokens
        """
        tokens = set()
        for result in results.values():
            tokens.update(result.surviving_unigrams)
        return tokens
    
    def stats(self, results: Dict[int, RegisterDisambiguationResult]) -> Dict[str, Any]:
        """
        Compute statistics from disambiguation results.
        
        Args:
            results: Dict from disambiguate_all() or disambiguate_parallel()
            
        Returns:
            Dict with statistics
        """
        total_initial = sum(r.initial_count for r in results.values())
        total_surviving = sum(len(r.surviving_unigrams) for r in results.values())
        
        confidences = [r.confidence for r in results.values() if r.initial_count > 0]
        
        return {
            'registers_processed': len(results),
            'total_initial_candidates': total_initial,
            'total_surviving_candidates': total_surviving,
            'overall_reduction_ratio': 1.0 - total_surviving / max(1, total_initial),
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'registers_with_unique_result': sum(1 for r in results.values() 
                                                 if len(r.surviving_unigrams) == 1),
            'registers_fully_disambiguated': sum(1 for r in results.values()
                                                  if r.confidence == 1.0),
        }
    
    def __repr__(self) -> str:
        active = len(self._find_active_registers())
        return f"ParallelDisambiguator(p_bits={self.p_bits}, active_registers={active})"
    
    # =========================================================================
    # Token Order Restoration (De Bruijn Graph)
    # =========================================================================
    
    def restore_token_order(self, 
                            surviving_unigrams: Optional[Set[str]] = None,
                            start_marker: str = START_TOKEN,
                            end_marker: str = END_TOKEN,
                            random_tiebreaker: bool = True,
                            use_eulerian: bool = True,
                            ) -> List[str]:
        """
        Restore original token order using De Bruijn graph and Eulerian path.
        
        A De Bruijn graph naturally represents overlapping k-mers:
        - Nodes are (k-1)-mers (bigrams for trigrams)
        - Edges are k-mers (trigrams)
        - Edge multiplicity = occurrence count
        
        The Eulerian path visits each edge exactly once per multiplicity,
        giving the optimal reconstruction that uses all evidence.
        
        Algorithm:
        1. Collect ALL trigrams from tensor (all registers)
        2. Filter: keep trigrams where ALL tokens survived
        3. Build De Bruijn graph with edge multiplicities
        4. Find Eulerian path from START to END
        5. Extract token sequence from path
        
        Args:
            surviving_unigrams: Set of tokens that passed disambiguation
                               If None, collect all unigrams from tensor
            start_marker: Token marking sequence start (default START_TOKEN)
            end_marker: Token marking sequence end (default END_TOKEN)
            random_tiebreaker: If True, randomize edge selection for variety
            use_eulerian: If True, find Eulerian path; else greedy
            
        Returns:
            Ordered list of tokens (without START/END markers)
        """
        from .debruijn import DeBruijnGraph
        
        # If no surviving set provided, collect all unigrams from tensor
        if surviving_unigrams is None:
            surviving_unigrams = set()
            for (layer, zeros, reg), entries in self.tensor._data.items():
                if layer == 0:  # unigrams only
                    for entry in entries:
                        surviving_unigrams.add(entry.token[0])
        
        # Step 1: Collect all trigrams (layer=2) with occurrence counts
        trigram_entries: List[TokenEntry] = []
        for (layer, zeros, reg), entries in self.tensor._data.items():
            if layer == 2:  # trigrams only
                trigram_entries.extend(entries)
        
        if not trigram_entries:
            # No trigrams, return unordered (without markers)
            return list(surviving_unigrams - {start_marker, end_marker})
        
        # Step 2: Filter trigrams - keep only where ALL tokens survived
        filter_set = surviving_unigrams | {start_marker, end_marker}
        
        # Build De Bruijn graph
        graph: DeBruijnGraph[str] = DeBruijnGraph(k=3)
        
        for entry in trigram_entries:
            tokens = entry.token
            if all(t in filter_set for t in tokens):
                # Get occurrence count (default 1 if not tracked)
                count = self._occurrence_counts.get(entry.hash_full, 1)
                graph.add_kmer(tokens, count)
        
        if graph.num_edges == 0:
            return list(surviving_unigrams - {start_marker, end_marker})
        
        # Step 3: Find path using De Bruijn graph
        if use_eulerian:
            result = graph.find_eulerian_path(
                start_marker=start_marker,
                end_marker=end_marker,
                randomize=random_tiebreaker,
            )
        else:
            result = graph.find_path_greedy(
                start_marker=start_marker,
                end_marker=end_marker,
                randomize=random_tiebreaker,
            )
        
        if result is None:
            # Fallback: return surviving tokens unordered
            return list(surviving_unigrams - {start_marker, end_marker})
        
        # Step 4: Extract sequence (strip markers)
        sequence = result.sequence
        if sequence and sequence[0] == start_marker:
            sequence = sequence[1:]
        if sequence and sequence[-1] == end_marker:
            sequence = sequence[:-1]
        
        return sequence
    
    def build_debruijn_graph(self,
                             surviving_unigrams: Optional[Set[str]] = None,
                             start_marker: str = START_TOKEN,
                             end_marker: str = END_TOKEN,
                             ) -> DeBruijnGraph[str]:
        """
        Build De Bruijn graph from tensor trigrams.
        
        Useful for analysis, visualization, or custom path finding.
        
        Args:
            surviving_unigrams: Optional filter set; if None, use all trigrams
            start_marker: Start marker token (default START_TOKEN)
            end_marker: End marker token (default END_TOKEN)
            
        Returns:
            DeBruijnGraph instance
        """
        from .debruijn import DeBruijnGraph
        
        graph: DeBruijnGraph[str] = DeBruijnGraph(k=3)
        
        for (layer, zeros, reg), entries in self.tensor._data.items():
            if layer == 2:  # trigrams only
                for entry in entries:
                    tokens = entry.token
                    
                    # Filter if surviving_unigrams provided
                    if surviving_unigrams is not None:
                        filter_set = surviving_unigrams | {start_marker, end_marker}
                        if not all(t in filter_set for t in tokens):
                            continue
                    
                    count = self._occurrence_counts.get(entry.hash_full, 1)
                    graph.add_kmer(tokens, count)
        
        return graph
    
    def disambiguate_and_restore(self,
                                  start_marker: str = "START",
                                  end_marker: str = "END",
                                  max_workers: int = 1,
                                  random_tiebreaker: bool = True,
                                  use_eulerian: bool = True,
                                  ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Full disambiguation pipeline: filter → collect → restore order.
        
        Convenience method combining:
        1. disambiguate_all() or disambiguate_parallel()
        2. collect_surviving_tokens()
        3. restore_token_order() using De Bruijn graph
        
        Args:
            start_marker: Token marking sequence start
            end_marker: Token marking sequence end
            max_workers: Number of workers (1 = sequential)
            random_tiebreaker: If True, shuffle candidates for variety
            use_eulerian: If True, find optimal Eulerian path; else greedy
            
        Returns:
            Tuple of (ordered_tokens, stats_dict)
        """
        # Phase 1: Parallel disambiguation
        if max_workers > 1:
            results = self.disambiguate_parallel(max_workers=max_workers)
        else:
            results = self.disambiguate_all()
        
        # Phase 2: Collect surviving tokens
        surviving = self.collect_surviving_tokens(results)
        
        # Phase 3: Restore order using De Bruijn graph
        ordered = self.restore_token_order(
            surviving, 
            start_marker=start_marker,
            end_marker=end_marker,
            random_tiebreaker=random_tiebreaker,
            use_eulerian=use_eulerian,
        )
        
        # Compute stats
        stats = self.stats(results)
        stats['ordered_tokens_count'] = len(ordered)
        stats['surviving_unigrams_count'] = len(surviving)
        
        return ordered, stats
