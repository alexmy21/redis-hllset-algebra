"""
Global N-gram Registry — Universe sets G₁, G₂, G₃ for the HLLSet framework.

Maintains persistent global HLLSets that accumulate ALL observed n-grams:

    G₁ = ∪ₜ unigrams(t)     (all 1-grams ever seen)
    G₂ = ∪ₜ bigrams(t)      (all 2-grams ever seen)
    G₃ = ∪ₜ trigrams(t)     (all 3-grams ever seen)

These serve as:
    - Universe sets for complement: ¬A = Gₖ \\ A
    - Membership tests (probabilistic): "has this n-gram been observed?"
    - Normalization for BSS: BSS_τ(A → Gₖ) measures coverage of vocabulary
    - Bloom-filter-like fast filtering before expensive disambiguation

Design:
    - GlobalNGramRegistry is an EXPLICIT object (no module-level singletons)
    - Applications create, populate, and pass it around
    - Supports snapshot/restore for persistence
    - Each layer is a standard immutable HLLSet (union-only accumulation)

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set, Any, Union
from dataclasses import dataclass
import numpy as np
import hashlib

from .hllset import HLLSet, HashConfig, DEFAULT_HASH_CONFIG, compute_sha1


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_N = 3   # Default maximum n-gram order
LAYER_NAMES = {0: "unigram", 1: "bigram", 2: "trigram"}


# =============================================================================
# Registry Snapshot (for serialization)
# =============================================================================

@dataclass(frozen=True)
class RegistrySnapshot:
    """
    Serializable snapshot of a GlobalNGramRegistry.
    
    Contains raw register bytes for each layer plus metadata.
    """
    layer_registers: Dict[int, bytes]   # layer → registers.tobytes()
    p_bits: int
    token_counts: Dict[int, int]        # layer → approximate count
    registry_id: str                    # SHA1 of combined state

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'layer_registers': {
                k: v.hex() for k, v in self.layer_registers.items()
            },
            'p_bits': self.p_bits,
            'token_counts': self.token_counts,
            'registry_id': self.registry_id,
        }


# =============================================================================
# Global N-gram Registry
# =============================================================================

class GlobalNGramRegistry:
    """
    Persistent global HLLSets for 1/2/3-gram layers.
    
    Accumulates all observed n-grams across the entire system lifetime.
    Each layer Gₖ is an HLLSet that only grows (union accumulation).
    
    Usage:
        registry = GlobalNGramRegistry(p_bits=10)
        
        # Ingest tokens from a document
        tokens = ["the", "quick", "brown", "fox"]
        registry.ingest(tokens, max_n=3)
        
        # Query
        print(registry.layer_cardinality(1))  # ~4 unigrams
        print(registry.layer_cardinality(2))  # ~3 bigrams
        
        # Use as universe for complement
        doc_hll = HLLSet.from_batch(["quick", "fox"])
        complement = registry.complement(doc_hll, layer=1)
        
        # Snapshot for persistence
        snap = registry.snapshot()
    
    The registry is an explicit object — no global state.
    Applications create it, populate it, pass it to components.
    """

    def __init__(
        self,
        p_bits: int = 10,
        max_n: int = DEFAULT_MAX_N,
        config: Optional[HashConfig] = None,
    ):
        """
        Create a new empty registry.
        
        Args:
            p_bits: HLL precision bits (default 10)
            max_n: Maximum n-gram order (default 3)
            config: Hash configuration (default: DEFAULT_HASH_CONFIG)
        """
        self._p_bits = p_bits
        self._max_n = max_n
        self._config = config or DEFAULT_HASH_CONFIG

        # One HLLSet per layer: G₁, G₂, ..., Gₙ
        # Layer index is 0-based: layer 0 = unigrams, layer 1 = bigrams, etc.
        self._layers: Dict[int, HLLSet] = {
            i: HLLSet(p_bits=p_bits) for i in range(max_n)
        }

        # Ingestion counts (approximate, for diagnostics)
        self._ingest_counts: Dict[int, int] = {i: 0 for i in range(max_n)}

    @property
    def p_bits(self) -> int:
        return self._p_bits

    @property
    def max_n(self) -> int:
        return self._max_n

    @property
    def config(self) -> HashConfig:
        return self._config

    # =========================================================================
    # Ingestion
    # =========================================================================

    def ingest(
        self,
        tokens: List[str],
        max_n: Optional[int] = None,
    ) -> Dict[int, int]:
        """
        Ingest a sequence of tokens, accumulating all n-grams into global layers.
        
        For a token sequence [t₁, t₂, t₃, t₄]:
            Layer 0 (unigrams): {t₁}, {t₂}, {t₃}, {t₄}
            Layer 1 (bigrams):  {t₁ t₂}, {t₂ t₃}, {t₃ t₄}
            Layer 2 (trigrams): {t₁ t₂ t₃}, {t₂ t₃ t₄}
        
        Args:
            tokens: Sequence of tokens
            max_n: Override max n-gram order (default: self._max_n)
            
        Returns:
            Dict[layer, count] — number of n-grams ingested per layer
        """
        max_n = min(max_n or self._max_n, self._max_n)
        counts = {}

        for n in range(1, max_n + 1):
            layer = n - 1
            ngrams = self._extract_ngrams(tokens, n)
            if ngrams:
                # Build an HLLSet from the n-gram strings and union into layer
                ngram_hll = HLLSet.from_batch(ngrams, p_bits=self._p_bits)
                self._layers[layer] = self._layers[layer].union(ngram_hll)
                self._ingest_counts[layer] += len(ngrams)
                counts[layer] = len(ngrams)
            else:
                counts[layer] = 0

        return counts

    def ingest_document(
        self,
        text: str,
        max_n: Optional[int] = None,
    ) -> Dict[int, int]:
        """
        Ingest a document (splits on whitespace).
        
        For more control over tokenization, use ingest() with pre-tokenized list.
        """
        tokens = text.split()
        return self.ingest(tokens, max_n)

    def ingest_batch_documents(
        self,
        documents: List[str],
        max_n: Optional[int] = None,
    ) -> Dict[int, int]:
        """
        Ingest multiple documents.
        
        Returns aggregate counts per layer.
        """
        total_counts: Dict[int, int] = {i: 0 for i in range(self._max_n)}
        for doc in documents:
            counts = self.ingest_document(doc, max_n)
            for layer, count in counts.items():
                total_counts[layer] += count
        return total_counts

    # =========================================================================
    # Universe Operations
    # =========================================================================

    def universe(self, layer: int = 0) -> HLLSet:
        """
        Get the universe set Gₖ for a given layer.
        
        Args:
            layer: 0=unigrams, 1=bigrams, 2=trigrams
            
        Returns:
            HLLSet containing all n-grams observed at this layer
        """
        if layer not in self._layers:
            raise ValueError(f"Layer {layer} not available (max: {self._max_n - 1})")
        return self._layers[layer]

    def complement(self, hllset: HLLSet, layer: int = 0) -> HLLSet:
        """
        Compute complement relative to universe: Gₖ \\ A.
        
        "All n-grams in the universe that are NOT in A."
        
        Args:
            hllset: The set A to complement
            layer: Which universe layer to use
            
        Returns:
            Gₖ \\ A
        """
        return self.universe(layer).diff(hllset)

    def coverage(self, hllset: HLLSet, layer: int = 0) -> float:
        """
        What fraction of the universe does this HLLSet cover?
        
        Computes |A ∩ Gₖ| / |Gₖ|  (this is BSS_τ(A → Gₖ))
        
        Args:
            hllset: The set to measure coverage of
            layer: Which universe layer to compare against
            
        Returns:
            Coverage ratio in [0, 1]
        """
        universe = self.universe(layer)
        u_card = universe.cardinality()
        if u_card <= 0:
            return 0.0

        intersection = hllset.intersect(universe)
        return min(1.0, intersection.cardinality() / u_card)

    def novelty(self, hllset: HLLSet, layer: int = 0) -> float:
        """
        What fraction of this HLLSet is novel (not in the universe)?
        
        Computes |A \\ Gₖ| / |A|
        
        Args:
            hllset: The set to check for novelty
            layer: Which universe layer to compare against
            
        Returns:
            Novelty ratio in [0, 1]
        """
        a_card = hllset.cardinality()
        if a_card <= 0:
            return 0.0

        diff = hllset.diff(self.universe(layer))
        return min(1.0, diff.cardinality() / a_card)

    # =========================================================================
    # Queries
    # =========================================================================

    def layer_cardinality(self, layer: int = 0) -> float:
        """Estimated cardinality of universe layer Gₖ."""
        return self.universe(layer).cardinality()

    def all_cardinalities(self) -> Dict[int, float]:
        """Cardinalities of all layers."""
        return {
            layer: hll.cardinality()
            for layer, hll in self._layers.items()
        }

    def layer_name(self, layer: int) -> str:
        """Human-readable name for a layer."""
        return LAYER_NAMES.get(layer, f"{layer+1}-gram")

    # =========================================================================
    # Snapshot / Restore
    # =========================================================================

    def snapshot(self) -> RegistrySnapshot:
        """
        Create a serializable snapshot of the current state.
        
        The snapshot contains raw register bytes — no HLLSet objects.
        """
        layer_registers = {}
        for layer, hll in self._layers.items():
            layer_registers[layer] = hll.dump_numpy().tobytes()

        # Combined ID: SHA1 of all layers concatenated
        combined = b''.join(
            layer_registers[i] for i in sorted(layer_registers.keys())
        )
        registry_id = hashlib.sha1(combined).hexdigest()

        return RegistrySnapshot(
            layer_registers=layer_registers,
            p_bits=self._p_bits,
            token_counts={k: v for k, v in self._ingest_counts.items()},
            registry_id=registry_id,
        )

    @classmethod
    def from_snapshot(cls, snap: RegistrySnapshot) -> 'GlobalNGramRegistry':
        """
        Restore a registry from a snapshot.
        
        Args:
            snap: RegistrySnapshot to restore from
            
        Returns:
            New GlobalNGramRegistry with restored state
        """
        max_n = len(snap.layer_registers)
        registry = cls(p_bits=snap.p_bits, max_n=max_n)

        for layer, reg_bytes in snap.layer_registers.items():
            registers = np.frombuffer(reg_bytes, dtype=np.uint32).copy()
            hll = HLLSet(p_bits=snap.p_bits)
            hll._core.set_registers(registers)
            hll._compute_name()
            registry._layers[layer] = hll

        registry._ingest_counts = dict(snap.token_counts)
        return registry

    # =========================================================================
    # Merge (combine two registries)
    # =========================================================================

    def merge(self, other: 'GlobalNGramRegistry') -> 'GlobalNGramRegistry':
        """
        Merge another registry into this one (union of all layers).
        
        Returns a NEW registry — does not modify self or other.
        """
        if self._p_bits != other._p_bits:
            raise ValueError("Cannot merge registries with different p_bits")

        max_n = max(self._max_n, other._max_n)
        result = GlobalNGramRegistry(p_bits=self._p_bits, max_n=max_n)

        for layer in range(max_n):
            if layer in self._layers and layer in other._layers:
                result._layers[layer] = self._layers[layer].union(other._layers[layer])
            elif layer in self._layers:
                result._layers[layer] = self._layers[layer]
            elif layer in other._layers:
                result._layers[layer] = other._layers[layer]

            result._ingest_counts[layer] = (
                self._ingest_counts.get(layer, 0) +
                other._ingest_counts.get(layer, 0)
            )

        return result

    # =========================================================================
    # Stats
    # =========================================================================

    def stats(self) -> Dict[str, Any]:
        """Return comprehensive registry statistics."""
        return {
            'p_bits': self._p_bits,
            'max_n': self._max_n,
            'layers': {
                layer: {
                    'name': self.layer_name(layer),
                    'cardinality': hll.cardinality(),
                    'ingest_count': self._ingest_counts.get(layer, 0),
                    'sha1': hll.name[:12],
                }
                for layer, hll in self._layers.items()
            },
        }

    # =========================================================================
    # Internal
    # =========================================================================

    @staticmethod
    def _extract_ngrams(tokens: List[str], n: int) -> List[str]:
        """Extract n-grams as space-joined strings."""
        if len(tokens) < n:
            return []
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def __repr__(self) -> str:
        cards = [f"G{i+1}≈{hll.cardinality():.0f}" for i, hll in self._layers.items()]
        return f"GlobalNGramRegistry(p={self._p_bits}, {', '.join(cards)})"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'GlobalNGramRegistry',
    'RegistrySnapshot',
    'DEFAULT_MAX_N',
    'LAYER_NAMES',
]
