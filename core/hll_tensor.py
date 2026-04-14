"""
HLLSet Tensor - 2D tensor view of HLLSet for disambiguation and semantic operations.

This module provides the bridge between:
- Ring layer (bitvector_ring.py): Pure bitwise operations
- Semantic layer (hllset.py): Token inscription, cardinality, BSS

The HLLSet fingerprint is viewed as a 2D tensor of shape (2^P, 32):
- First dimension: register index (determined by hash prefix)
- Second dimension: zeros count (determined by trailing zeros in hash)

Key Operations:
1. Inscription: hash(token) → (reg, zeros) → set bit
2. Active positions: Extract all (reg, zeros) pairs with bits set
3. Max zeros: For HLL cardinality estimation
4. Token LUT: (reg, zeros) → candidate tokens for disambiguation

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set, Iterator
from dataclasses import dataclass
import numpy as np

from .bitvector_ring import BitVector, BitVectorRing


class HLLTensor:
    """
    2D tensor view of HLLSet fingerprint for disambiguation operations.
    
    The tensor has shape (num_registers, 32) where:
    - num_registers = 2^p_bits (default 1024 for p_bits=10)
    - Each register is a 32-bit bitmap
    
    This provides the semantic bridge between:
    - BitVector (1D, for ring operations)
    - (reg, zeros) pairs (2D, for token disambiguation)
    
    Attributes:
        p_bits: Precision bits (determines number of registers)
        num_registers: Number of registers = 2^p_bits
        bits_per_register: Always 32 (uint32 bitmap)
        registers: numpy array of uint32 bitmaps
    """
    
    def __init__(self, p_bits: int = 10):
        """
        Create an HLLTensor with 2^p_bits registers.
        
        Args:
            p_bits: Precision bits (4-16, default 10)
        """
        if not 4 <= p_bits <= 16:
            raise ValueError("p_bits must be between 4 and 16")
        
        self.p_bits = p_bits
        self.num_registers = 1 << p_bits
        self.bits_per_register = 32
        self.total_bits = self.num_registers * self.bits_per_register
        
        # Internal storage: array of uint32 bitmaps
        self.registers = np.zeros(self.num_registers, dtype=np.uint32)
    
    @classmethod
    def from_registers(cls, registers: np.ndarray, p_bits: int = 10) -> 'HLLTensor':
        """Create from existing register array."""
        tensor = cls(p_bits)
        if len(registers) != tensor.num_registers:
            raise ValueError(f"Expected {tensor.num_registers} registers, got {len(registers)}")
        tensor.registers = registers.astype(np.uint32)
        return tensor
    
    @classmethod
    def from_numpy(cls, registers: np.ndarray, p_bits: int = 10) -> 'HLLTensor':
        """
        Create from numpy array of HLLSet registers.
        
        Alias for from_registers() - provided for compatibility with HLLSet.dump_numpy().
        
        Args:
            registers: numpy array of uint32 bitmaps from HLLSet.dump_numpy()
            p_bits: Precision bits (default 10)
        
        Returns:
            HLLTensor wrapping the registers
        """
        return cls.from_registers(registers, p_bits)
    
    @classmethod
    def from_bitvector(cls, bv: BitVector, p_bits: int = 10) -> 'HLLTensor':
        """Create from BitVector (unpacking into registers)."""
        tensor = cls(p_bits)
        value = bv.value
        for i in range(tensor.num_registers):
            tensor.registers[i] = (value >> (i * 32)) & 0xFFFFFFFF
        return tensor
    
    # === BitVector Interface (Ring Layer) ===
    
    def to_bitvector(self) -> BitVector:
        """Convert to flat BitVector for ring operations."""
        value = 0
        for i, reg in enumerate(self.registers):
            value |= (int(reg) << (i * 32))
        return BitVector(value, N=self.total_bits)
    
    def copy(self) -> 'HLLTensor':
        """Create a deep copy."""
        result = HLLTensor(self.p_bits)
        result.registers = self.registers.copy()
        return result
    
    # === 2D Semantic Interface ===
    
    def inscribe(self, reg: int, zeros: int):
        """
        Set bit at position (reg, zeros) — token inscription.
        
        This is the fundamental operation for adding a token to the HLLSet.
        Called after computing (reg, zeros) from hash(token).
        
        Args:
            reg: Register index [0, num_registers)
            zeros: Trailing zeros count [0, 31]
        """
        if not (0 <= reg < self.num_registers):
            raise ValueError(f"reg must be in [0, {self.num_registers}), got {reg}")
        if zeros < 32:
            self.registers[reg] |= np.uint32(1 << zeros)
    
    def inscribe_batch(self, positions: List[Tuple[int, int]]):
        """Inscribe multiple (reg, zeros) positions at once."""
        for reg, zeros in positions:
            self.inscribe(reg, zeros)
    
    def get_bit(self, reg: int, zeros: int) -> bool:
        """Check if bit at (reg, zeros) is set."""
        if zeros >= 32 or reg >= self.num_registers:
            return False
        return bool(self.registers[reg] & (1 << zeros))
    
    def clear_bit(self, reg: int, zeros: int):
        """Clear bit at position (reg, zeros)."""
        if zeros < 32 and reg < self.num_registers:
            self.registers[reg] &= ~np.uint32(1 << zeros)
    
    # === Active Positions (for disambiguation) ===
    
    def active_positions(self) -> List[Tuple[int, int]]:
        """
        Return all (reg, zeros) pairs where bits are set.
        
        This is used for disambiguation: each active position
        represents a potential token that was inscribed.
        
        Returns:
            List of (register_index, zeros_count) tuples
        """
        positions = []
        for reg in range(self.num_registers):
            val = int(self.registers[reg])
            zeros = 0
            while val:
                if val & 1:
                    positions.append((reg, zeros))
                val >>= 1
                zeros += 1
        return positions
    
    def active_positions_iter(self) -> Iterator[Tuple[int, int]]:
        """Iterator version of active_positions (memory efficient)."""
        for reg in range(self.num_registers):
            val = int(self.registers[reg])
            zeros = 0
            while val:
                if val & 1:
                    yield (reg, zeros)
                val >>= 1
                zeros += 1
    
    def active_registers(self) -> List[int]:
        """Return indices of non-zero registers."""
        return [i for i in range(self.num_registers) if self.registers[i] != 0]
    
    # === Cardinality Support ===
    
    def max_zeros_per_register(self) -> np.ndarray:
        """
        Compute highest set bit position per register.
        
        This is used for HLL cardinality estimation:
        - max_zeros[i] = bit_length(registers[i])
        - Equivalent to Julia's maxidx()
        
        Returns:
            Array of shape (num_registers,) with max zeros per register
        """
        result = np.zeros(self.num_registers, dtype=np.int32)
        for i, reg in enumerate(self.registers):
            val = int(reg)
            if val > 0:
                result[i] = val.bit_length()
        return result
    
    def popcount(self) -> int:
        """Total number of set bits (approximate cardinality proxy)."""
        return sum(int(r).bit_count() for r in self.registers)
    
    def register_popcount(self, reg: int) -> int:
        """Number of set bits in a specific register."""
        return int(self.registers[reg]).bit_count()
    
    # === Set Operations (Ring Layer) ===
    
    def union(self, other: 'HLLTensor') -> 'HLLTensor':
        """Bitwise OR — lattice join / set union."""
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot union tensors with different p_bits")
        result = HLLTensor(self.p_bits)
        result.registers = self.registers | other.registers
        return result
    
    def intersect(self, other: 'HLLTensor') -> 'HLLTensor':
        """Bitwise AND — ring multiplication / set intersection."""
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot intersect tensors with different p_bits")
        result = HLLTensor(self.p_bits)
        result.registers = self.registers & other.registers
        return result
    
    def symmetric_difference(self, other: 'HLLTensor') -> 'HLLTensor':
        """Bitwise XOR — ring addition / symmetric difference."""
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot XOR tensors with different p_bits")
        result = HLLTensor(self.p_bits)
        result.registers = self.registers ^ other.registers
        return result
    
    def difference(self, other: 'HLLTensor') -> 'HLLTensor':
        """Bitwise AND-NOT — set difference."""
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot diff tensors with different p_bits")
        result = HLLTensor(self.p_bits)
        result.registers = self.registers & ~other.registers
        return result
    
    def complement(self) -> 'HLLTensor':
        """Bitwise NOT — complement."""
        result = HLLTensor(self.p_bits)
        result.registers = ~self.registers
        return result
    
    # === Comparison ===
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HLLTensor):
            return NotImplemented
        return self.p_bits == other.p_bits and np.array_equal(self.registers, other.registers)
    
    def is_subset(self, other: 'HLLTensor') -> bool:
        """Check if self ⊆ other (all bits in self are in other)."""
        return np.all((self.registers & ~other.registers) == 0)
    
    def is_empty(self) -> bool:
        """Check if all registers are zero."""
        return np.all(self.registers == 0)
    
    # === Visualization ===
    
    def show_region(self, max_reg: int = 20, max_zeros: int = 10):
        """Display a region of the tensor as ASCII art."""
        print(f"HLLTensor[:{max_reg}, :{max_zeros}] (popcount={self.popcount()})")
        print("     " + " ".join(f"{z:2}" for z in range(max_zeros)))
        print("     " + "─" * (max_zeros * 3))
        for reg in range(min(max_reg, self.num_registers)):
            row = ""
            for zeros in range(max_zeros):
                bit = self.get_bit(reg, zeros)
                row += " ● " if bit else " · "
            print(f"{reg:3} │{row}")
    
    def __repr__(self) -> str:
        active = self.popcount()
        return f"HLLTensor(p_bits={self.p_bits}, active={active}/{self.total_bits})"


@dataclass
class TokenEntry:
    """
    Entry in the token lookup table for disambiguation.
    
    Stores the mapping: (reg, zeros) → token with metadata.
    
    For n-grams (layer > 0), first_token links back to the unigram
    that starts the n-gram, enabling efficient triangulation:
    - layer=0: "fox"           → first_token="fox"
    - layer=1: "brown fox"     → first_token="brown"
    - layer=2: "quick brown fox" → first_token="quick"
    
    This allows O(1) constraint checking during parallel disambiguation.
    """
    token: str
    reg: int
    zeros: int
    hash_full: int  # Full 64-bit hash for verification
    layer: int = 0  # N-gram layer (0=unigram, 1=bigram, 2=trigram)
    first_token: str = ""  # First token of n-gram (for triangulation linking)
    
    def __post_init__(self):
        """Auto-populate first_token if not provided."""
        if not self.first_token and self.token:
            # Extract first token from space or comma-separated n-gram
            sep = "," if "," in self.token else " "
            self.first_token = self.token.split(sep)[0].strip()
    
    @property
    def position(self) -> Tuple[int, int]:
        return (self.reg, self.zeros)
    
    @property
    def is_unigram(self) -> bool:
        """Check if this is a unigram (layer 0)."""
        return self.layer == 0


class TokenLUT:
    """
    Lookup table for (reg, zeros) → candidate tokens.
    
    This is the core data structure for disambiguation:
    - Multiple tokens may hash to the same (reg, zeros) position
    - During disambiguation, we retrieve all candidates and triangulate
    
    Attributes:
        p_bits: Precision bits (must match HLLTensor)
        entries: Dict mapping (reg, zeros) → list of TokenEntry
    """
    
    def __init__(self, p_bits: int = 10):
        self.p_bits = p_bits
        self.entries: Dict[Tuple[int, int], List[TokenEntry]] = {}
        self._token_count = 0
    
    def add_entry(self, entry: TokenEntry):
        """Add a token entry to the LUT."""
        pos = entry.position
        if pos not in self.entries:
            self.entries[pos] = []
        self.entries[pos].append(entry)
        self._token_count += 1
    
    def add_token(self, token: str, reg: int, zeros: int, 
                  hash_full: int = 0, layer: int = 0, first_token: str = ""):
        """Convenience method to add a token."""
        entry = TokenEntry(token=token, reg=reg, zeros=zeros, 
                          hash_full=hash_full, layer=layer, first_token=first_token)
        self.add_entry(entry)
    
    def lookup(self, reg: int, zeros: int) -> List[TokenEntry]:
        """Get all candidate tokens at (reg, zeros) position."""
        return self.entries.get((reg, zeros), [])
    
    def lookup_position(self, position: Tuple[int, int]) -> List[TokenEntry]:
        """Get all candidate tokens at position."""
        return self.entries.get(position, [])
    
    def has_candidates(self, reg: int, zeros: int) -> bool:
        """Check if position has any candidates."""
        return (reg, zeros) in self.entries
    
    def positions(self) -> Set[Tuple[int, int]]:
        """Return all positions that have entries."""
        return set(self.entries.keys())
    
    def entries_at_layer(self, layer: int) -> List[TokenEntry]:
        """Get all entries at a specific n-gram layer."""
        result = []
        for entries in self.entries.values():
            result.extend(e for e in entries if e.layer == layer)
        return result
    
    # === Register-based filtering (for parallel disambiguation) ===
    
    def entries_at_register(self, reg: int, layer: Optional[int] = None) -> List[TokenEntry]:
        """
        Get all entries at a specific register, optionally filtered by layer.
        
        This is the key method for parallel disambiguation:
        tokens at different registers are mutually exclusive.
        
        Args:
            reg: Register index
            layer: Optional layer filter (None = all layers)
            
        Returns:
            List of TokenEntry at the given register
        """
        result = []
        for (r, z), entries in self.entries.items():
            if r == reg:
                if layer is None:
                    result.extend(entries)
                else:
                    result.extend(e for e in entries if e.layer == layer)
        return result
    
    def first_tokens_at_register(self, reg: int, layer: int) -> Set[str]:
        """
        Get set of first_tokens for entries at a register and layer.
        
        Used for triangulation constraint checking:
        - Get bigrams at reg → extract their first_tokens
        - Unigrams must match these first_tokens to survive filtering
        
        Args:
            reg: Register index
            layer: N-gram layer (1=bigram, 2=trigram)
            
        Returns:
            Set of first_token strings
        """
        return {e.first_token for e in self.entries_at_register(reg, layer)}
    
    def unigrams_at_register(self, reg: int) -> Set[str]:
        """
        Get all unigram tokens at a specific register.
        
        Args:
            reg: Register index
            
        Returns:
            Set of unigram token strings
        """
        return {e.token for e in self.entries_at_register(reg, layer=0)}
    
    def active_registers(self) -> Set[int]:
        """Return set of all registers that have entries."""
        return {reg for (reg, zeros) in self.entries.keys()}
    
    def stats(self) -> Dict:
        """Return LUT statistics."""
        positions = len(self.entries)
        collisions = sum(1 for v in self.entries.values() if len(v) > 1)
        max_collision = max((len(v) for v in self.entries.values()), default=0)
        
        return {
            'total_tokens': self._token_count,
            'unique_positions': positions,
            'positions_with_collisions': collisions,
            'max_collision_depth': max_collision,
            'collision_rate': collisions / max(1, positions),
        }
    
    def __len__(self) -> int:
        return self._token_count
    
    def __repr__(self) -> str:
        return f"TokenLUT(p_bits={self.p_bits}, tokens={self._token_count}, positions={len(self.entries)})"


class TensorRingAdapter:
    """
    Adapter that connects HLLTensor with BitVectorRing for compression.
    
    This allows:
    - Registering HLLTensors in the ring algebra for basis compression
    - Retrieving compressed representations
    - Performing ring operations on compressed tensors
    """
    
    def __init__(self, p_bits: int = 10):
        self.p_bits = p_bits
        self.total_bits = (1 << p_bits) * 32
        self.ring = BitVectorRing(N=self.total_bits)
        self.tensor_cache: Dict[int, HLLTensor] = {}
    
    def register(self, tensor: HLLTensor) -> int:
        """
        Register tensor in the ring for compression.
        
        Returns:
            Integer ID for the compressed tensor
        """
        if tensor.p_bits != self.p_bits:
            raise ValueError(f"Tensor p_bits {tensor.p_bits} != adapter p_bits {self.p_bits}")
        
        bv = tensor.to_bitvector()
        vec_id = self.ring.compress(bv)
        self.tensor_cache[vec_id] = tensor
        return vec_id
    
    def get_tensor(self, vec_id: int) -> HLLTensor:
        """Retrieve tensor by ID."""
        if vec_id in self.tensor_cache:
            return self.tensor_cache[vec_id]
        # Reconstruct from coefficients
        bv = self.ring.decompress(vec_id)
        return HLLTensor.from_bitvector(bv, self.p_bits)
    
    def union_ids(self, id_a: int, id_b: int) -> int:
        """Union two compressed tensors."""
        tensor_a = self.get_tensor(id_a)
        tensor_b = self.get_tensor(id_b)
        result = tensor_a.union(tensor_b)
        return self.register(result)
    
    def intersect_ids(self, id_a: int, id_b: int) -> int:
        """Intersect two compressed tensors."""
        tensor_a = self.get_tensor(id_a)
        tensor_b = self.get_tensor(id_b)
        result = tensor_a.intersect(tensor_b)
        return self.register(result)
    
    def compression_stats(self) -> Dict:
        """Return compression statistics."""
        return {
            'p_bits': self.p_bits,
            'total_tensors': len(self.tensor_cache),
            'basis_rank': self.ring.rank(),
            'theoretical_bits': self.total_bits,
            'compression_ratio': self.total_bits / max(1, self.ring.rank()),
        }
