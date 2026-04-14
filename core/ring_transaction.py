"""
Ring Transaction — Ephemeral workspace for IICA-compliant HLLSet construction.

Implements the transaction protocol that enforces:

    I — Immutability:   All committed results are frozen.  Mid-transaction
                        state is ephemeral and discarded on rollback.
    I — Idempotence:    Re-ingesting the same tokens produces the same
                        bitvector (deterministic hash → same bits → same
                        ring vector → same SHA1 content identity).
    C — Composability:  Every stage consumes and produces HLLSets — the
                        universal connector.  Pipelines chain without glue.
    A — Algebraic       Ring algebra (Gaussian elimination over F₂)
        Closure:        provides lossless compression and basis extraction;
                        set operations (∪ ∩ ∖ ⊕) are closed.

Transaction Lifecycle
=====================

    ┌─ begin ──────────────────────────────────────────────────┐
    │  tx = RingTransaction(base_ring=ring, lattice=lat)       │
    │                                                          │
    │  tx.ingest(batch_1)            # tokens → ring           │
    │  tx.ingest(batch_2)            # accumulate more         │
    │                                                          │
    │  # Optional: filter base vectors for relevance           │
    │  idx = tx.filter_bases(predicate)                        │
    │  idx = tx.bases_by_overlap(ref_hllset)                   │
    │  idx = tx.bases_from_lattice(t0, t1)                     │
    │                                                          │
    │  result = tx.commit(basis_indices=idx)   # freeze        │
    └──────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌─ TransactionResult (frozen) ─────────────────────────────┐
    │  .merged_hllset      — the HLLSet                        │
    │  .ring_delta         — what changed in the ring          │
    │  .basis_vectors      — for merge into main ring          │
    │  .token_lut          — position → token map              │
    │  .ingest_records     — full provenance                   │
    └──────────────────────────────────────────────────────────┘
                             │
                             ▼
    ┌─ user decides ───────────────────────────────────────────┐
    │  lattice.append([result.merged_hllset], timestamp=now)   │
    │  RingTransaction.merge_into_ring(main_ring, result)      │
    │  registry.ingest(all_tokens)                             │
    └──────────────────────────────────────────────────────────┘

Data Flow inside a Transaction
==============================

    tokens ─hash─▶ (reg, zeros) ─inscribe─▶ HLLTensor ─flatten─▶ BitVector
                                                                       │
                                                              ring.compress()
                                                              (Gauss elim.)
                                                                       │
                                                                ◆ basis updated
                                                                       │
                                              filter / select basis indices
                                                                       │
                                                           OR selected ─▶ HLLSet
                                                                         (immutable)

The transaction is a LIBRARY component — no singletons, no event loops.
Applications create transactions, accumulate changes, commit, and handle
results at their own pace.

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import (
    List, Tuple, Optional, Dict, Any, Callable, Set, Union,
)
from dataclasses import dataclass, field
from enum import Enum
import time
import numpy as np

from .hllset import HLLSet, HashConfig, DEFAULT_HASH_CONFIG, compute_sha1
from .bitvector_ring import BitVector, BitVectorRing
from .hll_tensor import HLLTensor, TokenLUT, TokenEntry
from .bss import bss, BSSPair


# =============================================================================
# Enums & Value Types
# =============================================================================

class TransactionPhase(Enum):
    """Lifecycle phase of a RingTransaction."""
    ACTIVE = "active"          # Accepting ingest / filter operations
    COMMITTED = "committed"    # Frozen — result available
    ROLLED_BACK = "rolled_back"  # Discarded


@dataclass(frozen=True)
class IngestRecord:
    """
    Immutable record of a single ingest() call within a transaction.

    Provenance chain:
        tokens  →  reg_zeros  →  vector_id  →  basis coefficients

    Attributes:
        batch_id:            SHA1(sorted tokens) — content address of the batch
        vector_id:           Ring vector ID (from BitVectorRing.compress)
        tokens:              Frozen copy of original tokens
        reg_zeros:           Per-token (register, zeros) pairs
        popcount:            Number of set bits in the batch bitvector
        rank_delta:          Ring rank change caused by this batch
        label:               User-provided label (for provenance)
        timestamp:           Wall-clock time of the ingest call
    """
    batch_id: str
    vector_id: int
    tokens: Tuple[str, ...]
    reg_zeros: Tuple[Tuple[int, int], ...]
    popcount: int
    rank_delta: int
    label: str
    timestamp: float

    @property
    def is_basis_extending(self) -> bool:
        """Did this batch add new basis vectors to the ring?"""
        return self.rank_delta > 0

    def __repr__(self) -> str:
        n = len(self.tokens)
        return (
            f"IngestRecord({self.batch_id[:8]}…, {n} tokens, "
            f"pop={self.popcount}, Δrank={self.rank_delta:+d})"
        )


@dataclass(frozen=True)
class RingDelta:
    """
    What changed in the ring during a transaction (or merge).

    Attributes:
        rank_before:       Basis rank at start
        rank_after:        Basis rank at end
        new_basis_count:   rank_after − rank_before
        ingested_count:    Number of ingest() calls
        total_tokens:      Sum of tokens across all ingests
        independent_count: Batches that increased the rank
        dependent_count:   Batches fully in the existing span
    """
    rank_before: int
    rank_after: int
    new_basis_count: int
    ingested_count: int
    total_tokens: int
    independent_count: int
    dependent_count: int

    @property
    def compression_ratio(self) -> float:
        """How much the basis compresses the ingested data."""
        if self.rank_after == 0:
            return float('inf')
        return self.total_tokens / self.rank_after

    def __repr__(self) -> str:
        return (
            f"RingDelta(rank {self.rank_before}→{self.rank_after}, "
            f"+{self.new_basis_count} bases, "
            f"{self.total_tokens} tokens in {self.ingested_count} batches)"
        )


@dataclass(frozen=True)
class TransactionResult:
    """
    Immutable result of a committed transaction.

    This is what the user receives after calling ``tx.commit()``.
    Every field is frozen / tuple-wrapped — nothing can be mutated.

    Attributes:
        transaction_id:     SHA1 of merged_hllset registers (content address)
        timestamp:          Wall-clock time of commit
        ring_delta:         Ring change summary
        merged_hllset:      The primary HLLSet (union of selected vectors)
        per_ingest_hllsets: One HLLSet per ingest record (empty if not requested)
        basis_vectors:      Frozen copy of the draft ring's basis
        token_lut:          TokenLUT built during ingestion
        ingest_records:     Full provenance chain
    """
    transaction_id: str
    timestamp: float
    ring_delta: RingDelta
    merged_hllset: HLLSet
    per_ingest_hllsets: Tuple[HLLSet, ...]
    basis_vectors: Tuple[BitVector, ...]
    token_lut: TokenLUT
    ingest_records: Tuple[IngestRecord, ...]

    def __repr__(self) -> str:
        card = self.merged_hllset.cardinality()
        return (
            f"TransactionResult({self.transaction_id[:8]}…, "
            f"|M|≈{card:.0f}, {self.ring_delta})"
        )


@dataclass(frozen=True)
class SearchResult:
    """
    Result of a ``search()`` call within a RingTransaction.

    The search condition is always an HLLSet or BitVector (or a list
    of them), optionally narrowed by a cardinality range.  Positions
    in the query that have no backing entry in the transaction's
    TokenLUT are flagged as *orphans* and masked out of the overlap
    computation.

    Attributes:
        basis_indices:       Matching basis vector indices (for commit).
        vector_ids:          Matching ingested-vector IDs   (for commit).
        query_popcount:      Popcount of the *validated* query bitvector.
        orphan_positions:    (reg, zeros) in query NOT present in the LUT.
        warnings:            Human-readable warning strings.
    """
    basis_indices: Tuple[int, ...]
    vector_ids: Tuple[int, ...]
    query_popcount: int
    orphan_positions: Tuple[Tuple[int, int], ...]
    warnings: Tuple[str, ...]

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def empty(self) -> bool:
        return len(self.basis_indices) == 0 and len(self.vector_ids) == 0

    def __len__(self) -> int:
        return len(self.basis_indices)

    def __repr__(self) -> str:
        w = f", {len(self.warnings)} warn" if self.warnings else ""
        return (
            f"SearchResult({len(self.basis_indices)} bases, "
            f"{len(self.vector_ids)} vectors{w})"
        )


# =============================================================================
# Ring Transaction
# =============================================================================

class RingTransaction:
    """
    Ephemeral workspace for ring-based HLLSet construction.

    Lifecycle::

        with RingTransaction(base_ring=ring, lattice=lat) as tx:
            tx.ingest(["alpha", "beta", "gamma"])
            tx.ingest(["delta", "epsilon"], label="extra")

            # Filter bases by lattice time-slice relevance
            relevant = tx.bases_from_lattice(t0, t1)

            result = tx.commit(basis_indices=relevant)
        # auto-rollback on exception

    After commit:

        # The result is fully immutable
        lattice.append([result.merged_hllset], timestamp=now)
        RingTransaction.merge_into_ring(main_ring, result)

    The transaction deep-copies the base ring at creation time, so the
    original ring is never mutated.  All intermediate state lives only
    inside the transaction and is discarded on rollback.
    """

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    def __init__(
        self,
        base_ring: Optional[BitVectorRing] = None,
        lattice=None,           # Optional[HLLLattice] — kept untyped to avoid import
        p_bits: int = 10,
        config: Optional[HashConfig] = None,
    ):
        """
        Begin a new transaction.

        Args:
            base_ring:  Existing ring to branch from (deep-copied).
                        If None a fresh ring is created.
            lattice:    Read-only lattice reference for search / filter.
            p_bits:     HLL precision bits (used when base_ring is None).
            config:     Hash configuration (default: DEFAULT_HASH_CONFIG).
        """
        self._phase = TransactionPhase.ACTIVE
        self._p_bits = p_bits
        self._config = config or DEFAULT_HASH_CONFIG
        self._N = (1 << p_bits) * 32          # total bits per bitvector

        # Deep-copy the base ring (or create fresh)
        if base_ring is not None:
            if base_ring.N != self._N:
                raise ValueError(
                    f"base_ring.N={base_ring.N} does not match "
                    f"expected N={self._N} for p_bits={p_bits}"
                )
            self._draft_ring = _copy_ring(base_ring)
            self._rank_at_start = base_ring.rank()
        else:
            self._draft_ring = BitVectorRing(N=self._N)
            self._rank_at_start = 0

        # Lattice reference (read-only — never mutated)
        self._lattice = lattice

        # Accumulation buffers
        self._ingest_records: List[IngestRecord] = []
        self._token_lut = TokenLUT(p_bits=p_bits)
        self._original_bvs: Dict[int, BitVector] = {}   # vector_id → BV

        self._created_at = time.time()

    # -----------------------------------------------------------------
    # Context manager  (auto-rollback on exception)
    # -----------------------------------------------------------------

    def __enter__(self) -> 'RingTransaction':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._phase == TransactionPhase.ACTIVE:
            self.rollback()
        return False  # do not suppress exceptions

    # -----------------------------------------------------------------
    # Guard
    # -----------------------------------------------------------------

    def _assert_active(self) -> None:
        if self._phase != TransactionPhase.ACTIVE:
            raise RuntimeError(
                f"Transaction is {self._phase.value}; cannot modify."
            )

    # -----------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------

    def ingest(
        self,
        tokens: List[str],
        label: str = "",
    ) -> IngestRecord:
        """
        Ingest a batch of tokens into the draft ring.

        Steps performed:
            1. hash(token) → (reg, zeros) for each token
            2. Build an HLLTensor and populate the TokenLUT
            3. Flatten the tensor to a BitVector
            4. Compress the bitvector into the ring (Gaussian elimination)
            5. Record provenance in an IngestRecord

        Args:
            tokens: Batch of string tokens.
            label:  User-provided label (for provenance tracking).

        Returns:
            Frozen IngestRecord describing this ingestion step.
        """
        self._assert_active()
        if isinstance(tokens, str):
            tokens = [tokens]
        tokens = list(tokens)

        # 1. Hash every token → (reg, zeros)
        reg_zeros_list: List[Tuple[int, int]] = []
        for tok in tokens:
            rz = self._config.hash_to_reg_zeros(tok)
            reg_zeros_list.append(rz)
            # populate token LUT
            h_full = self._config.hash(tok)
            self._token_lut.add_token(tok, rz[0], rz[1], hash_full=h_full)

        # 2. Inscribe into tensor
        tensor = HLLTensor(p_bits=self._p_bits)
        tensor.inscribe_batch(reg_zeros_list)

        # 3. Flatten to bitvector
        bv = tensor.to_bitvector()

        # 4. Compress into ring
        rank_before = self._draft_ring.rank()
        vector_id = self._draft_ring.compress(bv)
        rank_after = self._draft_ring.rank()

        # 5. Store original bitvector (for vector-level HLLSet building)
        #    If the same bitvector was already seen, vector_id is reused
        #    and the stored BV is identical — idempotent.
        self._original_bvs[vector_id] = BitVector(bv.value, bv.N)

        # 6. Build frozen record
        batch_id = compute_sha1(" ".join(sorted(tokens)))
        record = IngestRecord(
            batch_id=batch_id,
            vector_id=vector_id,
            tokens=tuple(tokens),
            reg_zeros=tuple(reg_zeros_list),
            popcount=bv.popcount(),
            rank_delta=rank_after - rank_before,
            label=label,
            timestamp=time.time(),
        )
        self._ingest_records.append(record)
        return record

    def ingest_batches(
        self,
        batches: List[List[str]],
        labels: Optional[List[str]] = None,
    ) -> List[IngestRecord]:
        """
        Ingest multiple batches in order.

        Args:
            batches: List of token lists.
            labels:  Per-batch labels (defaults to empty strings).

        Returns:
            List of IngestRecord, one per batch.
        """
        self._assert_active()
        labels = labels or [""] * len(batches)
        return [
            self.ingest(batch, label=lbl)
            for batch, lbl in zip(batches, labels)
        ]

    # -----------------------------------------------------------------
    # Basis Inspection
    # -----------------------------------------------------------------

    @property
    def draft_ring(self) -> BitVectorRing:
        """Read access to the draft ring (for advanced introspection)."""
        self._assert_active()
        return self._draft_ring

    @property
    def basis_count(self) -> int:
        """Current rank of the draft ring."""
        return self._draft_ring.rank() if self._draft_ring else 0

    def get_basis_vectors(self) -> List[Tuple[int, BitVector]]:
        """
        Return (index, BitVector) pairs for every basis vector.

        Useful for manual inspection before filtering.
        """
        self._assert_active()
        return list(enumerate(self._draft_ring.basis))

    def get_basis_info(self) -> List[Dict[str, Any]]:
        """
        Metadata for each basis vector.

        Returns a list of dicts with keys:
            index, leading_bit, popcount, N
        """
        self._assert_active()
        return [
            {
                "index": i,
                "leading_bit": self._draft_ring.leading_bits[i],
                "popcount": bv.popcount(),
                "N": bv.N,
            }
            for i, bv in enumerate(self._draft_ring.basis)
        ]

    # -----------------------------------------------------------------
    # Search — unified filter: HLLSet / BitVector + cardinality range
    #
    # The algebraic types ARE the query language.
    # Any position in the query not backed by a LUT entry is an
    # orphan: warned about and masked out before overlap is computed.
    # -----------------------------------------------------------------

    def search(
        self,
        query: Union[HLLSet, BitVector, List[Union[HLLSet, BitVector]]],
        min_card: Optional[float] = None,
        max_card: Optional[float] = None,
    ) -> SearchResult:
        """
        Find basis vectors and ingested vectors that overlap with *query*.

        This is the **single entry-point** for all filtering.  The query
        is one or more HLLSets / BitVectors (unioned together); an
        optional cardinality range narrows the result further.

        LUT validation
        ~~~~~~~~~~~~~~
        Every active (reg, zeros) position in the query is checked
        against the transaction's TokenLUT.  Positions that have **no**
        LUT entry are *orphans*: they are reported in
        ``SearchResult.warnings`` and **masked out** so they cannot
        cause false-positive overlaps.

        Args:
            query:      An HLLSet, BitVector, or list of either.
                        Multiple items are unioned into one query mask.
            min_card:   Optional lower bound on the matching entity's
                        estimated cardinality.
            max_card:   Optional upper bound.

        Returns:
            Frozen ``SearchResult`` with ``basis_indices`` and
            ``vector_ids`` ready for ``commit()``.

        Example::

            # Search by a reference HLLSet (e.g. from lattice)
            ref = lattice.cumulative(t=1.0)
            found = tx.search(ref, min_card=5.0)
            result = tx.commit(basis_indices=found.basis_indices)
        """
        self._assert_active()

        # ── normalise query to a single BitVector ────────────────
        query_bv = self._normalize_query(query)

        # ── LUT validation: mask out orphan positions ────────────
        validated_bv, orphans, warnings = self._validate_query_lut(query_bv)

        if validated_bv.is_zero():
            return SearchResult(
                basis_indices=(),
                vector_ids=(),
                query_popcount=0,
                orphan_positions=tuple(orphans),
                warnings=tuple(warnings),
            )

        # ── helper: cardinality gate ─────────────────────────────
        def _card_ok(bv: BitVector) -> bool:
            if min_card is None and max_card is None:
                return True
            hll = self._bitvector_to_hllset(bv)
            c = hll.cardinality()
            if min_card is not None and c < min_card:
                return False
            if max_card is not None and c > max_card:
                return False
            return True

        # ── match basis vectors (AND with validated query ≠ 0) ───
        matching_bases: List[int] = []
        for i, bv in enumerate(self._draft_ring.basis):
            if not (bv & validated_bv).is_zero() and _card_ok(bv):
                matching_bases.append(i)

        # ── match ingested vectors ───────────────────────────────
        matching_vectors: List[int] = []
        seen_vids: Set[int] = set()
        for rec in self._ingest_records:
            vid = rec.vector_id
            if vid in seen_vids:
                continue
            seen_vids.add(vid)
            bv = self._original_bvs.get(vid)
            if bv is None:
                bv = self._draft_ring.decompress(vid)
            if not (bv & validated_bv).is_zero() and _card_ok(bv):
                matching_vectors.append(vid)

        return SearchResult(
            basis_indices=tuple(matching_bases),
            vector_ids=tuple(matching_vectors),
            query_popcount=validated_bv.popcount(),
            orphan_positions=tuple(orphans),
            warnings=tuple(warnings),
        )

    # -- query helpers (private) ------------------------------------

    def _normalize_query(
        self,
        query: Union[HLLSet, BitVector, List[Union[HLLSet, BitVector]]],
    ) -> BitVector:
        """
        Convert one or more HLLSets / BitVectors to a single BitVector.

        Multiple items are unioned (OR-ed) together.
        """
        if isinstance(query, list):
            if not query:
                return BitVector.zeros(self._N)
            parts = [self._normalize_query(q) for q in query]
            combined = parts[0]
            for p in parts[1:]:
                combined = combined | p
            return combined

        if isinstance(query, HLLSet):
            return BitVector.from_numpy(query.dump_numpy())

        if isinstance(query, BitVector):
            if query.N != self._N:
                raise ValueError(
                    f"BitVector N={query.N} does not match "
                    f"expected N={self._N} for p_bits={self._p_bits}"
                )
            return query

        raise TypeError(
            f"query must be HLLSet, BitVector, or list thereof; "
            f"got {type(query).__name__}"
        )

    def _validate_query_lut(
        self,
        query_bv: BitVector,
    ) -> Tuple[BitVector, List[Tuple[int, int]], List[str]]:
        """
        Mask out (reg, zeros) positions that have no TokenLUT entry.

        Returns:
            (validated_bitvector, orphan_positions, warning_messages)
        """
        raw_regs = query_bv.to_numpy()
        num_regs = 1 << self._p_bits
        validated_regs = np.zeros(num_regs, dtype=np.uint32)
        orphans: List[Tuple[int, int]] = []

        for reg_idx in range(min(len(raw_regs), num_regs)):
            val = int(raw_regs[reg_idx])
            keep = 0
            z = 0
            tmp = val
            while tmp:
                if tmp & 1:
                    if self._token_lut.has_candidates(reg_idx, z):
                        keep |= (1 << z)
                    else:
                        orphans.append((reg_idx, z))
                tmp >>= 1
                z += 1
            validated_regs[reg_idx] = np.uint32(keep)

        validated_bv = BitVector.from_numpy(validated_regs)

        warnings: List[str] = []
        if orphans:
            preview = orphans[:5]
            tail = "…" if len(orphans) > 5 else ""
            warnings.append(
                f"{len(orphans)} query position(s) not in LUT — "
                f"masked out: {preview}{tail}"
            )

        return validated_bv, orphans, warnings

    # -----------------------------------------------------------------
    # Building HLLSets  (pre-commit inspection or explicit selection)
    # -----------------------------------------------------------------

    def _bitvector_to_hllset(self, bv: BitVector) -> HLLSet:
        """
        Convert a BitVector back to an HLLSet.

        The bitvector is split into (2^p) uint32 registers, which are
        loaded directly into an HLLSet via the C backend.
        """
        registers = bv.to_numpy()
        expected = 1 << self._p_bits
        if len(registers) != expected:
            raise ValueError(
                f"BitVector gives {len(registers)} registers, "
                f"expected {expected} for p_bits={self._p_bits}"
            )
        hll = HLLSet(p_bits=self._p_bits)
        hll._core.set_registers(registers)
        hll._compute_name()
        return hll

    def build_hllset_from_bases(
        self,
        basis_indices: Optional[List[int]] = None,
    ) -> HLLSet:
        """
        Build a single HLLSet by OR-ing selected basis vectors.

        If *basis_indices* is None, all basis vectors are used.

        This produces an HLLSet whose registers are the bitwise union
        of the chosen bases — the lattice join in the ring.
        """
        self._assert_active()
        bases = self._draft_ring.basis
        if basis_indices is not None:
            bases = [bases[i] for i in basis_indices]

        if not bases:
            return HLLSet(p_bits=self._p_bits)

        combined = BitVector.zeros(self._N)
        for bv in bases:
            combined = combined | bv
        return self._bitvector_to_hllset(combined)

    def build_hllset_from_vectors(
        self,
        vector_ids: Optional[List[int]] = None,
    ) -> HLLSet:
        """
        Build a single HLLSet by OR-ing original ingested bitvectors.

        If *vector_ids* is None, all ingested vectors are used.

        Unlike ``build_hllset_from_bases`` this uses the **pre-elimination**
        bitvectors, preserving the exact register pattern of the ingested
        token batches.
        """
        self._assert_active()
        if vector_ids is None:
            vector_ids = [r.vector_id for r in self._ingest_records]

        if not vector_ids:
            return HLLSet(p_bits=self._p_bits)

        combined = BitVector.zeros(self._N)
        for vid in vector_ids:
            bv = self._original_bvs.get(vid)
            if bv is None:
                bv = self._draft_ring.decompress(vid)
            combined = combined | bv
        return self._bitvector_to_hllset(combined)

    def build_hllsets_per_ingest(self) -> List[Tuple[IngestRecord, HLLSet]]:
        """
        Build one HLLSet per ingest record (pre-elimination bitvectors).

        Returns:
            List of (IngestRecord, HLLSet) pairs.
        """
        self._assert_active()
        result: List[Tuple[IngestRecord, HLLSet]] = []
        for rec in self._ingest_records:
            bv = self._original_bvs.get(rec.vector_id)
            if bv is None:
                bv = self._draft_ring.decompress(rec.vector_id)
            hll = self._bitvector_to_hllset(bv)
            result.append((rec, hll))
        return result

    # -----------------------------------------------------------------
    # Commit / Rollback
    # -----------------------------------------------------------------

    def commit(
        self,
        basis_indices: Optional[List[int]] = None,
        vector_ids: Optional[List[int]] = None,
        include_per_ingest: bool = False,
    ) -> TransactionResult:
        """
        Commit the transaction, returning a frozen ``TransactionResult``.

        Exactly one of *basis_indices* / *vector_ids* / neither may be
        provided:

        - **basis_indices**: Build merged HLLSet from selected ring bases.
        - **vector_ids**: Build merged HLLSet from selected ingested vectors.
        - **neither**: Build merged HLLSet from ALL ingested vectors.

        Args:
            basis_indices:      Basis indices (from ``filter_bases`` etc.).
            vector_ids:         Ingested vector IDs (from ``filter_vectors``).
            include_per_ingest: Also build one HLLSet per ingest record.

        Returns:
            Frozen ``TransactionResult``.

        Raises:
            RuntimeError: If the transaction is not ACTIVE.
            ValueError:   If both basis_indices and vector_ids are given.
        """
        self._assert_active()

        if basis_indices is not None and vector_ids is not None:
            raise ValueError(
                "Provide basis_indices OR vector_ids, not both"
            )

        # Finalize ring (recompute all coefficient vectors)
        self._draft_ring.finalize()

        # ---- Build the primary merged HLLSet ----------------------------
        if basis_indices is not None:
            merged_hllset = self.build_hllset_from_bases(basis_indices)
        elif vector_ids is not None:
            merged_hllset = self.build_hllset_from_vectors(vector_ids)
        else:
            merged_hllset = self.build_hllset_from_vectors()  # all

        # ---- Per-ingest HLLSets (optional) -------------------------------
        per_ingest: Tuple[HLLSet, ...] = ()
        if include_per_ingest:
            pairs = self.build_hllsets_per_ingest()
            per_ingest = tuple(hll for _, hll in pairs)

        # ---- Ring delta --------------------------------------------------
        rank_after = self._draft_ring.rank()
        independent = sum(1 for r in self._ingest_records if r.is_basis_extending)
        ring_delta = RingDelta(
            rank_before=self._rank_at_start,
            rank_after=rank_after,
            new_basis_count=rank_after - self._rank_at_start,
            ingested_count=len(self._ingest_records),
            total_tokens=sum(len(r.tokens) for r in self._ingest_records),
            independent_count=independent,
            dependent_count=len(self._ingest_records) - independent,
        )

        # ---- Freeze basis vectors ----------------------------------------
        frozen_basis = tuple(
            BitVector(b.value, b.N) for b in self._draft_ring.basis
        )

        # ---- Transaction ID (content address of the result) --------------
        tx_id = compute_sha1(merged_hllset.dump_numpy())

        result = TransactionResult(
            transaction_id=tx_id,
            timestamp=time.time(),
            ring_delta=ring_delta,
            merged_hllset=merged_hllset,
            per_ingest_hllsets=per_ingest,
            basis_vectors=frozen_basis,
            token_lut=self._token_lut,
            ingest_records=tuple(self._ingest_records),
        )

        # ---- Transition to COMMITTED ------------------------------------
        self._phase = TransactionPhase.COMMITTED
        return result

    def rollback(self) -> None:
        """
        Discard all accumulated changes.

        After rollback the transaction object is dead — no further
        operations are allowed.
        """
        self._assert_active()
        self._phase = TransactionPhase.ROLLED_BACK
        # Release references
        self._draft_ring = None       # type: ignore[assignment]
        self._ingest_records.clear()
        self._original_bvs.clear()
        self._token_lut = None        # type: ignore[assignment]

    # -----------------------------------------------------------------
    # Static: merge committed result into an external ring
    # -----------------------------------------------------------------

    @staticmethod
    def merge_into_ring(
        target_ring: BitVectorRing,
        result: TransactionResult,
    ) -> RingDelta:
        """
        Merge a committed transaction's basis into *target_ring*.

        Uses Gaussian elimination: new independent vectors extend the
        target basis; dependent ones are absorbed.

        Args:
            target_ring:  The ring to merge into (mutated in-place).
            result:       A committed ``TransactionResult``.

        Returns:
            RingDelta describing what changed in *target_ring*.
        """
        rank_before = target_ring.rank()
        new_count = 0

        for bv in result.basis_vectors:
            if target_ring.add_to_basis(bv):
                new_count += 1

        target_ring.finalize()
        rank_after = target_ring.rank()

        return RingDelta(
            rank_before=rank_before,
            rank_after=rank_after,
            new_basis_count=new_count,
            ingested_count=len(result.ingest_records),
            total_tokens=sum(len(r.tokens) for r in result.ingest_records),
            independent_count=new_count,
            dependent_count=len(result.basis_vectors) - new_count,
        )

    # -----------------------------------------------------------------
    # Properties / Stats
    # -----------------------------------------------------------------

    @property
    def phase(self) -> TransactionPhase:
        return self._phase

    @property
    def p_bits(self) -> int:
        return self._p_bits

    @property
    def ingest_count(self) -> int:
        return len(self._ingest_records)

    @property
    def token_count(self) -> int:
        return sum(len(r.tokens) for r in self._ingest_records)

    @property
    def token_lut(self) -> TokenLUT:
        return self._token_lut

    @property
    def ingest_records(self) -> List[IngestRecord]:
        """Read-only view of accumulated ingest records."""
        return list(self._ingest_records)

    def stats(self) -> Dict[str, Any]:
        """Summary statistics of the transaction."""
        return {
            "phase": self._phase.value,
            "p_bits": self._p_bits,
            "ingest_count": len(self._ingest_records),
            "token_count": self.token_count,
            "rank_at_start": self._rank_at_start,
            "current_rank": (
                self._draft_ring.rank() if self._draft_ring else None
            ),
            "lut_size": len(self._token_lut) if self._token_lut else 0,
            "created_at": self._created_at,
        }

    def __len__(self) -> int:
        """Number of ingest records."""
        return len(self._ingest_records)

    def __repr__(self) -> str:
        rank = self._draft_ring.rank() if self._draft_ring else "?"
        return (
            f"RingTransaction(phase={self._phase.value}, "
            f"ingests={len(self._ingest_records)}, rank={rank})"
        )


# =============================================================================
# Convenience factory
# =============================================================================

def begin_transaction(
    base_ring: Optional[BitVectorRing] = None,
    lattice=None,
    p_bits: int = 10,
    config: Optional[HashConfig] = None,
) -> RingTransaction:
    """
    Begin a new ring transaction.

    Convenience factory that mirrors the conceptual
    ``begin_transaction`` / ``commit`` lifecycle.

    Args:
        base_ring:  Existing ring (deep-copied; None → fresh ring).
        lattice:    Read-only lattice for search / filter.
        p_bits:     HLL precision bits (when base_ring is None).
        config:     Hash configuration.

    Returns:
        An ACTIVE ``RingTransaction``.
    """
    return RingTransaction(
        base_ring=base_ring,
        lattice=lattice,
        p_bits=p_bits,
        config=config,
    )


# =============================================================================
# Internal helpers
# =============================================================================

def _copy_ring(ring: BitVectorRing) -> BitVectorRing:
    """
    Deep-copy a BitVectorRing.

    All containers are new objects; BitVector instances are recreated
    (even though their int attributes are immutable in Python).
    """
    new = BitVectorRing(N=ring.N)
    new.basis = [BitVector(b.value, b.N) for b in ring.basis]
    new.leading_bits = list(ring.leading_bits)
    new.vectors = {
        k: (BitVector(v.value, v.N), c)
        for k, (v, c) in ring.vectors.items()
    }
    new.next_id = ring.next_id
    new.vector_to_id = dict(ring.vector_to_id)
    return new


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TransactionPhase",
    "IngestRecord",
    "RingDelta",
    "TransactionResult",
    "SearchResult",
    "RingTransaction",
    "begin_transaction",
]
