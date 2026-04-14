"""HLLSetAlgebra – Python wrapper for the Redis HyperLogLog set-algebra Lua scripts.

All cardinality results are *estimates*; HyperLogLog is a probabilistic
data structure with a typical error rate of ~0.81 %.

Usage example
-------------
::

    import redis
    from redis_hllset_algebra import HLLSetAlgebra

    r   = redis.Redis()
    hll = HLLSetAlgebra(r)

    hll.add("users:2024", "alice", "bob", "carol")
    hll.add("users:2025", "bob", "carol", "dave")

    print(hll.union("users:all", "users:2024", "users:2025"))   # ≈ 4
    print(hll.intersect_card("users:2024", "users:2025"))       # ≈ 2
    print(hll.diff_card("users:2025", "users:2024"))            # ≈ 1  (dave)
    print(hll.symmdiff_card("users:2024", "users:2025"))        # ≈ 2  (alice, dave)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable

# Redis client type hint without a hard import-time dependency
try:
    from redis import Redis as _RedisType  # type: ignore[import]
except ImportError:  # pragma: no cover
    _RedisType = object  # type: ignore[misc,assignment]

_SCRIPTS_DIR = Path(__file__).parent / "scripts"


def _load_script(name: str) -> str:
    """Return the Lua source for *name* (without the .lua extension)."""
    return (_SCRIPTS_DIR / f"{name}.lua").read_text(encoding="utf-8")


class HLLSetAlgebra:
    """Set-algebra operations for Redis HyperLogLog keys.

    All expensive work is done atomically inside Redis via Lua scripts.
    Temporary keys created during computation are automatically removed
    before the script returns.

    Parameters
    ----------
    redis_client:
        A connected ``redis.Redis`` (or compatible) instance.
    tmp_key_prefix:
        Prefix used for transient keys created inside Lua scripts.
        Override this if the default prefix clashes with your keyspace.
    """

    def __init__(
        self,
        redis_client: _RedisType,
        tmp_key_prefix: str = "__hllset_tmp__:",
    ) -> None:
        self._redis = redis_client
        self._tmp_prefix = tmp_key_prefix

        # Pre-register all Lua scripts so subsequent calls use EVALSHA.
        self._script_union = redis_client.register_script(
            _load_script("hll_union")
        )
        self._script_intersect_card = redis_client.register_script(
            _load_script("hll_intersect_card")
        )
        self._script_diff_card = redis_client.register_script(
            _load_script("hll_diff_card")
        )
        self._script_symmdiff_card = redis_client.register_script(
            _load_script("hll_symmdiff_card")
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tmp_key(self) -> str:
        """Return a key name that is almost certainly unused in Redis."""
        return f"{self._tmp_prefix}{uuid.uuid4().hex}"

    # ------------------------------------------------------------------
    # Pass-through convenience wrappers for standard HLL commands
    # ------------------------------------------------------------------

    def add(self, key: str, *elements: str) -> int:
        """Add *elements* to the HLL at *key* (PFADD).

        Returns 1 if the HLL was modified, 0 otherwise.
        """
        return self._redis.pfadd(key, *elements)

    def count(self, *keys: str) -> int:
        """Return the approximate cardinality of the union of *keys* (PFCOUNT)."""
        return self._redis.pfcount(*keys)

    # ------------------------------------------------------------------
    # Set-algebra operations (implemented as Lua scripts)
    # ------------------------------------------------------------------

    def union(self, dest: str, *sources: str) -> int:
        """Merge *sources* into *dest* and return the resulting cardinality.

        This is a thin wrapper around Redis ``PFMERGE`` + ``PFCOUNT`` executed
        atomically in a single Lua call.

        Parameters
        ----------
        dest:
            Destination key.  Will be created or overwritten.
        *sources:
            One or more source HLL keys.

        Returns
        -------
        int
            Estimated cardinality of the union.
        """
        if not sources:
            raise ValueError("union() requires at least one source key")
        keys = [dest] + list(sources)
        return int(self._script_union(keys=keys))

    def union_card(self, *keys: str) -> int:
        """Return the estimated cardinality of the union of *keys*.

        Unlike :meth:`union`, no new key is stored in Redis.

        Parameters
        ----------
        *keys:
            Two or more HLL keys.

        Returns
        -------
        int
            Estimated cardinality of the union.
        """
        if len(keys) < 2:
            raise ValueError("union_card() requires at least 2 keys")
        # Delegate to PFCOUNT with multiple keys – Redis computes the union
        # internally without persisting anything.
        return self._redis.pfcount(*keys)

    def intersect_card(self, *keys: str) -> int:
        """Return the estimated cardinality of the intersection of *keys*.

        Uses the inclusion-exclusion principle computed atomically in Lua.
        The estimate may be slightly negative due to HLL approximation error;
        in that case 0 is returned.

        Parameters
        ----------
        *keys:
            One or more HLL keys.

        Returns
        -------
        int
            Estimated cardinality of the intersection.
        """
        if not keys:
            raise ValueError("intersect_card() requires at least 1 key")
        tmp = self._tmp_key()
        return int(self._script_intersect_card(keys=list(keys), args=[tmp]))

    def diff_card(self, key_a: str, key_b: str) -> int:
        """Return the estimated cardinality of the difference  A \\ B.

        Uses the identity  |A \\ B| = |A ∪ B| - |B|.

        Parameters
        ----------
        key_a:
            The HLL key representing set *A*.
        key_b:
            The HLL key representing set *B*.

        Returns
        -------
        int
            Estimated cardinality of elements in *A* but not in *B*.
        """
        tmp = self._tmp_key()
        return int(self._script_diff_card(keys=[key_a, key_b], args=[tmp]))

    def symmdiff_card(self, key_a: str, key_b: str) -> int:
        """Return the estimated cardinality of the symmetric difference A △ B.

        Uses the identity  |A △ B| = 2·|A ∪ B| - |A| - |B|.

        Parameters
        ----------
        key_a:
            The HLL key representing set *A*.
        key_b:
            The HLL key representing set *B*.

        Returns
        -------
        int
            Estimated cardinality of elements in exactly one of *A* or *B*.
        """
        tmp = self._tmp_key()
        return int(self._script_symmdiff_card(keys=[key_a, key_b], args=[tmp]))

    def is_subset(self, key_a: str, key_b: str) -> bool:
        """Return ``True`` if *A* is approximately a subset of *B*.

        This is probabilistic: if |A \\ B| ≈ 0 then *A* ⊆ *B*.

        Parameters
        ----------
        key_a:
            The candidate subset HLL key.
        key_b:
            The potential superset HLL key.
        """
        return self.diff_card(key_a, key_b) == 0
