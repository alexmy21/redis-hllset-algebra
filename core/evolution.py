"""
Evolution Module (A6): State transitions over time.

This module implements the L3 (Evolution) abstraction level, tracking
system state snapshots as a De Bruijn-style graph where:
- Nodes = State snapshots (union of all base HLLSets at time t)
- Edges = State transitions (delta = symmetric difference)

The same mathematical structure that enables token sequence reconstruction
(L1 De Bruijn) also enables system history replay (L3 De Bruijn).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Iterator, Protocol

from .hllset import HLLSet

if TYPE_CHECKING:
    from .hllset_store import HLLSetStore


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class StateCommit:
    """
    A committed system state snapshot.
    
    Analogous to a git commit, but for HLLSet system state.
    The state_id is the SHA1 of the union of all base HLLSets.
    """
    state_id: str           # SHA1 of the union HLLSet (content-addressable)
    timestamp: float        # Unix timestamp of commit
    parent_id: str | None   # Previous state ID (None for genesis)
    delta_id: str | None    # SHA1 of Δ(parent, self), None for genesis
    message: str = ""       # Human-readable commit message
    metadata: dict = field(default_factory=dict)  # Arbitrary metadata
    
    @property
    def is_genesis(self) -> bool:
        """True if this is the initial commit (no parent)."""
        return self.parent_id is None
    
    def __repr__(self) -> str:
        parent = self.parent_id[:8] + "..." if self.parent_id else "None"
        return f"StateCommit({self.state_id[:12]}... ← {parent}, msg={self.message!r})"


@dataclass
class StateEdge:
    """
    An edge in the evolution graph (state transition).
    
    Represents the transition from source state to target state,
    with the delta (symmetric difference) as the edge label.
    """
    source_id: str      # Parent state SHA1
    target_id: str      # Child state SHA1
    delta_id: str       # SHA1 of the delta HLLSet
    timestamp: float    # When transition occurred
    
    def __repr__(self) -> str:
        return f"Edge({self.source_id[:8]}→{self.target_id[:8]}, Δ={self.delta_id[:8]})"


class BranchStatus(Enum):
    """Status of evolution branches."""
    ACTIVE = auto()      # Current working branch
    MERGED = auto()      # Branch has been merged
    ABANDONED = auto()   # Branch is no longer active


@dataclass
class Branch:
    """
    A named branch in the evolution graph.
    
    Supports git-like branching for parallel development paths.
    """
    name: str
    head_id: str            # Current commit at branch tip
    created_at: float
    status: BranchStatus = BranchStatus.ACTIVE
    
    def __repr__(self) -> str:
        return f"Branch({self.name!r} → {self.head_id[:12]}...)"


# ═══════════════════════════════════════════════════════════════════════════════
# EVOLUTION GRAPH
# ═══════════════════════════════════════════════════════════════════════════════


class EvolutionGraph:
    """
    De Bruijn-style graph tracking system state evolution.
    
    The evolution graph forms a DAG where:
    - Each node is a StateCommit (snapshot of all base HLLSets)
    - Each edge is a state transition with delta (what changed)
    - The main branch tracks linear history
    - Optional branches support parallel development
    
    Key operations:
    - commit(): Snapshot current state, create new commit
    - diff(): Symmetric difference between any two states
    - rollback(): Restore system to a previous state
    - history(): Get commit chain
    
    The graph is content-addressable: same state content → same SHA1.
    This enables efficient deduplication and merge detection.
    
    Example:
        >>> store = HLLSetStore(p_bits=10)
        >>> evo = EvolutionGraph(store)
        >>> 
        >>> # Add some data
        >>> store.register_base(HLLSet.from_batch(["a", "b", "c"]))
        >>> c1 = evo.commit("Initial data")
        >>> 
        >>> # Add more data
        >>> store.register_base(HLLSet.from_batch(["d", "e"]))
        >>> c2 = evo.commit("Added more data")
        >>> 
        >>> # See what changed
        >>> delta = evo.diff(c1.state_id, c2.state_id)
        >>> print(f"Changed: {delta.cardinality():.0f} items")
    """
    
    def __init__(self, store: HLLSetStore):
        """
        Initialize evolution graph.
        
        Args:
            store: The HLLSetStore to track evolution for
        """
        self._store = store
        self._commits: dict[str, StateCommit] = {}      # state_id → commit
        self._edges: dict[str, StateEdge] = {}          # target_id → edge
        self._deltas: dict[str, HLLSet] = {}            # delta_id → delta HLLSet
        self._state_cache: dict[str, HLLSet] = {}       # state_id → union HLLSet
        self._branches: dict[str, Branch] = {}
        self._head: StateCommit | None = None
        self._current_branch: str = "main"
        
        # Create default main branch
        self._branches["main"] = Branch(
            name="main",
            head_id="",  # Will be set on first commit
            created_at=time.time(),
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Core Operations
    # ─────────────────────────────────────────────────────────────────────────
    
    def commit(self, message: str = "", **metadata) -> StateCommit:
        """
        Snapshot current system state and create a new commit.
        
        This computes the union of all base HLLSets in the store,
        generates a content-addressable SHA1, and records the delta
        from the previous state.
        
        Args:
            message: Human-readable commit message
            **metadata: Arbitrary metadata to attach to commit
            
        Returns:
            The new StateCommit object
        """
        # Compute current state: union of all base HLLSets
        state_hll = self._compute_state_union()
        state_id = self._compute_state_id(state_hll)
        
        # Check for duplicate commit (same state)
        if state_id in self._commits:
            # State unchanged, return existing commit
            return self._commits[state_id]
        
        # Compute delta from parent
        parent_id = self._head.state_id if self._head else None
        delta_id = None
        
        if parent_id:
            parent_hll = self._state_cache.get(parent_id)
            if parent_hll:
                delta_hll = state_hll.symmetric_difference(parent_hll)
                delta_id = self._compute_state_id(delta_hll)
                self._deltas[delta_id] = delta_hll
        
        # Create commit
        commit = StateCommit(
            state_id=state_id,
            timestamp=time.time(),
            parent_id=parent_id,
            delta_id=delta_id,
            message=message,
            metadata=metadata,
        )
        
        # Record commit and edge
        self._commits[state_id] = commit
        self._state_cache[state_id] = state_hll
        
        if parent_id:
            edge = StateEdge(
                source_id=parent_id,
                target_id=state_id,
                delta_id=delta_id,
                timestamp=commit.timestamp,
            )
            self._edges[state_id] = edge
        
        # Update head and branch
        self._head = commit
        self._branches[self._current_branch].head_id = state_id
        
        return commit
    
    def diff(self, commit_a: str, commit_b: str) -> HLLSet:
        """
        Compute symmetric difference between two states.
        
        This shows what changed between any two commits,
        regardless of their position in the history.
        
        Args:
            commit_a: First state ID
            commit_b: Second state ID
            
        Returns:
            HLLSet representing the symmetric difference
        """
        hll_a = self._get_state_hll(commit_a)
        hll_b = self._get_state_hll(commit_b)
        
        if hll_a is None or hll_b is None:
            raise KeyError(f"Unknown commit: {commit_a if hll_a is None else commit_b}")
        
        return hll_a.symmetric_difference(hll_b)
    
    def get_delta(self, commit_id: str) -> HLLSet | None:
        """
        Get the delta (what changed) for a specific commit.
        
        Args:
            commit_id: The commit to get delta for
            
        Returns:
            HLLSet delta, or None if genesis commit
        """
        commit = self._commits.get(commit_id)
        if not commit or not commit.delta_id:
            return None
        return self._deltas.get(commit.delta_id)
    
    def history(self, branch: str | None = None) -> list[StateCommit]:
        """
        Get commit history in chronological order (oldest first).
        
        Args:
            branch: Branch name, or None for current branch
            
        Returns:
            List of commits from genesis to head
        """
        branch_name = branch or self._current_branch
        branch_obj = self._branches.get(branch_name)
        
        if not branch_obj or not branch_obj.head_id:
            return []
        
        # Walk backwards from head to genesis
        commits = []
        current_id = branch_obj.head_id
        
        while current_id:
            commit = self._commits.get(current_id)
            if not commit:
                break
            commits.append(commit)
            current_id = commit.parent_id
        
        # Reverse to get chronological order
        commits.reverse()
        return commits
    
    def head(self) -> StateCommit | None:
        """Get the current head commit."""
        return self._head
    
    def get_commit(self, commit_id: str) -> StateCommit | None:
        """Get a specific commit by ID."""
        return self._commits.get(commit_id)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Rollback and Navigation
    # ─────────────────────────────────────────────────────────────────────────
    
    def rollback(self, target_commit: str) -> StateCommit:
        """
        Restore system to a previous state.
        
        This creates a NEW commit that represents the rollback,
        preserving full history (non-destructive).
        
        Args:
            target_commit: The commit ID to roll back to
            
        Returns:
            New commit representing the rollback
        """
        target = self._commits.get(target_commit)
        if not target:
            raise KeyError(f"Unknown commit: {target_commit}")
        
        # Get the target state
        target_hll = self._get_state_hll(target_commit)
        if target_hll is None:
            raise ValueError(f"Cannot reconstruct state for {target_commit}")
        
        # For now, we just record the rollback as metadata
        # Full implementation would restore the actual store state
        return self.commit(
            message=f"Rollback to {target_commit[:12]}",
            rollback_target=target_commit,
            rollback_from=self._head.state_id if self._head else None,
        )
    
    def checkout(self, commit_id: str) -> StateCommit:
        """
        Move HEAD to a specific commit (detached head state).
        
        Args:
            commit_id: The commit to checkout
            
        Returns:
            The checked-out commit
        """
        commit = self._commits.get(commit_id)
        if not commit:
            raise KeyError(f"Unknown commit: {commit_id}")
        
        self._head = commit
        return commit
    
    # ─────────────────────────────────────────────────────────────────────────
    # Branching
    # ─────────────────────────────────────────────────────────────────────────
    
    def create_branch(self, name: str, from_commit: str | None = None) -> Branch:
        """
        Create a new branch.
        
        Args:
            name: Branch name
            from_commit: Commit to branch from (default: HEAD)
            
        Returns:
            The new Branch object
        """
        if name in self._branches:
            raise ValueError(f"Branch {name!r} already exists")
        
        head_id = from_commit or (self._head.state_id if self._head else "")
        
        branch = Branch(
            name=name,
            head_id=head_id,
            created_at=time.time(),
        )
        self._branches[name] = branch
        return branch
    
    def switch_branch(self, name: str) -> Branch:
        """
        Switch to a different branch.
        
        Args:
            name: Branch name to switch to
            
        Returns:
            The Branch object
        """
        branch = self._branches.get(name)
        if not branch:
            raise KeyError(f"Unknown branch: {name}")
        
        self._current_branch = name
        if branch.head_id:
            self._head = self._commits.get(branch.head_id)
        
        return branch
    
    def list_branches(self) -> list[Branch]:
        """List all branches."""
        return list(self._branches.values())
    
    def current_branch(self) -> Branch:
        """Get the current branch."""
        return self._branches[self._current_branch]
    
    # ─────────────────────────────────────────────────────────────────────────
    # Merge Operations
    # ─────────────────────────────────────────────────────────────────────────
    
    def merge(self, source_branch: str, message: str = "") -> StateCommit:
        """
        Merge another branch into the current branch.
        
        The merge strategy uses HLLSet union: merged = current ∪ source.
        
        Args:
            source_branch: Branch name to merge from
            message: Merge commit message
            
        Returns:
            The merge commit
        """
        source = self._branches.get(source_branch)
        if not source:
            raise KeyError(f"Unknown branch: {source_branch}")
        
        current_hll = self._get_state_hll(self._head.state_id) if self._head else None
        source_hll = self._get_state_hll(source.head_id) if source.head_id else None
        
        if current_hll is None:
            # No current state, just adopt source
            if source_hll:
                self._state_cache["_merge_temp"] = source_hll
        elif source_hll is None:
            # No source state, nothing to merge
            pass
        else:
            # Merge via union
            merged = current_hll.union(source_hll)
            self._state_cache["_merge_temp"] = merged
        
        return self.commit(
            message=message or f"Merge {source_branch} into {self._current_branch}",
            merge_source=source_branch,
            merge_source_commit=source.head_id,
        )
    
    def find_common_ancestor(self, commit_a: str, commit_b: str) -> str | None:
        """
        Find the common ancestor of two commits.
        
        Args:
            commit_a: First commit ID
            commit_b: Second commit ID
            
        Returns:
            Commit ID of common ancestor, or None if none exists
        """
        # Build ancestor set for commit_a
        ancestors_a = set()
        current = commit_a
        while current:
            ancestors_a.add(current)
            commit = self._commits.get(current)
            current = commit.parent_id if commit else None
        
        # Walk commit_b's ancestors until we find one in ancestors_a
        current = commit_b
        while current:
            if current in ancestors_a:
                return current
            commit = self._commits.get(current)
            current = commit.parent_id if commit else None
        
        return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Analysis
    # ─────────────────────────────────────────────────────────────────────────
    
    def evolution_rate(self, window: int = 10) -> float:
        """
        Compute average cardinality change per commit.
        
        Args:
            window: Number of recent commits to analyze
            
        Returns:
            Average delta cardinality
        """
        history = self.history()
        if len(history) < 2:
            return 0.0
        
        recent = history[-window:] if len(history) > window else history
        
        total_delta = 0.0
        count = 0
        
        for commit in recent:
            if commit.delta_id:
                delta = self._deltas.get(commit.delta_id)
                if delta:
                    total_delta += delta.cardinality()
                    count += 1
        
        return total_delta / count if count > 0 else 0.0
    
    def state_cardinality(self, commit_id: str | None = None) -> float:
        """
        Get cardinality of a state.
        
        Args:
            commit_id: Commit to check, or None for HEAD
            
        Returns:
            Estimated cardinality
        """
        cid = commit_id or (self._head.state_id if self._head else None)
        if not cid:
            return 0.0
        
        hll = self._get_state_hll(cid)
        return hll.cardinality() if hll else 0.0
    
    def stats(self) -> dict:
        """Get evolution graph statistics."""
        return {
            "total_commits": len(self._commits),
            "total_branches": len(self._branches),
            "current_branch": self._current_branch,
            "head": self._head.state_id[:16] + "..." if self._head else None,
            "cached_states": len(self._state_cache),
            "cached_deltas": len(self._deltas),
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────────────────────────────────
    
    def to_dot(self) -> str:
        """
        Export evolution graph to Graphviz DOT format.
        
        Returns:
            DOT format string
        """
        lines = ["digraph EvolutionGraph {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box, style=rounded];")
        lines.append("")
        
        # Add nodes
        for state_id, commit in self._commits.items():
            short_id = state_id[:8]
            label = f"{short_id}\\n{commit.message[:20]}" if commit.message else short_id
            
            # Highlight HEAD
            if self._head and state_id == self._head.state_id:
                lines.append(f'  "{short_id}" [label="{label}", style="rounded,bold", color=blue];')
            else:
                lines.append(f'  "{short_id}" [label="{label}"];')
        
        lines.append("")
        
        # Add edges
        for target_id, edge in self._edges.items():
            source_short = edge.source_id[:8]
            target_short = target_id[:8]
            delta_short = edge.delta_id[:6] if edge.delta_id else "?"
            lines.append(f'  "{source_short}" -> "{target_short}" [label="Δ={delta_short}"];')
        
        lines.append("}")
        return "\n".join(lines)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────────────────────
    
    def _compute_state_union(self) -> HLLSet:
        """Compute union of all base HLLSets in the store."""
        base_ids = list(self._store.all_base_ids())
        
        if not base_ids:
            return HLLSet(p_bits=self._store._p_bits)
        
        # Start with first base
        result = self._store.get(base_ids[0])
        if result is None:
            return HLLSet(p_bits=self._store._p_bits)
        
        # Union with remaining (HLLSet.union returns a new HLLSet)
        for base_id in base_ids[1:]:
            hll = self._store.get(base_id)
            if hll:
                result = result.union(hll)
        
        return result
    
    def _compute_state_id(self, hll: HLLSet) -> str:
        """Compute content-addressable ID for an HLLSet state."""
        data = hll.dump_numpy().tobytes()
        return hashlib.sha1(data).hexdigest()
    
    def _get_state_hll(self, state_id: str) -> HLLSet | None:
        """Get HLLSet for a state, from cache or reconstruction."""
        if state_id in self._state_cache:
            return self._state_cache[state_id]
        
        # Could reconstruct from deltas, but for now just return None
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def create_evolution_tracker(store: HLLSetStore) -> EvolutionGraph:
    """
    Create an evolution tracker for an HLLSetStore.
    
    Args:
        store: The store to track
        
    Returns:
        Configured EvolutionGraph
    """
    return EvolutionGraph(store)
