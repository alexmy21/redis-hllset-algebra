"""
De Bruijn Graph — Generic implementation for sequence reconstruction.

A De Bruijn graph represents overlapping k-mers as a directed graph where:
- Nodes are (k-1)-mers (prefixes/suffixes of k-mers)
- Edges are k-mers, connecting prefix to suffix
- Edge multiplicity tracks how many times a k-mer appears

This structure enables:
1. Sequence reconstruction from overlapping fragments
2. Handling repeated patterns via edge multiplicities  
3. Finding Eulerian paths (visiting each edge exactly once per multiplicity)

Applications:
- Token order restoration from n-gram tensors
- Genome assembly from DNA reads
- Text reconstruction from overlapping fragments
- Any sequence reassembly problem with overlapping pieces

Example:
    # Reconstruct "a b a b a" from trigrams
    graph = DeBruijnGraph(k=3)
    graph.add_kmer(('START', 'a', 'b'))
    graph.add_kmer(('a', 'b', 'a'), count=2)  # appears twice
    graph.add_kmer(('b', 'a', 'b'))
    graph.add_kmer(('b', 'a', 'END'))
    
    path = graph.find_eulerian_path(start_prefix=('START', 'a'))
    # Returns: [('START', 'a'), ('a', 'b'), ('b', 'a'), ('a', 'b'), ('b', 'a'), ('a', 'END')]
    
    sequence = graph.path_to_sequence(path)
    # Returns: ['START', 'a', 'b', 'a', 'b', 'a', 'END']

Author: HLLSet Algebra Project
"""

from __future__ import annotations
from typing import (
    List, Tuple, Dict, Set, Optional, Iterator, Any, 
    TypeVar, Generic, Hashable, Callable, Union
)
from dataclasses import dataclass, field
from collections import defaultdict, deque
from copy import deepcopy
import random


# Type variable for node/edge labels
T = TypeVar('T', bound=Hashable)


@dataclass
class Edge(Generic[T]):
    """
    Directed edge in a De Bruijn graph.
    
    Attributes:
        source: Source node (k-1)-mer
        target: Target node (k-1)-mer
        label: Edge label (the k-th element of the k-mer)
        kmer: Full k-mer tuple
        multiplicity: How many times this edge should be traversed
    """
    source: Tuple[T, ...]
    target: Tuple[T, ...]
    label: T
    kmer: Tuple[T, ...]
    multiplicity: int = 1
    
    def __hash__(self):
        return hash(self.kmer)
    
    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.kmer == other.kmer


@dataclass 
class PathResult(Generic[T]):
    """
    Result of path finding in a De Bruijn graph.
    
    Attributes:
        path: List of nodes visited
        edges_used: List of edges traversed
        sequence: Reconstructed sequence
        is_eulerian: True if all edges were used exactly per their multiplicity
        unused_edges: Edges that weren't fully consumed
    """
    path: List[Tuple[T, ...]]
    edges_used: List[Edge[T]]
    sequence: List[T]
    is_eulerian: bool
    unused_edges: Dict[Tuple[T, ...], int] = field(default_factory=dict)


class DeBruijnGraph(Generic[T]):
    """
    Generic De Bruijn graph for sequence reconstruction.
    
    A De Bruijn graph of order k has:
    - Nodes: all (k-1)-mers that appear as prefix or suffix of a k-mer
    - Edges: k-mers, where k-mer (a₁,...,aₖ) creates edge (a₁,...,aₖ₋₁) → (a₂,...,aₖ)
    - Edge labels: the last element aₖ (for sequence reconstruction)
    - Multiplicities: how many times each k-mer appears
    
    For token restoration with trigrams (k=3):
    - Nodes are bigrams: (word1, word2)
    - Edges are trigrams: (word1, word2, word3)
    - Edge from (word1, word2) to (word2, word3), labeled word3
    """
    
    def __init__(self, k: int = 3):
        """
        Initialize De Bruijn graph.
        
        Args:
            k: K-mer size (default 3 for trigrams)
        """
        if k < 2:
            raise ValueError("k must be at least 2")
        
        self.k = k
        
        # Node set (for enumeration)
        self._nodes: Set[Tuple[T, ...]] = set()
        
        # Adjacency: source_node → list of (target_node, edge)
        self._adj: Dict[Tuple[T, ...], List[Edge[T]]] = defaultdict(list)
        
        # Reverse adjacency: target_node → list of (source_node, edge)
        self._rev_adj: Dict[Tuple[T, ...], List[Edge[T]]] = defaultdict(list)
        
        # Edge index: kmer → Edge
        self._edges: Dict[Tuple[T, ...], Edge[T]] = {}
        
        # Statistics
        self._total_edge_count = 0  # Sum of all multiplicities
    
    @property
    def nodes(self) -> Set[Tuple[T, ...]]:
        """All nodes in the graph."""
        return self._nodes
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return len(self._nodes)
    
    @property
    def num_edges(self) -> int:
        """Number of unique edges (k-mers)."""
        return len(self._edges)
    
    @property
    def total_edge_count(self) -> int:
        """Total edge count including multiplicities."""
        return self._total_edge_count
    
    def add_kmer(self, kmer: Tuple[T, ...], count: int = 1) -> Edge[T]:
        """
        Add a k-mer to the graph.
        
        Creates an edge from prefix (k-1)-mer to suffix (k-1)-mer.
        If the k-mer already exists, adds to its multiplicity.
        
        Args:
            kmer: Tuple of k elements
            count: Multiplicity (how many times this k-mer appears)
            
        Returns:
            The Edge object (new or updated)
            
        Raises:
            ValueError: If kmer length doesn't match k
        """
        if len(kmer) != self.k:
            raise ValueError(f"Expected {self.k}-mer, got {len(kmer)}-mer: {kmer}")
        
        # Split into prefix and suffix (k-1)-mers
        prefix = kmer[:-1]  # First k-1 elements
        suffix = kmer[1:]   # Last k-1 elements
        label = kmer[-1]    # Last element (edge label)
        
        # Add nodes
        self._nodes.add(prefix)
        self._nodes.add(suffix)
        
        # Check if edge already exists
        if kmer in self._edges:
            edge = self._edges[kmer]
            edge.multiplicity += count
            self._total_edge_count += count
            return edge
        
        # Create new edge
        edge = Edge(
            source=prefix,
            target=suffix,
            label=label,
            kmer=kmer,
            multiplicity=count,
        )
        
        self._edges[kmer] = edge
        self._adj[prefix].append(edge)
        self._rev_adj[suffix].append(edge)
        self._total_edge_count += count
        
        return edge
    
    def add_kmers(self, kmers: List[Tuple[T, ...]], counts: Optional[Dict[Tuple[T, ...], int]] = None):
        """
        Add multiple k-mers to the graph.
        
        Args:
            kmers: List of k-mer tuples
            counts: Optional dict of kmer → count (default 1 for each)
        """
        for kmer in kmers:
            count = counts.get(kmer, 1) if counts else 1
            self.add_kmer(kmer, count)
    
    def get_edge(self, kmer: Tuple[T, ...]) -> Optional[Edge[T]]:
        """Get edge by k-mer."""
        return self._edges.get(kmer)
    
    def out_edges(self, node: Tuple[T, ...]) -> List[Edge[T]]:
        """Get outgoing edges from a node."""
        return self._adj.get(node, [])
    
    def in_edges(self, node: Tuple[T, ...]) -> List[Edge[T]]:
        """Get incoming edges to a node."""
        return self._rev_adj.get(node, [])
    
    def out_degree(self, node: Tuple[T, ...], weighted: bool = True) -> int:
        """
        Get out-degree of a node.
        
        Args:
            node: The node
            weighted: If True, count multiplicities; if False, count unique edges
        """
        edges = self._adj.get(node, [])
        if weighted:
            return sum(e.multiplicity for e in edges)
        return len(edges)
    
    def in_degree(self, node: Tuple[T, ...], weighted: bool = True) -> int:
        """
        Get in-degree of a node.
        
        Args:
            node: The node  
            weighted: If True, count multiplicities; if False, count unique edges
        """
        edges = self._rev_adj.get(node, [])
        if weighted:
            return sum(e.multiplicity for e in edges)
        return len(edges)
    
    def degree_balance(self, node: Tuple[T, ...]) -> int:
        """
        Get degree balance: out_degree - in_degree.
        
        For Eulerian path:
        - Start node has balance +1
        - End node has balance -1
        - All other nodes have balance 0
        """
        return self.out_degree(node) - self.in_degree(node)
    
    # =========================================================================
    # Path Finding
    # =========================================================================
    
    def find_start_end_nodes(self, 
                             start_marker: Optional[T] = None,
                             end_marker: Optional[T] = None,
                             ) -> Tuple[Optional[Tuple[T, ...]], Optional[Tuple[T, ...]]]:
        """
        Find start and end nodes for Eulerian path.
        
        For an Eulerian path to exist:
        - At most one node has out_degree - in_degree = +1 (start)
        - At most one node has out_degree - in_degree = -1 (end)
        - All other nodes have equal in/out degree
        
        If start_marker/end_marker provided, looks for nodes containing them.
        
        Returns:
            (start_node, end_node) or (None, None) if no valid path exists
        """
        start_node = None
        end_node = None
        
        for node in self._nodes:
            balance = self.degree_balance(node)
            
            # Check for marker-based identification
            if start_marker is not None and node[0] == start_marker:
                start_node = node
            if end_marker is not None and node[-1] == end_marker:
                end_node = node
            
            # Check degree-based identification
            if balance == 1:
                if start_node is None or start_marker is None:
                    start_node = node
            elif balance == -1:
                if end_node is None or end_marker is None:
                    end_node = node
            elif balance != 0:
                # Unbalanced node - no Eulerian path possible
                return None, None
        
        return start_node, end_node
    
    def find_eulerian_path(self,
                           start_node: Optional[Tuple[T, ...]] = None,
                           start_marker: Optional[T] = None,
                           end_marker: Optional[T] = None,
                           randomize: bool = False,
                           ) -> Optional[PathResult[T]]:
        """
        Find Eulerian path using Hierholzer's algorithm.
        
        An Eulerian path visits every edge exactly once (per multiplicity).
        This is the optimal reconstruction that uses all k-mer evidence.
        
        Args:
            start_node: Explicit start node (if known)
            start_marker: Token marking sequence start (e.g., "START")
            end_marker: Token marking sequence end (e.g., "END")
            randomize: If True, randomize edge selection (for multiple solutions)
            
        Returns:
            PathResult with path, sequence, and metadata, or None if no path exists
        """
        if not self._edges:
            return None
        
        # Find start/end nodes if not provided
        if start_node is None:
            start_node, _ = self.find_start_end_nodes(start_marker, end_marker)
        
        if start_node is None:
            # No valid start - pick any node with outgoing edges
            for node in self._nodes:
                if self._adj.get(node):
                    start_node = node
                    break
        
        if start_node is None:
            return None
        
        # Hierholzer's algorithm with multiplicities
        # Track remaining capacity for each edge
        remaining: Dict[Tuple[T, ...], int] = {
            kmer: edge.multiplicity for kmer, edge in self._edges.items()
        }
        
        # Stack for DFS
        stack: List[Tuple[T, ...]] = [start_node]
        path: List[Tuple[T, ...]] = []
        edges_used: List[Edge[T]] = []
        
        while stack:
            current = stack[-1]
            
            # Find an edge with remaining capacity
            available = [
                e for e in self._adj.get(current, [])
                if remaining.get(e.kmer, 0) > 0
            ]
            
            if available:
                # Pick edge (random or deterministic)
                if randomize:
                    edge = random.choice(available)
                else:
                    edge = available[0]
                
                # Consume capacity
                remaining[edge.kmer] -= 1
                
                # Move to next node
                stack.append(edge.target)
            else:
                # No more edges from this node - add to path
                path.append(stack.pop())
        
        # Reverse to get correct order
        path = path[::-1]
        
        # Reconstruct edges from path
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            # Find edge connecting src to tgt
            for edge in self._adj.get(src, []):
                if edge.target == tgt:
                    edges_used.append(edge)
                    break
        
        # Check if all edges were used
        unused = {k: v for k, v in remaining.items() if v > 0}
        is_eulerian = len(unused) == 0
        
        # Convert path to sequence
        sequence = self.path_to_sequence(path)
        
        return PathResult(
            path=path,
            edges_used=edges_used,
            sequence=sequence,
            is_eulerian=is_eulerian,
            unused_edges=unused,
        )
    
    def find_path_greedy(self,
                         start_node: Optional[Tuple[T, ...]] = None,
                         start_marker: Optional[T] = None,
                         end_marker: Optional[T] = None,
                         randomize: bool = False,
                         ) -> Optional[PathResult[T]]:
        """
        Find path using greedy traversal with capacity consumption.
        
        Unlike Eulerian path, this stops when reaching an end node or
        when no more edges are available. Good for partial reconstruction.
        
        Args:
            start_node: Explicit start node
            start_marker: Token marking sequence start
            end_marker: Token marking sequence end
            randomize: If True, randomize choices
            
        Returns:
            PathResult with path and sequence
        """
        if not self._edges:
            return None
        
        # Find start node
        if start_node is None:
            for node in self._nodes:
                if start_marker is not None and node[0] == start_marker:
                    start_node = node
                    break
        
        if start_node is None:
            start_node, _ = self.find_start_end_nodes(start_marker, end_marker)
        
        if start_node is None:
            return None
        
        # Track remaining capacity
        remaining: Dict[Tuple[T, ...], int] = {
            kmer: edge.multiplicity for kmer, edge in self._edges.items()
        }
        
        # Greedy traversal
        path: List[Tuple[T, ...]] = [start_node]
        edges_used: List[Edge[T]] = []
        current = start_node
        
        max_steps = self._total_edge_count + 10
        
        for _ in range(max_steps):
            # Find edges with capacity
            available = [
                e for e in self._adj.get(current, [])
                if remaining.get(e.kmer, 0) > 0
            ]
            
            if not available:
                break
            
            # Pick edge
            if randomize and len(available) > 1:
                edge = random.choice(available)
            else:
                edge = available[0]
            
            # Consume and move
            remaining[edge.kmer] -= 1
            edges_used.append(edge)
            current = edge.target
            path.append(current)
            
            # Check for end marker
            if end_marker is not None and current[-1] == end_marker:
                break
        
        # Check completeness
        unused = {k: v for k, v in remaining.items() if v > 0}
        is_eulerian = len(unused) == 0
        
        sequence = self.path_to_sequence(path)
        
        return PathResult(
            path=path,
            edges_used=edges_used,
            sequence=sequence,
            is_eulerian=is_eulerian,
            unused_edges=unused,
        )
    
    def path_to_sequence(self, path: List[Tuple[T, ...]]) -> List[T]:
        """
        Convert node path to token sequence.
        
        Path: [(a,b), (b,c), (c,d), ...]
        Sequence: [a, b, c, d, ...]
        
        The sequence is built by:
        1. Taking all elements of the first node
        2. Adding the last element of each subsequent node
        
        Args:
            path: List of (k-1)-mer nodes
            
        Returns:
            Reconstructed sequence
        """
        if not path:
            return []
        
        # Start with first node
        sequence = list(path[0])
        
        # Add last element of each subsequent node
        for node in path[1:]:
            sequence.append(node[-1])
        
        return sequence
    
    # =========================================================================
    # Graph Analysis
    # =========================================================================
    
    def is_eulerian(self) -> bool:
        """
        Check if graph has an Eulerian path.
        
        Requirements:
        - Graph is connected (ignoring isolated nodes)
        - At most 2 nodes have unbalanced degree
        - If 2 unbalanced: one has +1, one has -1
        """
        plus_one = 0
        minus_one = 0
        
        for node in self._nodes:
            balance = self.degree_balance(node)
            if balance == 1:
                plus_one += 1
            elif balance == -1:
                minus_one += 1
            elif balance != 0:
                return False
        
        # Valid: (0,0) for cycle, or (1,1) for path
        return (plus_one == 0 and minus_one == 0) or (plus_one == 1 and minus_one == 1)
    
    def is_connected(self) -> bool:
        """Check if all nodes with edges are connected."""
        if not self._nodes:
            return True
        
        # Find nodes with edges
        active = {n for n in self._nodes 
                  if self._adj.get(n) or self._rev_adj.get(n)}
        
        if not active:
            return True
        
        # BFS from any active node
        start = next(iter(active))
        visited = set()
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            
            # Add neighbors (both directions for connectivity)
            for edge in self._adj.get(node, []):
                if edge.target not in visited:
                    queue.append(edge.target)
            for edge in self._rev_adj.get(node, []):
                if edge.source not in visited:
                    queue.append(edge.source)
        
        return visited == active
    
    def stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        in_degrees = [self.in_degree(n) for n in self._nodes]
        out_degrees = [self.out_degree(n) for n in self._nodes]
        
        return {
            'k': self.k,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'total_edge_count': self.total_edge_count,
            'is_eulerian': self.is_eulerian(),
            'is_connected': self.is_connected(),
            'max_in_degree': max(in_degrees) if in_degrees else 0,
            'max_out_degree': max(out_degrees) if out_degrees else 0,
            'avg_multiplicity': self.total_edge_count / max(1, self.num_edges),
        }
    
    def __repr__(self) -> str:
        return (f"DeBruijnGraph(k={self.k}, nodes={self.num_nodes}, "
                f"edges={self.num_edges}, total={self.total_edge_count})")
    
    # =========================================================================
    # Visualization and Export
    # =========================================================================
    
    def to_dot(self, max_nodes: int = 50) -> str:
        """
        Export graph to DOT format for visualization.
        
        Args:
            max_nodes: Maximum nodes to include (for large graphs)
            
        Returns:
            DOT format string
        """
        lines = ['digraph DeBruijn {']
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box];')
        
        shown_nodes = set()
        edge_count = 0
        
        for kmer, edge in self._edges.items():
            if len(shown_nodes) >= max_nodes:
                break
            
            src_label = ','.join(str(x) for x in edge.source)
            tgt_label = ','.join(str(x) for x in edge.target)
            
            # Escape quotes
            src_id = f'"{src_label}"'
            tgt_id = f'"{tgt_label}"'
            
            mult = f' [{edge.multiplicity}]' if edge.multiplicity > 1 else ''
            label = f'{edge.label}{mult}'
            
            lines.append(f'  {src_id} -> {tgt_id} [label="{label}"];')
            shown_nodes.add(edge.source)
            shown_nodes.add(edge.target)
            edge_count += 1
        
        if edge_count < len(self._edges):
            lines.append(f'  // ... {len(self._edges) - edge_count} more edges')
        
        lines.append('}')
        return '\n'.join(lines)


# =============================================================================
# Factory Functions
# =============================================================================

def build_debruijn_from_sequence(sequence: List[T], k: int = 3) -> DeBruijnGraph[T]:
    """
    Build De Bruijn graph from a token sequence.
    
    Automatically counts k-mer occurrences.
    
    Args:
        sequence: List of tokens
        k: K-mer size
        
    Returns:
        DeBruijnGraph with all k-mers and their counts
        
    Example:
        graph = build_debruijn_from_sequence(['START', 'a', 'b', 'a', 'b', 'a', 'END'], k=3)
    """
    graph = DeBruijnGraph(k=k)
    
    # Extract k-mers and count occurrences
    kmer_counts: Dict[Tuple[T, ...], int] = defaultdict(int)
    
    for i in range(len(sequence) - k + 1):
        kmer = tuple(sequence[i:i + k])
        kmer_counts[kmer] += 1
    
    # Add to graph
    for kmer, count in kmer_counts.items():
        graph.add_kmer(kmer, count)
    
    return graph


def build_debruijn_from_kmers(kmers: List[Tuple[T, ...]], 
                              counts: Optional[Dict[Tuple[T, ...], int]] = None,
                              ) -> DeBruijnGraph[T]:
    """
    Build De Bruijn graph from k-mers directly.
    
    Args:
        kmers: List of k-mer tuples
        counts: Optional multiplicity dict
        
    Returns:
        DeBruijnGraph
    """
    if not kmers:
        raise ValueError("Need at least one k-mer")
    
    k = len(kmers[0])
    graph = DeBruijnGraph(k=k)
    graph.add_kmers(kmers, counts)
    return graph


# =============================================================================
# Integration with HLLSet Disambiguation  
# =============================================================================

def restore_sequence_debruijn(
    trigrams: List[Tuple[str, str, str]],
    trigram_counts: Optional[Dict[Tuple[str, str, str], int]] = None,
    start_marker: str = "START",
    end_marker: str = "END",
    use_eulerian: bool = True,
    randomize: bool = False,
    strip_markers: bool = True,
) -> List[str]:
    """
    Restore token sequence from trigrams using De Bruijn graph.
    
    This is the main interface for HLLSet disambiguation integration.
    
    Args:
        trigrams: List of trigram tuples
        trigram_counts: Optional occurrence counts per trigram
        start_marker: Token marking sequence start
        end_marker: Token marking sequence end
        use_eulerian: If True, use Eulerian path (optimal); else greedy
        randomize: If True, randomize choices for variety
        strip_markers: If True, remove START/END from result
        
    Returns:
        Ordered list of tokens
        
    Example:
        trigrams = [('START', 'a', 'b'), ('a', 'b', 'a'), ('b', 'a', 'END')]
        counts = {('a', 'b', 'a'): 2}  # appears twice
        
        result = restore_sequence_debruijn(trigrams, counts)
        # Returns: ['a', 'b', 'a', 'b', 'a']
    """
    if not trigrams:
        return []
    
    # Build graph
    graph = DeBruijnGraph(k=3)
    
    for trigram in trigrams:
        count = trigram_counts.get(trigram, 1) if trigram_counts else 1
        graph.add_kmer(trigram, count)
    
    # Find path
    if use_eulerian:
        result = graph.find_eulerian_path(
            start_marker=start_marker,
            end_marker=end_marker,
            randomize=randomize,
        )
    else:
        result = graph.find_path_greedy(
            start_marker=start_marker,
            end_marker=end_marker,
            randomize=randomize,
        )
    
    if result is None:
        # Fallback: return unique tokens
        tokens = set()
        for t in trigrams:
            tokens.update(t)
        tokens.discard(start_marker)
        tokens.discard(end_marker)
        return list(tokens)
    
    sequence = result.sequence
    
    # Strip markers if requested
    if strip_markers:
        if sequence and sequence[0] == start_marker:
            sequence = sequence[1:]
        if sequence and sequence[-1] == end_marker:
            sequence = sequence[:-1]
    
    return sequence
