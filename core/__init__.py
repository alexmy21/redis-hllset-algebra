"""
HLLSet Algebra — Core package.

Layer stack:
    L0  bitvector_ring     Boolean ring (Z/2Z)^N — XOR, AND, Gaussian elimination
    L0  bitvector_core     Cython acceleration of L0
    L1  hll_tensor         2D tensor view (2^p × 32) — inscription, active positions
    L2  hllset             Immutable anti-set — from_batch, union/intersect/diff/xor
    L2r hllset_redis       Redis-backed HLLSet — distributed, persistent
    L3  disambiguation     Triangulation tensor — token recovery from fingerprints
    L4  bss                Bell State Similarity — directed (τ,ρ) metric, morphisms
    L4  noether            Noether steering law — conservation monitoring, flux
    L4  global_registry    Global n-gram HLLSets — universe sets G₁, G₂, G₃
    L4  hll_lattice        Temporal W lattice — content-addressed nodes, join/meet
    L8  bayesian           Bayesian interpretation — priors, posteriors, KL divergence
    L9  bayesian_network   Bayesian Network on HLLSets — d-separation, Markov blanket, belief propagation
    L10 markov_hll         Markov constructs on HLLSets — MC, HMM, MRF, PageRank, causal do-calculus
"""

# L2 — Core anti-set
from .hllset import (
    HLLSet,
    HashConfig,
    HashType,
    DEFAULT_HASH_CONFIG,
    REGISTER_DTYPE,
    compute_sha1,
    C_BACKEND_AVAILABLE,
)

# L2r — Redis-backed HLLSet (optional)
try:
    from .hllset_redis import (
        HLLSetRedis,
        RedisClientManager,
        load_functions as load_redis_functions,
        check_redis_modules,
    )
    REDIS_BACKEND_AVAILABLE = True
except ImportError:
    REDIS_BACKEND_AVAILABLE = False

# L2r — TokenLUT Redis (RediSearch-backed)
try:
    from .tokenlut_redis import (
        TokenLUTRedis,
        TokenEntry,
    )
    from .tokenlut_stream import (
        TokenLUTStream,
        StreamConfig,
        IngestStats,
        HashMode,
        create_stream_lut,
        ingest_file,
    )
    from .tokenlut_session import (
        TokenLUTSession,
        SessionProducer,
        SessionConfig,
        CheckpointResult,
        CommitResult,
        SessionStats,
        create_session,
        quick_ingest,
    )
    TOKENLUT_AVAILABLE = True
except ImportError:
    TOKENLUT_AVAILABLE = False

# L0 — Ring algebra
from .bitvector_ring import BitVector, BitVectorRing

# L1 — Tensor view
from .hll_tensor import HLLTensor, TokenLUT, TokenEntry, TensorRingAdapter

# L3 — Disambiguation
from .disambiguation import (
    DisambiguationEngine, 
    DisambiguationResult,
    ParallelDisambiguator,
    RegisterDisambiguationResult,
    START_TOKEN,
    END_TOKEN,
    NUM_LAYERS,
)

# L3 — De Bruijn Graph (sequence reconstruction)
from .debruijn import (
    DeBruijnGraph,
    Edge,
    PathResult,
    build_debruijn_from_sequence,
    build_debruijn_from_kmers,
    restore_sequence_debruijn,
)

# L4 — BSS
from .bss import (
    BSSPair,
    MorphismResult,
    bss,
    bss_symmetric,
    test_morphism,
    bss_matrix,
    morphism_graph,
    bss_from_registers,
    bss_summary,
)

# L4 — Noether steering
from .noether import (
    NoetherEvolution,
    SteeringPhase,
    FluxRecord,
    SteeringDiagnostics,
    compute_flux,
    apply_transition,
    is_balanced,
)

# L4 — Global registry
from .global_registry import (
    GlobalNGramRegistry,
    RegistrySnapshot,
)

# L4 — Lattice
from .hll_lattice import (
    LatticeNode,
    HLLLattice,
    InMemoryStorage,
)

# L4 — Lattice Reconstruction (De Bruijn analogy for BSS morphisms)
from .lattice_reconstruction import (
    LatticeEdge,
    LatticeLevel,
    ReconstructedLattice,
    BSSMorphismGraph,
    LatticeReconstructor,
    reconstruct_lattice,
    lattice_to_dot,
    reconstruct_from_ring,
)

# L4 — HLLSet De Bruijn (two-level: token + HLLSet evolution)
from .hllset_debruijn import (
    DRNTriple,
    FullDRNTriple,
    HLLSetEdge,
    HLLSetDeBruijnGraph,
    decompose_transition,
    full_decompose_transition,
    verify_reconstruction,
    build_hllset_debruijn,
    find_evolution_path,
    recover_tokens_from_drn,
    EvolutionPhase,
    classify_transition,
    EvolutionSummary,
    analyze_evolution,
)

# L4 — HLLSet Dynamics (monitoring, planning, steering)
from .hllset_dynamics import (
    SystemObservation,
    SystemMonitor,
    TransitionPlan,
    PathPlan,
    plan_transition,
    find_path,
    SteeringMode,
    SteeringAction,
    SystemController,
    HLLSetDynamicSystem,
    # Bernoulli / Symbolic Dynamics
    BitVectorState,
    ShiftTransition,
    BernoulliAnalyzer,
    UnifiedSystemController,
)

# L5 — Transaction (IICA protocol)
from .ring_transaction import (
    RingTransaction,
    TransactionPhase,
    TransactionResult,
    SearchResult,
    RingDelta,
    IngestRecord,
    begin_transaction,
)

# L6 — HLLSet Store (Base HLLSets + Derivation LUT)
from .hllset_store import (
    HLLSetID,
    Operation,
    Derivation,
    StorageBackend,
    InMemoryBackend,
    HLLSetLUT,
    HLLSetStore,
    EphemeralLattice,
)

# L7 — Evolution (State De Bruijn Graph)
from .evolution import (
    StateCommit,
    StateEdge,
    Branch,
    BranchStatus,
    EvolutionGraph,
    create_evolution_tracker,
)

# L8 — Bayesian Interpretation
from .bayesian import (
    BayesianResult,
    BayesTheoremResult,
    InterpretationComparison,
    TemporalBayesRecord,
    prior,
    conditional,
    joint,
    bayes_theorem,
    surprise,
    entropy_of_partition,
    kl_divergence,
    bayesian_surprise_temporal,
    temporal_posterior,
    temporal_trajectory,
    interpretation_divergence,
    edge_probability,
    path_log_likelihood,
)

# L9 — Bayesian Network (Love-Hate Triangle: BSS ↔ BN ↔ Ring)
from .bayesian_network import (
    CPTEntry,
    IndependenceResult,
    MarkovBlanket,
    IsomorphismWitness,
    BeliefState,
    HLLBayesNet,
    hllset_mutual_information,
    conditional_mutual_information,
    ring_to_bn_functor,
)

# L10 — Markov Constructs (MC, HMM, MRF, Causal)
from .markov_hll import (
    StationaryResult,
    PageRankResult,
    HittingTimeResult,
    CommunicatingClass,
    MixingResult,
    RandomWalkTrace,
    EntropyRateResult,
    ForwardResult,
    ViterbiResult,
    HLLMarkovChain,
    HLLHiddenMarkov,
    MarkovRandomField,
    CausalHLL,
    hllset_pagerank,
    markov_from_lattice,
    information_flow_rate,
)

# L11 — HLLSet Transformer (Complement-based temporal attention)
from .hllset_transformer import (
    TransformerPhase,
    ConvergenceReason,
    AttentionRecord,
    HLLSetEdge,
    HLLSetGraph,
    TransformerResult,
    DocumentNode,
    DocumentEdge,
    DocumentGraph,
    HLLSetTransformer,
    transformer_forward,
)

__all__ = [
    # L2
    'HLLSet', 'HashConfig', 'HashType', 'DEFAULT_HASH_CONFIG',
    'REGISTER_DTYPE', 'compute_sha1', 'C_BACKEND_AVAILABLE',
    # L2r — Redis backend
    'HLLSetRedis', 'RedisClientManager', 'load_redis_functions',
    'check_redis_modules', 'REDIS_BACKEND_AVAILABLE',
    # L0
    'BitVector', 'BitVectorRing',
    # L1
    'HLLTensor', 'TokenLUT', 'TokenEntry', 'TensorRingAdapter',
    # L3
    'DisambiguationEngine', 'DisambiguationResult',
    'ParallelDisambiguator', 'RegisterDisambiguationResult',
    'START_TOKEN', 'END_TOKEN', 'NUM_LAYERS',
    # L3 — De Bruijn
    'DeBruijnGraph', 'Edge', 'PathResult',
    'build_debruijn_from_sequence', 'build_debruijn_from_kmers',
    'restore_sequence_debruijn',
    # L4 — BSS
    'BSSPair', 'MorphismResult', 'bss', 'bss_symmetric',
    'test_morphism', 'bss_matrix', 'morphism_graph',
    'bss_from_registers', 'bss_summary',
    # L4 — Noether
    'NoetherEvolution', 'SteeringPhase', 'FluxRecord',
    'SteeringDiagnostics', 'compute_flux', 'apply_transition', 'is_balanced',
    # L4 — Registry
    'GlobalNGramRegistry', 'RegistrySnapshot',
    # L4 — Lattice
    'LatticeNode', 'HLLLattice', 'InMemoryStorage',
    # L4 — Lattice Reconstruction
    'LatticeEdge', 'LatticeLevel', 'ReconstructedLattice',
    'BSSMorphismGraph', 'LatticeReconstructor',
    'reconstruct_lattice', 'lattice_to_dot', 'reconstruct_from_ring',
    # L4 — HLLSet De Bruijn
    'DRNTriple', 'FullDRNTriple', 'HLLSetEdge', 'HLLSetDeBruijnGraph',
    'decompose_transition', 'full_decompose_transition', 'verify_reconstruction',
    'build_hllset_debruijn', 'find_evolution_path', 'recover_tokens_from_drn',
    'EvolutionPhase', 'classify_transition', 'EvolutionSummary', 'analyze_evolution',
    # L4 — HLLSet Dynamics
    'SystemObservation', 'SystemMonitor',
    'TransitionPlan', 'PathPlan', 'plan_transition', 'find_path',
    'SteeringMode', 'SteeringAction', 'SystemController',
    'HLLSetDynamicSystem',
    # L4 — Bernoulli / Symbolic Dynamics
    'BitVectorState', 'ShiftTransition', 'BernoulliAnalyzer', 'UnifiedSystemController',
    # L5 — Transaction
    'RingTransaction', 'TransactionPhase', 'TransactionResult',
    'SearchResult', 'RingDelta', 'IngestRecord', 'begin_transaction',
    # L6 — HLLSet Store
    'HLLSetID', 'Operation', 'Derivation',
    'StorageBackend', 'InMemoryBackend', 'HLLSetLUT', 'HLLSetStore', 'EphemeralLattice',
    # L7 — Evolution
    'StateCommit', 'StateEdge', 'Branch', 'BranchStatus',
    'EvolutionGraph', 'create_evolution_tracker',
    # L8 — Bayesian Interpretation
    'BayesianResult', 'BayesTheoremResult', 'InterpretationComparison',
    'TemporalBayesRecord', 'prior', 'conditional', 'joint', 'bayes_theorem',
    'surprise', 'entropy_of_partition', 'kl_divergence',
    'bayesian_surprise_temporal', 'temporal_posterior', 'temporal_trajectory',
    'interpretation_divergence', 'edge_probability', 'path_log_likelihood',
    # L9 — Bayesian Network (Love-Hate Triangle)
    'CPTEntry', 'IndependenceResult', 'MarkovBlanket',
    'IsomorphismWitness', 'BeliefState', 'HLLBayesNet',
    'hllset_mutual_information', 'conditional_mutual_information',
    'ring_to_bn_functor',
    # L10 — Markov Constructs (MC, HMM, MRF, Causal)
    'StationaryResult', 'PageRankResult', 'HittingTimeResult',
    'CommunicatingClass', 'MixingResult', 'RandomWalkTrace',
    'EntropyRateResult', 'ForwardResult', 'ViterbiResult',
    'HLLMarkovChain', 'HLLHiddenMarkov', 'MarkovRandomField',
    'CausalHLL', 'hllset_pagerank', 'markov_from_lattice',
    'information_flow_rate',
    # L11 — HLLSet Transformer
    'TransformerPhase', 'ConvergenceReason', 'AttentionRecord',
    'HLLSetEdge', 'HLLSetGraph', 'TransformerResult',
    'DocumentNode', 'DocumentEdge', 'DocumentGraph',
    'HLLSetTransformer', 'transformer_forward',
]
