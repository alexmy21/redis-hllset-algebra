//! HLLSet - HyperLogLog with Set Algebra Operations
//!
//! This module provides a native Redis data type for probabilistic set operations.
//! HLLSet combines HyperLogLog cardinality estimation with full set algebra:
//! union, intersection, difference, and symmetric difference.
//!
//! # Key Features
//!
//! - **Content-Addressable**: Keys are SHA-1 hashes of sorted tokens
//! - **Set Algebra**: Full support for ∪, ∩, \, ⊕ operations
//! - **Probabilistic**: O(1) space complexity with ~2% error rate
//! - **Native Redis Type**: Proper RDB persistence and memory reporting
//!
//! # Commands
//!
//! - `HLLSET.CREATE token1 [token2 ...]` - Create HLLSet from tokens
//! - `HLLSET.CREATEHASH hash1 [hash2 ...]` - Create from pre-computed hashes
//! - `HLLSET.CARD key` - Get estimated cardinality
//! - `HLLSET.UNION key1 key2` - Union of two sets (A ∪ B)
//! - `HLLSET.INTER key1 key2` - Intersection (A ∩ B)
//! - `HLLSET.DIFF key1 key2` - Difference (A \ B)
//! - `HLLSET.XOR key1 key2` - Symmetric difference (A ⊕ B)
//! - `HLLSET.SIM key1 key2` - Jaccard similarity
//! - `HLLSET.INFO key` - Get HLLSet metadata
//! - `HLLSET.DUMP key` - Dump register positions
//! - `HLLSET.DEL key` - Delete HLLSet

mod hllset;
mod commands;
mod rdb;
mod bias;

use redis_module::{redis_module, Context, RedisResult, Status};

pub use hllset::HLLSet;
pub use commands::*;
pub use rdb::HLLSET_TYPE;

/// Module initialization
fn init(ctx: &Context, _args: &[redis_module::RedisString]) -> Status {
    ctx.log_notice("HLLSet Algebra module v0.1.0 loaded");
    ctx.log_notice(&format!(
        "Configuration: M={} registers, {} bits precision",
        hllset::M,
        hllset::P
    ));
    Status::Ok
}

// Register the Redis module
redis_module! {
    name: "hllset",
    version: 1,
    allocator: (redis_module::alloc::RedisAlloc, redis_module::alloc::RedisAlloc),
    data_types: [HLLSET_TYPE],
    init: init,
    commands: [
        // Creation
        ["hllset.create", commands::hllset_create, "write fast", 0, 0, 0],
        ["hllset.createhash", commands::hllset_create_hash, "write fast", 0, 0, 0],
        
        // Cardinality
        ["hllset.card", commands::hllset_card, "readonly fast", 1, 1, 1],
        
        // Set operations (create new keys)
        ["hllset.union", commands::hllset_union, "write", 1, 2, 1],
        ["hllset.inter", commands::hllset_inter, "write", 1, 2, 1],
        ["hllset.diff", commands::hllset_diff, "write", 1, 2, 1],
        ["hllset.xor", commands::hllset_xor, "write", 1, 2, 1],
        
        // Similarity
        ["hllset.sim", commands::hllset_similarity, "readonly fast", 1, 2, 1],
        ["hllset.jaccard", commands::hllset_similarity, "readonly fast", 1, 2, 1],
        
        // Info and debug
        ["hllset.info", commands::hllset_info, "readonly fast", 1, 1, 1],
        ["hllset.dump", commands::hllset_dump, "readonly", 1, 1, 1],
        
        // Management
        ["hllset.del", commands::hllset_del, "write fast", 1, 1, 1],
        ["hllset.exists", commands::hllset_exists, "readonly fast", 1, 1, 1],
        
        // Merge (in-place union)
        ["hllset.merge", commands::hllset_merge, "write", 1, -1, 1],
    ],
}
