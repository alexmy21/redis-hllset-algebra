//! Redis command implementations for HLLSET.RING operations
//!
//! Commands:
//! - HLLSET.RING.INIT - Initialize a ring
//! - HLLSET.RING.INGEST - Ingest token and decompose
//! - HLLSET.RING.DECOMPOSE - Decompose existing HLLSet
//! - HLLSET.RING.EXPRESS - Get XOR expression for HLLSet
//! - HLLSET.RING.BASIS - Get current basis SHA1s
//! - HLLSET.RING.RANK - Get current rank
//! - HLLSET.W.COMMIT - Create W lattice commit
//! - HLLSET.W.DIFF - Compute diff between W commits
//! - HLLSET.RECONSTRUCT - Reconstruct HLLSet from XOR of bases

use redis_module::{Context, RedisError, RedisResult, RedisString, RedisValue};
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::hllset::HLLSet;
use crate::ring::{RingState, WCommit, WDiff};
use crate::rdb::HLLSET_TYPE;

// Global cache for ring states (to avoid repeated Redis reads during batch operations)
lazy_static::lazy_static! {
    static ref RING_CACHE: RwLock<HashMap<String, RingState>> = RwLock::new(HashMap::new());
}

/// Key prefixes
const KEY_RING: &str = "hllring:ring:";
const KEY_BASE: &str = "hllring:base:";
const KEY_LUT: &str = "hllring:lut:";
const KEY_META: &str = "hllring:meta:";
const KEY_W: &str = "hllring:W:";

/// Get current timestamp
fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Helper to get string from RedisString
fn to_string(rs: &RedisString) -> String {
    rs.to_string_lossy()
}

/// Get HLLSet from key (readonly)
fn get_hllset(ctx: &Context, key: &RedisString) -> Result<Option<HLLSet>, RedisError> {
    let key_handle = ctx.open_key(key);
    match key_handle.get_value::<HLLSet>(&HLLSET_TYPE)? {
        Some(hll) => Ok(Some(hll.clone())),
        None => Ok(None),
    }
}

/// Load ring state from Redis (or cache)
#[allow(dead_code)]
fn _load_ring(ctx: &Context, ring_id: &str) -> Result<Option<RingState>, RedisError> {
    // Check cache first
    if let Ok(cache) = RING_CACHE.read() {
        if let Some(ring) = cache.get(ring_id) {
            return Ok(Some(ring.clone()));
        }
    }
    
    // Load from Redis
    let key = format!("{}{}", KEY_RING, ring_id);
    let redis_key = ctx.create_string(key.as_bytes());
    let key_handle = ctx.open_key(&redis_key);
    
    if key_handle.key_type() == redis_module::KeyType::Empty {
        return Ok(None);
    }
    
    // Read hash fields
    // Note: This is simplified - in production, use HGETALL
    let _ring_id_key = ctx.create_string(b"ring_id");
    let _p_bits_key = ctx.create_string(b"p_bits");
    let _basis_key = ctx.create_string(b"basis_sha1s");
    let _created_key = ctx.create_string(b"created_at");
    let _updated_key = ctx.create_string(b"updated_at");
    let _matrix_key = ctx.create_string(b"matrix_data");
    let _pivots_key = ctx.create_string(b"pivots");
    
    // For now, construct manually from hash
    // In actual implementation, would use ctx.call("HGETALL", ...)
    
    // Simplified: create new ring if can't parse
    // Real implementation would properly deserialize
    Ok(None)
}

/// Save ring state to Redis (and cache)
#[allow(dead_code)]
fn _save_ring(ctx: &Context, ring: &RingState) -> Result<(), RedisError> {
    let key = format!("{}{}", KEY_RING, ring.ring_id);
    let redis_key = ctx.create_string(key.as_bytes());
    
    // Convert to hash fields
    let hash_data = ring.to_redis_hash();
    
    // Store each field
    let _key_handle = ctx.open_key_writable(&redis_key);
    for (field, value) in hash_data {
        let _field_key = ctx.create_string(field.as_bytes());
        let _value_str = ctx.create_string(value.as_bytes());
        // Would use HSET here
    }
    
    // Update cache
    if let Ok(mut cache) = RING_CACHE.write() {
        cache.insert(ring.ring_id.clone(), ring.clone());
    }
    
    Ok(())
}

/// Store base HLLSet bytes
fn store_base(ctx: &Context, sha1: &str, hll: &HLLSet) -> Result<(), RedisError> {
    let key = format!("{}{}", KEY_BASE, sha1);
    let redis_key = ctx.create_string(key.as_bytes());
    let key_handle = ctx.open_key_writable(&redis_key);
    
    // Store HLLSet using native type
    key_handle.set_value(&HLLSET_TYPE, hll.clone())?;
    Ok(())
}

/// Store derivation in LUT
fn store_derivation(ctx: &Context, sha1: &str, is_base: bool, bases: &[String]) -> Result<(), RedisError> {
    let key = format!("{}{}", KEY_LUT, sha1);
    let redis_key = ctx.create_string(key.as_bytes());
    
    let op = if is_base { "base" } else { "xor" };
    let bases_json = serde_json::to_string(bases).unwrap_or_default();
    
    // Store as hash
    // In real implementation: HSET key op "base/xor" bases "[...]"
    ctx.call("HSET", &[
        &redis_key,
        &ctx.create_string(b"op"),
        &ctx.create_string(op.as_bytes()),
        &ctx.create_string(b"bases"),
        &ctx.create_string(bases_json.as_bytes()),
    ])?;
    
    Ok(())
}

/// Store metadata
fn store_meta(
    ctx: &Context,
    sha1: &str,
    source: Option<&str>,
    cardinality: f64,
    is_base: bool,
    tags: Option<&[String]>,
) -> Result<(), RedisError> {
    let key = format!("{}{}", KEY_META, sha1);
    let redis_key = ctx.create_string(key.as_bytes());
    
    let timestamp = now();
    let source_str = source.unwrap_or("unknown");
    let tags_json = serde_json::to_string(tags.unwrap_or(&[])).unwrap_or_default();
    
    ctx.call("HSET", &[
        &redis_key,
        &ctx.create_string(b"sha1"),
        &ctx.create_string(sha1.as_bytes()),
        &ctx.create_string(b"source"),
        &ctx.create_string(source_str.as_bytes()),
        &ctx.create_string(b"cardinality"),
        &ctx.create_string(cardinality.to_string().as_bytes()),
        &ctx.create_string(b"is_base"),
        &ctx.create_string(if is_base { b"1" } else { b"0" }),
        &ctx.create_string(b"created_at"),
        &ctx.create_string(timestamp.to_string().as_bytes()),
        &ctx.create_string(b"tags"),
        &ctx.create_string(tags_json.as_bytes()),
    ])?;
    
    Ok(())
}

// =============================================================================
// HLLSET.RING.INIT
// =============================================================================

/// HLLSET.RING.INIT <ring_key> [PBITS <p>]
/// Initialize a new ring for decomposition
pub fn ring_init(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 2 {
        return Err(RedisError::WrongArity);
    }
    
    let ring_id = to_string(&args[1]);
    
    // Parse optional PBITS
    let mut p_bits: u8 = 10;
    let mut i = 2;
    while i < args.len() {
        let arg = to_string(&args[i]).to_uppercase();
        if arg == "PBITS" && i + 1 < args.len() {
            p_bits = to_string(&args[i + 1]).parse().unwrap_or(10);
            i += 2;
        } else {
            i += 1;
        }
    }
    
    // Create ring state
    let mut ring = RingState::new(ring_id.clone(), p_bits);
    ring.created_at = now();
    ring.updated_at = now();
    
    // Store in Redis
    let key = format!("{}{}", KEY_RING, ring_id);
    let redis_key = ctx.create_string(key.as_bytes());
    
    let hash_data = ring.to_redis_hash();
    for (field, value) in &hash_data {
        ctx.call("HSET", &[
            &redis_key,
            &ctx.create_string(field.as_bytes()),
            &ctx.create_string(value.as_bytes()),
        ])?;
    }
    
    // Cache
    if let Ok(mut cache) = RING_CACHE.write() {
        cache.insert(ring.ring_id.clone(), ring);
    }
    
    Ok(RedisValue::SimpleString("OK".into()))
}

// =============================================================================
// HLLSET.RING.INGEST
// =============================================================================

/// HLLSET.RING.INGEST <ring_key> <token> [SOURCE <source>] [TAGS <tags>]
/// Create HLLSet from token, decompose, and store
pub fn ring_ingest(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 3 {
        return Err(RedisError::WrongArity);
    }
    
    let ring_id = to_string(&args[1]);
    let token = to_string(&args[2]);
    
    // Parse optional arguments
    let mut source: Option<String> = None;
    let mut tags: Option<Vec<String>> = None;
    
    let mut i = 3;
    while i < args.len() {
        let arg = to_string(&args[i]).to_uppercase();
        if arg == "SOURCE" && i + 1 < args.len() {
            source = Some(to_string(&args[i + 1]));
            i += 2;
        } else if arg == "TAGS" && i + 1 < args.len() {
            let tags_str = to_string(&args[i + 1]);
            tags = Some(tags_str.split(',').map(|s| s.trim().to_string()).collect());
            i += 2;
        } else {
            i += 1;
        }
    }
    
    // Load ring from cache or Redis
    let ring_key = format!("{}{}", KEY_RING, ring_id);
    let redis_ring_key = ctx.create_string(ring_key.as_bytes());
    
    // Check if ring exists
    let exists = ctx.call("EXISTS", &[&redis_ring_key])?;
    if matches!(exists, RedisValue::Integer(0)) {
        return Err(RedisError::Str("ERR Ring not initialized. Use HLLSET.RING.INIT first"));
    }
    
    // Load ring state from cache
    let mut ring = {
        let cache = RING_CACHE.read().map_err(|_| RedisError::Str("ERR Cache lock failed"))?;
        cache.get(&ring_id).cloned()
    };
    
    // If not in cache, load from Redis
    if ring.is_none() {
        // Read p_bits from Redis
        let p_bits_result = ctx.call("HGET", &[
            &redis_ring_key,
            &ctx.create_string(b"p_bits"),
        ])?;
        
        let p_bits: u8 = match p_bits_result {
            RedisValue::BulkString(s) => s.parse().unwrap_or(10),
            _ => 10,
        };
        
        ring = Some(RingState::new(ring_id.clone(), p_bits));
        
        // TODO: Load full matrix state from Redis
        // For now, starting fresh each time (will fix with proper serialization)
    }
    
    let mut ring = ring.ok_or(RedisError::Str("ERR Ring not found"))?;
    
    // Create HLLSet from token
    let hll = HLLSet::from_tokens(&[token.as_str()]);
    let sha1 = HLLSet::content_key(&[&token]);
    let cardinality = hll.cardinality();
    let dense_regs = hll.to_dense();
    
    // Decompose
    let result = ring.decompose(&dense_regs, sha1.clone());
    
    // Store base if new
    if result.is_new_base {
        store_base(ctx, &sha1, &hll)?;
    }
    
    // Store derivation
    store_derivation(ctx, &sha1, result.is_new_base, &result.expression)?;
    
    // Store metadata
    store_meta(
        ctx,
        &sha1,
        source.as_deref(),
        cardinality,
        result.is_new_base,
        tags.as_deref(),
    )?;
    
    // Update ring state
    ring.updated_at = now();
    
    // Save ring back to Redis and cache
    let hash_data = ring.to_redis_hash();
    for (field, value) in &hash_data {
        ctx.call("HSET", &[
            &redis_ring_key,
            &ctx.create_string(field.as_bytes()),
            &ctx.create_string(value.as_bytes()),
        ])?;
    }
    
    if let Ok(mut cache) = RING_CACHE.write() {
        cache.insert(ring_id, ring);
    }
    
    // Return result array: [sha1, is_new_base, num_bases, base1, base2, ...]
    let mut response = vec![
        RedisValue::BulkString(sha1.clone()),
        RedisValue::Integer(if result.is_new_base { 1 } else { 0 }),
        RedisValue::Integer(result.expression.len() as i64),
    ];
    
    for base_sha1 in &result.expression {
        response.push(RedisValue::BulkString(base_sha1.clone()));
    }
    
    Ok(RedisValue::Array(response))
}

// =============================================================================
// HLLSET.RING.DECOMPOSE
// =============================================================================

/// HLLSET.RING.DECOMPOSE <ring_key> <hllset_key> [SOURCE <source>] [TAGS <tags>]
/// Decompose an existing HLLSet
pub fn ring_decompose(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 3 {
        return Err(RedisError::WrongArity);
    }
    
    let ring_id = to_string(&args[1]);
    let hllset_key = &args[2];
    
    // Get HLLSet
    let hll = get_hllset(ctx, hllset_key)?
        .ok_or(RedisError::Str("ERR HLLSet not found"))?;
    
    // Parse optional arguments
    let mut source: Option<String> = None;
    let mut tags: Option<Vec<String>> = None;
    
    let mut i = 3;
    while i < args.len() {
        let arg = to_string(&args[i]).to_uppercase();
        if arg == "SOURCE" && i + 1 < args.len() {
            source = Some(to_string(&args[i + 1]));
            i += 2;
        } else if arg == "TAGS" && i + 1 < args.len() {
            let tags_str = to_string(&args[i + 1]);
            tags = Some(tags_str.split(',').map(|s| s.trim().to_string()).collect());
            i += 2;
        } else {
            i += 1;
        }
    }
    
    // Load ring
    let ring_key = format!("{}{}", KEY_RING, ring_id);
    let redis_ring_key = ctx.create_string(ring_key.as_bytes());
    
    let mut ring = {
        let cache = RING_CACHE.read().map_err(|_| RedisError::Str("ERR Cache lock failed"))?;
        cache.get(&ring_id).cloned()
    }.ok_or(RedisError::Str("ERR Ring not found"))?;
    
    // Compute SHA1 and decompose
    let dense_regs = hll.to_dense();
    let sha1 = format!("hllset:{:x}", md5::compute(format!("{:?}", dense_regs)));
    let cardinality = hll.cardinality();
    
    let result = ring.decompose(&dense_regs, sha1.clone());
    
    // Store
    if result.is_new_base {
        store_base(ctx, &sha1, &hll)?;
    }
    store_derivation(ctx, &sha1, result.is_new_base, &result.expression)?;
    store_meta(ctx, &sha1, source.as_deref(), cardinality, result.is_new_base, tags.as_deref())?;
    
    // Update ring
    ring.updated_at = now();
    let hash_data = ring.to_redis_hash();
    for (field, value) in &hash_data {
        ctx.call("HSET", &[
            &redis_ring_key,
            &ctx.create_string(field.as_bytes()),
            &ctx.create_string(value.as_bytes()),
        ])?;
    }
    
    if let Ok(mut cache) = RING_CACHE.write() {
        cache.insert(ring_id, ring);
    }
    
    // Return result
    let mut response = vec![
        RedisValue::BulkString(sha1.clone()),
        RedisValue::Integer(if result.is_new_base { 1 } else { 0 }),
        RedisValue::Integer(result.expression.len() as i64),
    ];
    for base in &result.expression {
        response.push(RedisValue::BulkString(base.clone()));
    }
    
    Ok(RedisValue::Array(response))
}

// =============================================================================
// HLLSET.RING.BASIS
// =============================================================================

/// HLLSET.RING.BASIS <ring_key>
/// Get current basis SHA1s
pub fn ring_basis(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 2 {
        return Err(RedisError::WrongArity);
    }
    
    let ring_id = to_string(&args[1]);
    
    // Get from cache
    let ring = {
        let cache = RING_CACHE.read().map_err(|_| RedisError::Str("ERR Cache lock failed"))?;
        cache.get(&ring_id).cloned()
    };
    
    if let Some(ring) = ring {
        let bases: Vec<RedisValue> = ring.basis()
            .iter()
            .map(|s| RedisValue::BulkString(s.clone()))
            .collect();
        return Ok(RedisValue::Array(bases));
    }
    
    // Try loading from Redis
    let ring_key = format!("{}{}", KEY_RING, ring_id);
    let redis_ring_key = ctx.create_string(ring_key.as_bytes());
    
    let basis_result = ctx.call("HGET", &[
        &redis_ring_key,
        &ctx.create_string(b"basis_sha1s"),
    ])?;
    
    match basis_result {
        RedisValue::BulkString(s) => {
            let basis: Vec<String> = serde_json::from_str(&s).unwrap_or_default();
            let values: Vec<RedisValue> = basis
                .iter()
                .map(|s| RedisValue::BulkString(s.clone()))
                .collect();
            Ok(RedisValue::Array(values))
        }
        _ => Ok(RedisValue::Array(vec![])),
    }
}

// =============================================================================
// HLLSET.RING.RANK
// =============================================================================

/// HLLSET.RING.RANK <ring_key>
/// Get current rank
pub fn ring_rank(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 2 {
        return Err(RedisError::WrongArity);
    }
    
    let ring_id = to_string(&args[1]);
    
    // Get from cache
    let ring = {
        let cache = RING_CACHE.read().map_err(|_| RedisError::Str("ERR Cache lock failed"))?;
        cache.get(&ring_id).cloned()
    };
    
    if let Some(ring) = ring {
        return Ok(RedisValue::Integer(ring.rank() as i64));
    }
    
    // Try loading from Redis
    let ring_key = format!("{}{}", KEY_RING, ring_id);
    let redis_ring_key = ctx.create_string(ring_key.as_bytes());
    
    let rank_result = ctx.call("HGET", &[
        &redis_ring_key,
        &ctx.create_string(b"rank"),
    ])?;
    
    match rank_result {
        RedisValue::BulkString(s) => {
            let rank: i64 = s.parse().unwrap_or(0);
            Ok(RedisValue::Integer(rank))
        }
        _ => Ok(RedisValue::Integer(0)),
    }
}

// =============================================================================
// HLLSET.W.COMMIT
// =============================================================================

/// HLLSET.W.COMMIT <ring_key> [META <json>]
/// Create W lattice commit
pub fn w_commit(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 2 {
        return Err(RedisError::WrongArity);
    }
    
    let ring_id = to_string(&args[1]);
    
    // Parse optional META
    let mut metadata: Option<String> = None;
    let mut i = 2;
    while i < args.len() {
        let arg = to_string(&args[i]).to_uppercase();
        if arg == "META" && i + 1 < args.len() {
            metadata = Some(to_string(&args[i + 1]));
            i += 2;
        } else {
            i += 1;
        }
    }
    
    // Get ring state
    let ring = {
        let cache = RING_CACHE.read().map_err(|_| RedisError::Str("ERR Cache lock failed"))?;
        cache.get(&ring_id).cloned()
    }.ok_or(RedisError::Str("ERR Ring not found"))?;
    
    // Find next time index
    let pattern = format!("{}{}:*", KEY_W, ring_id);
    let pattern_key = ctx.create_string(pattern.as_bytes());
    
    // Count existing W commits
    let keys_result = ctx.call("KEYS", &[&pattern_key])?;
    let time_index = match keys_result {
        RedisValue::Array(arr) => arr.len() as u64,
        _ => 0,
    };
    
    // Create commit
    let mut commit = WCommit::from_ring(&ring, time_index, metadata);
    commit.timestamp = now();
    
    // Store
    let w_key = format!("{}{}:{}", KEY_W, ring_id, time_index);
    let redis_w_key = ctx.create_string(w_key.as_bytes());
    
    let hash_data = commit.to_redis_hash();
    for (field, value) in &hash_data {
        ctx.call("HSET", &[
            &redis_w_key,
            &ctx.create_string(field.as_bytes()),
            &ctx.create_string(value.as_bytes()),
        ])?;
    }
    
    Ok(RedisValue::Integer(time_index as i64))
}

// =============================================================================
// HLLSET.W.DIFF
// =============================================================================

/// HLLSET.W.DIFF <ring_key> <t1> <t2>
/// Compute diff between W commits
pub fn w_diff(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 4 {
        return Err(RedisError::WrongArity);
    }
    
    let ring_id = to_string(&args[1]);
    let t1: u64 = to_string(&args[2]).parse().map_err(|_| RedisError::Str("ERR Invalid t1"))?;
    let t2: u64 = to_string(&args[3]).parse().map_err(|_| RedisError::Str("ERR Invalid t2"))?;
    
    // Load both commits
    let w1_key = format!("{}{}:{}", KEY_W, ring_id, t1);
    let w2_key = format!("{}{}:{}", KEY_W, ring_id, t2);
    
    let redis_w1_key = ctx.create_string(w1_key.as_bytes());
    let redis_w2_key = ctx.create_string(w2_key.as_bytes());
    
    // Get basis_sha1s from each
    let w1_basis = ctx.call("HGET", &[&redis_w1_key, &ctx.create_string(b"basis_sha1s")])?;
    let w2_basis = ctx.call("HGET", &[&redis_w2_key, &ctx.create_string(b"basis_sha1s")])?;
    
    let basis1: Vec<String> = match w1_basis {
        RedisValue::BulkString(s) => serde_json::from_str(&s).unwrap_or_default(),
        _ => vec![],
    };
    
    let basis2: Vec<String> = match w2_basis {
        RedisValue::BulkString(s) => serde_json::from_str(&s).unwrap_or_default(),
        _ => vec![],
    };
    
    // Create dummy commits for diff computation
    let commit1 = WCommit {
        time_index: t1,
        ring_id: ring_id.clone(),
        basis_sha1s: basis1,
        rank: 0,
        timestamp: 0.0,
        metadata: None,
    };
    
    let commit2 = WCommit {
        time_index: t2,
        ring_id,
        basis_sha1s: basis2,
        rank: 0,
        timestamp: 0.0,
        metadata: None,
    };
    
    let diff = WDiff::compute(&commit1, &commit2);
    
    // Return: [added_count, removed_count, shared_count, delta_rank, added..., removed..., shared...]
    let mut response = vec![
        RedisValue::Integer(diff.added_bases.len() as i64),
        RedisValue::Integer(diff.removed_bases.len() as i64),
        RedisValue::Integer(diff.shared_bases.len() as i64),
        RedisValue::Integer(diff.delta_rank),
    ];
    
    for sha1 in &diff.added_bases {
        response.push(RedisValue::BulkString(sha1.clone()));
    }
    for sha1 in &diff.removed_bases {
        response.push(RedisValue::BulkString(sha1.clone()));
    }
    for sha1 in &diff.shared_bases {
        response.push(RedisValue::BulkString(sha1.clone()));
    }
    
    Ok(RedisValue::Array(response))
}

// =============================================================================
// HLLSET.RECONSTRUCT
// =============================================================================

/// HLLSET.RECONSTRUCT <sha1>
/// Reconstruct HLLSet from XOR of bases
pub fn reconstruct(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 2 {
        return Err(RedisError::WrongArity);
    }
    
    let sha1 = to_string(&args[1]);
    
    // Get derivation
    let lut_key = format!("{}{}", KEY_LUT, sha1);
    let redis_lut_key = ctx.create_string(lut_key.as_bytes());
    
    let op_result = ctx.call("HGET", &[&redis_lut_key, &ctx.create_string(b"op")])?;
    let bases_result = ctx.call("HGET", &[&redis_lut_key, &ctx.create_string(b"bases")])?;
    
    let op = match op_result {
        RedisValue::BulkString(s) => s,
        _ => return Err(RedisError::Str("ERR Derivation not found")),
    };
    
    if op == "base" {
        // It's a base - return directly
        let base_key = format!("{}{}", KEY_BASE, sha1);
        let redis_base_key = ctx.create_string(base_key.as_bytes());
        
        let key_handle = ctx.open_key(&redis_base_key);
        match key_handle.get_value::<HLLSet>(&HLLSET_TYPE)? {
            Some(_hll) => {
                // Return the SHA1 key (caller can retrieve using HLLSET.* commands)
                Ok(RedisValue::BulkString(sha1))
            }
            None => Err(RedisError::Str("ERR Base HLLSet not found")),
        }
    } else {
        // XOR of bases
        let bases: Vec<String> = match bases_result {
            RedisValue::BulkString(s) => serde_json::from_str(&s).unwrap_or_default(),
            _ => return Err(RedisError::Str("ERR Invalid bases")),
        };
        
        if bases.is_empty() {
            return Err(RedisError::Str("ERR No bases to reconstruct from"));
        }
        
        // Load and XOR all bases using symmetric_difference
        let mut result: Option<HLLSet> = None;
        
        for base_sha1 in &bases {
            let base_key = format!("{}{}", KEY_BASE, base_sha1);
            let redis_base_key = ctx.create_string(base_key.as_bytes());
            
            let key_handle = ctx.open_key(&redis_base_key);
            match key_handle.get_value::<HLLSet>(&HLLSET_TYPE)? {
                Some(hll) => {
                    result = Some(match result {
                        Some(r) => r.symmetric_difference(&hll),
                        None => hll.clone(),
                    });
                }
                None => return Err(RedisError::Str("ERR Base not found during reconstruction")),
            }
        }
        
        // Store reconstructed HLLSet temporarily and return key
        if let Some(hll) = result {
            let temp_key = format!("hllring:temp:{}", sha1);
            let redis_temp_key = ctx.create_string(temp_key.as_bytes());
            
            let key_handle = ctx.open_key_writable(&redis_temp_key);
            key_handle.set_value(&HLLSET_TYPE, hll)?;
            
            // Set TTL (60 seconds)
            ctx.call("EXPIRE", &[
                &redis_temp_key,
                &ctx.create_string(b"60"),
            ])?;
            
            Ok(RedisValue::BulkString(temp_key))
        } else {
            Err(RedisError::Str("ERR Reconstruction failed"))
        }
    }
}
