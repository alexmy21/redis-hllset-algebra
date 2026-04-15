//! Redis command implementations for HLLSet

use redis_module::{Context, RedisError, RedisResult, RedisString, RedisValue, NotifyEvent};
use crate::hllset::HLLSet;
use crate::rdb::HLLSET_TYPE;

/// Helper to get HLLSet from a key (read-only)
fn get_hllset_readonly(ctx: &Context, key: &RedisString) -> Result<Option<HLLSet>, RedisError> {
    let key_handle = ctx.open_key(key);
    
    match key_handle.get_value::<HLLSet>(&HLLSET_TYPE)? {
        Some(hll) => Ok(Some(hll.clone())),
        None => Ok(None),
    }
}

/// Helper to check if key exists
fn key_exists(ctx: &Context, key: &RedisString) -> bool {
    let key_handle = ctx.open_key(key);
    key_handle.key_type() != redis_module::KeyType::Empty
}

/// HLLSET.CREATE token1 [token2 ...]
/// Creates an HLLSet from tokens and returns the content-addressable key
pub fn hllset_create(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 2 {
        return Err(RedisError::WrongArity);
    }

    // Collect tokens (skip command name)
    let tokens: Vec<String> = args[1..]
        .iter()
        .map(|s| s.to_string_lossy())
        .collect();

    // Generate content-addressable key
    let key_str = HLLSet::content_key(&tokens);
    
    // Check if key already exists
    let key = ctx.create_string(key_str.as_bytes());
    if key_exists(ctx, &key) {
        // Key exists - that's fine for content-addressable storage
        return Ok(RedisValue::BulkString(key_str));
    }
    
    // Create HLLSet from tokens
    let hllset = HLLSet::from_tokens(&tokens);
    
    // Store in Redis
    let key_handle = ctx.open_key_writable(&key);
    key_handle.set_value(&HLLSET_TYPE, hllset)?;
    
    Ok(RedisValue::BulkString(key_str))
}

/// HLLSET.CREATEHASH hash1 [hash2 ...]
/// Creates an HLLSet from pre-computed 64-bit hashes
pub fn hllset_create_hash(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 2 {
        return Err(RedisError::WrongArity);
    }

    // Parse hashes
    let hashes: Vec<u64> = args[1..]
        .iter()
        .filter_map(|s| s.to_string_lossy().parse::<u64>().ok())
        .collect();

    if hashes.is_empty() {
        return Err(RedisError::Str("ERR No valid hashes provided"));
    }

    // Generate key from hash values (sorted)
    let mut sorted_hashes = hashes.clone();
    sorted_hashes.sort();
    let hash_strings: Vec<String> = sorted_hashes.iter().map(|h| h.to_string()).collect();
    let key_str = HLLSet::content_key(&hash_strings);
    
    // Check if key already exists
    let key = ctx.create_string(key_str.as_bytes());
    if key_exists(ctx, &key) {
        return Ok(RedisValue::BulkString(key_str));
    }
    
    // Create HLLSet from hashes
    let hllset = HLLSet::from_hashes(hashes);
    
    // Store in Redis
    let key_handle = ctx.open_key_writable(&key);
    key_handle.set_value(&HLLSET_TYPE, hllset)?;
    
    Ok(RedisValue::BulkString(key_str))
}

/// HLLSET.CARD key
/// Returns the estimated cardinality of an HLLSet
pub fn hllset_card(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 2 {
        return Err(RedisError::WrongArity);
    }

    let key = &args[1];
    
    match get_hllset_readonly(ctx, key)? {
        Some(hll) => {
            let card = hll.cardinality();
            Ok(RedisValue::Float(card))
        }
        None => Ok(RedisValue::Float(0.0)),
    }
}

/// HLLSET.UNION key1 key2
/// Returns the union of two HLLSets as a new content-addressable key
pub fn hllset_union(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 3 {
        return Err(RedisError::WrongArity);
    }

    let key1 = &args[1];
    let key2 = &args[2];

    // Get both HLLSets
    let hll1 = get_hllset_readonly(ctx, key1)?.unwrap_or_else(HLLSet::new);
    let hll2 = get_hllset_readonly(ctx, key2)?.unwrap_or_else(HLLSet::new);

    // Compute union
    let result = hll1.union(&hll2);
    
    // Generate key from operation
    let k1_short = extract_key_hash(&key1.to_string_lossy());
    let k2_short = extract_key_hash(&key2.to_string_lossy());
    let key_str = format!("hllset:union:{}:{}", k1_short, k2_short);
    
    // Store result
    let result_key = ctx.create_string(key_str.as_bytes());
    let key_handle = ctx.open_key_writable(&result_key);
    key_handle.set_value(&HLLSET_TYPE, result)?;
    
    Ok(RedisValue::BulkString(key_str))
}

/// HLLSET.INTER key1 key2
/// Returns the intersection of two HLLSets
pub fn hllset_inter(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 3 {
        return Err(RedisError::WrongArity);
    }

    let key1 = &args[1];
    let key2 = &args[2];

    let hll1 = get_hllset_readonly(ctx, key1)?.unwrap_or_else(HLLSet::new);
    let hll2 = get_hllset_readonly(ctx, key2)?.unwrap_or_else(HLLSet::new);

    let result = hll1.intersection(&hll2);
    
    let k1_short = extract_key_hash(&key1.to_string_lossy());
    let k2_short = extract_key_hash(&key2.to_string_lossy());
    let key_str = format!("hllset:inter:{}:{}", k1_short, k2_short);
    
    let result_key = ctx.create_string(key_str.as_bytes());
    let key_handle = ctx.open_key_writable(&result_key);
    key_handle.set_value(&HLLSET_TYPE, result)?;
    
    Ok(RedisValue::BulkString(key_str))
}

/// HLLSET.DIFF key1 key2
/// Returns the difference A \ B
pub fn hllset_diff(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 3 {
        return Err(RedisError::WrongArity);
    }

    let key1 = &args[1];
    let key2 = &args[2];

    let hll1 = get_hllset_readonly(ctx, key1)?.unwrap_or_else(HLLSet::new);
    let hll2 = get_hllset_readonly(ctx, key2)?.unwrap_or_else(HLLSet::new);

    let result = hll1.difference(&hll2);
    
    let k1_short = extract_key_hash(&key1.to_string_lossy());
    let k2_short = extract_key_hash(&key2.to_string_lossy());
    let key_str = format!("hllset:diff:{}:{}", k1_short, k2_short);
    
    let result_key = ctx.create_string(key_str.as_bytes());
    let key_handle = ctx.open_key_writable(&result_key);
    key_handle.set_value(&HLLSET_TYPE, result)?;
    
    Ok(RedisValue::BulkString(key_str))
}

/// HLLSET.XOR key1 key2
/// Returns the symmetric difference A ⊕ B
pub fn hllset_xor(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 3 {
        return Err(RedisError::WrongArity);
    }

    let key1 = &args[1];
    let key2 = &args[2];

    let hll1 = get_hllset_readonly(ctx, key1)?.unwrap_or_else(HLLSet::new);
    let hll2 = get_hllset_readonly(ctx, key2)?.unwrap_or_else(HLLSet::new);

    let result = hll1.symmetric_difference(&hll2);
    
    let k1_short = extract_key_hash(&key1.to_string_lossy());
    let k2_short = extract_key_hash(&key2.to_string_lossy());
    let key_str = format!("hllset:xor:{}:{}", k1_short, k2_short);
    
    let result_key = ctx.create_string(key_str.as_bytes());
    let key_handle = ctx.open_key_writable(&result_key);
    key_handle.set_value(&HLLSET_TYPE, result)?;
    
    Ok(RedisValue::BulkString(key_str))
}

/// HLLSET.SIM key1 key2
/// Returns the Jaccard similarity between two HLLSets
pub fn hllset_similarity(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 3 {
        return Err(RedisError::WrongArity);
    }

    let key1 = &args[1];
    let key2 = &args[2];

    let hll1 = get_hllset_readonly(ctx, key1)?.unwrap_or_else(HLLSet::new);
    let hll2 = get_hllset_readonly(ctx, key2)?.unwrap_or_else(HLLSet::new);

    let similarity = hll1.jaccard_similarity(&hll2);
    
    Ok(RedisValue::Float(similarity))
}

/// HLLSET.INFO key
/// Returns metadata about an HLLSet
pub fn hllset_info(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 2 {
        return Err(RedisError::WrongArity);
    }

    let key = &args[1];
    
    match get_hllset_readonly(ctx, key)? {
        Some(hll) => {
            let card = hll.cardinality();
            let non_zero = hll.non_zero_registers();
            let memory = hll.memory_usage();
            
            Ok(RedisValue::Array(vec![
                RedisValue::BulkString("key".to_string()),
                RedisValue::BulkString(key.to_string_lossy()),
                RedisValue::BulkString("cardinality".to_string()),
                RedisValue::Float(card),
                RedisValue::BulkString("registers".to_string()),
                RedisValue::Integer(crate::hllset::M as i64),
                RedisValue::BulkString("non_zero_registers".to_string()),
                RedisValue::Integer(non_zero as i64),
                RedisValue::BulkString("precision_bits".to_string()),
                RedisValue::Integer(crate::hllset::P as i64),
                RedisValue::BulkString("memory_bytes".to_string()),
                RedisValue::Integer(memory as i64),
            ]))
        }
        None => Err(RedisError::Str("ERR Key does not exist")),
    }
}

/// HLLSET.DUMP key
/// Returns all register positions for debugging
pub fn hllset_dump(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 2 {
        return Err(RedisError::WrongArity);
    }

    let key = &args[1];
    
    match get_hllset_readonly(ctx, key)? {
        Some(hll) => {
            let positions = hll.dump_positions();
            let result: Vec<RedisValue> = positions
                .iter()
                .map(|(bucket, value)| {
                    RedisValue::Array(vec![
                        RedisValue::Integer(*bucket as i64),
                        RedisValue::Integer(*value as i64),
                    ])
                })
                .collect();
            
            Ok(RedisValue::Array(result))
        }
        None => Err(RedisError::Str("ERR Key does not exist")),
    }
}

/// HLLSET.DEL key
/// Deletes an HLLSet
pub fn hllset_del(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 2 {
        return Err(RedisError::WrongArity);
    }

    let key = &args[1];
    
    if !key_exists(ctx, key) {
        return Ok(RedisValue::Integer(0));
    }
    
    let key_handle = ctx.open_key_writable(key);
    key_handle.delete()?;
    Ok(RedisValue::Integer(1))
}

/// HLLSET.EXISTS key
/// Check if an HLLSet exists
pub fn hllset_exists(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 2 {
        return Err(RedisError::WrongArity);
    }

    let key = &args[1];
    
    match get_hllset_readonly(ctx, key)? {
        Some(_) => Ok(RedisValue::Integer(1)),
        None => Ok(RedisValue::Integer(0)),
    }
}

/// HLLSET.MERGE destkey key1 [key2 ...]
/// Merges multiple HLLSets into destkey (in-place union)
pub fn hllset_merge(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 3 {
        return Err(RedisError::WrongArity);
    }

    let dest_key = &args[1];
    
    // Start with destination or empty set
    let mut result = get_hllset_readonly(ctx, dest_key)?.unwrap_or_else(HLLSet::new);
    
    // Merge all source keys
    for i in 2..args.len() {
        let src_key = &args[i];
        if let Some(src) = get_hllset_readonly(ctx, src_key)? {
            result.merge(&src);
        }
    }
    
    // Store result
    let key_handle = ctx.open_key_writable(dest_key);
    key_handle.set_value(&HLLSET_TYPE, result)?;
    
    Ok(RedisValue::SimpleStringStatic("OK"))
}

/// Extract short hash from key for derived key names
fn extract_key_hash(key: &str) -> String {
    if let Some(hash_part) = key.strip_prefix("hllset:") {
        hash_part.chars().take(8).collect()
    } else {
        key.chars().take(8).collect()
    }
}

// === Tensor / Active Positions Commands ===

/// HLLSET.POSITIONS key
/// Returns all active (reg, zeros) positions as flat array [reg1, zeros1, reg2, zeros2, ...]
/// This is the primary command for disambiguation lookup
pub fn hllset_positions(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 2 {
        return Err(RedisError::WrongArity);
    }

    let key = &args[1];
    
    match get_hllset_readonly(ctx, key)? {
        Some(hll) => {
            let positions = hll.active_positions();
            let result: Vec<RedisValue> = positions
                .iter()
                .flat_map(|(reg, zeros)| vec![
                    RedisValue::Integer(*reg as i64),
                    RedisValue::Integer(*zeros as i64),
                ])
                .collect();
            Ok(RedisValue::Array(result))
        }
        None => Ok(RedisValue::Array(vec![])),
    }
}

/// HLLSET.POPCOUNT key
/// Returns total number of set bits (popcount)
pub fn hllset_popcount(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 2 {
        return Err(RedisError::WrongArity);
    }

    let key = &args[1];
    
    match get_hllset_readonly(ctx, key)? {
        Some(hll) => Ok(RedisValue::Integer(hll.popcount() as i64)),
        None => Ok(RedisValue::Integer(0)),
    }
}

/// HLLSET.BITCOUNTS key
/// Returns c_s counts (number of registers with bit s set) for each bit position
/// Used for Horvitz-Thompson cardinality calculation
pub fn hllset_bitcounts(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 2 {
        return Err(RedisError::WrongArity);
    }

    let key = &args[1];
    
    match get_hllset_readonly(ctx, key)? {
        Some(hll) => {
            let counts = hll.bit_counts();
            let result: Vec<RedisValue> = counts
                .iter()
                .map(|&c| RedisValue::Integer(c as i64))
                .collect();
            Ok(RedisValue::Array(result))
        }
        None => Ok(RedisValue::Array(vec![RedisValue::Integer(0); 32])),
    }
}

/// HLLSET.REGISTER key reg
/// Returns the bitmap value for a specific register
pub fn hllset_register(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 3 {
        return Err(RedisError::WrongArity);
    }

    let key = &args[1];
    let reg: usize = args[2].to_string_lossy().parse()
        .map_err(|_| RedisError::Str("ERR invalid register index"))?;
    
    if reg >= 1024 {
        return Err(RedisError::Str("ERR register index out of range (0-1023)"));
    }
    
    match get_hllset_readonly(ctx, key)? {
        Some(hll) => {
            let bitmap = hll.get_register_bitmap(reg);
            Ok(RedisValue::Integer(bitmap as i64))
        }
        None => Ok(RedisValue::Integer(0)),
    }
}

/// HLLSET.HASBIT key reg zeros
/// Check if bit at (reg, zeros) position is set
pub fn hllset_hasbit(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 4 {
        return Err(RedisError::WrongArity);
    }

    let key = &args[1];
    let reg: usize = args[2].to_string_lossy().parse()
        .map_err(|_| RedisError::Str("ERR invalid register index"))?;
    let zeros: u32 = args[3].to_string_lossy().parse()
        .map_err(|_| RedisError::Str("ERR invalid zeros count"))?;
    
    if reg >= 1024 || zeros >= 32 {
        return Err(RedisError::Str("ERR position out of range"));
    }
    
    match get_hllset_readonly(ctx, key)? {
        Some(hll) => {
            let bitmap = hll.get_register_bitmap(reg);
            let has_bit = (bitmap >> zeros) & 1 == 1;
            Ok(RedisValue::Integer(has_bit as i64))
        }
        None => Ok(RedisValue::Integer(0)),
    }
}
