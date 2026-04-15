//! Disambiguation commands for HLLSet
//!
//! These commands enable zero-copy disambiguation of HLLSet positions
//! against TokenLUT entries stored as Redis hashes.
//!
//! Key design principle: Stream results immediately to release Redis ASAP.
//! The filtering/triangulation happens client-side or in subsequent consumers.
//!
//! Commands:
//! - HLLSET.CANDIDATES - Stream LUT entries matching HLLSet positions
//! - HLLSET.FILTER - Filter candidates by layer/criteria
//! - HLLSET.TRIANGULATE - Validate bigrams against unigrams

use redis_module::{Context, RedisError, RedisResult, RedisString, RedisValue};
use crate::hllset::HLLSet;
use crate::rdb::HLLSET_TYPE;
use std::collections::HashSet;

/// Helper to get HLLSet from a key (read-only)
fn get_hllset_readonly(ctx: &Context, key: &RedisString) -> Result<Option<HLLSet>, RedisError> {
    let key_handle = ctx.open_key(key);
    
    match key_handle.get_value::<HLLSet>(&HLLSET_TYPE)? {
        Some(hll) => Ok(Some(hll.clone())),
        None => Ok(None),
    }
}

/// HLLSET.CANDIDATES <hllset_key> <lut_prefix> [STREAM <stream_key>] [LAYER <n>] [LIMIT <n>]
///
/// Finds LUT entries (Redis hashes) matching positions in the HLLSet.
/// 
/// If STREAM is specified, results are written to the stream and count is returned.
/// Otherwise, returns array of matching entry keys.
///
/// Arguments:
///   hllset_key  - Key of the HLLSet to get positions from
///   lut_prefix  - Prefix for LUT hash keys (e.g., "tokenlut:entry:session123:")
///   STREAM      - Optional: Stream key to write results to
///   LAYER       - Optional: Filter by layer (0=unigram, 1=bigram, etc.)
///   LIMIT       - Optional: Max entries to return
///
/// Returns:
///   Without STREAM: Array of [key, token, layer, first_token, ...]
///   With STREAM: Integer count of entries streamed
///
/// Example:
///   HLLSET.CANDIDATES hllset:abc123 tokenlut:entry:sess1: STREAM candidates:out
pub fn hllset_candidates(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 3 {
        return Err(RedisError::WrongArity);
    }

    let hllset_key = &args[1];
    let lut_prefix = args[2].to_string_lossy();
    
    // Parse optional arguments
    let mut stream_key: Option<String> = None;
    let mut layer_filter: Option<i64> = None;
    let mut limit: Option<usize> = None;
    
    let mut i = 3;
    while i < args.len() {
        let arg = args[i].to_string_lossy().to_uppercase();
        match arg.as_str() {
            "STREAM" => {
                if i + 1 >= args.len() {
                    return Err(RedisError::Str("ERR STREAM requires a key argument"));
                }
                stream_key = Some(args[i + 1].to_string_lossy());
                i += 2;
            }
            "LAYER" => {
                if i + 1 >= args.len() {
                    return Err(RedisError::Str("ERR LAYER requires a number argument"));
                }
                layer_filter = Some(args[i + 1].to_string_lossy().parse()
                    .map_err(|_| RedisError::Str("ERR invalid layer number"))?);
                i += 2;
            }
            "LIMIT" => {
                if i + 1 >= args.len() {
                    return Err(RedisError::Str("ERR LIMIT requires a number argument"));
                }
                limit = Some(args[i + 1].to_string_lossy().parse()
                    .map_err(|_| RedisError::Str("ERR invalid limit number"))?);
                i += 2;
            }
            _ => {
                return Err(RedisError::Str("ERR unknown argument"));
            }
        }
    }

    // Get positions from HLLSet (zero-copy access to Roaring bitmap)
    let hllset = match get_hllset_readonly(ctx, hllset_key)? {
        Some(hll) => hll,
        None => return Err(RedisError::Str("ERR Key does not exist")),
    };
    
    let positions = hllset.active_positions();
    
    // Build position set for O(1) lookup (u32, u32) = (reg, zeros)
    let pos_set: HashSet<(u32, u32)> = positions.into_iter().collect();
    
    // Scan LUT entries matching prefix
    let scan_pattern = format!("{}*", lut_prefix);
    let mut count = 0;
    let mut results: Vec<RedisValue> = Vec::new();
    
    // Use SCAN to iterate over matching keys
    let scan_result = ctx.call("SCAN", &["0", "MATCH", &scan_pattern, "COUNT", "1000"]);
    
    match scan_result {
        Ok(RedisValue::Array(arr)) if arr.len() >= 2 => {
            // Process keys from scan result
            if let RedisValue::Array(keys) = &arr[1] {
                for key_val in keys {
                    if let Some(lim) = limit {
                        if count >= lim {
                            break;
                        }
                    }
                    
                    // Handle both BulkString and SimpleString
                    let key_str = match key_val {
                        RedisValue::BulkString(s) => s.clone(),
                        RedisValue::SimpleString(s) => s.clone(),
                        _ => continue,
                    };
                    
                    // Get hash fields: reg, zeros, layer, token, first_token
                    let hash_result = ctx.call("HMGET", &[
                        key_str.as_str(),
                        "reg", "zeros", "layer", "token", "first_token"
                    ]);
                    
                    if let Ok(RedisValue::Array(fields)) = hash_result {
                        if fields.len() >= 5 {
                            // Parse reg and zeros as u32
                            let reg = parse_int_field(&fields[0]).unwrap_or(0) as u32;
                            let zeros = parse_int_field(&fields[1]).unwrap_or(0) as u32;
                            let layer = parse_int_field(&fields[2]).unwrap_or(0);
                            
                            // Check position match
                            if !pos_set.contains(&(reg, zeros)) {
                                continue;
                            }
                            
                            // Check layer filter
                            if let Some(lf) = layer_filter {
                                if layer != lf {
                                    continue;
                                }
                            }
                            
                            // Match found!
                            if let Some(ref stream) = stream_key {
                                // Stream the result immediately (release Redis faster)
                                let token = extract_string_field(&fields[3]);
                                let first_token = extract_string_field(&fields[4]);
                                
                                let _ = ctx.call("XADD", &[
                                    stream.as_str(),
                                    "*",
                                    "key", key_str.as_str(),
                                    "reg", &reg.to_string(),
                                    "zeros", &zeros.to_string(),
                                    "layer", &layer.to_string(),
                                    "token", &token,
                                    "first_token", &first_token,
                                ]);
                            } else {
                                // Accumulate for array response
                                results.push(RedisValue::BulkString(key_str.clone()));
                                results.push(fields[3].clone()); // token
                                results.push(RedisValue::Integer(layer));
                                results.push(fields[4].clone()); // first_token
                            }
                            
                            count += 1;
                        }
                    }
                }
            }
        }
        _ => {}
    }
    
    if stream_key.is_some() {
        Ok(RedisValue::Integer(count as i64))
    } else {
        Ok(RedisValue::Array(results))
    }
}

/// HLLSET.SCANMATCH <hllset_key> <lut_prefix> <stream_key> [LAYER <n>] [BATCH <n>]
///
/// Full scan with cursor iteration - streams ALL matching entries.
/// More efficient for large datasets.
///
/// This command iterates through all LUT entries, checking each against
/// the HLLSet positions. Matched entries are immediately streamed.
///
/// Returns: Total count of matched entries
pub fn hllset_scanmatch(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 4 {
        return Err(RedisError::WrongArity);
    }

    let hllset_key = &args[1];
    let lut_prefix = args[2].to_string_lossy();
    let stream_key = args[3].to_string_lossy();
    
    // Parse optional arguments
    let mut layer_filter: Option<i64> = None;
    let mut batch_size: usize = 1000;
    
    let mut i = 4;
    while i < args.len() {
        let arg = args[i].to_string_lossy().to_uppercase();
        match arg.as_str() {
            "LAYER" => {
                if i + 1 >= args.len() {
                    return Err(RedisError::Str("ERR LAYER requires a number argument"));
                }
                layer_filter = Some(args[i + 1].to_string_lossy().parse()
                    .map_err(|_| RedisError::Str("ERR invalid layer number"))?);
                i += 2;
            }
            "BATCH" => {
                if i + 1 >= args.len() {
                    return Err(RedisError::Str("ERR BATCH requires a number argument"));
                }
                batch_size = args[i + 1].to_string_lossy().parse()
                    .map_err(|_| RedisError::Str("ERR invalid batch size"))?;
                i += 2;
            }
            _ => {
                return Err(RedisError::Str("ERR unknown argument"));
            }
        }
    }

    // Get positions from HLLSet
    let hllset = match get_hllset_readonly(ctx, hllset_key)? {
        Some(hll) => hll,
        None => return Err(RedisError::Str("ERR Key does not exist")),
    };
    
    let positions = hllset.active_positions();
    let pos_set: HashSet<(u32, u32)> = positions.into_iter().collect();
    
    let scan_pattern = format!("{}*", lut_prefix);
    let batch_str = batch_size.to_string();
    let mut cursor = "0".to_string();
    let mut total_count: i64 = 0;
    
    // Full cursor iteration
    loop {
        let scan_result = ctx.call("SCAN", &[&cursor, "MATCH", &scan_pattern, "COUNT", &batch_str]);
        
        match scan_result {
            Ok(RedisValue::Array(arr)) if arr.len() >= 2 => {
                // Update cursor (handle both BulkString and SimpleString)
                cursor = match &arr[0] {
                    RedisValue::BulkString(s) => s.clone(),
                    RedisValue::SimpleString(s) => s.clone(),
                    _ => break,
                };
                
                // Process keys
                if let RedisValue::Array(keys) = &arr[1] {
                    for key_val in keys {
                        // Handle both BulkString and SimpleString
                        let key_str = match key_val {
                            RedisValue::BulkString(s) => s.clone(),
                            RedisValue::SimpleString(s) => s.clone(),
                            _ => continue,
                        };
                        
                        // Get hash fields
                        let hash_result = ctx.call("HMGET", &[
                            key_str.as_str(),
                            "reg", "zeros", "layer", "token", "first_token"
                        ]);
                        
                        if let Ok(RedisValue::Array(fields)) = hash_result {
                            if fields.len() >= 5 {
                                let reg = parse_int_field(&fields[0]).unwrap_or(0) as u32;
                                let zeros = parse_int_field(&fields[1]).unwrap_or(0) as u32;
                                let layer = parse_int_field(&fields[2]).unwrap_or(0);
                                
                                // Check position match
                                if !pos_set.contains(&(reg, zeros)) {
                                    continue;
                                }
                                
                                // Check layer filter
                                if let Some(lf) = layer_filter {
                                    if layer != lf {
                                        continue;
                                    }
                                }
                                
                                // Stream the match
                                let token = extract_string_field(&fields[3]);
                                let first_token = extract_string_field(&fields[4]);
                                
                                let _ = ctx.call("XADD", &[
                                    stream_key.as_ref(),
                                    "*",
                                    "key", key_str.as_str(),
                                    "reg", &reg.to_string(),
                                    "zeros", &zeros.to_string(),
                                    "layer", &layer.to_string(),
                                    "token", &token,
                                    "first_token", &first_token,
                                ]);
                                
                                total_count += 1;
                            }
                        }
                    }
                }
                
                // Check if scan complete
                if cursor == "0" {
                    break;
                }
            }
            _ => break,
        }
    }
    
    Ok(RedisValue::Integer(total_count))
}

/// HLLSET.POSINDEX <hllset_key> <index_key>
///
/// Create a sorted set index of positions for fast range queries.
/// Score = reg * 32 + zeros (linearized position)
///
/// This enables efficient position-based queries without scanning.
pub fn hllset_posindex(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 3 {
        return Err(RedisError::WrongArity);
    }

    let hllset_key = &args[1];
    let index_key = &args[2];
    
    // Get positions from HLLSet
    let hllset = match get_hllset_readonly(ctx, hllset_key)? {
        Some(hll) => hll,
        None => return Err(RedisError::Str("ERR Key does not exist")),
    };
    
    let positions = hllset.active_positions();
    
    // Delete existing index
    let _ = ctx.call("DEL", &[&index_key.to_string_lossy()]);
    
    // Add positions to sorted set
    for (reg, zeros) in &positions {
        let score = (*reg as f64) * 32.0 + (*zeros as f64);
        let member = format!("{}:{}", reg, zeros);
        let _ = ctx.call("ZADD", &[
            &index_key.to_string_lossy(),
            &score.to_string(),
            &member
        ]);
    }
    
    Ok(RedisValue::Integer(positions.len() as i64))
}

// =============================================================================
// Helper functions
// =============================================================================

fn parse_int_field(val: &RedisValue) -> Option<i64> {
    match val {
        RedisValue::BulkString(s) => s.parse().ok(),
        RedisValue::SimpleString(s) => s.parse().ok(),
        RedisValue::Integer(i) => Some(*i),
        _ => None,
    }
}

fn extract_string_field(val: &RedisValue) -> String {
    match val {
        RedisValue::BulkString(s) => s.clone(),
        RedisValue::SimpleString(s) => s.clone(),
        _ => String::new(),
    }
}
