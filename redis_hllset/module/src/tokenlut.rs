//! TokenLUT Commands for HLLSet
//!
//! Commands for managing TokenLUT entries (Redis hashes indexed by RediSearch).
//! Each entry stores n-gram information with term frequency (TF) tracking.
//!
//! Key design: TF is atomically incremented on each add/touch operation.
//! This allows tracking token frequency across the corpus without expensive
//! document frequency (DF) calculations.
//!
//! Commands:
//! - TOKENLUT.ADD - Add/update token entry with TF increment
//! - TOKENLUT.INCR - Increment TF for existing entry
//! - TOKENLUT.GET - Get entry by hash key
//! - TOKENLUT.MGET - Batch get entries

use redis_module::{Context, RedisError, RedisResult, RedisString, RedisValue};

/// Helper to convert RedisString to String
fn to_string(s: &RedisString) -> String {
    s.to_string_lossy()
}

/// TOKENLUT.ADD <prefix> <hash_full> <reg> <zeros> <layer> <first_token> [TOKEN t1 t2 ...] [TF n]
///
/// Add or update a TokenLUT entry. If the entry exists, merges tokens and increments TF.
///
/// Arguments:
///   prefix      - Key prefix (e.g., "tokenlut:entry:session123:")
///   hash_full   - Full 64-bit hash (becomes part of key)
///   reg         - Register index [0, 1023]
///   zeros       - Trailing zeros count [0, 31]
///   layer       - N-gram layer (0=unigram, 1=bigram, 2=trigram)
///   first_token - First token (the main token for unigrams)
///   TOKEN       - Optional: Additional tokens for n-grams
///   TF          - Optional: TF increment amount (default: 1)
///
/// Returns:
///   Array: [collision_count, tf]
///
/// Example:
///   TOKENLUT.ADD tokenlut:entry:sess1: 12345678 42 3 0 hello
///   TOKENLUT.ADD tokenlut:entry:sess1: 87654321 100 5 1 quick TOKEN quick brown TF 1
pub fn tokenlut_add(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 7 {
        return Err(RedisError::WrongArity);
    }

    let prefix = to_string(&args[1]);
    let hash_full = to_string(&args[2]);
    let reg = to_string(&args[3]);
    let zeros = to_string(&args[4]);
    let layer = to_string(&args[5]);
    let first_token = to_string(&args[6]);
    
    // Parse optional arguments
    let mut token_parts: Vec<String> = Vec::new();
    let mut tf_increment: i64 = 1;
    
    let mut i = 7;
    while i < args.len() {
        let arg = to_string(&args[i]).to_uppercase();
        match arg.as_str() {
            "TOKEN" => {
                // Collect all following tokens until TF or end
                i += 1;
                while i < args.len() {
                    let next = to_string(&args[i]);
                    if next.to_uppercase() == "TF" {
                        break;
                    }
                    token_parts.push(next);
                    i += 1;
                }
            }
            "TF" => {
                if i + 1 >= args.len() {
                    return Err(RedisError::Str("ERR TF requires a number argument"));
                }
                tf_increment = to_string(&args[i + 1]).parse()
                    .map_err(|_| RedisError::Str("ERR invalid TF number"))?;
                i += 2;
            }
            _ => {
                return Err(RedisError::Str("ERR unknown argument"));
            }
        }
    }

    // Construct the key
    let key = format!("{}{}", prefix, hash_full);
    
    // Check if key exists
    let exists_result = ctx.call("EXISTS", &[key.as_str()])?;
    let exists = matches!(exists_result, RedisValue::Integer(1));
    
    if !exists {
        // New entry - create hash with all fields
        let first_tokens_json = format!("[\"{}\"]", escape_json(&first_token));
        
        let tokens_json = if token_parts.is_empty() {
            "[]".to_string()
        } else {
            let parts: Vec<String> = token_parts.iter()
                .map(|t| format!("\"{}\"", escape_json(t)))
                .collect();
            format!("[[{}]]", parts.join(","))
        };
        
        ctx.call("HSET", &[
            key.as_str(),
            "reg", reg.as_str(),
            "zeros", zeros.as_str(),
            "hash_full", hash_full.as_str(),
            "layer", layer.as_str(),
            "first_tokens", first_tokens_json.as_str(),
            "tokens", tokens_json.as_str(),
            "first_tokens_tag", first_token.as_str(),
            "collision_count", "1",
            "tf", &tf_increment.to_string(),
        ])?;
        
        return Ok(RedisValue::Array(vec![
            RedisValue::Integer(1),  // collision_count
            RedisValue::Integer(tf_increment),  // tf
        ]));
    }
    
    // Existing entry - need to merge and increment TF
    // Get current values
    let current = ctx.call("HMGET", &[
        key.as_str(),
        "first_tokens",
        "tokens",
        "tf",
    ])?;
    
    let (old_first_tokens_json, old_tokens_json, old_tf) = match current {
        RedisValue::Array(arr) if arr.len() >= 3 => {
            let ft = extract_string(&arr[0]).unwrap_or_else(|| "[]".to_string());
            let tk = extract_string(&arr[1]).unwrap_or_else(|| "[]".to_string());
            let tf: i64 = extract_string(&arr[2])
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            (ft, tk, tf)
        }
        _ => ("[]".to_string(), "[]".to_string(), 0),
    };
    
    // Parse and merge first_tokens
    let mut first_tokens = parse_json_string_array(&old_first_tokens_json);
    if !first_tokens.contains(&first_token) {
        first_tokens.push(first_token.clone());
    }
    
    // Parse and merge tokens (n-gram arrays)
    let mut tokens_arrays = parse_json_array_of_arrays(&old_tokens_json);
    if !token_parts.is_empty() && !tokens_arrays.contains(&token_parts) {
        tokens_arrays.push(token_parts);
    }
    
    // Serialize back to JSON
    let new_first_tokens_json = json_string_array(&first_tokens);
    let new_tokens_json = json_array_of_arrays(&tokens_arrays);
    let first_tokens_tag = first_tokens.join(",");
    let collision_count = first_tokens.len() as i64;
    let new_tf = old_tf + tf_increment;
    
    // Update the hash
    ctx.call("HSET", &[
        key.as_str(),
        "first_tokens", new_first_tokens_json.as_str(),
        "tokens", new_tokens_json.as_str(),
        "first_tokens_tag", first_tokens_tag.as_str(),
        "collision_count", &collision_count.to_string(),
        "tf", &new_tf.to_string(),
    ])?;
    
    Ok(RedisValue::Array(vec![
        RedisValue::Integer(collision_count),
        RedisValue::Integer(new_tf),
    ]))
}


/// TOKENLUT.INCR <key> [BY n]
///
/// Increment TF for an existing TokenLUT entry.
///
/// Arguments:
///   key  - Full key of the entry
///   BY   - Optional increment amount (default: 1)
///
/// Returns:
///   Integer: new TF value, or -1 if key doesn't exist
pub fn tokenlut_incr(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 2 {
        return Err(RedisError::WrongArity);
    }
    
    let key = to_string(&args[1]);
    
    // Parse optional BY argument
    let increment: i64 = if args.len() >= 4 {
        let by_arg = to_string(&args[2]).to_uppercase();
        if by_arg == "BY" {
            to_string(&args[3]).parse()
                .map_err(|_| RedisError::Str("ERR invalid increment"))?
        } else {
            1
        }
    } else {
        1
    };
    
    // Check if key exists
    let exists = ctx.call("EXISTS", &[key.as_str()])?;
    if matches!(exists, RedisValue::Integer(0)) {
        return Ok(RedisValue::Integer(-1));
    }
    
    // Increment TF using HINCRBY
    let result = ctx.call("HINCRBY", &[
        key.as_str(),
        "tf",
        &increment.to_string(),
    ])?;
    
    Ok(result)
}


/// TOKENLUT.GET <key>
///
/// Get a TokenLUT entry by its full key.
///
/// Returns:
///   Array of field-value pairs, or nil if not found
pub fn tokenlut_get(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 2 {
        return Err(RedisError::WrongArity);
    }
    
    let key = to_string(&args[1]);
    
    ctx.call("HGETALL", &[key.as_str()])
}


/// TOKENLUT.MGET <key1> [key2] ...
///
/// Batch get multiple TokenLUT entries.
///
/// Returns:
///   Array of arrays (field-value pairs for each key)
pub fn tokenlut_mget(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 2 {
        return Err(RedisError::WrongArity);
    }
    
    let mut results: Vec<RedisValue> = Vec::new();
    
    for i in 1..args.len() {
        let key = to_string(&args[i]);
        let entry = ctx.call("HGETALL", &[key.as_str()])?;
        results.push(entry);
    }
    
    Ok(RedisValue::Array(results))
}


// =============================================================================
// Helper functions for JSON handling
// =============================================================================

/// Escape special characters for JSON string
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Extract string from RedisValue
fn extract_string(val: &RedisValue) -> Option<String> {
    match val {
        RedisValue::BulkString(s) => Some(s.clone()),
        RedisValue::SimpleString(s) => Some(s.clone()),
        _ => None,
    }
}

/// Parse JSON array of strings: ["a", "b", "c"]
fn parse_json_string_array(json: &str) -> Vec<String> {
    // Simple parser for ["str1", "str2", ...]
    let trimmed = json.trim();
    if trimmed == "[]" || trimmed.is_empty() {
        return Vec::new();
    }
    
    let inner = trimmed.trim_start_matches('[').trim_end_matches(']');
    if inner.is_empty() {
        return Vec::new();
    }
    
    // Split by comma, handling quoted strings
    let mut result = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut escape_next = false;
    
    for c in inner.chars() {
        if escape_next {
            current.push(c);
            escape_next = false;
            continue;
        }
        
        match c {
            '\\' => {
                escape_next = true;
            }
            '"' => {
                in_string = !in_string;
            }
            ',' if !in_string => {
                if !current.trim().is_empty() {
                    result.push(current.trim().trim_matches('"').to_string());
                }
                current = String::new();
            }
            _ => {
                current.push(c);
            }
        }
    }
    
    if !current.trim().is_empty() {
        result.push(current.trim().trim_matches('"').to_string());
    }
    
    result
}

/// Parse JSON array of arrays: [["a", "b"], ["c", "d"]]
fn parse_json_array_of_arrays(json: &str) -> Vec<Vec<String>> {
    let trimmed = json.trim();
    if trimmed == "[]" || trimmed.is_empty() {
        return Vec::new();
    }
    
    // Remove outer brackets
    let inner = trimmed.trim_start_matches('[').trim_end_matches(']').trim();
    if inner.is_empty() {
        return Vec::new();
    }
    
    let mut result = Vec::new();
    let mut depth = 0;
    let mut current = String::new();
    let mut in_string = false;
    let mut escape_next = false;
    
    for c in inner.chars() {
        if escape_next {
            current.push(c);
            escape_next = false;
            continue;
        }
        
        match c {
            '\\' => {
                escape_next = true;
                current.push(c);
            }
            '"' => {
                in_string = !in_string;
                current.push(c);
            }
            '[' if !in_string => {
                depth += 1;
                if depth == 1 {
                    current = String::new();
                } else {
                    current.push(c);
                }
            }
            ']' if !in_string => {
                if depth == 1 {
                    // End of inner array
                    let arr = parse_json_string_array(&format!("[{}]", current));
                    if !arr.is_empty() {
                        result.push(arr);
                    }
                } else {
                    current.push(c);
                }
                depth -= 1;
            }
            ',' if !in_string && depth == 0 => {
                // Skip commas between inner arrays
            }
            _ => {
                current.push(c);
            }
        }
    }
    
    result
}

/// Serialize Vec<String> to JSON array
fn json_string_array(arr: &[String]) -> String {
    if arr.is_empty() {
        return "[]".to_string();
    }
    
    let parts: Vec<String> = arr.iter()
        .map(|s| format!("\"{}\"", escape_json(s)))
        .collect();
    
    format!("[{}]", parts.join(","))
}

/// Serialize Vec<Vec<String>> to JSON array of arrays
fn json_array_of_arrays(arr: &[Vec<String>]) -> String {
    if arr.is_empty() {
        return "[]".to_string();
    }
    
    let parts: Vec<String> = arr.iter()
        .map(|inner| json_string_array(inner))
        .collect();
    
    format!("[{}]", parts.join(","))
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_string_array() {
        assert_eq!(parse_json_string_array("[]"), Vec::<String>::new());
        assert_eq!(parse_json_string_array("[\"hello\"]"), vec!["hello"]);
        assert_eq!(parse_json_string_array("[\"a\",\"b\",\"c\"]"), vec!["a", "b", "c"]);
    }
    
    #[test]
    fn test_parse_array_of_arrays() {
        assert_eq!(parse_json_array_of_arrays("[]"), Vec::<Vec<String>>::new());
        assert_eq!(
            parse_json_array_of_arrays("[[\"a\",\"b\"]]"),
            vec![vec!["a", "b"]]
        );
        assert_eq!(
            parse_json_array_of_arrays("[[\"a\",\"b\"],[\"c\",\"d\"]]"),
            vec![vec!["a", "b"], vec!["c", "d"]]
        );
    }
    
    #[test]
    fn test_json_string_array() {
        assert_eq!(json_string_array(&[]), "[]");
        assert_eq!(json_string_array(&["hello".to_string()]), "[\"hello\"]");
    }
}
