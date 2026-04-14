//! RDB persistence support for HLLSet
//!
//! This module registers HLLSet as a native Redis data type with proper
//! serialization for RDB persistence and memory reporting.

use redis_module::native_types::RedisType;
use redis_module::raw;
use std::os::raw::c_void;

use crate::hllset::HLLSet;

/// Type version for RDB encoding
const HLLSET_ENCODING_VERSION: i32 = 1;

/// Native Redis type definition for HLLSet
pub static HLLSET_TYPE: RedisType = RedisType::new(
    "hllset-rs",
    HLLSET_ENCODING_VERSION,
    raw::RedisModuleTypeMethods {
        version: raw::REDISMODULE_TYPE_METHOD_VERSION as u64,
        
        // RDB persistence
        rdb_load: Some(hllset_rdb_load),
        rdb_save: Some(hllset_rdb_save),
        
        // AOF rewrite (optional - uses commands by default)
        aof_rewrite: None,
        
        // Memory management
        mem_usage: Some(hllset_mem_usage),
        mem_usage2: None,
        
        // Free callback
        free: Some(hllset_free),
        
        // Digest for DEBUG DIGEST
        digest: None,
        
        // Aux data for RDB (not used)
        aux_load: None,
        aux_save: None,
        aux_save2: None,
        aux_save_triggers: 0,
        
        // Defrag callback
        free_effort: None,
        free_effort2: None,
        unlink: None,
        unlink2: None,
        copy: Some(hllset_copy),
        copy2: None,
        defrag: None,
    },
);

/// Load HLLSet from RDB
unsafe extern "C" fn hllset_rdb_load(
    rdb: *mut raw::RedisModuleIO,
    _encver: i32,
) -> *mut c_void {
    // Read serialized data length
    let len = raw::RedisModule_LoadUnsigned.unwrap()(rdb) as usize;
    
    if len == 0 {
        // Empty HLLSet
        let hllset = Box::new(HLLSet::new());
        return Box::into_raw(hllset) as *mut c_void;
    }
    
    // Read serialized data
    let mut out_len: usize = 0;
    let read_ptr = raw::RedisModule_LoadStringBuffer.unwrap()(
        rdb,
        &mut out_len as *mut usize,
    );
    
    if read_ptr.is_null() {
        // Failed to read, return empty
        let hllset = Box::new(HLLSet::new());
        return Box::into_raw(hllset) as *mut c_void;
    }
    
    // Copy data from Redis buffer
    let buffer = std::slice::from_raw_parts(read_ptr as *const u8, out_len);
    let result = HLLSet::from_bytes(buffer);
    
    // Free Redis buffer
    raw::RedisModule_Free.unwrap()(read_ptr as *mut c_void);
    
    // Deserialize
    match result {
        Some(hllset) => Box::into_raw(Box::new(hllset)) as *mut c_void,
        None => {
            // Deserialization failed, return empty
            let hllset = Box::new(HLLSet::new());
            Box::into_raw(hllset) as *mut c_void
        }
    }
}

/// Save HLLSet to RDB
unsafe extern "C" fn hllset_rdb_save(rdb: *mut raw::RedisModuleIO, value: *mut c_void) {
    let hllset = &*(value as *const HLLSet);
    
    // Serialize to bytes
    let bytes = hllset.to_bytes();
    
    // Write length
    raw::RedisModule_SaveUnsigned.unwrap()(rdb, bytes.len() as u64);
    
    if !bytes.is_empty() {
        // Write data
        raw::RedisModule_SaveStringBuffer.unwrap()(
            rdb,
            bytes.as_ptr() as *const i8,
            bytes.len(),
        );
    }
}

/// Report memory usage
unsafe extern "C" fn hllset_mem_usage(value: *const c_void) -> usize {
    let hllset = &*(value as *const HLLSet);
    hllset.memory_usage()
}

/// Free HLLSet memory
unsafe extern "C" fn hllset_free(value: *mut c_void) {
    if !value.is_null() {
        drop(Box::from_raw(value as *mut HLLSet));
    }
}

/// Copy HLLSet (for COPY command)
unsafe extern "C" fn hllset_copy(
    _from_key: *mut raw::RedisModuleString,
    _to_key: *mut raw::RedisModuleString,
    value: *const c_void,
) -> *mut c_void {
    let hllset = &*(value as *const HLLSet);
    let copy = Box::new(hllset.clone());
    Box::into_raw(copy) as *mut c_void
}
