#!lua name=hllset_lib

--[[
HLLSet Algebra — Redis Function Library (Minimal POC)

Implements core HLLSet operations using Redis Roaring Bitmaps.
This is a simplified version for testing the Redis integration.

Usage:
    FCALL hllset_create 0 token1 token2 token3 ...
    FCALL hllset_union 2 key1 key2
    FCALL hllset_cardinality 1 key
]]

-- Constants
local P_BITS = 10
local M = 1024                       -- 2^10 registers
local SEED = 42
local BITS_PER_REG = 32

-- Key prefixes
local PREFIX_HLLSET = "hllset:"
local PREFIX_META = "hllset:meta:"
local PREFIX_TEMP = "hllset:temp:"

-- Alpha constants for HLL bias correction
local ALPHA_INF = 0.7213 / (1.0 + 1.079 / 65536.0)


-- Simple hash function (for POC - Python will provide real hashes)
local function simple_hash(str, seed)
    local h = seed
    for i = 1, #str do
        h = bit.bxor(h, string.byte(str, i))
        h = h * 31
        h = bit.band(h, 0xFFFFFFFF)
    end
    return h
end


-- Convert hash to register index and trailing zeros
local function hash_to_reg_zeros(hash)
    local reg = bit.band(hash, M - 1)  -- Bottom P_BITS
    local remaining = bit.rshift(hash, P_BITS)
    
    -- Count trailing zeros
    local zeros = 0
    if remaining == 0 then
        zeros = 32 - P_BITS
    else
        while bit.band(remaining, 1) == 0 and zeros < 31 do
            zeros = zeros + 1
            remaining = bit.rshift(remaining, 1)
        end
    end
    
    return reg, zeros
end


-- Highest set bit position (1-indexed, 0 if no bits set)
local function highest_set_bit(value)
    if value == 0 then return 0 end
    local pos = 0
    while value > 0 do
        value = bit.rshift(value, 1)
        pos = pos + 1
    end
    return pos
end


-- Set bit in roaring bitmap
local function set_bit(key, position)
    redis.call('R.SETBIT', key, position, 1)
end


-- Get all set positions from roaring bitmap
local function get_positions(key)
    local ok, result = pcall(redis.call, 'R.GETINTARRAY', key)
    if ok and result then
        return result
    end
    return {}
end


-- Convert positions to register array
local function positions_to_registers(positions)
    local registers = {}
    for i = 1, M do
        registers[i] = 0
    end
    
    for _, pos in ipairs(positions) do
        local reg_idx = math.floor(pos / BITS_PER_REG) + 1
        local bit_idx = pos % BITS_PER_REG
        if reg_idx >= 1 and reg_idx <= M then
            registers[reg_idx] = bit.bor(registers[reg_idx], bit.lshift(1, bit_idx))
        end
    end
    
    return registers
end


-- Compute SHA1 from bitmap key
local function compute_sha1_from_key(bitmap_key)
    local positions = get_positions(bitmap_key)
    local registers = positions_to_registers(positions)
    
    -- Serialize registers to binary
    local parts = {}
    for i = 1, M do
        local val = registers[i] or 0
        table.insert(parts, string.char(
            bit.band(val, 0xFF),
            bit.band(bit.rshift(val, 8), 0xFF),
            bit.band(bit.rshift(val, 16), 0xFF),
            bit.band(bit.rshift(val, 24), 0xFF)
        ))
    end
    
    return redis.sha1hex(table.concat(parts))
end


-- Estimate cardinality
local function estimate_cardinality(bitmap_key)
    local positions = get_positions(bitmap_key)
    local registers = positions_to_registers(positions)
    
    local raw_sum = 0.0
    local zero_count = 0
    
    for i = 1, M do
        local val = registers[i] or 0
        local hsb = highest_set_bit(val)
        raw_sum = raw_sum + math.pow(2, -hsb)
        if val == 0 then
            zero_count = zero_count + 1
        end
    end
    
    if zero_count == M then
        return 0
    end
    
    local estimate = ALPHA_INF * M * M / raw_sum
    
    -- Linear counting for small cardinalities
    if estimate <= 2.5 * M and zero_count > 0 then
        estimate = M * math.log(M / zero_count)
    end
    
    return math.max(0, math.floor(estimate + 0.5))
end


-- Create HLLSet from tokens
local function hllset_create(keys, args)
    if #args == 0 then
        local empty_sha1 = redis.sha1hex(string.rep('\0', M * 4))
        return PREFIX_HLLSET .. empty_sha1
    end
    
    local temp_key = PREFIX_TEMP .. redis.sha1hex(table.concat(args, ':'))
    
    for _, token in ipairs(args) do
        local hash = simple_hash(token, SEED)
        local reg, zeros = hash_to_reg_zeros(hash)
        local position = reg * BITS_PER_REG + zeros
        set_bit(temp_key, position)
    end
    
    local sha1 = compute_sha1_from_key(temp_key)
    local final_key = PREFIX_HLLSET .. sha1
    
    if redis.call('EXISTS', final_key) == 1 then
        redis.call('DEL', temp_key)
        return final_key
    end
    
    redis.call('RENAME', temp_key, final_key)
    
    redis.call('HSET', PREFIX_META .. sha1,
        'p_bits', P_BITS,
        'seed', SEED,
        'created_at', redis.call('TIME')[1]
    )
    
    return final_key
end


-- Create HLLSet from pre-computed hashes (Python integration)
local function hllset_create_from_hashes(keys, args)
    if #args == 0 then
        local empty_sha1 = redis.sha1hex(string.rep('\0', M * 4))
        return PREFIX_HLLSET .. empty_sha1
    end
    
    local temp_key = PREFIX_TEMP .. "h:" .. redis.sha1hex(table.concat(args, ':'))
    
    for _, hash_str in ipairs(args) do
        local hash = tonumber(hash_str)
        if hash then
            local reg, zeros = hash_to_reg_zeros(hash)
            local position = reg * BITS_PER_REG + zeros
            set_bit(temp_key, position)
        end
    end
    
    local sha1 = compute_sha1_from_key(temp_key)
    local final_key = PREFIX_HLLSET .. sha1
    
    if redis.call('EXISTS', final_key) == 1 then
        redis.call('DEL', temp_key)
        return final_key
    end
    
    redis.call('RENAME', temp_key, final_key)
    
    redis.call('HSET', PREFIX_META .. sha1,
        'p_bits', P_BITS,
        'seed', SEED,
        'created_at', redis.call('TIME')[1]
    )
    
    return final_key
end


-- Union of two HLLSets
local function hllset_union(keys, args)
    if #keys ~= 2 then
        return redis.error_reply("Expected exactly 2 keys")
    end
    
    local temp_key = PREFIX_TEMP .. "union:" .. redis.sha1hex(keys[1] .. keys[2])
    redis.call('R.BITOP', 'OR', temp_key, keys[1], keys[2])
    
    local sha1 = compute_sha1_from_key(temp_key)
    local final_key = PREFIX_HLLSET .. sha1
    
    if redis.call('EXISTS', final_key) == 1 then
        redis.call('DEL', temp_key)
        return final_key
    end
    
    redis.call('RENAME', temp_key, final_key)
    
    redis.call('HSET', PREFIX_META .. sha1,
        'p_bits', P_BITS,
        'operation', 'union',
        'parent1', keys[1],
        'parent2', keys[2]
    )
    
    return final_key
end


-- Intersection of two HLLSets
local function hllset_intersect(keys, args)
    if #keys ~= 2 then
        return redis.error_reply("Expected exactly 2 keys")
    end
    
    local temp_key = PREFIX_TEMP .. "inter:" .. redis.sha1hex(keys[1] .. keys[2])
    redis.call('R.BITOP', 'AND', temp_key, keys[1], keys[2])
    
    local sha1 = compute_sha1_from_key(temp_key)
    local final_key = PREFIX_HLLSET .. sha1
    
    if redis.call('EXISTS', final_key) == 1 then
        redis.call('DEL', temp_key)
        return final_key
    end
    
    redis.call('RENAME', temp_key, final_key)
    
    redis.call('HSET', PREFIX_META .. sha1,
        'p_bits', P_BITS,
        'operation', 'intersect',
        'parent1', keys[1],
        'parent2', keys[2]
    )
    
    return final_key
end


-- Difference A - B
local function hllset_diff(keys, args)
    if #keys ~= 2 then
        return redis.error_reply("Expected exactly 2 keys")
    end
    
    local temp_key = PREFIX_TEMP .. "diff:" .. redis.sha1hex(keys[1] .. keys[2])
    redis.call('R.DIFF', temp_key, keys[1], keys[2])
    
    local sha1 = compute_sha1_from_key(temp_key)
    local final_key = PREFIX_HLLSET .. sha1
    
    if redis.call('EXISTS', final_key) == 1 then
        redis.call('DEL', temp_key)
        return final_key
    end
    
    redis.call('RENAME', temp_key, final_key)
    
    redis.call('HSET', PREFIX_META .. sha1,
        'p_bits', P_BITS,
        'operation', 'diff',
        'parent1', keys[1],
        'parent2', keys[2]
    )
    
    return final_key
end


-- Symmetric difference (XOR)
local function hllset_xor(keys, args)
    if #keys ~= 2 then
        return redis.error_reply("Expected exactly 2 keys")
    end
    
    local temp_key = PREFIX_TEMP .. "xor:" .. redis.sha1hex(keys[1] .. keys[2])
    redis.call('R.BITOP', 'XOR', temp_key, keys[1], keys[2])
    
    local sha1 = compute_sha1_from_key(temp_key)
    local final_key = PREFIX_HLLSET .. sha1
    
    if redis.call('EXISTS', final_key) == 1 then
        redis.call('DEL', temp_key)
        return final_key
    end
    
    redis.call('RENAME', temp_key, final_key)
    
    redis.call('HSET', PREFIX_META .. sha1,
        'p_bits', P_BITS,
        'operation', 'xor',
        'parent1', keys[1],
        'parent2', keys[2]
    )
    
    return final_key
end


-- Get cardinality
local function hllset_cardinality(keys, args)
    if #keys ~= 1 then
        return redis.error_reply("Expected exactly 1 key")
    end
    return estimate_cardinality(keys[1])
end


-- Jaccard similarity
local function hllset_similarity(keys, args)
    if #keys ~= 2 then
        return redis.error_reply("Expected exactly 2 keys")
    end
    
    local union_key = PREFIX_TEMP .. "sim_u:" .. redis.sha1hex(keys[1] .. keys[2])
    redis.call('R.BITOP', 'OR', union_key, keys[1], keys[2])
    local card_union = estimate_cardinality(union_key)
    
    if card_union == 0 then
        redis.call('DEL', union_key)
        return "0.0"
    end
    
    local inter_key = PREFIX_TEMP .. "sim_i:" .. redis.sha1hex(keys[1] .. keys[2])
    redis.call('R.BITOP', 'AND', inter_key, keys[1], keys[2])
    local card_inter = estimate_cardinality(inter_key)
    
    redis.call('DEL', union_key, inter_key)
    
    return tostring(card_inter / card_union)
end


-- Get info about HLLSet
local function hllset_info(keys, args)
    if #keys ~= 1 then
        return redis.error_reply("Expected exactly 1 key")
    end
    
    local key = keys[1]
    if redis.call('EXISTS', key) == 0 then
        return redis.error_reply("Key does not exist")
    end
    
    local card = estimate_cardinality(key)
    local sha1 = string.sub(key, #PREFIX_HLLSET + 1)
    local meta = redis.call('HGETALL', PREFIX_META .. sha1)
    
    return {
        'key', key,
        'sha1', sha1,
        'cardinality', card,
        'p_bits', P_BITS,
        'registers', M,
        'metadata', meta
    }
end


-- Dump positions (for debugging)
local function hllset_dump(keys, args)
    if #keys ~= 1 then
        return redis.error_reply("Expected exactly 1 key")
    end
    return get_positions(keys[1])
end


-- Delete HLLSet
local function hllset_delete(keys, args)
    if #keys ~= 1 then
        return redis.error_reply("Expected exactly 1 key")
    end
    
    local key = keys[1]
    local sha1 = string.sub(key, #PREFIX_HLLSET + 1)
    return redis.call('DEL', key, PREFIX_META .. sha1)
end


-- Register all functions
redis.register_function('hllset_create', hllset_create)
redis.register_function('hllset_create_from_hashes', hllset_create_from_hashes)
redis.register_function('hllset_union', hllset_union)
redis.register_function('hllset_intersect', hllset_intersect)
redis.register_function('hllset_diff', hllset_diff)
redis.register_function('hllset_xor', hllset_xor)
redis.register_function('hllset_cardinality', hllset_cardinality)
redis.register_function('hllset_similarity', hllset_similarity)
redis.register_function('hllset_info', hllset_info)
redis.register_function('hllset_dump', hllset_dump)
redis.register_function('hllset_delete', hllset_delete)
