-- hll_intersect_card.lua
-- Estimate the cardinality of the intersection of two or more HLL keys using
-- the inclusion-exclusion principle:
--
--   |A1 ∩ A2 ∩ ... ∩ An| = Σ_{S ⊆ {1..n}, S≠∅} (-1)^(|S|+1) · |⋃_{i∈S} Ai|
--
-- KEYS: key1, key2 [, key3, ...]
-- ARGV[1]: a unique temporary key prefix (caller must ensure no collision)
-- Returns: integer – estimated cardinality of the intersection (>= 0)

if #KEYS < 1 then
    return redis.error_reply("ERR hll_intersect_card requires at least 1 key")
end

if #KEYS == 1 then
    return redis.call('PFCOUNT', KEYS[1])
end

local n       = #KEYS
local tmp_pfx = ARGV[1]
local result  = 0

-- Compute 2^n without floating-point math
local max_mask = 1
for i = 1, n do
    max_mask = max_mask * 2
end
max_mask = max_mask - 1

for mask = 1, max_mask do
    -- Collect the subset of KEYS whose bit is set in 'mask'
    local subset = {}
    local size   = 0
    local m      = mask
    for i = 1, n do
        if m % 2 == 1 then
            subset[#subset + 1] = KEYS[i]
            size = size + 1
        end
        m = math.floor(m / 2)
    end

    -- Cardinality of the union of this subset
    local union_count
    if size == 1 then
        union_count = redis.call('PFCOUNT', subset[1])
    else
        local tmp_key   = tmp_pfx .. mask
        local merge_cmd = {tmp_key}
        for _, k in ipairs(subset) do
            merge_cmd[#merge_cmd + 1] = k
        end
        redis.call('PFMERGE', unpack(merge_cmd))
        union_count = redis.call('PFCOUNT', tmp_key)
        redis.call('DEL', tmp_key)
    end

    -- Inclusion-exclusion sign: + for odd-size subsets, – for even-size subsets
    if size % 2 == 1 then
        result = result + union_count
    else
        result = result - union_count
    end
end

if result < 0 then result = 0 end
return result
