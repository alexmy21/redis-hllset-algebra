-- hll_diff_card.lua
-- Estimate the cardinality of the set difference  A \ B  using the identity:
--
--   |A \ B| = |A ∪ B| − |B|
--
-- This follows directly from  |A ∪ B| = |A \ B| + |A ∩ B| + |B \ A|  and
-- |B| = |A ∩ B| + |B \ A|.
--
-- KEYS[1]: key A
-- KEYS[2]: key B
-- ARGV[1]: a unique temporary key (caller must ensure no collision)
-- Returns: integer – estimated cardinality of A \ B  (>= 0)

if #KEYS ~= 2 then
    return redis.error_reply("ERR hll_diff_card requires exactly 2 keys")
end

local key_a   = KEYS[1]
local key_b   = KEYS[2]
local tmp_key = ARGV[1]

redis.call('PFMERGE', tmp_key, key_a, key_b)
local union_count = redis.call('PFCOUNT', tmp_key)
local count_b     = redis.call('PFCOUNT', key_b)
redis.call('DEL', tmp_key)

local diff = union_count - count_b
if diff < 0 then diff = 0 end
return diff
