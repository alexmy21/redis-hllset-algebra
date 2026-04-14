-- hll_symmdiff_card.lua
-- Estimate the cardinality of the symmetric difference  A △ B  using:
--
--   |A △ B| = |A ∪ B| − |A ∩ B|
--           = |A ∪ B| − (|A| + |B| − |A ∪ B|)
--           = 2·|A ∪ B| − |A| − |B|
--
-- KEYS[1]: key A
-- KEYS[2]: key B
-- ARGV[1]: a unique temporary key (caller must ensure no collision)
-- Returns: integer – estimated cardinality of A △ B  (>= 0)

if #KEYS ~= 2 then
    return redis.error_reply("ERR hll_symmdiff_card requires exactly 2 keys")
end

local key_a   = KEYS[1]
local key_b   = KEYS[2]
local tmp_key = ARGV[1]

redis.call('PFMERGE', tmp_key, key_a, key_b)
local union_count = redis.call('PFCOUNT', tmp_key)
local count_a     = redis.call('PFCOUNT', key_a)
local count_b     = redis.call('PFCOUNT', key_b)
redis.call('DEL', tmp_key)

local symmdiff = 2 * union_count - count_a - count_b
if symmdiff < 0 then symmdiff = 0 end
return symmdiff
