-- hll_union.lua
-- Merge two or more HyperLogLog keys into a destination key and return
-- the resulting cardinality estimate.
--
-- KEYS: dest, src1 [, src2, ...]
-- ARGV: (none)
-- Returns: integer – cardinality of the resulting union HLL

if #KEYS < 2 then
    return redis.error_reply("ERR hll_union requires at least 2 keys (dest + 1 source)")
end

local args = {}
for i = 1, #KEYS do
    args[i] = KEYS[i]
end

redis.call('PFMERGE', unpack(args))
return redis.call('PFCOUNT', KEYS[1])
