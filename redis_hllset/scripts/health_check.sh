#!/bin/bash
# Health check for Redis and HLLSet functions
#
# Usage: ./health_check.sh [--quiet]
#
# Exit codes:
#   0 - All checks passed
#   1 - Redis not reachable
#   2 - Required modules not loaded
#   3 - HLLSet functions not loaded

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

QUIET="${1:-}"
errors=0

print() {
    [ "$QUIET" != "--quiet" ] && echo -e "$@"
}

# Check 1: Redis connectivity
print "${BLUE}Checking Redis connectivity...${NC}"
if redis_cmd PING > /dev/null 2>&1; then
    print "${GREEN}✓${NC} Redis reachable at $REDIS_HOST:$REDIS_PORT"
else
    print "${RED}✗${NC} Cannot connect to Redis at $REDIS_HOST:$REDIS_PORT"
    exit 1
fi

# Check 2: Redis version
version=$(redis_cmd INFO server | grep redis_version | cut -d: -f2 | tr -d '\r')
print "${GREEN}✓${NC} Redis version: $version"

# Check 3: Required modules
print ""
print "${BLUE}Checking required modules...${NC}"

modules=$(redis_cmd MODULE LIST)
required_modules=("redis-roaring")
optional_modules=("redisgraph" "search")

for mod in "${required_modules[@]}"; do
    if echo "$modules" | grep -qi "$mod"; then
        print "${GREEN}✓${NC} Module loaded: $mod"
    else
        print "${RED}✗${NC} Missing required module: $mod"
        ((errors++))
    fi
done

for mod in "${optional_modules[@]}"; do
    if echo "$modules" | grep -qi "$mod"; then
        print "${GREEN}✓${NC} Module loaded: $mod (optional)"
    else
        print "${YELLOW}○${NC} Module not loaded: $mod (optional)"
    fi
done

[ $errors -gt 0 ] && exit 2

# Check 4: HLLSet functions
print ""
print "${BLUE}Checking HLLSet functions...${NC}"

functions=$(redis_cmd FUNCTION LIST LIBRARYNAME hllset_lib 2>/dev/null)
if [ -z "$functions" ] || [[ "$functions" == *"empty"* ]]; then
    print "${RED}✗${NC} HLLSet functions not loaded"
    print "   Run: ./load_functions.sh"
    exit 3
fi

# Count functions
func_count=$(echo "$functions" | grep -c "name" || echo 0)
print "${GREEN}✓${NC} HLLSet library loaded with $func_count functions"

# Check 5: Quick functional test
print ""
print "${BLUE}Running functional test...${NC}"
test_result=$(redis_cmd FCALL hllset_create 0 health_check_test 2>&1)
if [[ "$test_result" == hllset:* ]]; then
    print "${GREEN}✓${NC} hllset_create works"
    # Clean up
    redis_cmd FCALL hllset_delete 1 "$test_result" > /dev/null 2>&1
else
    print "${RED}✗${NC} hllset_create failed: $test_result"
    ((errors++))
fi

# Check 6: Memory stats
print ""
print "${BLUE}Memory usage...${NC}"
used_memory=$(redis_cmd INFO memory | grep "used_memory_human" | cut -d: -f2 | tr -d '\r')
print "   Used memory: $used_memory"

# Check 7: Key counts
hllset_keys=$(redis_cmd KEYS "hllset:*" | wc -l)
temp_keys=$(redis_cmd KEYS "hllset:temp:*" | wc -l)
meta_keys=$(redis_cmd KEYS "hllset:meta:*" | wc -l)
print "   HLLSet keys: $hllset_keys (temp: $temp_keys, meta: $meta_keys)"

# Summary
print ""
if [ $errors -eq 0 ]; then
    print "${GREEN}═══════════════════════════════════════${NC}"
    print "${GREEN}  All health checks passed!${NC}"
    print "${GREEN}═══════════════════════════════════════${NC}"
    exit 0
else
    print "${RED}═══════════════════════════════════════${NC}"
    print "${RED}  $errors check(s) failed${NC}"
    print "${RED}═══════════════════════════════════════${NC}"
    exit 3
fi
