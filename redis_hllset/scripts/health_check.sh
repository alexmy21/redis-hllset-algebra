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

# Check 4: HLLSet functions (Lua) or Rust RING commands
print ""
print "${BLUE}Checking HLLSet functions/commands...${NC}"

functions=$(redis_cmd FUNCTION LIST LIBRARYNAME hllset_lib 2>/dev/null)
if [ -n "$functions" ] && [[ "$functions" != *"empty"* ]]; then
    # Lua functions are loaded
    func_count=$(echo "$functions" | grep -c "name" || echo 0)
    print "${GREEN}✓${NC} HLLSet Lua library loaded with $func_count functions"
else
    # Check for native Rust RING commands (module v2+)
    ring_cmd=$(redis_cmd COMMAND INFO hllset.ring.init 2>/dev/null)
    if [[ "$ring_cmd" != *"nil"* ]] && [ -n "$ring_cmd" ]; then
        print "${GREEN}✓${NC} Native Rust RING commands available (HLLSET.RING.*)"
    else
        # Check hllset module version
        hllset_ver=$(redis_cmd MODULE LIST | grep -A2 "hllset" | grep "ver" | awk '{print $NF}')
        if [ -n "$hllset_ver" ] && [ "$hllset_ver" -ge 2 ] 2>/dev/null; then
            print "${GREEN}✓${NC} HLLSet module v$hllset_ver loaded (Rust commands)"
        else
            print "${RED}✗${NC} HLLSet functions not loaded"
            print "   Run: ./load_functions.sh  (for Lua)"
            print "   Or reload module v2+ for native Rust RING commands"
            exit 3
        fi
    fi
fi

# Check 5: Quick functional test
print ""
print "${BLUE}Running functional test...${NC}"

functions=$(redis_cmd FUNCTION LIST LIBRARYNAME hllset_lib 2>/dev/null)
if [ -n "$functions" ] && [[ "$functions" != *"empty"* ]]; then
    # Test Lua path
    test_result=$(redis_cmd FCALL hllset_create 0 health_check_test 2>&1)
    if [[ "$test_result" == hllset:* ]]; then
        print "${GREEN}✓${NC} hllset_create (Lua) works"
        redis_cmd FCALL hllset_delete 1 "$test_result" > /dev/null 2>&1
    else
        print "${RED}✗${NC} hllset_create failed: $test_result"
        ((errors++))
    fi
else
    # Test Rust path
    test_result=$(redis_cmd HLLSET.RING.INIT health_check_ring P_BITS 10 2>&1)
    rank_result=$(redis_cmd HLLSET.RING.RANK health_check_ring 2>&1)
    if [[ "$test_result" == "OK" ]] && [[ "$rank_result" == "0" ]]; then
        print "${GREEN}✓${NC} HLLSET.RING.INIT / RING.RANK (Rust) work"
        redis_cmd DEL "hllring:ring:health_check_ring" > /dev/null 2>&1
    else
        print "${RED}✗${NC} Rust RING commands failed: init=$test_result rank=$rank_result"
        ((errors++))
    fi
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
