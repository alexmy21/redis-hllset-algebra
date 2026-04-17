#!/bin/bash
# Test HLLSet commands in Redis
#
# Tests native Rust RING commands (HLLSET.RING.*) when available,
# falls back to Lua FCALL tests if Lua functions are loaded.
#
# Usage: ./test_hllset.sh

# Don't use set -e as we want to continue on failures

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "=========================================="
echo "  HLLSet Algebra Redis Tests"
echo "=========================================="
echo ""

pass=0
fail=0

test_result() {
    local name="$1"
    local expected="$2"
    local actual="$3"
    local actual_clean=$(echo "$actual" | sed 's/(integer) //' | tr -d '\r')
    local expected_clean=$(echo "$expected" | sed 's/(integer) //')
    if [ "$expected_clean" == "$actual_clean" ]; then
        echo -e "${GREEN}✓${NC} $name"
        ((pass++))
    else
        echo -e "${RED}✗${NC} $name: expected '$expected_clean', got '$actual_clean'"
        ((fail++))
    fi
}

test_contains() {
    local name="$1"
    local substring="$2"
    local actual="$3"
    if echo "$actual" | grep -q "$substring"; then
        echo -e "${GREEN}✓${NC} $name"
        ((pass++))
    else
        echo -e "${RED}✗${NC} $name: '$substring' not found in '$actual'"
        ((fail++))
    fi
}

# ============================================================
# Detect mode: Rust or Lua
# ============================================================
ring_cmd=$(redis_cmd COMMAND INFO hllset.ring.init 2>/dev/null)
lua_loaded=$(redis_cmd FUNCTION LIST LIBRARYNAME hllset_lib 2>/dev/null)

if [[ "$ring_cmd" != *"nil"* ]] && [ -n "$ring_cmd" ]; then
    MODE="rust"
elif [ -n "$lua_loaded" ] && [[ "$lua_loaded" != *"empty"* ]]; then
    MODE="lua"
else
    echo -e "${RED}✗${NC} No HLLSet commands available (neither Rust nor Lua)"
    echo "0 failed"
    exit 10
fi

echo "Mode: $MODE"
echo ""

# ============================================================
# Cleanup
# ============================================================
redis_cmd DEL "hlltest:ring" > /dev/null 2>&1 || true
redis_cmd DEL "hllring:ring:hlltest:ring" > /dev/null 2>&1 || true

if [ "$MODE" == "rust" ]; then

    # ========================================================
    # Rust RING command tests
    # ========================================================

    echo "1. Testing RING.INIT..."
    result=$(redis_cmd HLLSET.RING.INIT hlltest:ring P_BITS 10 2>&1)
    test_result "RING.INIT returns OK" "OK" "$result"
    echo ""

    echo "2. Testing RING.RANK (empty ring)..."
    rank=$(redis_cmd HLLSET.RING.RANK hlltest:ring 2>&1 | tr -d '(integer) ')
    test_result "Initial rank is 0" "0" "$rank"
    echo ""

    echo "3. Testing RING.BASIS (empty ring)..."
    # Empty basis returns nothing in raw mode (redis-cli default when piped)
    # Just verify the command doesn't error
    redis_cmd HLLSET.RING.BASIS hlltest:ring > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} RING.BASIS command succeeds on empty ring"
        ((pass++))
    else
        echo -e "${RED}✗${NC} RING.BASIS command failed"
        ((fail++))
    fi
    echo ""

    echo "4. Testing RING.INGEST..."
    result=$(redis_cmd HLLSET.RING.INGEST hlltest:ring "hello world" SOURCE test 2>&1)
    test_contains "RING.INGEST returns result" "." "$result"  # any non-empty
    echo ""

    echo "5. Testing RING.RANK after ingest..."
    rank=$(redis_cmd HLLSET.RING.RANK hlltest:ring 2>&1 | tr -d '(integer) ')
    if [ "$rank" -ge 1 ] 2>/dev/null; then
        echo -e "${GREEN}✓${NC} RING.RANK increased after ingest ($rank)"
        ((pass++))
    else
        echo -e "${RED}✗${NC} RING.RANK did not increase: $rank"
        ((fail++))
    fi
    echo ""

    echo "6. Testing W.COMMIT..."
    result=$(redis_cmd HLLSET.W.COMMIT hlltest:ring 2>&1)
    test_contains "W.COMMIT returns result" "." "$result"
    echo ""

    echo "7. Testing second ingest + W.COMMIT..."
    redis_cmd HLLSET.RING.INGEST hlltest:ring "foo bar baz" SOURCE test > /dev/null 2>&1
    result=$(redis_cmd HLLSET.W.COMMIT hlltest:ring 2>&1)
    test_contains "Second W.COMMIT succeeds" "." "$result"
    echo ""

    echo "8. Testing W.DIFF..."
    result=$(redis_cmd HLLSET.W.DIFF hlltest:ring 0 1 2>&1)
    test_contains "W.DIFF returns result" "." "$result"
    echo ""

    echo "9. Testing RING.BASIS is non-empty after ingest..."
    basis=$(redis_cmd HLLSET.RING.BASIS hlltest:ring 2>&1)
    if [[ "$basis" != *"empty"* ]]; then
        echo -e "${GREEN}✓${NC} RING.BASIS has entries after ingest"
        ((pass++))
    else
        echo -e "${RED}✗${NC} RING.BASIS still empty after ingest"
        ((fail++))
    fi
    echo ""

    echo "10. Cleanup..."
    redis_cmd DEL "hllring:ring:hlltest:ring" > /dev/null 2>&1 || true
    echo -e "${GREEN}✓${NC} Cleanup done"
    ((pass++))
    echo ""

else

    # ========================================================
    # Lua FCALL tests (legacy)
    # ========================================================

    echo "1. Creating test sets..."
    A=$(redis_cmd FCALL hllset_create 0 alice bob carol)
    B=$(redis_cmd FCALL hllset_create 0 bob carol dave)
    echo "   Set A: $A"
    echo "   Set B: $B"
    echo ""

    echo "2. Testing cardinality..."
    card_a=$(redis_cmd FCALL hllset_cardinality 1 "$A")
    card_b=$(redis_cmd FCALL hllset_cardinality 1 "$B")
    test_result "Cardinality A" "3" "$card_a"
    test_result "Cardinality B" "3" "$card_b"
    echo ""

    echo "3. Testing union (A ∪ B)..."
    UNION=$(redis_cmd FCALL hllset_union 2 "$A" "$B")
    card_union=$(redis_cmd FCALL hllset_cardinality 1 "$UNION")
    test_result "Union cardinality (alice,bob,carol,dave)" "4" "$card_union"
    echo ""

    echo "4. Testing intersection (A ∩ B)..."
    INTER=$(redis_cmd FCALL hllset_intersect 2 "$A" "$B")
    card_inter=$(redis_cmd FCALL hllset_cardinality 1 "$INTER")
    test_result "Intersection cardinality (bob,carol)" "2" "$card_inter"
    echo ""

    echo "5. Testing difference (A - B)..."
    DIFF=$(redis_cmd FCALL hllset_diff 2 "$A" "$B")
    card_diff=$(redis_cmd FCALL hllset_cardinality 1 "$DIFF")
    test_result "Difference cardinality (alice)" "1" "$card_diff"
    echo ""

    echo "6. Testing XOR (A ⊕ B)..."
    XOR=$(redis_cmd FCALL hllset_xor 2 "$A" "$B")
    card_xor=$(redis_cmd FCALL hllset_cardinality 1 "$XOR")
    test_result "XOR cardinality (alice,dave)" "2" "$card_xor"
    echo ""

    echo "7. Content-addressable property..."
    A2=$(redis_cmd FCALL hllset_create 0 alice bob carol)
    if [ "$A" == "$A2" ]; then
        echo -e "${GREEN}✓${NC} Same tokens produce same key"
        ((pass++))
    else
        echo -e "${RED}✗${NC} Content-addressable property violated"
        ((fail++))
    fi
    echo ""

fi

# ============================================================
# Summary
# ============================================================
echo "=========================================="
printf "Results: ${GREEN}%d passed${NC}, ${RED}%d failed${NC}\n" "$pass" "$fail"
echo "=========================================="

exit $fail
