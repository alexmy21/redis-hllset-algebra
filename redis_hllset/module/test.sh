#!/bin/bash
# Integration tests for Redis HLLSet module
#
# Run after loading the module into Redis:
#   redis-server --loadmodule ./libredis_hllset.so
#
# Usage: ./test.sh [redis-host] [redis-port]

set -e

REDIS_HOST="${1:-127.0.0.1}"
REDIS_PORT="${2:-6379}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

PASSED=0
FAILED=0

redis_cmd() {
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" "$@"
}

test_pass() {
    echo -e "${GREEN}✓${NC} $1"
    PASSED=$((PASSED + 1))
}

test_fail() {
    echo -e "${RED}✗${NC} $1: $2"
    FAILED=$((FAILED + 1))
}

assert_eq() {
    if [ "$1" = "$2" ]; then
        test_pass "$3"
    else
        test_fail "$3" "expected '$2', got '$1'"
    fi
}

assert_between() {
    local val="$1"
    local min="$2"
    local max="$3"
    local msg="$4"
    
    if (( $(echo "$val >= $min && $val <= $max" | bc -l) )); then
        test_pass "$msg"
    else
        test_fail "$msg" "expected $min-$max, got $val"
    fi
}

echo ""
echo "=========================================="
echo "  HLLSet Redis Module Tests"
echo "=========================================="
echo ""

# Check module is loaded
echo "1. Checking module is loaded..."
MODULES=$(redis_cmd MODULE LIST | grep -c "hllset" || echo "0")
if [ "$MODULES" -gt 0 ]; then
    test_pass "HLLSet module loaded"
else
    test_fail "HLLSet module not loaded" "Run: redis-server --loadmodule ./libredis_hllset.so"
    exit 1
fi

# Cleanup
redis_cmd DEL hllset:test:* 2>/dev/null || true

# Test CREATE
echo ""
echo "2. Testing HLLSET.CREATE..."
KEY_A=$(redis_cmd HLLSET.CREATE alice bob carol)
KEY_B=$(redis_cmd HLLSET.CREATE bob carol dave)

if [[ "$KEY_A" == hllset:* ]]; then
    test_pass "CREATE returns hllset: key"
else
    test_fail "CREATE key format" "got: $KEY_A"
fi

# Test content-addressable
KEY_A2=$(redis_cmd HLLSET.CREATE carol alice bob)  # Same tokens, different order
assert_eq "$KEY_A" "$KEY_A2" "Content-addressable keys match"

# Test CARD
echo ""
echo "3. Testing HLLSET.CARD..."
CARD_A=$(redis_cmd HLLSET.CARD "$KEY_A")
assert_between "$CARD_A" 2.5 3.5 "Cardinality A ≈ 3"

CARD_B=$(redis_cmd HLLSET.CARD "$KEY_B")
assert_between "$CARD_B" 2.5 3.5 "Cardinality B ≈ 3"

# Test UNION
echo ""
echo "4. Testing HLLSET.UNION..."
UNION_KEY=$(redis_cmd HLLSET.UNION "$KEY_A" "$KEY_B")
UNION_CARD=$(redis_cmd HLLSET.CARD "$UNION_KEY")
assert_between "$UNION_CARD" 3.5 4.5 "Union cardinality ≈ 4"

# Test INTER
echo ""
echo "5. Testing HLLSET.INTER..."
INTER_KEY=$(redis_cmd HLLSET.INTER "$KEY_A" "$KEY_B")
INTER_CARD=$(redis_cmd HLLSET.CARD "$INTER_KEY")
assert_between "$INTER_CARD" 1.5 2.5 "Intersection cardinality ≈ 2"

# Test DIFF
echo ""
echo "6. Testing HLLSET.DIFF..."
DIFF_KEY=$(redis_cmd HLLSET.DIFF "$KEY_A" "$KEY_B")
DIFF_CARD=$(redis_cmd HLLSET.CARD "$DIFF_KEY")
assert_between "$DIFF_CARD" 0.5 1.5 "Difference cardinality ≈ 1"

# Test XOR
echo ""
echo "7. Testing HLLSET.XOR..."
XOR_KEY=$(redis_cmd HLLSET.XOR "$KEY_A" "$KEY_B")
XOR_CARD=$(redis_cmd HLLSET.CARD "$XOR_KEY")
assert_between "$XOR_CARD" 1.5 2.5 "XOR cardinality ≈ 2"

# Test SIM (Jaccard similarity)
echo ""
echo "8. Testing HLLSET.SIM..."
SIM=$(redis_cmd HLLSET.SIM "$KEY_A" "$KEY_B")
assert_between "$SIM" 0.3 0.7 "Jaccard similarity ≈ 0.5"

# Test INFO
echo ""
echo "9. Testing HLLSET.INFO..."
INFO=$(redis_cmd HLLSET.INFO "$KEY_A")
if echo "$INFO" | grep -q "cardinality"; then
    test_pass "INFO returns cardinality"
else
    test_fail "INFO missing cardinality" "$INFO"
fi

# Test EXISTS
echo ""
echo "10. Testing HLLSET.EXISTS..."
EXISTS=$(redis_cmd HLLSET.EXISTS "$KEY_A")
assert_eq "$EXISTS" "1" "EXISTS returns 1 for existing key"

NOT_EXISTS=$(redis_cmd HLLSET.EXISTS "hllset:nonexistent")
assert_eq "$NOT_EXISTS" "0" "EXISTS returns 0 for non-existing key"

# Test DEL
echo ""
echo "11. Testing HLLSET.DEL..."
redis_cmd HLLSET.DEL "$UNION_KEY" >/dev/null
AFTER_DEL=$(redis_cmd HLLSET.EXISTS "$UNION_KEY")
assert_eq "$AFTER_DEL" "0" "DEL removes key"

# Test MERGE
echo ""
echo "12. Testing HLLSET.MERGE..."
MERGE_DEST="hllset:merge:test"
redis_cmd HLLSET.CREATE a b > /dev/null
KEY_C=$(redis_cmd HLLSET.CREATE c d)
KEY_D=$(redis_cmd HLLSET.CREATE e f)

redis_cmd DEL "$MERGE_DEST" 2>/dev/null || true
redis_cmd HLLSET.MERGE "$MERGE_DEST" "$KEY_A" "$KEY_C" "$KEY_D" >/dev/null
MERGE_CARD=$(redis_cmd HLLSET.CARD "$MERGE_DEST")
assert_between "$MERGE_CARD" 5 8 "MERGE cardinality reasonable"

# Test large set accuracy
echo ""
echo "13. Testing large set accuracy..."
TOKENS=""
for i in $(seq 1 1000); do
    TOKENS="$TOKENS token_$i"
done
LARGE_KEY=$(redis_cmd HLLSET.CREATE $TOKENS)
LARGE_CARD=$(redis_cmd HLLSET.CARD "$LARGE_KEY")
ERROR=$(echo "scale=4; ($LARGE_CARD - 1000) / 1000 * 100" | bc)
ERROR_ABS=$(echo "$ERROR" | tr -d -)

if (( $(echo "$ERROR_ABS < 5" | bc -l) )); then
    test_pass "Large set error < 5% (got ${ERROR}%)"
else
    test_fail "Large set error" "expected < 5%, got ${ERROR}%"
fi

# Test empty set
echo ""
echo "14. Testing empty operations..."
EMPTY_CARD=$(redis_cmd HLLSET.CARD "hllset:nonexistent")
assert_eq "$EMPTY_CARD" "0" "Non-existent key has cardinality 0"

# Cleanup
echo ""
echo "Cleaning up test keys..."
redis_cmd KEYS "hllset:*" | xargs -r redis_cmd DEL 2>/dev/null || true

# Results
echo ""
echo "=========================================="
echo "  Results: $PASSED passed, $FAILED failed"
echo "=========================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi
