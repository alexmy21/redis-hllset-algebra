#!/bin/bash
# Test HLLSet functions in Redis
#
# Usage: ./test_hllset.sh

# Don't use set -e as we want to continue on failures

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  HLLSet Algebra Redis Function Tests"
echo "=========================================="
echo ""

pass=0
fail=0

test_result() {
    local name="$1"
    local expected="$2"
    local actual="$3"
    
    # Strip "(integer) " prefix if present for numeric comparison
    local actual_clean=$(echo "$actual" | sed 's/(integer) //')
    local expected_clean=$(echo "$expected" | sed 's/(integer) //')
    
    if [ "$expected_clean" == "$actual_clean" ]; then
        echo -e "${GREEN}✓${NC} $name: $actual_clean"
        ((pass++))
    else
        echo -e "${RED}✗${NC} $name: expected $expected_clean, got $actual_clean"
        ((fail++))
    fi
}

# Clean up any existing test keys
redis-cli DEL hllset:test:* > /dev/null 2>&1 || true

# Test 1: Create sets
echo "1. Creating test sets..."
A=$(redis-cli FCALL hllset_create 0 alice bob carol)
B=$(redis-cli FCALL hllset_create 0 bob carol dave)
echo "   Set A (alice, bob, carol): $A"
echo "   Set B (bob, carol, dave): $B"
echo ""

# Test 2: Cardinality
echo "2. Testing cardinality..."
card_a=$(redis-cli FCALL hllset_cardinality 1 "$A")
card_b=$(redis-cli FCALL hllset_cardinality 1 "$B")
test_result "Cardinality A" "3" "$card_a"
test_result "Cardinality B" "3" "$card_b"
echo ""

# Test 3: Union
echo "3. Testing union (A ∪ B)..."
UNION=$(redis-cli FCALL hllset_union 2 "$A" "$B")
card_union=$(redis-cli FCALL hllset_cardinality 1 "$UNION")
test_result "Union cardinality (alice,bob,carol,dave)" "4" "$card_union"
echo ""

# Test 4: Intersection
echo "4. Testing intersection (A ∩ B)..."
INTER=$(redis-cli FCALL hllset_intersect 2 "$A" "$B")
card_inter=$(redis-cli FCALL hllset_cardinality 1 "$INTER")
test_result "Intersection cardinality (bob,carol)" "2" "$card_inter"
echo ""

# Test 5: Difference
echo "5. Testing difference (A - B)..."
DIFF=$(redis-cli FCALL hllset_diff 2 "$A" "$B")
card_diff=$(redis-cli FCALL hllset_cardinality 1 "$DIFF")
test_result "Difference cardinality (alice)" "1" "$card_diff"
echo ""

# Test 6: XOR (symmetric difference)
echo "6. Testing XOR (A ⊕ B)..."
XOR=$(redis-cli FCALL hllset_xor 2 "$A" "$B")
card_xor=$(redis-cli FCALL hllset_cardinality 1 "$XOR")
test_result "XOR cardinality (alice,dave)" "2" "$card_xor"
echo ""

# Test 7: Similarity
echo "7. Testing Jaccard similarity..."
sim=$(redis-cli FCALL hllset_similarity 2 "$A" "$B")
test_result "Jaccard similarity (2/4=0.5)" "0.5" "$sim"
echo ""

# Test 8: Content-addressable property
echo "8. Testing content-addressable property..."
A2=$(redis-cli FCALL hllset_create 0 alice bob carol)
if [ "$A" == "$A2" ]; then
    echo -e "${GREEN}✓${NC} Same tokens produce same key"
    ((pass++))
else
    echo -e "${RED}✗${NC} Content-addressable property violated"
    ((fail++))
fi
echo ""

# Test 9: Info
echo "9. Testing info..."
info=$(redis-cli FCALL hllset_info 1 "$A")
if [[ "$info" == *"cardinality"* ]] && [[ "$info" == *"sha1"* ]]; then
    echo -e "${GREEN}✓${NC} Info returns expected fields"
    ((pass++))
else
    echo -e "${RED}✗${NC} Info missing expected fields"
    ((fail++))
fi
echo ""

# Test 10: Empty set
echo "10. Testing empty set..."
EMPTY=$(redis-cli FCALL hllset_create 0)
card_empty=$(redis-cli FCALL hllset_cardinality 1 "$EMPTY")
test_result "Empty set cardinality" "0" "$card_empty"
echo ""

# Test 11: Large set
echo "11. Testing larger set..."
LARGE=$(redis-cli FCALL hllset_create 0 $(seq 1 100 | tr '\n' ' '))
card_large=$(redis-cli FCALL hllset_cardinality 1 "$LARGE")
echo "   Large set (100 items) cardinality: $card_large"
# HLL has ~2-3% error, so 95-105 is acceptable
card_val=$(echo "$card_large" | grep -o '[0-9]*')
if [ "$card_val" -ge 95 ] && [ "$card_val" -le 105 ]; then
    echo -e "${GREEN}✓${NC} Large set cardinality within expected range (95-105)"
    ((pass++))
else
    echo -e "${RED}✗${NC} Large set cardinality $card_val outside expected range"
    ((fail++))
fi
echo ""

# Summary
echo "=========================================="
printf "Results: ${GREEN}%d passed${NC}, ${RED}%d failed${NC}\n" "$pass" "$fail"
echo "=========================================="

exit $fail
