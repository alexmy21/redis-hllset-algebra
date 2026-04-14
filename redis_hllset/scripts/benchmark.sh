#!/bin/bash
# Benchmark HLLSet operations
#
# Usage: ./benchmark.sh [iterations]
#
# Default: 100 iterations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

ITERATIONS="${1:-100}"

echo "═══════════════════════════════════════════════════════════════"
echo "  HLLSet Benchmark - $ITERATIONS iterations"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Helper for timing
time_operation() {
    local name="$1"
    shift
    local start=$(date +%s%N)
    for i in $(seq 1 "$ITERATIONS"); do
        "$@" > /dev/null 2>&1
    done
    local end=$(date +%s%N)
    local total_ms=$(( (end - start) / 1000000 ))
    local avg_us=$(( (end - start) / ITERATIONS / 1000 ))
    printf "  %-30s %6d ms total, %6d μs/op\n" "$name" "$total_ms" "$avg_us"
}

# Create test data
log_info "Creating test data..."
A=$(redis_cmd FCALL hllset_create 0 $(seq 1 50))
B=$(redis_cmd FCALL hllset_create 0 $(seq 25 75))
echo ""

echo "${BLUE}Operations:${NC}"
echo "───────────────────────────────────────────────────────────────"

# Benchmark each operation
time_operation "hllset_create (10 tokens)" \
    redis_cmd FCALL hllset_create 0 a b c d e f g h i j

time_operation "hllset_cardinality" \
    redis_cmd FCALL hllset_cardinality 1 "$A"

time_operation "hllset_union" \
    redis_cmd FCALL hllset_union 2 "$A" "$B"

time_operation "hllset_intersect" \
    redis_cmd FCALL hllset_intersect 2 "$A" "$B"

time_operation "hllset_diff" \
    redis_cmd FCALL hllset_diff 2 "$A" "$B"

time_operation "hllset_xor" \
    redis_cmd FCALL hllset_xor 2 "$A" "$B"

time_operation "hllset_similarity" \
    redis_cmd FCALL hllset_similarity 2 "$A" "$B"

time_operation "hllset_info" \
    redis_cmd FCALL hllset_info 1 "$A"

echo ""
echo "${BLUE}Scaling:${NC}"
echo "───────────────────────────────────────────────────────────────"

# Test creation with different sizes
for size in 10 100 1000; do
    tokens=$(seq 1 $size | tr '\n' ' ')
    start=$(date +%s%N)
    redis_cmd FCALL hllset_create 0 $tokens > /dev/null
    end=$(date +%s%N)
    ms=$(( (end - start) / 1000000 ))
    printf "  Create with %4d tokens:     %6d ms\n" "$size" "$ms"
done

echo ""

# Cleanup temp keys
log_info "Cleaning up..."
"$SCRIPT_DIR/cleanup.sh" --dry-run 2>/dev/null | tail -5

echo ""
log_ok "Benchmark complete"
