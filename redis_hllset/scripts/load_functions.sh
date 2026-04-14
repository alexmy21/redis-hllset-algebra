#!/bin/bash
# Load HLLSet Algebra functions into Redis
#
# Usage: ./load_functions.sh [host] [port]
#
# Examples:
#   ./load_functions.sh                    # localhost:6379
#   ./load_functions.sh redis.local 6380   # custom host/port

set -e

HOST=${1:-localhost}
PORT=${2:-6379}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUNCTIONS_DIR="$SCRIPT_DIR/../functions"

echo "Loading HLLSet Algebra functions into Redis at $HOST:$PORT"
echo "=========================================================="

# Load hllset.lua
echo "Loading hllset.lua..."
result=$(redis-cli -h "$HOST" -p "$PORT" FUNCTION LOAD REPLACE "$(cat "$FUNCTIONS_DIR/hllset.lua")" 2>&1)
if [[ "$result" == *"hllset_lib"* ]]; then
    echo "✓ Loaded hllset_lib successfully"
else
    echo "✗ Failed to load: $result"
    exit 1
fi

echo ""
echo "Verifying loaded functions..."
redis-cli -h "$HOST" -p "$PORT" FUNCTION LIST LIBRARYNAME hllset_lib | grep -E "name|library"

echo ""
echo "Done! Functions loaded successfully."
echo ""
echo "Quick test:"
echo "  redis-cli FCALL hllset_create 0 hello world test"
