#!/bin/bash
# Watch Lua files and auto-reload on changes
#
# Usage: ./watch.sh [--test]
#
# Options:
#   --test    Run test suite after reload
#
# This creates a development loop:
#   1. Edit hllset.lua
#   2. Script detects change
#   3. Auto-reloads into Redis
#   4. Optionally runs tests
#
# Press Ctrl+C to stop

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

RUN_TESTS="${1:-}"
WATCH_FILE="$FUNCTIONS_DIR/hllset.lua"

echo "═══════════════════════════════════════════════════════"
echo "  HLLSet Development Watcher"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Watching: $WATCH_FILE"
echo "  Redis:    $REDIS_HOST:$REDIS_PORT"
echo "  Tests:    ${RUN_TESTS:-disabled}"
echo ""
echo "  Press Ctrl+C to stop"
echo "═══════════════════════════════════════════════════════"
echo ""

# Get initial checksum
if [ -f "$WATCH_FILE" ]; then
    last_checksum=$(md5sum "$WATCH_FILE" | cut -d' ' -f1)
else
    log_error "Watch file not found: $WATCH_FILE"
    exit 1
fi

log_info "Initial load..."
"$SCRIPT_DIR/load_functions.sh" > /dev/null && log_ok "Functions loaded"

reload_count=0

while true; do
    sleep "$WATCH_INTERVAL"
    
    if [ -f "$WATCH_FILE" ]; then
        current_checksum=$(md5sum "$WATCH_FILE" | cut -d' ' -f1)
        
        if [ "$current_checksum" != "$last_checksum" ]; then
            ((reload_count++))
            echo ""
            log_info "[$(date '+%H:%M:%S')] Change detected (reload #$reload_count)"
            
            # Reload functions
            if "$SCRIPT_DIR/load_functions.sh" > /dev/null 2>&1; then
                log_ok "Functions reloaded"
                
                # Run tests if requested
                if [ "$RUN_TESTS" == "--test" ]; then
                    echo ""
                    "$SCRIPT_DIR/test_hllset.sh"
                fi
            else
                log_error "Reload failed!"
                # Show error details
                "$SCRIPT_DIR/load_functions.sh" 2>&1 | head -20
            fi
            
            last_checksum="$current_checksum"
        fi
    fi
done
