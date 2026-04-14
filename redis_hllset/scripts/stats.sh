#!/bin/bash
# Show Redis and HLLSet statistics
#
# Usage: ./stats.sh [--watch]
#
# Options:
#   --watch    Refresh stats every 2 seconds

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

show_stats() {
    clear
    echo "═══════════════════════════════════════════════════════════════"
    echo "  HLLSet Algebra - Redis Statistics"
    echo "  $(date '+%Y-%m-%d %H:%M:%S') @ $REDIS_HOST:$REDIS_PORT"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    
    # Connection info
    echo "${BLUE}Server${NC}"
    echo "───────────────────────────────────────────────────────────────"
    redis_cmd INFO server | grep -E "redis_version|uptime_in_days|connected_clients" | sed 's/^/  /'
    echo ""
    
    # Memory
    echo "${BLUE}Memory${NC}"
    echo "───────────────────────────────────────────────────────────────"
    redis_cmd INFO memory | grep -E "used_memory_human|used_memory_peak_human|maxmemory_human" | sed 's/^/  /'
    echo ""
    
    # Keys
    echo "${BLUE}HLLSet Keys${NC}"
    echo "───────────────────────────────────────────────────────────────"
    hllset_count=$(redis_cmd KEYS "hllset:*" | grep -v "temp:" | grep -v "meta:" | wc -l)
    meta_count=$(redis_cmd KEYS "hllset:meta:*" | wc -l)
    temp_count=$(redis_cmd KEYS "hllset:temp:*" | wc -l)
    printf "  %-20s %d\n" "HLLSet objects:" "$hllset_count"
    printf "  %-20s %d\n" "Metadata keys:" "$meta_count"
    printf "  %-20s %d\n" "Temporary keys:" "$temp_count"
    echo ""
    
    # Functions
    echo "${BLUE}Functions${NC}"
    echo "───────────────────────────────────────────────────────────────"
    func_count=$(redis_cmd FUNCTION LIST LIBRARYNAME hllset_lib 2>/dev/null | grep -c "name" || echo 0)
    printf "  %-20s %d\n" "Registered:" "$func_count"
    echo ""
    
    # Modules
    echo "${BLUE}Modules${NC}"
    echo "───────────────────────────────────────────────────────────────"
    redis_cmd MODULE LIST | grep -E "name|ver" | paste - - | sed 's/^/  /'
    echo ""
    
    # Command stats (last 10)
    echo "${BLUE}Recent Command Stats${NC}"
    echo "───────────────────────────────────────────────────────────────"
    redis_cmd INFO commandstats | grep -E "fcall|hllset" | head -5 | sed 's/^/  /'
    echo ""
}

if [ "$1" == "--watch" ]; then
    while true; do
        show_stats
        echo "  Refreshing every ${WATCH_INTERVAL}s... Press Ctrl+C to stop"
        sleep "$WATCH_INTERVAL"
    done
else
    show_stats
fi
