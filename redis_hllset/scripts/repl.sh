#!/bin/bash
# Interactive HLLSet REPL helper
#
# Usage: ./repl.sh
#
# Provides shortcuts for common HLLSet operations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

show_help() {
    cat << 'EOF'
═══════════════════════════════════════════════════════════════
  HLLSet REPL - Interactive Redis Shell
═══════════════════════════════════════════════════════════════

Quick Commands (prefix with @):
  @create <tokens...>     Create HLLSet from tokens
  @card <key>             Get cardinality
  @union <key1> <key2>    Union of two sets
  @inter <key1> <key2>    Intersection of two sets
  @diff <key1> <key2>     Difference (key1 - key2)
  @sim <key1> <key2>      Jaccard similarity
  @info <key>             Get HLLSet info
  @list                   List all HLLSet keys
  @help                   Show this help

Examples:
  @create apple banana cherry
  @card hllset:abc123...
  @sim hllset:abc... hllset:def...
  
Or use raw Redis commands:
  FCALL hllset_create 0 hello world
  KEYS hllset:*
  
Press Ctrl+D to exit
═══════════════════════════════════════════════════════════════
EOF
}

process_shortcut() {
    local cmd="$1"
    shift
    
    case "$cmd" in
        @create)
            redis_cmd FCALL hllset_create 0 "$@"
            ;;
        @card)
            redis_cmd FCALL hllset_cardinality 1 "$1"
            ;;
        @union)
            redis_cmd FCALL hllset_union 2 "$1" "$2"
            ;;
        @inter)
            redis_cmd FCALL hllset_intersect 2 "$1" "$2"
            ;;
        @diff)
            redis_cmd FCALL hllset_diff 2 "$1" "$2"
            ;;
        @xor)
            redis_cmd FCALL hllset_xor 2 "$1" "$2"
            ;;
        @sim)
            redis_cmd FCALL hllset_similarity 2 "$1" "$2"
            ;;
        @info)
            redis_cmd FCALL hllset_info 1 "$1"
            ;;
        @dump)
            redis_cmd FCALL hllset_dump 1 "$1"
            ;;
        @del)
            redis_cmd FCALL hllset_delete 1 "$1"
            ;;
        @list)
            redis_cmd KEYS "hllset:*" | grep -v "temp:" | grep -v "meta:"
            ;;
        @count)
            redis_cmd KEYS "hllset:*" | grep -v "temp:" | grep -v "meta:" | wc -l
            ;;
        @help|@h|@?)
            show_help
            ;;
        *)
            echo "Unknown shortcut: $cmd (try @help)"
            ;;
    esac
}

# Show intro
show_help

# Start interactive loop
while true; do
    echo -ne "${GREEN}hllset>${NC} "
    read -r line || break
    
    [ -z "$line" ] && continue
    
    if [[ "$line" == @* ]]; then
        # Process shortcut
        # shellcheck disable=SC2086
        process_shortcut $line
    else
        # Pass through to redis-cli
        # shellcheck disable=SC2086
        redis_cmd $line
    fi
done

echo ""
log_info "Goodbye!"
