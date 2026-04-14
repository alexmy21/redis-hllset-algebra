#!/bin/bash
# Clean up temporary and orphaned HLLSet keys
#
# Usage: ./cleanup.sh [--dry-run] [--all]
#
# Options:
#   --dry-run   Show what would be deleted without deleting
#   --all       Delete ALL hllset keys (use with caution!)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

DRY_RUN=false
DELETE_ALL=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --all) DELETE_ALL=true ;;
    esac
done

echo "═══════════════════════════════════════"
echo "  HLLSet Cleanup"
echo "═══════════════════════════════════════"
echo ""

# Count keys before
temp_keys=$(redis_cmd KEYS "hllset:temp:*")
temp_count=$(echo "$temp_keys" | grep -c "hllset:" || echo 0)

if [ "$DELETE_ALL" == "true" ]; then
    all_keys=$(redis_cmd KEYS "hllset:*")
    all_count=$(echo "$all_keys" | grep -c "hllset:" || echo 0)
    
    log_warn "DELETE ALL mode: $all_count total hllset keys"
    
    if [ "$DRY_RUN" == "true" ]; then
        log_info "Dry run - would delete:"
        echo "$all_keys" | head -20
        [ "$all_count" -gt 20 ] && echo "  ... and $((all_count - 20)) more"
    else
        read -p "Are you sure you want to delete ALL $all_count keys? [y/N] " confirm
        if [ "$confirm" == "y" ] || [ "$confirm" == "Y" ]; then
            if [ "$all_count" -gt 0 ]; then
                echo "$all_keys" | xargs -r redis_cmd DEL > /dev/null
                log_ok "Deleted $all_count keys"
            fi
        else
            log_info "Cancelled"
        fi
    fi
else
    # Only delete temp keys
    log_info "Found $temp_count temporary keys"
    
    if [ "$temp_count" -eq 0 ]; then
        log_ok "No temporary keys to clean up"
        exit 0
    fi
    
    if [ "$DRY_RUN" == "true" ]; then
        log_info "Dry run - would delete:"
        echo "$temp_keys" | head -20
        [ "$temp_count" -gt 20 ] && echo "  ... and $((temp_count - 20)) more"
    else
        echo "$temp_keys" | xargs -r redis_cmd DEL > /dev/null
        log_ok "Deleted $temp_count temporary keys"
    fi
fi

# Show remaining counts
echo ""
log_info "Remaining keys:"
echo "   hllset:*      $(redis_cmd KEYS 'hllset:*' | grep -v temp: | grep -v meta: | wc -l)"
echo "   hllset:meta:* $(redis_cmd KEYS 'hllset:meta:*' | wc -l)"
echo "   hllset:temp:* $(redis_cmd KEYS 'hllset:temp:*' | wc -l)"
