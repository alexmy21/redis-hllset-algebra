#!/bin/bash
# Restore HLLSet functions from backup
#
# Usage: ./restore.sh [backup_dir]
#
# If no backup_dir specified, uses latest backup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

BACKUP_DIR="${1:-$REDIS_DIR/backups/latest}"

echo "═══════════════════════════════════════"
echo "  HLLSet Restore"
echo "═══════════════════════════════════════"
echo ""

if [ ! -d "$BACKUP_DIR" ]; then
    log_error "Backup not found: $BACKUP_DIR"
    echo ""
    echo "Available backups:"
    ls -1 "$REDIS_DIR/backups" 2>/dev/null || echo "  (none)"
    exit 1
fi

# Resolve symlink
BACKUP_DIR=$(readlink -f "$BACKUP_DIR")
log_info "Restoring from: $BACKUP_DIR"

# Show manifest
if [ -f "$BACKUP_DIR/manifest.json" ]; then
    echo ""
    cat "$BACKUP_DIR/manifest.json"
    echo ""
fi

read -p "Continue with restore? [y/N] " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    log_info "Cancelled"
    exit 0
fi

# Restore Lua source
if [ -f "$BACKUP_DIR/hllset.lua" ]; then
    cp "$BACKUP_DIR/hllset.lua" "$FUNCTIONS_DIR/hllset.lua"
    log_ok "Restored Lua source file"
fi

# Reload functions
log_info "Reloading functions into Redis..."
"$SCRIPT_DIR/load_functions.sh"

# Verify
log_info "Verifying restore..."
"$SCRIPT_DIR/health_check.sh" --quiet && log_ok "Health check passed" || log_error "Health check failed"

echo ""
log_ok "Restore complete"
