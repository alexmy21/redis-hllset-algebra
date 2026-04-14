#!/bin/bash
# Backup HLLSet functions and data
#
# Usage: ./backup.sh [backup_dir]
#
# Creates:
#   - functions.lua     - Current Lua library
#   - functions.rdb     - Redis function dump
#   - data.rdb          - Full Redis dump (if requested)
#   - manifest.json     - Backup metadata

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

BACKUP_DIR="${1:-$REDIS_DIR/backups}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
BACKUP_PATH="$BACKUP_DIR/$TIMESTAMP"

echo "═══════════════════════════════════════"
echo "  HLLSet Backup"
echo "═══════════════════════════════════════"
echo ""

mkdir -p "$BACKUP_PATH"
log_info "Backup directory: $BACKUP_PATH"

# Backup 1: Copy Lua source file
if [ -f "$FUNCTIONS_DIR/hllset.lua" ]; then
    cp "$FUNCTIONS_DIR/hllset.lua" "$BACKUP_PATH/hllset.lua"
    log_ok "Backed up Lua source"
fi

# Backup 2: Dump functions from Redis
log_info "Dumping functions from Redis..."
redis_cmd FUNCTION DUMP > "$BACKUP_PATH/functions.rdb" 2>/dev/null
if [ -s "$BACKUP_PATH/functions.rdb" ]; then
    log_ok "Backed up Redis functions"
else
    log_warn "Function dump is empty"
    rm -f "$BACKUP_PATH/functions.rdb"
fi

# Backup 3: Export HLLSet keys info
log_info "Exporting HLLSet key info..."
{
    echo "# HLLSet Keys Export"
    echo "# Generated: $(date)"
    echo ""
    
    for key in $(redis_cmd KEYS "hllset:*" | grep -v "temp:" | grep -v "meta:"); do
        card=$(redis_cmd FCALL hllset_cardinality 1 "$key" 2>/dev/null)
        echo "$key $card"
    done
} > "$BACKUP_PATH/keys.txt"
log_ok "Exported key list"

# Backup 4: Create manifest
cat > "$BACKUP_PATH/manifest.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "date": "$(date -Iseconds)",
    "redis_host": "$REDIS_HOST",
    "redis_port": "$REDIS_PORT",
    "redis_version": "$(redis_cmd INFO server | grep redis_version | cut -d: -f2 | tr -d '\r')",
    "hllset_keys": $(redis_cmd KEYS "hllset:*" | grep -v temp: | grep -v meta: | wc -l),
    "meta_keys": $(redis_cmd KEYS "hllset:meta:*" | wc -l),
    "files": [
        "hllset.lua",
        "functions.rdb",
        "keys.txt",
        "manifest.json"
    ]
}
EOF
log_ok "Created manifest"

# Summary
echo ""
echo "═══════════════════════════════════════"
log_ok "Backup complete: $BACKUP_PATH"
ls -la "$BACKUP_PATH"
echo "═══════════════════════════════════════"

# Create latest symlink
ln -sfn "$BACKUP_PATH" "$BACKUP_DIR/latest"
