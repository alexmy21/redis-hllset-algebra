#!/bin/bash
# Wrapper for redis-cli with project defaults
#
# Usage: ./redis-cli.sh [redis-cli args...]
#
# Examples:
#   ./redis-cli.sh PING
#   ./redis-cli.sh FCALL hllset_create 0 hello world
#   ./redis-cli.sh                      # Interactive mode

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh" 2>/dev/null || true

HOST="${REDIS_HOST:-localhost}"
PORT="${REDIS_PORT:-6379}"

exec redis-cli -h "$HOST" -p "$PORT" "$@"
