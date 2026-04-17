#!/bin/bash
# Environment configuration for Redis HLLSet development
#
# Source this file in other scripts or your shell:
#   source env.sh
#
# Override defaults by setting environment variables:
#   REDIS_HOST=192.168.1.100 ./health_check.sh

# Redis connection
export REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
export REDIS_PORT="${REDIS_PORT:-6379}"
export REDIS_PASSWORD="${REDIS_PASSWORD:-}"

# Container settings
export CONTAINER_NAME="${CONTAINER_NAME:-redis-server}"
export CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-podman}"  # podman or docker
export IMAGE_NAME="${IMAGE_NAME:-localhost/redis-hllset:latest}"

# Project paths
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export REDIS_DIR="$PROJECT_ROOT/redis_hllset"
export FUNCTIONS_DIR="$REDIS_DIR/functions"
export DATA_DIR="$REDIS_DIR/data"
export BACKUP_DIR="$REDIS_DIR/backups"

# Development settings
export WATCH_INTERVAL="${WATCH_INTERVAL:-2}"  # seconds between file checks

# Colors for output
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[0;33m'
export BLUE='\033[0;34m'
export NC='\033[0m'  # No Color

# Helper function: run redis-cli with auth if needed
redis_cmd() {
    if [ -n "$REDIS_PASSWORD" ]; then
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning "$@"
    else
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" "$@"
    fi
}

# Helper function: print status messages
log_info() { echo -e "${BLUE}ℹ${NC} $*"; }
log_ok() { echo -e "${GREEN}✓${NC} $*"; }
log_warn() { echo -e "${YELLOW}⚠${NC} $*"; }
log_error() { echo -e "${RED}✗${NC} $*"; }

# Export helpers
export -f redis_cmd log_info log_ok log_warn log_error
