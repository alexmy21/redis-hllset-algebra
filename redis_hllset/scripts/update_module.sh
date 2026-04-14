#!/bin/bash
# Update HLLSet module from build directory to deploy directory
#
# This script:
#   1. Copies the built module to deploy/ for Docker image building
#   2. Optionally rebuilds the Docker image
#   3. Optionally restarts the container
#
# Usage:
#   ./update_module.sh              # Just copy the module
#   ./update_module.sh --build      # Copy and rebuild Docker image
#   ./update_module.sh --restart    # Copy and restart container (dev mode)
#   ./update_module.sh --all        # Copy, rebuild, and restart

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# Paths
MODULE_BUILD="$REDIS_DIR/module/target/release/libredis_hllset.so"
MODULE_DEV="$REDIS_DIR/module/libredis_hllset.so"
DEPLOY_DIR="$REDIS_DIR/deploy"

# Options
BUILD_IMAGE=false
RESTART_CONTAINER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build|-b)   BUILD_IMAGE=true; shift ;;
        --restart|-r) RESTART_CONTAINER=true; shift ;;
        --all|-a)     BUILD_IMAGE=true; RESTART_CONTAINER=true; shift ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --build, -b     Rebuild Docker image after copying"
            echo "  --restart, -r   Restart container after copying (dev mode)"
            echo "  --all, -a       Build image and restart container"
            echo "  --help, -h      Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Step 1: Find and copy the module
log_info "Looking for built module..."

if [ -f "$MODULE_BUILD" ]; then
    SOURCE="$MODULE_BUILD"
    log_ok "Found release build: $MODULE_BUILD"
elif [ -f "$MODULE_DEV" ]; then
    SOURCE="$MODULE_DEV"
    log_ok "Found dev build: $MODULE_DEV"
else
    log_error "No module found. Build it first with:"
    echo "  cd $REDIS_DIR/module && cargo build --release"
    exit 1
fi

# Create deploy directory if needed
mkdir -p "$DEPLOY_DIR"

# Copy the module
cp "$SOURCE" "$DEPLOY_DIR/libredis_hllset.so"
log_ok "Module copied to: $DEPLOY_DIR/libredis_hllset.so"

# Show module info
ls -lh "$DEPLOY_DIR/libredis_hllset.so"

# Step 2: Build Docker image (if requested)
if $BUILD_IMAGE; then
    log_info "Building Docker image..."
    cd "$REDIS_DIR"
    
    $CONTAINER_RUNTIME build -t localhost/redis-hllset:latest .
    log_ok "Docker image built: localhost/redis-hllset:latest"
fi

# Step 3: Restart container (if requested)
if $RESTART_CONTAINER; then
    log_info "Restarting container..."
    
    if $CONTAINER_RUNTIME container exists "$CONTAINER_NAME" 2>/dev/null; then
        $CONTAINER_RUNTIME restart "$CONTAINER_NAME"
        log_ok "Container restarted: $CONTAINER_NAME"
        
        # Wait for Redis to be ready
        log_info "Waiting for Redis..."
        for i in {1..30}; do
            if $CONTAINER_RUNTIME exec "$CONTAINER_NAME" redis-cli ping 2>/dev/null | grep -q PONG; then
                log_ok "Redis is ready"
                break
            fi
            sleep 1
        done
    else
        log_warn "Container not running: $CONTAINER_NAME"
        log_info "Start with: $CONTAINER_RUNTIME run -d --name $CONTAINER_NAME -p 6379:6379 localhost/redis-hllset:latest"
    fi
fi

echo ""
log_ok "Module update complete!"
