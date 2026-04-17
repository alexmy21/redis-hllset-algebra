#!/bin/bash
# Start Redis container with HLLSet modules using Podman
#
# Usage: ./start_redis.sh [build]
#
# Options:
#   build   - Rebuild the container image before starting

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
CONTAINER_NAME="${CONTAINER_NAME:-redis-server}"
IMAGE_NAME="${IMAGE_NAME:-localhost/redis-hllset:latest}"

cd "$REDIS_DIR"

# Check if we should rebuild
if [ "$1" == "build" ] || ! podman image exists "$IMAGE_NAME"; then
    echo "Building Redis image with modules..."
    podman build -t "$IMAGE_NAME" .
fi

# Stop existing container if running
if podman container exists "$CONTAINER_NAME"; then
    echo "Stopping existing container..."
    podman stop "$CONTAINER_NAME" 2>/dev/null || true
    podman rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Start container
echo "Starting Redis container..."
podman run -d \
    --name "$CONTAINER_NAME" \
    -p 6379:6379 \
    -v redis-hllset-data:/data:Z \
    -v "$REDIS_DIR/functions:/usr/local/lib/redis_hllset/functions:ro,Z" \
    --restart unless-stopped \
    "$IMAGE_NAME"

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
for i in {1..30}; do
    if podman exec "$CONTAINER_NAME" redis-cli ping 2>/dev/null | grep -q PONG; then
        echo "Redis is ready!"
        break
    fi
    sleep 1
done

# Show status
echo ""
echo "Container status:"
podman ps --filter name="$CONTAINER_NAME"

echo ""
echo "Redis modules loaded:"
podman exec "$CONTAINER_NAME" redis-cli MODULE LIST

echo ""
echo "To load HLLSet functions, run:"
echo "  ./load_functions.sh"
