#!/bin/bash
# Build script for Redis HLLSet module
#
# This script builds the Rust module and optionally creates a Docker image.
#
# Usage:
#   ./build.sh              # Build module only
#   ./build.sh --docker     # Build module and Docker image
#   ./build.sh --test       # Build and run tests
#   ./build.sh --release    # Build optimized release
#   ./build.sh --clean      # Clean build artifacts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ${NC} $*"; }
log_ok() { echo -e "${GREEN}✓${NC} $*"; }
log_warn() { echo -e "${YELLOW}⚠${NC} $*"; }
log_error() { echo -e "${RED}✗${NC} $*"; }

# Parse arguments
BUILD_TYPE="release"
BUILD_DOCKER=false
RUN_TESTS=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)     BUILD_TYPE="debug"; shift ;;
        --release)   BUILD_TYPE="release"; shift ;;
        --docker)    BUILD_DOCKER=true; shift ;;
        --test)      RUN_TESTS=true; shift ;;
        --clean)     CLEAN=true; shift ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --debug     Build debug version"
            echo "  --release   Build optimized release (default)"
            echo "  --docker    Build Docker image after compiling"
            echo "  --test      Run tests after building"
            echo "  --clean     Clean build artifacts"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Clean
if $CLEAN; then
    log_info "Cleaning build artifacts..."
    cargo clean
    rm -f *.so
    log_ok "Clean complete"
    exit 0
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     HLLSet Redis Module Build                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Rust toolchain
if ! command -v cargo &>/dev/null; then
    log_error "Rust/Cargo not found. Install from https://rustup.rs"
    exit 1
fi
log_ok "Rust toolchain: $(rustc --version)"

# Build
log_info "Building $BUILD_TYPE..."
if [ "$BUILD_TYPE" = "release" ]; then
    cargo build --release
    MODULE_PATH="target/release/libredis_hllset.so"
else
    cargo build
    MODULE_PATH="target/debug/libredis_hllset.so"
fi

# Check build output
if [ -f "$MODULE_PATH" ]; then
    log_ok "Module built: $MODULE_PATH"
    ls -lh "$MODULE_PATH"
    
    # Copy to current directory for easy access
    cp "$MODULE_PATH" ./libredis_hllset.so
    log_ok "Copied to ./libredis_hllset.so"
else
    log_error "Build failed - module not found"
    exit 1
fi

# Run tests
if $RUN_TESTS; then
    echo ""
    log_info "Running tests..."
    cargo test --$BUILD_TYPE
    log_ok "All tests passed"
fi

# Build Docker image
if $BUILD_DOCKER; then
    echo ""
    log_info "Building Docker image..."
    
    # Check for podman or docker
    if command -v podman &>/dev/null; then
        CONTAINER_CMD="podman"
    elif command -v docker &>/dev/null; then
        CONTAINER_CMD="docker"
    else
        log_error "Neither podman nor docker found"
        exit 1
    fi
    
    $CONTAINER_CMD build -t hllset-redis:latest .
    log_ok "Docker image built: hllset-redis:latest"
    
    # Show image info
    echo ""
    $CONTAINER_CMD images hllset-redis:latest
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Build Complete!                                ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""

echo "Module: ./libredis_hllset.so"
echo ""
echo "To load in Redis:"
echo "  redis-server --loadmodule ./libredis_hllset.so"
echo ""
echo "To test:"
echo "  redis-cli MODULE LIST"
echo "  redis-cli HLLSET.CREATE apple banana cherry"
