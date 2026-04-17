#!/bin/bash
# Deploy Redis container with tested HLLSet features
#
# This script handles the complete deployment workflow:
#   1. Pre-flight checks (syntax validation, tests)
#   2. Backup current state (optional)
#   3. Stop existing container
#   4. Rebuild image (if requested)
#   5. Start new container
#   6. Load Lua functions
#   7. Run verification tests
#   8. Rollback on failure (optional)
#
# Usage: ./deploy.sh [options]
#
# Options:
#   --build, -b      Rebuild container image before deploy
#   --force, -f      Skip pre-flight tests (use with caution)
#   --backup, -B     Create backup before deploy
#   --rollback, -r   Enable automatic rollback on failure
#   --dry-run, -n    Show what would be done without executing
#   --help, -h       Show this help message
#
# Examples:
#   ./deploy.sh                    # Deploy with pre-flight tests
#   ./deploy.sh --build            # Rebuild image and deploy
#   ./deploy.sh --backup --build   # Full production deploy
#   ./deploy.sh --dry-run          # Preview deployment steps

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# Parse arguments
BUILD=false
FORCE=false
BACKUP=false
ROLLBACK=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build|-b)    BUILD=true; shift ;;
        --force|-f)    FORCE=true; shift ;;
        --backup|-B)   BACKUP=true; shift ;;
        --rollback|-r) ROLLBACK=true; BACKUP=true; shift ;;  # rollback requires backup
        --dry-run|-n)  DRY_RUN=true; shift ;;
        --help|-h)
            head -30 "$0" | grep -E "^#" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Configuration (use env.sh values, with fallbacks)
REDIS_DIR="${REDIS_DIR:-$SCRIPT_DIR/..}"
FUNCTIONS_DIR="${FUNCTIONS_DIR:-$REDIS_DIR/functions}"
BACKUP_DIR="${BACKUP_DIR:-$REDIS_DIR/backups}"
IMAGE_NAME="${IMAGE_NAME:-localhost/redis-hllset:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-redis-server}"
DEPLOY_TAG=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE=""

# Dry-run wrapper
run() {
    if $DRY_RUN; then
        echo "  [DRY-RUN] $*"
    else
        "$@"
    fi
}

# Step tracking
STEP=0
step() {
    STEP=$((STEP + 1))
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step $STEP: $*${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Cleanup function for rollback
cleanup_on_failure() {
    if $ROLLBACK && [ -n "$BACKUP_FILE" ] && [ -f "$BACKUP_FILE" ]; then
        echo ""
        log_warn "Deployment failed! Initiating rollback..."
        "$SCRIPT_DIR/restore.sh" "$BACKUP_FILE"
        log_ok "Rollback completed"
    fi
}

# Set trap for cleanup
trap 'cleanup_on_failure' ERR

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     HLLSet Redis Deployment                        ║${NC}"
echo -e "${GREEN}║     Deploy ID: $DEPLOY_TAG                  ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"

if $DRY_RUN; then
    echo ""
    log_warn "DRY-RUN MODE - No changes will be made"
fi

# ============================================================
# Step 1: Pre-flight checks
# ============================================================
step "Pre-flight checks"

# Check Lua syntax (optional — skipped if no Lua functions present)
log_info "Checking Lua syntax..."
LUA_FILE="$FUNCTIONS_DIR/hllset.lua"
if [ ! -f "$LUA_FILE" ]; then
    log_warn "No Lua functions found (using native Rust commands), skipping Lua check"
else
    # Basic Lua syntax check (if lua is available)
    if command -v lua5.1 &>/dev/null; then
        if ! lua5.1 -p "$LUA_FILE" 2>/dev/null; then
            log_error "Lua syntax error in $LUA_FILE"
            exit 1
        fi
        log_ok "Lua syntax OK"
    elif command -v luac &>/dev/null; then
        if ! luac -p "$LUA_FILE" 2>/dev/null; then
            log_error "Lua syntax error in $LUA_FILE"
            exit 1
        fi
        log_ok "Lua syntax OK"
    else
        log_warn "No Lua checker available, skipping syntax check"
    fi
fi

# Check container runtime
if ! command -v "$CONTAINER_RUNTIME" &>/dev/null; then
    log_error "Container runtime not found: $CONTAINER_RUNTIME"
    exit 1
fi
log_ok "Container runtime: $CONTAINER_RUNTIME"

# Check if Redis is currently running (for pre-deploy tests)
if ! $FORCE; then
    if "$CONTAINER_RUNTIME" container exists "$CONTAINER_NAME" 2>/dev/null; then
        log_info "Running pre-deploy tests on current instance..."
        
        # Load functions to current instance and test
        if ! $DRY_RUN; then
            if "$SCRIPT_DIR/load_functions.sh" &>/dev/null; then
                log_ok "Functions loaded successfully"
                
                # Run quick smoke test
                TEST_OUTPUT=$("$SCRIPT_DIR/test_hllset.sh" 2>&1) || true
                if echo "$TEST_OUTPUT" | grep -q "0 failed"; then
                    log_ok "Pre-deploy tests passed"
                else
                    log_error "Pre-deploy tests failed!"
                    echo "$TEST_OUTPUT" | tail -10
                    echo ""
                    log_warn "Use --force to skip pre-flight tests"
                    exit 1
                fi
            else
                log_warn "Could not load functions for pre-deploy test"
            fi
        else
            echo "  [DRY-RUN] Would run load_functions.sh"
            echo "  [DRY-RUN] Would run test_hllset.sh"
        fi
    else
        log_info "No running container - skipping pre-deploy tests"
    fi
else
    log_warn "Skipping pre-flight tests (--force)"
fi

# ============================================================
# Step 2: Backup (if requested)
# ============================================================
if $BACKUP; then
    step "Creating backup"
    
    mkdir -p "$BACKUP_DIR"
    BACKUP_FILE="$BACKUP_DIR/pre_deploy_$DEPLOY_TAG.tar.gz"
    
    if ! $DRY_RUN; then
        if "$CONTAINER_RUNTIME" container exists "$CONTAINER_NAME" 2>/dev/null; then
            "$SCRIPT_DIR/backup.sh" "$BACKUP_FILE"
            log_ok "Backup created: $BACKUP_FILE"
        else
            # Just backup the functions
            tar -czf "$BACKUP_FILE" -C "$REDIS_DIR" functions/
            log_ok "Functions backup created: $BACKUP_FILE"
        fi
    else
        echo "  [DRY-RUN] Would create backup: $BACKUP_FILE"
    fi
fi

# ============================================================
# Step 3: Stop existing container
# ============================================================
step "Stopping existing container"

if "$CONTAINER_RUNTIME" container exists "$CONTAINER_NAME" 2>/dev/null; then
    run "$CONTAINER_RUNTIME" stop "$CONTAINER_NAME" || true
    log_ok "Container stopped"
    
    # Remove container for clean start
    run "$CONTAINER_RUNTIME" rm "$CONTAINER_NAME" || true
    log_ok "Container removed"
else
    log_info "No existing container to stop"
fi

# ============================================================
# Step 4: Build image (if requested)
# ============================================================
if $BUILD; then
    step "Building container image"
    
    cd "$REDIS_DIR"
    
    # Tag old image for potential rollback
    if "$CONTAINER_RUNTIME" image exists "$IMAGE_NAME" 2>/dev/null; then
        run "$CONTAINER_RUNTIME" tag "$IMAGE_NAME" "${IMAGE_NAME%.latest}:pre_$DEPLOY_TAG" || true
        log_info "Tagged previous image for rollback"
    fi
    
    run "$CONTAINER_RUNTIME" build -t "$IMAGE_NAME" .
    log_ok "Image built: $IMAGE_NAME"
fi

# ============================================================
# Step 5: Start new container
# ============================================================
step "Starting new container"

cd "$REDIS_DIR"

run "$CONTAINER_RUNTIME" run -d \
    --name "$CONTAINER_NAME" \
    -p "${REDIS_PORT:-6379}:6379" \
    -v redis-hllset-data:/data:Z \
    -v "$FUNCTIONS_DIR:/usr/local/lib/redis_hllset/functions:ro,Z" \
    --restart unless-stopped \
    "$IMAGE_NAME"

if ! $DRY_RUN; then
    log_ok "Container started"
    
    # Wait for Redis to be ready
    log_info "Waiting for Redis to be ready..."
    for i in {1..30}; do
        if "$CONTAINER_RUNTIME" exec "$CONTAINER_NAME" redis-cli ping 2>/dev/null | grep -q PONG; then
            log_ok "Redis is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Redis failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
else
    echo "  [DRY-RUN] Would wait for Redis to be ready"
fi

# ============================================================
# Step 6: Load Lua functions
# ============================================================
step "Loading Lua functions"

if [ ! -f "$FUNCTIONS_DIR/hllset.lua" ]; then
    log_warn "No Lua functions found (using native Rust commands), skipping"
elif ! $DRY_RUN; then
    # Give Redis a moment to fully initialize
    sleep 1
    
    "$SCRIPT_DIR/load_functions.sh"
    log_ok "Functions loaded"
else
    echo "  [DRY-RUN] Would run load_functions.sh"
fi

# ============================================================
# Step 7: Post-deploy verification
# ============================================================
step "Post-deploy verification"

if ! $DRY_RUN; then
    # Run health check
    if "$SCRIPT_DIR/health_check.sh" &>/dev/null; then
        log_ok "Health check passed"
    else
        log_error "Health check failed!"
        exit 1
    fi
    
    # Run test suite
    log_info "Running verification tests..."
    TEST_OUTPUT=$("$SCRIPT_DIR/test_hllset.sh" 2>&1)
    if echo "$TEST_OUTPUT" | grep -q "0 failed"; then
        log_ok "All verification tests passed"
    else
        log_error "Verification tests failed!"
        echo "$TEST_OUTPUT"
        exit 1
    fi
else
    echo "  [DRY-RUN] Would run health_check.sh"
    echo "  [DRY-RUN] Would run test_hllset.sh"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Deployment Complete!                           ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""

if ! $DRY_RUN; then
    echo "Container status:"
    "$CONTAINER_RUNTIME" ps --filter name="$CONTAINER_NAME" --format "  Name: {{.Names}}\n  Status: {{.Status}}\n  Ports: {{.Ports}}"
    echo ""
    
    echo "Redis info:"
    redis_cmd INFO server 2>/dev/null | grep -E "redis_version|uptime_in_seconds" | sed 's/^/  /'
    echo ""
    
    echo "Loaded functions:"
    redis_cmd FUNCTION LIST 2>/dev/null | grep -E "library_name|name" | head -20 | sed 's/^/  /'
    echo ""
    
    if [ -n "$BACKUP_FILE" ]; then
        log_info "Backup saved to: $BACKUP_FILE"
    fi
    
    log_ok "Deploy ID: $DEPLOY_TAG"
else
    echo "Dry-run complete. Run without --dry-run to execute."
fi
