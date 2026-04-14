#!/bin/bash
# Master control script for Redis HLLSet development
#
# Usage: ./hllset.sh <command> [args...]
#
# Commands:
#   start       Start Redis container
#   stop        Stop Redis container
#   restart     Restart Redis container
#   status      Show container and Redis status
#   
#   deploy      Deploy with tests and optional rebuild
#   load        Load/reload Lua functions
#   test        Run test suite
#   watch       Watch files and auto-reload
#   
#   health      Run health checks
#   stats       Show statistics
#   benchmark   Run performance benchmarks
#   
#   backup      Backup functions and data
#   restore     Restore from backup
#   cleanup     Clean up temporary keys
#   
#   repl        Interactive REPL
#   cli         Redis CLI with project defaults
#
#   module      Build native Rust module
#   
#   help        Show this help

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

show_help() {
    head -30 "$0" | grep -E "^#" | sed 's/^# //' | sed 's/^#//'
}

case "${1:-help}" in
    start)
        "$SCRIPT_DIR/start_redis.sh"
        ;;
    stop)
        $CONTAINER_RUNTIME stop "$CONTAINER_NAME"
        log_ok "Stopped $CONTAINER_NAME"
        ;;
    restart)
        $CONTAINER_RUNTIME restart "$CONTAINER_NAME"
        log_ok "Restarted $CONTAINER_NAME"
        sleep 2
        "$SCRIPT_DIR/load_functions.sh"
        ;;
    status)
        echo "Container:"
        $CONTAINER_RUNTIME ps -a --filter "name=$CONTAINER_NAME" --format "  {{.Names}}: {{.Status}}"
        echo ""
        "$SCRIPT_DIR/health_check.sh"
        ;;
    
    deploy)
        "$SCRIPT_DIR/deploy.sh" "${@:2}"
        ;;
    load)
        "$SCRIPT_DIR/load_functions.sh"
        ;;
    test)
        "$SCRIPT_DIR/test_hllset.sh"
        ;;
    watch)
        "$SCRIPT_DIR/watch.sh" "${@:2}"
        ;;
    
    health)
        "$SCRIPT_DIR/health_check.sh"
        ;;
    stats)
        "$SCRIPT_DIR/stats.sh" "${@:2}"
        ;;
    benchmark)
        "$SCRIPT_DIR/benchmark.sh" "${@:2}"
        ;;
    
    backup)
        "$SCRIPT_DIR/backup.sh" "${@:2}"
        ;;
    restore)
        "$SCRIPT_DIR/restore.sh" "${@:2}"
        ;;
    cleanup)
        "$SCRIPT_DIR/cleanup.sh" "${@:2}"
        ;;
    
    repl)
        "$SCRIPT_DIR/repl.sh"
        ;;
    cli)
        "$SCRIPT_DIR/redis-cli.sh" "${@:2}"
        ;;
    
    module)
        MODULE_DIR="$SCRIPT_DIR/../module"
        case "${2:-build}" in
            build)
                cd "$MODULE_DIR" && ./build.sh --release
                ;;
            test)
                cd "$MODULE_DIR" && ./test.sh 127.0.0.1 6379
                ;;
            deploy)
                log_info "Building and deploying native module..."
                cd "$MODULE_DIR" && ./build.sh --release
                log_info "Restarting Redis with native module..."
                $CONTAINER_RUNTIME stop "$CONTAINER_NAME" 2>/dev/null || true
                $CONTAINER_RUNTIME rm "$CONTAINER_NAME" 2>/dev/null || true
                $CONTAINER_RUNTIME run -d \
                    --name "$CONTAINER_NAME" \
                    -p 6379:6379 \
                    -v "$MODULE_DIR:/modules:ro,Z" \
                    docker.io/library/redis:7.0.15 \
                    redis-server --loadmodule /modules/libredis_hllset.so
                sleep 2
                log_ok "Module deployed"
                cd "$MODULE_DIR" && ./test.sh 127.0.0.1 6379
                ;;
            *)
                echo "Usage: ./hllset.sh module [build|test|deploy]"
                ;;
        esac
        ;;
    
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
