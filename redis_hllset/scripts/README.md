# HLLSet Redis DevOps Toolkit

A complete development environment for the HLLSet Algebra Redis functions.

## Quick Start

```bash
cd redis_hllset/scripts

# Deploy (start container, load functions, run tests)
./hllset.sh deploy

# Or step by step:
./hllset.sh start     # Start Redis container
./hllset.sh load      # Load Lua functions
./hllset.sh test      # Run tests

# Start development watch (auto-reload on file changes)
./hllset.sh watch --test
```

## Scripts Overview

### Master Control (`hllset.sh`)

Single entry point for all operations:

```bash
./hllset.sh <command> [args...]
```

| Command | Description |
|---------|-------------|
| `deploy` | **Deploy with tests and optional rebuild** |
| `start` | Start Redis container |
| `stop` | Stop Redis container |
| `restart` | Restart and reload functions |
| `status` | Show container and health status |
| `load` | Load/reload Lua functions |
| `test` | Run test suite |
| `watch` | Watch files and auto-reload |
| `health` | Run health checks |
| `stats` | Show statistics |
| `benchmark` | Run performance benchmarks |
| `backup` | Backup functions and data |
| `restore` | Restore from backup |
| `cleanup` | Clean up temporary keys |
| `repl` | Interactive REPL |
| `cli` | Redis CLI with project defaults |

### Production Deployment

```bash
# Simple deploy (start, load, test)
./hllset.sh deploy

# Deploy with image rebuild
./hllset.sh deploy --build

# Full production deploy (backup + rebuild + rollback on failure)
./hllset.sh deploy --backup --build --rollback

# Preview what deploy would do
./hllset.sh deploy --dry-run

# Force deploy (skip pre-flight tests)
./hllset.sh deploy --force
```

### Development Workflow

#### 1. Start Environment

```bash
./hllset.sh start      # Start Redis container
./hllset.sh health     # Verify everything is working
```

#### 2. Edit-Reload-Test Loop

**Option A: Manual**
```bash
# Edit redis/functions/hllset.lua
./hllset.sh load       # Reload functions
./hllset.sh test       # Run tests
```

**Option B: Auto-reload (recommended)**
```bash
./hllset.sh watch --test
# Now edit hllset.lua - changes auto-reload and tests run
```

#### 3. Interactive Testing

```bash
./hllset.sh repl
```

In REPL:
```
hllset> @create apple banana cherry
"hllset:abc123..."

hllset> @card hllset:abc123...
3

hllset> @help
```

### Individual Scripts

| Script | Purpose |
|--------|---------|
| `env.sh` | Environment configuration (source this) |
| `deploy.sh` | **Full deployment with pre-flight tests** |
| `start_redis.sh` | Start Redis with Podman |
| `load_functions.sh` | Load Lua functions into Redis |
| `test_hllset.sh` | Run test suite |
| `health_check.sh` | Comprehensive health checks |
| `watch.sh` | File watcher with auto-reload |
| `stats.sh` | Show Redis/HLLSet statistics |
| `benchmark.sh` | Performance benchmarks |
| `backup.sh` | Backup functions and data |
| `restore.sh` | Restore from backup |
| `cleanup.sh` | Clean up temporary keys |
| `repl.sh` | Interactive REPL |
| `redis-cli.sh` | Wrapper for redis-cli |

## Configuration

Edit `env.sh` or set environment variables:

```bash
# Connection
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_PASSWORD=

# Container
export CONTAINER_NAME=redis-server
export CONTAINER_RUNTIME=podman  # or docker

# Development
export WATCH_INTERVAL=2  # seconds
```

## REPL Shortcuts

| Shortcut | Full Command |
|----------|--------------|
| `@create <tokens>` | `FCALL hllset_create 0 <tokens>` |
| `@card <key>` | `FCALL hllset_cardinality 1 <key>` |
| `@union <k1> <k2>` | `FCALL hllset_union 2 <k1> <k2>` |
| `@inter <k1> <k2>` | `FCALL hllset_intersect 2 <k1> <k2>` |
| `@diff <k1> <k2>` | `FCALL hllset_diff 2 <k1> <k2>` |
| `@xor <k1> <k2>` | `FCALL hllset_xor 2 <k1> <k2>` |
| `@sim <k1> <k2>` | `FCALL hllset_similarity 2 <k1> <k2>` |
| `@info <key>` | `FCALL hllset_info 1 <key>` |
| `@list` | List all HLLSet keys |
| `@del <key>` | Delete HLLSet |

## Backup & Restore

```bash
# Create backup
./hllset.sh backup

# Create backup to specific directory
./hllset.sh backup /path/to/backup

# Restore latest backup
./hllset.sh restore

# Restore specific backup
./hllset.sh restore /path/to/backup/20260414_120000
```

Backups include:
- `hllset.lua` - Lua source code
- `functions.rdb` - Redis function dump
- `keys.txt` - List of HLLSet keys
- `manifest.json` - Backup metadata

## Tips

### Quick Test After Edit

```bash
./hllset.sh load && ./hllset.sh test
```

### Check for Orphaned Temp Keys

```bash
./hllset.sh cleanup --dry-run
```

### Monitor in Real-Time

```bash
./hllset.sh stats --watch
```

### Debug a Specific HLLSet

```bash
./hllset.sh cli FCALL hllset_dump 1 "hllset:abc..."
```
