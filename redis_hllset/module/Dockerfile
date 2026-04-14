# Multi-stage build for Redis HLLSet module
#
# Stage 1: Build the Rust module
# Stage 2: Create minimal Redis image with the module

# =============================================================================
# Stage 1: Build Environment
# =============================================================================
FROM rust:1.75-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    clang \
    libclang-dev \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /build

# Copy Cargo files first (for caching)
COPY Cargo.toml Cargo.lock* ./

# Create dummy src to cache dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn dummy() {}" > src/lib.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release 2>/dev/null || true

# Copy actual source
COPY src/ src/

# Build the module
RUN cargo build --release

# Verify the module was built
RUN ls -la target/release/*.so || ls -la target/release/*.dylib || echo "No .so found"

# =============================================================================
# Stage 2: Runtime Image
# =============================================================================
FROM redis:7.0.15-bookworm

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create module directory
RUN mkdir -p /usr/local/lib/redis/modules

# Copy the built module
COPY --from=builder /build/target/release/libredis_hllset.so /usr/local/lib/redis/modules/

# Note: Other modules (roaring, redisearch, redisgraph) should be copied
# from the parent redis_hllset directory when building the full image

# Copy Redis configuration
COPY redis.conf /usr/local/etc/redis/redis.conf

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD redis-cli ping | grep PONG || exit 1

# Start Redis with config
CMD ["redis-server", "/usr/local/etc/redis/redis.conf"]
