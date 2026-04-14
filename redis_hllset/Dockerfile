FROM redis:7.0.15

# Install runtime dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Create module directory
WORKDIR /usr/local/lib/redis

# Copy all Redis modules
COPY libredis-roaring.so .
COPY redisearch.so .
COPY redisgraph.so .

# Copy HLLSet module (built from module/ directory)
COPY deploy/libredis_hllset.so .

# Copy Redis configuration
COPY redis.conf /usr/local/etc/redis/redis.conf

# Expose Redis port
EXPOSE 6379

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD redis-cli ping | grep PONG || exit 1

CMD ["redis-server", "/usr/local/etc/redis/redis.conf"]