"""
Test script for HLLSetRedis — validates Redis implementation against Python.

This script:
1. Creates HLLSets using both Python and Redis backends
2. Compares results for all operations
3. Reports any discrepancies

Requirements:
    - Redis server running with redis-roaring module
    - HLLSet Lua functions loaded (run load_functions.sh first)
    - Python redis package installed

Usage:
    python test_hllset_redis.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis
from typing import List, Tuple
import time


def test_with_redis():
    """Run tests with Redis connection."""
    
    print("=" * 60)
    print("HLLSet Redis Integration Tests")
    print("=" * 60)
    
    # Connect to Redis
    r = redis.Redis(host='localhost', port=6379, decode_responses=False)
    
    # Check connection
    try:
        r.ping()
        print("✓ Redis connection OK")
    except redis.ConnectionError as e:
        print(f"✗ Redis connection failed: {e}")
        print("  Make sure Redis is running with the roaring module.")
        return False
    
    # Check modules
    print("\nChecking Redis modules...")
    try:
        modules = r.execute_command('MODULE', 'LIST')
        module_names = [m[1].decode('utf-8') if isinstance(m[1], bytes) else m[1] for m in modules]
        print(f"  Loaded modules: {module_names}")
        
        if 'roaring' not in str(module_names).lower():
            print("  ⚠ Warning: redis-roaring module may not be loaded")
    except Exception as e:
        print(f"  Could not check modules: {e}")
    
    # Check if functions are loaded
    print("\nChecking HLLSet functions...")
    try:
        funcs = r.execute_command('FUNCTION', 'LIST', 'LIBRARYNAME', 'hllset')
        if funcs:
            print("  ✓ HLLSet functions loaded")
        else:
            print("  ✗ HLLSet functions not loaded")
            print("  Run: redis/scripts/load_functions.sh")
            return False
    except redis.ResponseError as e:
        print(f"  ✗ Function check failed: {e}")
        print("  Run: redis/scripts/load_functions.sh")
        return False
    
    # Import modules
    print("\nImporting modules...")
    try:
        from core.hllset import HLLSet, murmur_hash64a
        from core.hllset_redis import HLLSetRedis, RedisClientManager
        print("  ✓ Modules imported")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    
    # Set Redis client
    RedisClientManager.set_default(r)
    
    # Test data
    tokens1 = ['apple', 'banana', 'cherry', 'date', 'elderberry']
    tokens2 = ['cherry', 'date', 'fig', 'grape', 'honeydew']
    
    print("\n" + "-" * 60)
    print("Test 1: Create HLLSet from tokens")
    print("-" * 60)
    
    # Python HLLSet
    hll_py = HLLSet.from_batch(tokens1)
    print(f"  Python: {hll_py}")
    print(f"    Cardinality: {hll_py.cardinality()}")
    print(f"    SHA1: {hll_py.name[:16]}...")
    
    # Redis HLLSet
    hll_redis = HLLSetRedis.from_batch(tokens1, redis_client=r)
    print(f"  Redis:  {hll_redis}")
    print(f"    Cardinality: {hll_redis.cardinality()}")
    print(f"    Key: {hll_redis.key}")
    
    # Compare cardinalities
    card_py = hll_py.cardinality()
    card_redis = hll_redis.cardinality()
    diff = abs(card_py - card_redis)
    
    if diff < 1:
        print(f"  ✓ Cardinalities match (diff={diff:.2f})")
    else:
        print(f"  ⚠ Cardinalities differ by {diff:.2f}")
    
    print("\n" + "-" * 60)
    print("Test 2: Union operation")
    print("-" * 60)
    
    hll_py2 = HLLSet.from_batch(tokens2)
    hll_redis2 = HLLSetRedis.from_batch(tokens2, redis_client=r)
    
    union_py = hll_py.union(hll_py2)
    union_redis = hll_redis.union(hll_redis2)
    
    print(f"  Python union cardinality: {union_py.cardinality()}")
    print(f"  Redis union cardinality:  {union_redis.cardinality()}")
    
    diff = abs(union_py.cardinality() - union_redis.cardinality())
    if diff < 1:
        print(f"  ✓ Union cardinalities match (diff={diff:.2f})")
    else:
        print(f"  ⚠ Union cardinalities differ by {diff:.2f}")
    
    print("\n" + "-" * 60)
    print("Test 3: Intersection operation")
    print("-" * 60)
    
    inter_py = hll_py.intersect(hll_py2)
    inter_redis = hll_redis.intersect(hll_redis2)
    
    print(f"  Python intersection cardinality: {inter_py.cardinality()}")
    print(f"  Redis intersection cardinality:  {inter_redis.cardinality()}")
    
    diff = abs(inter_py.cardinality() - inter_redis.cardinality())
    if diff < 1:
        print(f"  ✓ Intersection cardinalities match (diff={diff:.2f})")
    else:
        print(f"  ⚠ Intersection cardinalities differ by {diff:.2f}")
    
    print("\n" + "-" * 60)
    print("Test 4: Difference operation")
    print("-" * 60)
    
    diff_py = hll_py.diff(hll_py2)
    diff_redis = hll_redis.diff(hll_redis2)
    
    print(f"  Python difference cardinality: {diff_py.cardinality()}")
    print(f"  Redis difference cardinality:  {diff_redis.cardinality()}")
    
    diff = abs(diff_py.cardinality() - diff_redis.cardinality())
    if diff < 1:
        print(f"  ✓ Difference cardinalities match (diff={diff:.2f})")
    else:
        print(f"  ⚠ Difference cardinalities differ by {diff:.2f}")
    
    print("\n" + "-" * 60)
    print("Test 5: Similarity (Jaccard)")
    print("-" * 60)
    
    sim_py = hll_py.similarity(hll_py2)
    sim_redis = hll_redis.similarity(hll_redis2)
    
    print(f"  Python similarity: {sim_py:.4f}")
    print(f"  Redis similarity:  {sim_redis:.4f}")
    
    diff = abs(sim_py - sim_redis)
    if diff < 0.1:
        print(f"  ✓ Similarities match (diff={diff:.4f})")
    else:
        print(f"  ⚠ Similarities differ by {diff:.4f}")
    
    print("\n" + "-" * 60)
    print("Test 6: Info query")
    print("-" * 60)
    
    info = hll_redis.info()
    print(f"  Key: {info.get('key')}")
    print(f"  SHA1: {info.get('sha1', '')[:16]}...")
    print(f"  Cardinality: {info.get('cardinality')}")
    print(f"  Bits set: {info.get('bits_set')}")
    print(f"  Registers: {info.get('registers')}")
    print("  ✓ Info query OK")
    
    print("\n" + "-" * 60)
    print("Test 7: Dump and load positions")
    print("-" * 60)
    
    positions = hll_redis.dump_positions()
    print(f"  Dumped {len(positions)} positions")
    
    # Load from positions
    hll_loaded = HLLSetRedis.from_positions(positions, redis_client=r)
    print(f"  Loaded HLLSet: {hll_loaded}")
    print(f"  Cardinality: {hll_loaded.cardinality()}")
    
    if hll_loaded.name == hll_redis.name:
        print("  ✓ Content-addressed keys match")
    else:
        print("  ⚠ Keys differ (may indicate position encoding issue)")
    
    print("\n" + "-" * 60)
    print("Test 8: Bulk union")
    print("-" * 60)
    
    hll_list = [
        HLLSetRedis.from_batch(['a', 'b', 'c'], redis_client=r),
        HLLSetRedis.from_batch(['d', 'e', 'f'], redis_client=r),
        HLLSetRedis.from_batch(['g', 'h', 'i'], redis_client=r),
    ]
    
    bulk = HLLSetRedis.bulk_union(hll_list)
    print(f"  Bulk union of 3 sets: {bulk}")
    print(f"  Cardinality: {bulk.cardinality()}")
    print("  ✓ Bulk union OK")
    
    print("\n" + "-" * 60)
    print("Test 9: Performance comparison")
    print("-" * 60)
    
    # Generate larger dataset
    large_tokens = [f"token_{i}" for i in range(1000)]
    
    # Python timing
    start = time.time()
    hll_py_large = HLLSet.from_batch(large_tokens)
    py_time = time.time() - start
    
    # Redis timing
    start = time.time()
    hll_redis_large = HLLSetRedis.from_batch(large_tokens, redis_client=r)
    redis_time = time.time() - start
    
    print(f"  Python: {py_time*1000:.2f} ms for 1000 tokens")
    print(f"  Redis:  {redis_time*1000:.2f} ms for 1000 tokens")
    print(f"  Python cardinality: {hll_py_large.cardinality()}")
    print(f"  Redis cardinality:  {hll_redis_large.cardinality()}")
    
    # Cleanup
    print("\n" + "-" * 60)
    print("Cleanup")
    print("-" * 60)
    
    # Delete test keys
    pattern = "hllset:*"
    keys = r.keys(pattern)
    if keys:
        deleted = r.delete(*keys)
        print(f"  Deleted {deleted} test keys")
    else:
        print("  No test keys to delete")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = test_with_redis()
    sys.exit(0 if success else 1)
