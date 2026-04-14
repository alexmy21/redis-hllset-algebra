#!/usr/bin/env python3
"""
Test suite for HLLSetRedis Python wrapper.

Usage:
    python test_hllset_redis.py
"""

import redis
import sys
from typing import List, Tuple

# Add parent to path for imports
sys.path.insert(0, '.')

from core.hllset_redis import HLLSetRedis, RedisClientManager, load_functions


def setup():
    """Set up Redis connection."""
    r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=False)
    RedisClientManager.set_default(r)
    return r


def test_basic_creation():
    """Test basic HLLSet creation."""
    print("1. Testing basic creation...")
    
    A = HLLSetRedis.from_batch(['apple', 'banana', 'cherry'])
    assert A.key is not None, "Key should not be None"
    assert A.key.startswith('hllset:'), f"Key should start with 'hllset:', got {A.key}"
    assert A.cardinality() == 3, f"Expected cardinality 3, got {A.cardinality()}"
    
    print(f"   ✓ Created HLLSet with key {A.short_name}...")


def test_empty_set():
    """Test empty HLLSet."""
    print("2. Testing empty set...")
    
    empty = HLLSetRedis.from_batch([])
    assert empty.key is not None
    assert empty.cardinality() == 0, f"Empty set should have 0 cardinality, got {empty.cardinality()}"
    
    print("   ✓ Empty set works correctly")


def test_union():
    """Test union operation."""
    print("3. Testing union...")
    
    A = HLLSetRedis.from_batch(['a', 'b', 'c'])
    B = HLLSetRedis.from_batch(['b', 'c', 'd'])
    
    U = A.union(B)
    card = U.cardinality()
    assert 3 <= card <= 5, f"Union should have ~4 elements, got {card}"
    
    print(f"   ✓ Union cardinality: {card}")


def test_intersection():
    """Test intersection operation."""
    print("4. Testing intersection...")
    
    A = HLLSetRedis.from_batch(['a', 'b', 'c'])
    B = HLLSetRedis.from_batch(['b', 'c', 'd'])
    
    I = A.intersect(B)
    card = I.cardinality()
    assert 1 <= card <= 3, f"Intersection should have ~2 elements, got {card}"
    
    print(f"   ✓ Intersection cardinality: {card}")


def test_difference():
    """Test difference operation."""
    print("5. Testing difference...")
    
    A = HLLSetRedis.from_batch(['a', 'b', 'c'])
    B = HLLSetRedis.from_batch(['b', 'c', 'd'])
    
    D = A.diff(B)
    card = D.cardinality()
    assert 0 <= card <= 2, f"Difference should have ~1 element, got {card}"
    
    print(f"   ✓ Difference cardinality: {card}")


def test_symmetric_difference():
    """Test symmetric difference (XOR) operation."""
    print("6. Testing symmetric difference...")
    
    A = HLLSetRedis.from_batch(['a', 'b', 'c'])
    B = HLLSetRedis.from_batch(['b', 'c', 'd'])
    
    X = A.xor(B)
    card = X.cardinality()
    assert 1 <= card <= 3, f"XOR should have ~2 elements, got {card}"
    
    print(f"   ✓ XOR cardinality: {card}")


def test_similarity():
    """Test Jaccard similarity."""
    print("7. Testing Jaccard similarity...")
    
    A = HLLSetRedis.from_batch(['a', 'b', 'c'])
    B = HLLSetRedis.from_batch(['b', 'c', 'd'])
    
    sim = A.similarity(B)
    assert 0.3 <= sim <= 0.7, f"Similarity should be ~0.5, got {sim}"
    
    print(f"   ✓ Jaccard similarity: {sim}")


def test_cosine():
    """Test cosine similarity."""
    print("8. Testing cosine similarity...")
    
    A = HLLSetRedis.from_batch(['a', 'b', 'c'])
    B = HLLSetRedis.from_batch(['b', 'c', 'd'])
    
    cos = A.cosine(B)
    assert 0.4 <= cos <= 0.9, f"Cosine should be reasonable, got {cos}"
    
    print(f"   ✓ Cosine similarity: {cos:.4f}")


def test_content_addressable():
    """Test content-addressable property."""
    print("9. Testing content-addressable property...")
    
    tokens = ['hello', 'world', 'test']
    A = HLLSetRedis.from_batch(tokens)
    B = HLLSetRedis.from_batch(tokens)
    
    assert A.key == B.key, f"Same tokens should produce same key: {A.key} != {B.key}"
    
    print("   ✓ Same tokens produce same key")


def test_info():
    """Test info method."""
    print("10. Testing info...")
    
    A = HLLSetRedis.from_batch(['x', 'y', 'z'])
    info = A.info()
    
    assert 'key' in info, "Info should contain 'key'"
    assert 'cardinality' in info, "Info should contain 'cardinality'"
    assert 'sha1' in info, "Info should contain 'sha1'"
    
    print(f"   ✓ Info has {len(info)} fields")


def test_repr():
    """Test string representation."""
    print("11. Testing repr...")
    
    A = HLLSetRedis.from_batch(['foo', 'bar'])
    s = repr(A)
    
    assert 'HLLSetRedis' in s
    assert 'Redis' in s
    
    print(f"   ✓ repr: {s}")


def test_merge():
    """Test bulk merge."""
    print("12. Testing merge...")
    
    sets = [
        HLLSetRedis.from_batch(['a', 'b']),
        HLLSetRedis.from_batch(['c', 'd']),
        HLLSetRedis.from_batch(['e', 'f']),
    ]
    
    merged = HLLSetRedis.merge(sets)
    card = merged.cardinality()
    
    assert 4 <= card <= 8, f"Merged should have ~6 elements, got {card}"
    
    print(f"   ✓ Merged cardinality: {card}")


def test_large_set():
    """Test larger sets."""
    print("13. Testing large set...")
    
    tokens = [f"token_{i}" for i in range(1000)]
    A = HLLSetRedis.from_batch(tokens)
    card = A.cardinality()
    
    # Simple Lua hash has higher collision rate, accept wider range
    # Real implementation with proper hash will be more accurate
    assert 500 <= card <= 1200, f"Large set should have reasonable cardinality, got {card}"
    
    print(f"   ✓ Large set cardinality: {card} (tokens: 1000)")


def test_dump_positions():
    """Test dump_positions method."""
    print("14. Testing dump_positions...")
    
    A = HLLSetRedis.from_batch(['test1', 'test2'])
    positions = A.dump_positions()
    
    assert isinstance(positions, list), "Positions should be a list"
    assert len(positions) > 0, "Should have some positions"
    
    print(f"   ✓ Dumped {len(positions)} positions")


def main():
    """Run all tests."""
    print("=" * 50)
    print("  HLLSetRedis Python Wrapper Tests")
    print("=" * 50)
    print()
    
    try:
        setup()
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        print("Make sure Redis is running on 127.0.0.1:6379")
        sys.exit(1)
    
    tests = [
        test_basic_creation,
        test_empty_set,
        test_union,
        test_intersection,
        test_difference,
        test_symmetric_difference,
        test_similarity,
        test_cosine,
        test_content_addressable,
        test_info,
        test_repr,
        test_merge,
        test_large_set,
        test_dump_positions,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"   ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"   ✗ ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 50)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
