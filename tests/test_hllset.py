"""Tests for redis_hllset_algebra using fakeredis (in-process, no real Redis needed)."""

import pytest
import fakeredis

from redis_hllset_algebra import HLLSetAlgebra


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def redis_client():
    """Return a fresh in-process fake Redis instance per test."""
    return fakeredis.FakeRedis()


@pytest.fixture
def hll(redis_client):
    """Return an HLLSetAlgebra bound to the fake Redis instance."""
    return HLLSetAlgebra(redis_client)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _approx(expected: int, value: int, tolerance: float = 0.1) -> bool:
    """True when *value* is within *tolerance* (10 % by default) of *expected*."""
    if expected == 0:
        return value == 0
    return abs(value - expected) / expected <= tolerance


# ---------------------------------------------------------------------------
# add / count (pass-through wrappers)
# ---------------------------------------------------------------------------

class TestAddCount:
    def test_add_returns_1_on_change(self, hll):
        assert hll.add("k", "a", "b", "c") == 1

    def test_add_returns_0_when_unchanged(self, hll):
        hll.add("k", "a")
        assert hll.add("k", "a") == 0

    def test_count_empty_key(self, hll):
        assert hll.count("nonexistent") == 0

    def test_count_single_key(self, hll):
        hll.add("k", *[str(i) for i in range(100)])
        assert hll.count("k") == 100

    def test_count_multiple_keys(self, hll):
        hll.add("a", *[str(i) for i in range(100)])
        hll.add("b", *[str(i) for i in range(50, 150)])
        # union should be ~150
        assert _approx(150, hll.count("a", "b"))


# ---------------------------------------------------------------------------
# union
# ---------------------------------------------------------------------------

class TestUnion:
    def test_union_stored_in_dest(self, hll, redis_client):
        hll.add("a", "x", "y")
        hll.add("b", "y", "z")
        card = hll.union("dest", "a", "b")
        # dest key should now exist
        assert redis_client.exists("dest")
        assert card == redis_client.pfcount("dest")

    def test_union_cardinality(self, hll):
        hll.add("a", *[str(i) for i in range(100)])
        hll.add("b", *[str(i) for i in range(50, 150)])
        card = hll.union("u", "a", "b")
        assert _approx(150, card)

    def test_union_identical_sets(self, hll):
        elems = [str(i) for i in range(100)]
        hll.add("a", *elems)
        hll.add("b", *elems)
        assert hll.union("u", "a", "b") == 100

    def test_union_requires_sources(self, hll):
        with pytest.raises(ValueError):
            hll.union("dest")

    def test_union_three_sources(self, hll):
        hll.add("a", *[str(i) for i in range(50)])
        hll.add("b", *[str(i) for i in range(50, 100)])
        hll.add("c", *[str(i) for i in range(100, 150)])
        card = hll.union("u", "a", "b", "c")
        assert _approx(150, card)


# ---------------------------------------------------------------------------
# union_card
# ---------------------------------------------------------------------------

class TestUnionCard:
    def test_union_card_no_side_effects(self, hll, redis_client):
        hll.add("a", "x", "y")
        hll.add("b", "y", "z")
        hll.union_card("a", "b")
        # No extra keys should have been created
        assert set(redis_client.keys()) == {b"a", b"b"}

    def test_union_card_value(self, hll):
        hll.add("a", *[str(i) for i in range(100)])
        hll.add("b", *[str(i) for i in range(50, 150)])
        assert _approx(150, hll.union_card("a", "b"))

    def test_union_card_requires_two_keys(self, hll):
        with pytest.raises(ValueError):
            hll.union_card("a")


# ---------------------------------------------------------------------------
# intersect_card
# ---------------------------------------------------------------------------

class TestIntersectCard:
    def test_disjoint_sets(self, hll):
        hll.add("a", "x", "y")
        hll.add("b", "z", "w")
        assert hll.intersect_card("a", "b") == 0

    def test_identical_sets(self, hll):
        elems = [str(i) for i in range(100)]
        hll.add("a", *elems)
        hll.add("b", *elems)
        assert _approx(100, hll.intersect_card("a", "b"))

    def test_partial_overlap(self, hll):
        hll.add("a", *[str(i) for i in range(100)])    # 0–99
        hll.add("b", *[str(i) for i in range(50, 150)])  # 50–149
        # Intersection = 50–99 ≈ 50
        assert _approx(50, hll.intersect_card("a", "b"))

    def test_single_key_returns_its_cardinality(self, hll):
        hll.add("a", "x", "y", "z")
        assert hll.intersect_card("a") == 3

    def test_no_side_effects(self, hll, redis_client):
        hll.add("a", "x", "y")
        hll.add("b", "y", "z")
        hll.intersect_card("a", "b")
        assert set(redis_client.keys()) == {b"a", b"b"}

    def test_three_sets(self, hll):
        # A = 0–99, B = 50–149, C = 75–174
        # A ∩ B ∩ C = 75–99 ≈ 25
        hll.add("a", *[str(i) for i in range(100)])
        hll.add("b", *[str(i) for i in range(50, 150)])
        hll.add("c", *[str(i) for i in range(75, 175)])
        assert _approx(25, hll.intersect_card("a", "b", "c"))


# ---------------------------------------------------------------------------
# diff_card
# ---------------------------------------------------------------------------

class TestDiffCard:
    def test_identical_sets_diff_zero(self, hll):
        elems = [str(i) for i in range(100)]
        hll.add("a", *elems)
        hll.add("b", *elems)
        assert hll.diff_card("a", "b") == 0

    def test_disjoint_sets_diff_equals_a(self, hll):
        hll.add("a", *[str(i) for i in range(100)])
        hll.add("b", *[str(i) for i in range(100, 200)])
        assert _approx(100, hll.diff_card("a", "b"))

    def test_partial_overlap(self, hll):
        # A = 0–99, B = 50–149 → A \ B = 0–49 ≈ 50
        hll.add("a", *[str(i) for i in range(100)])
        hll.add("b", *[str(i) for i in range(50, 150)])
        assert _approx(50, hll.diff_card("a", "b"))

    def test_b_superset_of_a(self, hll):
        hll.add("a", *[str(i) for i in range(50)])
        hll.add("b", *[str(i) for i in range(100)])
        assert hll.diff_card("a", "b") == 0

    def test_no_side_effects(self, hll, redis_client):
        hll.add("a", "x")
        hll.add("b", "y")
        hll.diff_card("a", "b")
        assert set(redis_client.keys()) == {b"a", b"b"}


# ---------------------------------------------------------------------------
# symmdiff_card
# ---------------------------------------------------------------------------

class TestSymmDiffCard:
    def test_identical_sets_symmdiff_zero(self, hll):
        elems = [str(i) for i in range(100)]
        hll.add("a", *elems)
        hll.add("b", *elems)
        assert hll.symmdiff_card("a", "b") == 0

    def test_disjoint_sets_symmdiff_equals_union(self, hll):
        hll.add("a", *[str(i) for i in range(100)])
        hll.add("b", *[str(i) for i in range(100, 200)])
        assert _approx(200, hll.symmdiff_card("a", "b"))

    def test_partial_overlap(self, hll):
        # A = 0–99, B = 50–149 → A △ B = {0–49} ∪ {100–149} ≈ 100
        hll.add("a", *[str(i) for i in range(100)])
        hll.add("b", *[str(i) for i in range(50, 150)])
        assert _approx(100, hll.symmdiff_card("a", "b"))

    def test_no_side_effects(self, hll, redis_client):
        hll.add("a", "x")
        hll.add("b", "y")
        hll.symmdiff_card("a", "b")
        assert set(redis_client.keys()) == {b"a", b"b"}


# ---------------------------------------------------------------------------
# is_subset
# ---------------------------------------------------------------------------

class TestIsSubset:
    def test_true_when_a_subset_of_b(self, hll):
        hll.add("a", *[str(i) for i in range(50)])
        hll.add("b", *[str(i) for i in range(100)])
        assert hll.is_subset("a", "b") is True

    def test_false_when_a_not_subset_of_b(self, hll):
        hll.add("a", *[str(i) for i in range(100)])
        hll.add("b", *[str(i) for i in range(50)])
        assert hll.is_subset("a", "b") is False

    def test_identical_sets_are_subsets_of_each_other(self, hll):
        elems = [str(i) for i in range(100)]
        hll.add("a", *elems)
        hll.add("b", *elems)
        assert hll.is_subset("a", "b") is True
        assert hll.is_subset("b", "a") is True
