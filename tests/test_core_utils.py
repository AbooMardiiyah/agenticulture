"""
tests/test_core_utils.py — Unit tests for core/utils.py

All tests are pure-logic — no LLM calls, no network, no dataset needed.
Run with:
    uv run pytest tests/test_core_utils.py -v
"""
import math
import pytest
from core.utils import (
    detect_platform,
    compute_cf_statistics,
    variance_adjusted_rating,
    select_informative_reviews,
    borda_count_aggregation,
)
from core.config import GLOBAL_MEAN_RATING


# ---------------------------------------------------------------------------
# detect_platform
# ---------------------------------------------------------------------------

class TestDetectPlatform:
    def test_yelp_detected_by_categories_and_attributes(self):
        item = {"categories": "Restaurants, Italian", "attributes": {"WiFi": "free"}}
        assert detect_platform(item) == "yelp"

    def test_goodreads_detected_by_title_without_series(self):
        item = {"title_without_series": "Harry Potter", "average_rating": 4.5}
        assert detect_platform(item) == "goodreads"

    def test_goodreads_detected_by_ratings_count(self):
        item = {"ratings_count": 50000, "title": "Dune"}
        assert detect_platform(item) == "goodreads"

    def test_amazon_is_default(self):
        item = {"title": "USB Cable", "price": 9.99, "description": "Fast charging"}
        assert detect_platform(item) == "amazon"

    def test_amazon_explicit_fields(self):
        item = {"parent_asin": "B001XYZ", "price": 19.99, "store": "ElectroStore"}
        assert detect_platform(item) == "amazon"

    def test_empty_item_defaults_to_amazon(self):
        assert detect_platform({}) == "amazon"

    def test_yelp_requires_both_categories_and_attributes(self):
        # Only 'categories' alone should NOT be yelp
        item = {"categories": "Restaurants"}
        assert detect_platform(item) == "amazon"


# ---------------------------------------------------------------------------
# compute_cf_statistics
# ---------------------------------------------------------------------------

class TestComputeCFStatistics:
    def _make_reviews(self, stars_list):
        return [{"stars": s} for s in stars_list]

    def test_correct_user_mean(self):
        reviews = self._make_reviews([1, 2, 3, 4, 5])
        stats = compute_cf_statistics(reviews, [])
        assert stats["user_mean"] == pytest.approx(3.0)

    def test_correct_item_mean(self):
        reviews = self._make_reviews([4, 4, 5])
        stats = compute_cf_statistics([], reviews)
        assert stats["item_mean"] == pytest.approx(13/3)

    def test_empty_user_reviews_returns_global_mean(self):
        stats = compute_cf_statistics([], [])
        assert stats["user_mean"] == GLOBAL_MEAN_RATING

    def test_empty_item_reviews_returns_global_mean(self):
        stats = compute_cf_statistics([], [])
        assert stats["item_mean"] == GLOBAL_MEAN_RATING

    def test_single_review_variance_is_one(self):
        # Variance of a single rating is undefined — should return 1.0
        stats = compute_cf_statistics(self._make_reviews([5]), [])
        assert stats["user_variance"] == pytest.approx(1.0)

    def test_identical_ratings_low_variance(self):
        # All 5s → variance should be 0.0
        reviews = self._make_reviews([5, 5, 5, 5, 5])
        stats = compute_cf_statistics(reviews, [])
        assert stats["user_variance"] == pytest.approx(0.0)

    def test_spread_ratings_higher_variance(self):
        low_spread  = self._make_reviews([3, 3, 3, 3])
        high_spread = self._make_reviews([1, 2, 4, 5])
        stats_low  = compute_cf_statistics(low_spread,  [])
        stats_high = compute_cf_statistics(high_spread, [])
        assert stats_high["user_variance"] > stats_low["user_variance"]

    def test_missing_stars_field_uses_global_mean(self):
        reviews = [{"text": "Great place"}, {"text": "Bad service"}]
        stats = compute_cf_statistics(reviews, [])
        assert stats["user_mean"] == pytest.approx(GLOBAL_MEAN_RATING)

    def test_returns_all_four_keys(self):
        stats = compute_cf_statistics([], [])
        assert set(stats.keys()) == {"user_mean", "user_variance", "item_mean", "item_variance"}


# ---------------------------------------------------------------------------
# variance_adjusted_rating
# ---------------------------------------------------------------------------

class TestVarianceAdjustedRating:
    def test_output_clamped_to_minimum_1(self):
        result = variance_adjusted_rating(1.0, 1.0, 0.1, 0.1)
        assert result >= 1.0

    def test_output_clamped_to_maximum_5(self):
        result = variance_adjusted_rating(5.0, 5.0, 0.1, 0.1)
        assert result <= 5.0

    def test_output_rounded_to_nearest_half(self):
        result = variance_adjusted_rating(3.0, 3.0, 1.0, 1.0)
        # Result should be a multiple of 0.5
        assert (result * 2) == round(result * 2)

    def test_consistent_user_gets_higher_weight(self):
        # User with low variance (consistent) should pull result closer to their mean
        # User mean=5, item mean=1 — consistent user (low variance) should win
        high_weight = variance_adjusted_rating(5.0, 1.0, 0.1, 10.0)
        low_weight  = variance_adjusted_rating(5.0, 1.0, 10.0, 0.1)
        assert high_weight > low_weight

    def test_symmetric_means_return_close_to_mean(self):
        result = variance_adjusted_rating(4.0, 4.0, 1.0, 1.0)
        assert result == pytest.approx(4.0, abs=0.5)

    def test_global_mean_acts_as_prior(self):
        # With very high variances (low weights), result should be pulled toward global mean
        result = variance_adjusted_rating(1.0, 5.0, 100.0, 100.0)
        assert abs(result - GLOBAL_MEAN_RATING) < 1.5

    def test_valid_range_for_all_extremes(self):
        for u in [1.0, 3.0, 5.0]:
            for i in [1.0, 3.0, 5.0]:
                for var in [0.1, 1.0, 5.0]:
                    r = variance_adjusted_rating(u, i, var, var)
                    assert 1.0 <= r <= 5.0


# ---------------------------------------------------------------------------
# select_informative_reviews
# ---------------------------------------------------------------------------

class TestSelectInformativeReviews:
    def _make_review(self, stars, text="good", useful=0, funny=0, cool=0,
                     verified=False, n_votes=0):
        return {
            "stars": stars, "text": text,
            "useful": useful, "funny": funny, "cool": cool,
            "verified_purchase": verified, "n_votes": n_votes,
        }

    def test_returns_empty_for_no_reviews(self):
        assert select_informative_reviews([], "yelp") == []

    def test_respects_max_reviews(self):
        reviews = [self._make_review(3) for _ in range(20)]
        result = select_informative_reviews(reviews, "yelp", max_reviews=5)
        assert len(result) <= 5

    def test_rating_diversity_max_two_per_star(self):
        # 10 reviews all at 5 stars — with max_reviews=4, diversity cap (2) is
        # the binding constraint (max_reviews//2 = 2, so soft bypass ends at 2)
        reviews = [self._make_review(5, text="x" * 100) for _ in range(10)]
        result = select_informative_reviews(reviews, "yelp", max_reviews=4)
        five_star_count = sum(1 for r in result if r["stars"] == 5)
        assert five_star_count <= 2

    def test_yelp_useful_votes_boost_score(self):
        low  = self._make_review(3, text="okay", useful=0)
        high = self._make_review(3, text="okay", useful=100)
        result = select_informative_reviews([low, high], "yelp", max_reviews=1)
        assert result[0]["useful"] == 100

    def test_amazon_verified_purchase_boosted(self):
        unverified = self._make_review(4, text="good product", verified=False)
        verified   = self._make_review(4, text="good product", verified=True)
        result = select_informative_reviews([unverified, verified], "amazon", max_reviews=1)
        assert result[0]["verified_purchase"] is True

    def test_longer_reviews_preferred(self):
        short = self._make_review(3, text="ok")
        long  = self._make_review(3, text="This is a very detailed review " * 10)
        result = select_informative_reviews([short, long], "yelp", max_reviews=1)
        assert len(result[0]["text"]) > len(short["text"])

    def test_returns_list_of_dicts(self):
        reviews = [self._make_review(i) for i in range(1, 6)]
        result = select_informative_reviews(reviews, "amazon", max_reviews=3)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_goodreads_n_votes_boost(self):
        low  = self._make_review(4, text="great book", n_votes=0)
        high = self._make_review(4, text="great book", n_votes=500)
        result = select_informative_reviews([low, high], "goodreads", max_reviews=1)
        assert result[0]["n_votes"] == 500


# ---------------------------------------------------------------------------
# borda_count_aggregation
# ---------------------------------------------------------------------------

class TestBordaCountAggregation:
    CANDIDATES = ["A", "B", "C", "D", "E"]

    def test_returns_all_candidates(self):
        rankings = [["A", "B", "C", "D", "E"]]
        result = borda_count_aggregation(rankings, self.CANDIDATES)
        assert set(result) == set(self.CANDIDATES)

    def test_unanimous_top_item_wins(self):
        rankings = [
            ["A", "B", "C", "D", "E"],
            ["A", "C", "B", "E", "D"],
            ["A", "D", "E", "B", "C"],
        ]
        result = borda_count_aggregation(rankings, self.CANDIDATES)
        assert result[0] == "A"

    def test_majority_vote_overrides_minority(self):
        # B is top in 2/3 rankings, A is top in 1/3
        rankings = [
            ["A", "B", "C", "D", "E"],
            ["B", "A", "C", "D", "E"],
            ["B", "C", "A", "D", "E"],
        ]
        result = borda_count_aggregation(rankings, self.CANDIDATES)
        assert result[0] == "B"

    def test_empty_rankings_returns_candidates_unchanged(self):
        result = borda_count_aggregation([], self.CANDIDATES)
        assert result == self.CANDIDATES

    def test_single_ranking_preserved(self):
        ranking = ["E", "D", "C", "B", "A"]
        result = borda_count_aggregation([ranking], self.CANDIDATES)
        assert result[0] == "E"
        assert result[-1] == "A"

    def test_consistent_last_place(self):
        # E is last in all three rankings
        rankings = [
            ["A", "B", "C", "D", "E"],
            ["B", "A", "D", "C", "E"],
            ["C", "D", "A", "B", "E"],
        ]
        result = borda_count_aggregation(rankings, self.CANDIDATES)
        assert result[-1] == "E"

    def test_two_candidates(self):
        rankings = [["X", "Y"], ["X", "Y"], ["Y", "X"]]
        result = borda_count_aggregation(rankings, ["X", "Y"])
        assert result[0] == "X"

    def test_items_not_in_candidates_ignored(self):
        rankings = [["A", "B", "UNKNOWN", "C", "D", "E"]]
        result = borda_count_aggregation(rankings, self.CANDIDATES)
        assert "UNKNOWN" not in result
        assert set(result) == set(self.CANDIDATES)
