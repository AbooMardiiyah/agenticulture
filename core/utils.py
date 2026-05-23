"""
core/utils.py — Shared maths and helpers used by both agents.

Functions:
  detect_platform          — infer yelp/amazon/goodreads from item keys
  compute_cf_statistics    — mean + variance for user and item ratings
  variance_adjusted_rating — ASC's key formula (inverse-variance weighted blend)
  select_informative_reviews — platform-aware review scoring + selection
  borda_count_aggregation  — RecHackers voting method
"""
import math
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from core.config import PLATFORM_FEATURES, GLOBAL_MEAN_RATING

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def detect_platform(item_info: Dict) -> str:
    """
    Infer whether an item comes from Yelp, Goodreads, or Amazon
    by inspecting its metadata keys.
    """
    keys = set(item_info.keys())
    if "categories" in keys and "attributes" in keys:
        return "yelp"
    if "title_without_series" in keys or "ratings_count" in keys:
        return "goodreads"
    return "amazon"


# ---------------------------------------------------------------------------
# Collaborative filtering
# ---------------------------------------------------------------------------

def compute_cf_statistics(
    user_reviews: List[Dict],
    item_reviews: List[Dict],
    global_mean: float = GLOBAL_MEAN_RATING
) -> Dict:
    """
    Compute mean and variance for a user and an item from their review lists.

    Returns:
        {user_mean, user_variance, item_mean, item_variance}
    """
    def _stats(reviews: List[Dict]) -> Tuple[float, float]:
        if not reviews:
            return global_mean, 1.0
        ratings = [r.get("stars", global_mean) for r in reviews]
        mean = sum(ratings) / len(ratings)
        variance = (
            math.sqrt(sum((r - mean) ** 2 for r in ratings) / len(ratings))
            if len(ratings) > 1 else 1.0
        )
        return mean, variance

    user_mean, user_var = _stats(user_reviews)
    item_mean, item_var = _stats(item_reviews)

    return {
        "user_mean":     user_mean,
        "user_variance": user_var,
        "item_mean":     item_mean,
        "item_variance": item_var,
    }


def variance_adjusted_rating(
    user_mean: float,
    item_mean: float,
    user_variance: float,
    item_variance: float,
    global_mean: float = GLOBAL_MEAN_RATING
) -> float:
    """
    Users who rate everything the same (low variance → high weight).
    Items with very spread ratings (high variance → low weight).

    """
    w_u = 1.0 / max(user_variance, 0.1)
    w_i = 1.0 / max(item_variance, 0.1)
    total = w_u + w_i + 0.5

    predicted = (user_mean * w_u + item_mean * w_i + global_mean * 0.5) / total
    predicted = max(1.0, min(5.0, predicted))
    return round(predicted * 2) / 2   


# ---------------------------------------------------------------------------
# Review selection  (platform-aware, used by both agents)
# ---------------------------------------------------------------------------

def _score_review(review: Dict, platform: str) -> float:
    """Score a review for informativeness using platform-specific signals."""
    config  = PLATFORM_FEATURES.get(platform, PLATFORM_FEATURES["amazon"])
    weights = config["review_weights"]
    score   = 0.0
    text    = review.get("text", "")

    score += min(len(text.split()) / 50.0, 1.0) * weights.get("length", 0.1)
    score += (review.get("stars", 3) / 5.0) * weights.get("stars", 0.1)

    if platform == "yelp":
        score += review.get("useful", 0) * weights.get("useful", 0)
        score += review.get("funny",  0) * weights.get("funny",  0)
        score += review.get("cool",   0) * weights.get("cool",   0)
        if review.get("stars") in [1, 2, 5]:
            score += 0.5
    elif platform == "amazon":
        if review.get("verified_purchase", False):
            score += weights.get("verified_purchase", 0)
        if review.get("timestamp"):
            score += 0.2
    elif platform == "goodreads":
        score += review.get("n_votes",    0) * 0.01 * weights.get("n_votes",    0)
        score += review.get("n_comments", 0) * 0.01 * weights.get("n_comments", 0)
        if review.get("read_at"):
            score += 0.3

    return score


def select_informative_reviews(
    reviews: List[Dict],
    platform: str,
    max_reviews: int = 5
) -> List[Dict]:
    """
    Return the most informative reviews using platform-specific signals,
    while ensuring rating diversity (no more than 2 reviews per star value).
    """
    if not reviews:
        return []

    scored = sorted(reviews, key=lambda r: _score_review(r, platform), reverse=True)

    selected: List[Dict] = []
    rating_counts: Dict[int, int] = defaultdict(int)

    for review in scored:
        stars = int(review.get("stars", 3))
        if rating_counts[stars] < 2 or len(selected) < max_reviews // 2:
            selected.append(review)
            rating_counts[stars] += 1
        if len(selected) >= max_reviews:
            break

    return selected


# ---------------------------------------------------------------------------
# Borda count
# ---------------------------------------------------------------------------

def borda_count_aggregation(
    rankings: List[List[str]],
    candidates: List[str]
) -> List[str]:
    """
    Merge multiple ranked lists into one via Borda count voting.

    Each item earns (n - rank_position) points per ranking list.
    The item with the most total points wins.
    """
    if not rankings:
        return candidates

    n = len(candidates)
    scores: Dict[str, float] = defaultdict(float)

    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            if item_id in candidates:
                scores[item_id] += (n - rank)

    return sorted(candidates, key=lambda x: scores.get(x, 0), reverse=True)
