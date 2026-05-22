import os

LLM_MODEL    = os.environ.get("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY", "")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

COLD_START_THRESHOLD = 3  
GLOBAL_MEAN_RATING   = 3.5

# ---------------------------------------------------------------------------
# Platform-aware feature config 
# Defines which item fields and review fields matter per platform,
# and how to weight informativeness signals.
# ---------------------------------------------------------------------------

PLATFORM_FEATURES = {
    "yelp": {
        "item_fields":    ["item_id", "name", "stars", "review_count", "categories", "attributes"],
        "review_fields":  ["text", "stars", "useful", "funny", "cool"],
        "review_weights": {"useful": 0.4, "funny": 0.2, "cool": 0.2, "stars": 0.2}
    },
    "amazon": {
        "item_fields":    ["item_id", "name", "stars", "review_count", "description", "price", "categories"],
        "review_fields":  ["text", "stars", "verified_purchase", "timestamp"],
        "review_weights": {"verified_purchase": 0.5, "stars": 0.3, "length": 0.2}
    },
    "goodreads": {
        "item_fields":    ["item_id", "name", "stars", "review_count", "authors", "publication_year", "similar_books"],
        "review_fields":  ["text", "stars", "n_votes", "n_comments", "date_added", "read_at"],
        "review_weights": {"n_votes": 0.3, "n_comments": 0.3, "stars": 0.2, "length": 0.2}
    }
}
