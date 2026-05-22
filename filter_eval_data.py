#!/usr/bin/env python3
"""
filter_eval_data.py — Extract only the users/items/reviews needed by the
evaluation task files, producing a small ./data/eval/ subset that loads
instantly into the Simulator without running out of memory.

Usage:
    uv run python filter_eval_data.py
"""
import glob
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR     = "./data"
EVAL_DIR     = "./data/eval"
TASK_BASE    = "./evaluation/AgentSocietyChallenge/example"


def collect_needed_ids():
    """Scan all task files and collect every user_id and item_id referenced."""
    user_ids: set = set()
    item_ids: set = set()

    for track in ["track1", "track2"]:
        for platform in ["yelp", "amazon", "goodreads"]:
            path = os.path.join(TASK_BASE, track, platform, "tasks")
            if not os.path.exists(path):
                continue
            for f in glob.glob(os.path.join(path, "*.json")):
                with open(f) as fh:
                    t = json.load(fh)
                    if t.get("user_id"):
                        user_ids.add(t["user_id"])
                    if t.get("item_id"):
                        item_ids.add(t["item_id"])
                    for c in t.get("candidate_list", []):
                        item_ids.add(c)

    logger.info(f"Tasks reference {len(user_ids):,} users and {len(item_ids):,} items")
    return user_ids, item_ids


def filter_file(src, dst, keep_ids, id_field, label):
    kept = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get(id_field) in keep_ids:
                fout.write(line)
                kept += 1
    logger.info(f"  {label}: {kept:,} records kept")
    return kept


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    user_ids, item_ids = collect_needed_ids()

    logger.info("Filtering users…")
    filter_file(
        os.path.join(DATA_DIR, "user.json"),
        os.path.join(EVAL_DIR, "user.json"),
        user_ids, "user_id", "users"
    )

    logger.info("Filtering items…")
    filter_file(
        os.path.join(DATA_DIR, "item.json"),
        os.path.join(EVAL_DIR, "item.json"),
        item_ids, "item_id", "items"
    )

    logger.info("Filtering reviews (by user_id OR item_id)…")
    kept = 0
    with open(os.path.join(DATA_DIR, "review.json")) as fin, \
         open(os.path.join(EVAL_DIR, "review.json"), "w") as fout:
        for line in fin:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("user_id") in user_ids or obj.get("item_id") in item_ids:
                fout.write(line)
                kept += 1
    logger.info(f"  reviews: {kept:,} records kept")

    logger.info("\nOutput sizes:")
    for f in ("user.json", "item.json", "review.json"):
        path = os.path.join(EVAL_DIR, f)
        mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"  {path}  ({mb:.1f} MB)")

    logger.info("\nDone. Point eval scripts at --data_dir ./data/eval")


if __name__ == "__main__":
    main()
