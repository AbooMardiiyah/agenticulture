#!/usr/bin/env python3
"""
stream_data.py — Streaming data pipeline for AgentiCulture evaluation setup.

Downloads and processes Yelp, Amazon, and Goodreads datasets without keeping
raw files on disk permanently.

Disk usage:
  Yelp    — needs ~4.5 GB temp space for the ZIP (deleted after processing)
  Amazon  — zero temp disk space (true line-by-line streaming)
  Goodreads — zero temp disk space (true line-by-line streaming)
  Output  — ~500 MB–1 GB total (item.json + review.json + user.json)

Usage:
  pip install requests tqdm
  python stream_data.py --output_dir ./data/eval
  python stream_data.py --output_dir ./data/eval --skip goodreads
  python stream_data.py --output_dir ./data/eval --only amazon
"""

import argparse
import gzip
import io
import json
import logging
import os
import tempfile
import uuid
import zipfile
from typing import Iterator, Set

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


YELP_ZIP_URL = "https://business.yelp.com/external-assets/files/Yelp-JSON.zip"
YELP_LOCAL_ZIP = os.path.join(os.path.dirname(__file__), "Yelp-JSON.zip")
YELP_TOP_CITIES = {"Philadelphia", "Tampa", "Tucson"}

AMAZON_REVIEW_URLS = {
    "Industrial_and_Scientific": (
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
        "/raw/review_categories/Industrial_and_Scientific.jsonl.gz"
    ),
    "Musical_Instruments": (
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
        "/raw/review_categories/Musical_Instruments.jsonl.gz"
    ),
    "Video_Games": (
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
        "/raw/review_categories/Video_Games.jsonl.gz"
    ),
}

AMAZON_META_URLS = {
    "Industrial_and_Scientific": (
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
        "/raw/meta_categories/meta_Industrial_and_Scientific.jsonl.gz"
    ),
    "Musical_Instruments": (
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
        "/raw/meta_categories/meta_Musical_Instruments.jsonl.gz"
    ),
    "Video_Games": (
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
        "/raw/meta_categories/meta_Video_Games.jsonl.gz"
    ),
}

GOODREADS_REVIEW_URLS = {
    "children": (
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"
        "/byGenre/goodreads_reviews_children.json.gz"
    ),
    "comics_graphic": (
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"
        "/byGenre/goodreads_reviews_comics_graphic.json.gz"
    ),
    "poetry": (
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"
        "/byGenre/goodreads_reviews_poetry.json.gz"
    ),
}

GOODREADS_BOOK_URLS = {
    "children": (
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"
        "/byGenre/goodreads_books_children.json.gz"
    ),
    "comics_graphic": (
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"
        "/byGenre/goodreads_books_comics_graphic.json.gz"
    ),
    "poetry": (
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"
        "/byGenre/goodreads_books_poetry.json.gz"
    ),
}


def _stream_gz_lines(url: str, desc: str = "") -> Iterator[dict]:
    """
    Stream a remote .jsonl.gz (or .json.gz) file line-by-line without writing
    it to disk. Yields parsed dicts.

    Sets decode_content=False so urllib3 doesn't auto-decompress the HTTP
    transfer — we handle decompression ourselves via gzip.open.
    """
    with requests.get(url, stream=True, timeout=(15, 600)) as r:
        r.raise_for_status()
        r.raw.decode_content = False        
        with gzip.open(r.raw, "rt", encoding="utf-8", errors="replace") as gz:
            for line in tqdm(gz, desc=desc or url.split("/")[-1], unit=" lines", leave=False):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _download_to_temp(url: str, desc: str) -> str:
    """
    Download a file to a named temp file on disk (used only for the Yelp ZIP
    which cannot be seeked without full download first).
    Returns the temp file path — caller is responsible for deleting it.
    """
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        path = tmp.name
        with requests.get(url, stream=True, timeout=(15, 3600)) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20): 
                    tmp.write(chunk)
                    bar.update(len(chunk))
    return path



def process_yelp(out_items, out_reviews, out_users) -> None:
    logger.info("━━━  YELP  ━━━")

    if os.path.exists(YELP_LOCAL_ZIP):
        logger.info(f"Using local ZIP: {YELP_LOCAL_ZIP}")
        tmp_path = YELP_LOCAL_ZIP
        delete_after = False
    else:
        logger.info("No local ZIP found — downloading (needs ~4.5 GB temp disk space).")
        tmp_path = _download_to_temp(YELP_ZIP_URL, "Yelp ZIP ↓")
        delete_after = True

    try:
        import tarfile

        with zipfile.ZipFile(tmp_path, "r") as zf:
            tar_entry = next(n for n in zf.namelist() if n.endswith(".tar"))
            logger.info(f"Extracting inner TAR: {tar_entry}")
            tar_tmp = os.path.join(tempfile.gettempdir(), "yelp_dataset.tar")
            with zf.open(tar_entry) as src, open(tar_tmp, "wb") as dst:
                total = zf.getinfo(tar_entry).file_size
                with tqdm(total=total, unit="B", unit_scale=True, desc="Extracting TAR") as bar:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
                        bar.update(len(chunk))

        def _iter_tar_member(tf, keyword):
            member = next(m for m in tf.getmembers()
                         if keyword in m.name.lower() and m.name.endswith(".json"))
            logger.info(f"  reading {member.name}")
            return io.TextIOWrapper(tf.extractfile(member), encoding="utf-8")

        with tarfile.open(tar_tmp, "r") as tf:
            members = [m.name for m in tf.getmembers()]
            logger.info(f"TAR contents: {members}")

            logger.info("Pass 1/3  filtering businesses for target cities…")
            biz_ids: Set[str] = set()
            with tarfile.open(tar_tmp, "r") as tf:
                for line in tqdm(_iter_tar_member(tf, "business"),
                                 desc="businesses", unit=" lines", leave=False):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("city") not in YELP_TOP_CITIES:
                        continue
                    biz_ids.add(obj["business_id"])
                    item = {**obj, "item_id": obj["business_id"], "source": "yelp", "type": "business"}
                    del item["business_id"]
                    out_items.write(json.dumps(item) + "\n")

            logger.info(f"  → {len(biz_ids):,} businesses kept")

            logger.info("Pass 2/3  filtering reviews…")
            user_ids: Set[str] = set()
            with tarfile.open(tar_tmp, "r") as tf:
                for line in tqdm(_iter_tar_member(tf, "review"),
                                 desc="reviews", unit=" lines", leave=False):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("business_id") not in biz_ids:
                        continue
                    user_ids.add(obj["user_id"])
                    review = {**obj, "item_id": obj["business_id"], "source": "yelp", "type": "business"}
                    del review["business_id"]
                    out_reviews.write(json.dumps(review) + "\n")

            logger.info(f"  → {len(user_ids):,} unique users in filtered reviews")

            logger.info("Pass 3/3  filtering users…")
            kept = 0
            with tarfile.open(tar_tmp, "r") as tf:
                for line in tqdm(_iter_tar_member(tf, "user"),
                                 desc="users", unit=" lines", leave=False):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("user_id") not in user_ids:
                        continue
                    out_users.write(json.dumps({**obj, "source": "yelp"}) + "\n")
                    kept += 1

            logger.info(f"  → {kept:,} user records written")

        os.unlink(tar_tmp)
        logger.info("Yelp temp TAR deleted.")

    finally:
        if delete_after:
            os.unlink(tmp_path)
            logger.info("Yelp temp ZIP deleted.")


def process_amazon(out_items, out_reviews, out_users) -> None:
    logger.info("━━━  AMAZON  ━━━")

    logger.info("Pass 1/2  streaming reviews (collecting item + user IDs)…")
    valid_items: Set[str] = set()
    valid_users: Set[str] = set()

    for category, url in AMAZON_REVIEW_URLS.items():
        logger.info(f"  {category}")
        count = 0
        for obj in _stream_gz_lines(url, desc=f"  {category}"):
            parent_asin = obj.get("parent_asin")
            user_id     = obj.get("user_id")
            if not parent_asin or not user_id:
                continue
            valid_items.add(parent_asin)
            valid_users.add(user_id)
            review = {
                "review_id":        obj.get("review_id") or str(uuid.uuid4()),
                "user_id":          user_id,
                "item_id":          parent_asin,
                "sub_item_id":      obj.get("asin", ""),
                "stars":            float(obj.get("rating") or obj.get("stars") or 0),
                "text":             obj.get("text", ""),
                "title":            obj.get("title", ""),
                "timestamp":        obj.get("timestamp", ""),
                "verified_purchase": obj.get("verified_purchase", False),
                "helpful_vote":     obj.get("helpful_vote", 0),
                "source":           "amazon",
                "type":             "product",
            }
            out_reviews.write(json.dumps(review) + "\n")
            count += 1
        logger.info(f"    → {count:,} reviews")

    logger.info(f"  Total unique items: {len(valid_items):,}  |  users: {len(valid_users):,}")

    for uid in valid_users:
        out_users.write(json.dumps({"user_id": uid, "source": "amazon"}) + "\n")

    logger.info("Pass 2/2  streaming item metadata…")
    for category, url in AMAZON_META_URLS.items():
        logger.info(f"  {category}")
        count = 0
        for obj in _stream_gz_lines(url, desc=f"  meta_{category}"):
            if obj.get("parent_asin") not in valid_items:
                continue
            item = {**obj, "item_id": obj["parent_asin"], "source": "amazon", "type": "product"}
            item.pop("parent_asin", None)
            out_items.write(json.dumps(item) + "\n")
            count += 1
        logger.info(f"    → {count:,} items")



def process_goodreads(out_items, out_reviews, out_users) -> None:
    logger.info("━━━  GOODREADS  ━━━")

    logger.info("Pass 1/2  streaming reviews (collecting book + user IDs)…")
    valid_books: Set[str] = set()
    valid_users: Set[str] = set()

    for genre, url in GOODREADS_REVIEW_URLS.items():
        logger.info(f"  {genre}")
        count = 0
        for obj in _stream_gz_lines(url, desc=f"  {genre}"):
            book_id = obj.get("book_id")
            user_id = obj.get("user_id")
            if not book_id or not user_id:
                continue
            valid_books.add(book_id)
            valid_users.add(user_id)
            review = {
                "review_id":  obj.get("review_id") or str(uuid.uuid4()),
                "user_id":    user_id,
                "item_id":    book_id,
                "stars":      float(obj.get("rating") or obj.get("stars") or 0),
                "text":       obj.get("review_text") or obj.get("text", ""),
                "date_added": obj.get("date_added", ""),
                "source":     "goodreads",
                "type":       "book",
            }
            out_reviews.write(json.dumps(review) + "\n")
            count += 1
        logger.info(f"    → {count:,} reviews")

    logger.info(f"  Total unique books: {len(valid_books):,}  |  users: {len(valid_users):,}")

    for uid in valid_users:
        out_users.write(json.dumps({"user_id": uid, "source": "goodreads"}) + "\n")
    logger.info("Pass 2/2  streaming book metadata…")
    for genre, url in GOODREADS_BOOK_URLS.items():
        logger.info(f"  {genre}")
        count = 0
        for obj in _stream_gz_lines(url, desc=f"  books_{genre}"):
            if obj.get("book_id") not in valid_books:
                continue
            item = {**obj, "item_id": obj["book_id"], "source": "goodreads", "type": "book"}
            item.pop("book_id", None)
            out_items.write(json.dumps(item) + "\n")
            count += 1
        logger.info(f"    → {count:,} books")



SOURCES = ("yelp", "amazon", "goodreads")
PROCESSORS = {
    "yelp":      process_yelp,
    "amazon":    process_amazon,
    "goodreads": process_goodreads,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream Yelp + Amazon + Goodreads datasets and produce data/ files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output_dir", default="./data/eval",
                        help="Directory to write item.json, review.json, user.json (default: ./data/eval)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--skip", nargs="+", choices=SOURCES, metavar="SOURCE",
                       help="Skip one or more sources, e.g. --skip goodreads")
    group.add_argument("--only", nargs="+", choices=SOURCES, metavar="SOURCE",
                       help="Run only these sources, e.g. --only amazon")
    args = parser.parse_args()

    to_run = set(SOURCES)
    if args.skip:
        to_run -= set(args.skip)
    if args.only:
        to_run = set(args.only)

    os.makedirs(args.output_dir, exist_ok=True)
    item_path   = os.path.join(args.output_dir, "item.json")
    review_path = os.path.join(args.output_dir, "review.json")
    user_path   = os.path.join(args.output_dir, "user.json")

    files_exist = any(os.path.exists(p) for p in (item_path, review_path, user_path))
    mode = "a" if files_exist else "w"
    if files_exist:
        logger.info("Existing data files found — appending (not overwriting).")

    for source in SOURCES:
        if source not in to_run:
            logger.info(f"Skipping {source}.")
            continue
        with (
            open(item_path,   mode, encoding="utf-8") as out_items,
            open(review_path, mode, encoding="utf-8") as out_reviews,
            open(user_path,   mode, encoding="utf-8") as out_users,
        ):
            PROCESSORS[source](out_items, out_reviews, out_users)
        mode = "a"  

    logger.info("\nOutput files:")
    for path in (item_path, review_path, user_path):
        if os.path.exists(path):
            mb = os.path.getsize(path) / (1024 * 1024)
            logger.info(f"  {path}  ({mb:.0f} MB)")


if __name__ == "__main__":
    main()
