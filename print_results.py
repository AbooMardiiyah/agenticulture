#!/usr/bin/env python3
"""
print_results.py — Print evaluation metrics from saved result files.

Usage:
    python print_results.py
    python print_results.py --results_dir ./eval_results
    python print_results.py --per-task          # also print per-task breakdown
    python print_results.py --per-task --platform yelp
"""
import argparse
import json
import os

def load(path):
    with open(path) as f:
        return json.load(f)

def print_task_a_per_task(r, platform):
    outputs = r.get("simulation_outputs", [])
    if not outputs:
        print("  (no per-task data)")
        return
    print(f"\n  --- Task A per-task: {platform} ---")
    print(f"  {'#':>4} {'Pred':>6} {'Actual':>7} {'Error':>7}  Review (first 60 chars)")
    print("  " + "-" * 80)
    for i, entry in enumerate(outputs):
        pred   = entry.get("output", {}).get("stars", "?")
        actual = entry.get("groundtruth", {}).get("stars", "?")
        review = entry.get("output", {}).get("review", "")[:60]
        try:
            err = f"{abs(float(pred) - float(actual)):>7.2f}"
        except Exception:
            err = "     ?"
        pred_s   = f"{float(pred):>6.1f}" if pred != "?" else "     ?"
        actual_s = f"{float(actual):>7.1f}" if actual != "?" else "      ?"
        print(f"  {i+1:>4} {pred_s} {actual_s} {err}  {review}")

def print_task_b_per_task(r, platform):
    outputs = r.get("simulation_outputs", [])
    if not outputs:
        print("  (no per-task data)")
        return
    print(f"\n  --- Task B per-task: {platform} ---")
    print(f"  {'#':>4} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'GT Rank':>8}  Ground-truth item")
    print("  " + "-" * 72)
    for i, entry in enumerate(outputs):
        pred_list = entry.get("output", [])
        gt        = entry.get("groundtruth", "")
        try:
            rank = pred_list.index(gt) + 1  # 1-indexed
        except ValueError:
            rank = None
        h1 = "Y" if rank is not None and rank <= 1  else "-"
        h3 = "Y" if rank is not None and rank <= 3  else "-"
        h5 = "Y" if rank is not None and rank <= 5  else "-"
        rank_s = f"{rank:>8}" if rank is not None else "   >len"
        gt_short = gt[:40] if gt else "(none)"
        print(f"  {i+1:>4} {h1:>6} {h3:>6} {h5:>6} {rank_s}  {gt_short}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./eval_results")
    parser.add_argument("--per-task", action="store_true",
                        help="Print per-task breakdown after aggregate tables")
    parser.add_argument("--platform", choices=["yelp", "amazon", "goodreads"],
                        default=None, help="Limit per-task output to one platform")
    args = parser.parse_args()
    d = args.results_dir
    platforms = [args.platform] if args.platform else ["yelp", "amazon", "goodreads"]

    print("\n=== TASK A: User Modeling ===")
    print(f"{'Platform':<12} {'Pref Est':>10} {'RMSE':>8} {'Rev Gen':>10} {'ROUGE-1':>9} {'ROUGE-L':>9} {'Overall':>9}")
    print("-" * 72)
    task_a_data = {}
    for platform in ["yelp", "amazon", "goodreads"]:
        path = os.path.join(d, f"results_task_a_{platform}.json")
        if not os.path.exists(path):
            print(f"{platform:<12} (not found)")
            continue
        r = load(path)
        task_a_data[platform] = r
        m = r.get("metrics", {})
        e = r.get("extra_metrics", {})
        print(f"{platform:<12}"
              f" {m.get('preference_estimation',0):>10.4f}"
              f" {e.get('rmse') or 0:>8.4f}"
              f" {m.get('review_generation',0):>10.4f}"
              f" {e.get('rouge1') or 0:>9.4f}"
              f" {e.get('rougeL') or 0:>9.4f}"
              f" {m.get('overall_quality',0):>9.4f}")

    print("\n=== TASK B: Recommendation ===")
    print(f"{'Platform':<12} {'HR@1':>8} {'HR@3':>8} {'HR@5':>8} {'Avg HR':>9} {'NDCG@10':>10}")
    print("-" * 58)
    task_b_data = {}
    for platform in ["yelp", "amazon", "goodreads"]:
        path = os.path.join(d, f"results_task_b_{platform}.json")
        if not os.path.exists(path):
            print(f"{platform:<12} (not found)")
            continue
        r = load(path)
        task_b_data[platform] = r
        m = r.get("metrics", {})
        e = r.get("extra_metrics", {})
        print(f"{platform:<12}"
              f" {m.get('top_1_hit_rate',0):>8.4f}"
              f" {m.get('top_3_hit_rate',0):>8.4f}"
              f" {m.get('top_5_hit_rate',0):>8.4f}"
              f" {m.get('average_hit_rate',0):>9.4f}"
              f" {e.get('ndcg_at_10') or 0:>10.4f}")

    if args.per_task:
        for platform in platforms:
            if platform in task_a_data:
                print_task_a_per_task(task_a_data[platform], platform)
        for platform in platforms:
            if platform in task_b_data:
                print_task_b_per_task(task_b_data[platform], platform)

    print()

if __name__ == "__main__":
    main()
