#!/usr/bin/env python3
"""
eval_task_b.py — Run Task B (recommendation) evaluation.

Usage:
    uv run python eval_task_b.py --platform yelp --tasks 100
    uv run python eval_task_b.py --platform all --tasks 100
    uv run python eval_task_b.py --platform yelp --tasks 100 --ablation cosine_only
    uv run python eval_task_b.py --platform yelp --tasks 100 --no-nigerian

Ablation modes:
    full          Full system (default)
    cosine_only   Embedding cosine similarity only — no LLM at all
    single_llm    Single LLM call, no voting
    borda_only    Borda voting, no cross-domain boost

Metrics reported:
    hr@1, hr@3, hr@5    Hit rate at K (framework default)
    average_hit_rate    Mean of HR@1/3/5 (framework default)
    ndcg@10             Normalised Discounted Cumulative Gain at 10
"""

import argparse
import json
import logging
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "evaluation", "AgentSocietyChallenge"))
sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.llm import LLMBase

from task_b.agent import Baseline666RecHackersAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------

class OpenAICompatWrapper(LLMBase):
    def __init__(self, api_key: str, model: str, base_url: str = None):
        super().__init__(model=model)
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def __call__(self, messages, model=None, temperature=0.0, max_tokens=500,
                 stop_strs=None, n=1, **_):
        last_exc = None
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=model or self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop_strs,
                    n=n,
                )
                if n == 1:
                    return resp.choices[0].message.content
                return [c.message.content for c in resp.choices]
            except Exception as exc:
                last_exc = exc
                wait = 10 * (2 ** attempt)
                logger.warning(f"LLM call failed (attempt {attempt+1}/5): {exc} — retrying in {wait}s")
                time.sleep(wait)
        raise last_exc

    def get_embedding_model(self):
        return None


# ---------------------------------------------------------------------------
# Ablation-aware agent wrapper
# ---------------------------------------------------------------------------

class TaskBSimWrapper(RecommendationAgent):
    """
    Wraps Baseline666RecHackersAgent and supports ablation modes
    that selectively disable voting or cross-domain boosting.
    """
    _shared_agent: Baseline666RecHackersAgent = None
    ablation_mode: str = "full"
    nigerian_context: bool = True

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        if TaskBSimWrapper._shared_agent is None:
            TaskBSimWrapper._shared_agent = Baseline666RecHackersAgent()
            logger.info("Baseline666RecHackersAgent initialised (shared instance).")
        self._agent = TaskBSimWrapper._shared_agent
        outer_llm = llm

        def llm_fn(messages, temperature=0.1, max_tokens=500, **_):
            return outer_llm(messages=messages, temperature=temperature, max_tokens=max_tokens)

        self._agent.llm = llm_fn

    def set_interaction_tool(self, tool):
        super().set_interaction_tool(tool)
        self._agent.set_interaction_tool(tool)

    def workflow(self) -> list:
        user_id        = self.task.get("user_id", "")
        candidate_list = self.task.get("candidate_list", [])
        mode           = TaskBSimWrapper.ablation_mode
        nigerian       = TaskBSimWrapper.nigerian_context

        result = self._agent.workflow(
            user_id=user_id,
            candidate_list=candidate_list,
            nigerian_context=nigerian,
            ablation_mode=mode,
        )
        return result


# ---------------------------------------------------------------------------
# NDCG@K computation
# ---------------------------------------------------------------------------

def compute_ndcg(outputs: list, groundtruths: list, k: int = 10) -> float:
    """
    Compute mean NDCG@K across all tasks.

    For each task, the ground-truth item has relevance=1; all others are 0.
    NDCG@K = DCG@K / IDCG@K where IDCG@K = 1/log2(2) = 1.0 (best case:
    ground truth at rank 1).
    If the ground-truth item does not appear in the top-K predictions,
    NDCG for that task is 0.
    """
    scores = []
    for gt_item, pred_list in zip(groundtruths, outputs):
        top_k = pred_list[:k]
        if gt_item in top_k:
            rank = top_k.index(gt_item) + 1          # 1-indexed
            dcg  = 1.0 / math.log2(rank + 1)         # relevance=1
            idcg = 1.0 / math.log2(2)                 # best possible = rank 1
            scores.append(dcg / idcg)
        else:
            scores.append(0.0)

    return round(sum(scores) / len(scores), 4) if scores else float("nan")


# ---------------------------------------------------------------------------
# Extended simulator that saves raw outputs for post-processing
# ---------------------------------------------------------------------------

class ExtendedSimulator(Simulator):
    def evaluate(self):
        base = super().evaluate()

        paired = []
        gt_count = len(self.groundtruth_data)
        for i, out in enumerate(self.simulation_outputs):
            gt_item = self.groundtruth_data[i].get("ground truth", "") if i < gt_count else ""
            paired.append({
                "task":        out.get("task", {}) if out else {},
                "output":      out.get("output", []) if out else [],
                "groundtruth": gt_item,
            })
        base["simulation_outputs"] = paired
        return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_platform(platform, n_tasks, llm, data_dir, ablation_mode, nigerian_context, output_path, args_workers=3, shared_simulator=None):
    base = os.path.join(
        os.path.dirname(__file__),
        "evaluation", "AgentSocietyChallenge", "example", "track2", platform,
    )
    TaskBSimWrapper._shared_agent  = None
    TaskBSimWrapper.ablation_mode  = ablation_mode
    TaskBSimWrapper.nigerian_context = nigerian_context

    if shared_simulator is not None:
        simulator = shared_simulator
        simulator.simulation_outputs = []
        simulator.tasks = []
        simulator.groundtruth_data = []
    else:
        simulator = ExtendedSimulator(data_dir=data_dir, device="cpu", cache=False)

    simulator.set_task_and_groundtruth(
        task_dir=os.path.join(base, "tasks"),
        groundtruth_dir=os.path.join(base, "groundtruth"),
    )
    simulator.set_agent(TaskBSimWrapper)
    simulator.set_llm(llm)
    simulator.run_simulation(number_of_tasks=n_tasks, enable_threading=True, max_workers=args_workers)

    # Save raw outputs before evaluation so a crash does not lose API calls
    raw_path = output_path.replace(".json", "_raw.json")
    raw_outputs = []
    gt_data = simulator.groundtruth_data
    for i, out in enumerate(simulator.simulation_outputs):
        gt_item = gt_data[i].get("ground truth", "") if i < len(gt_data) else ""
        raw_outputs.append({
            "task":        out.get("task", {}) if out else {},
            "output":      out.get("output", []) if out else [],
            "groundtruth": gt_item,
        })
    with open(raw_path, "w") as f:
        json.dump(raw_outputs, f, indent=2)
    logger.info(f"Raw outputs saved to {raw_path} ({len(raw_outputs)} tasks)")

    results = simulator.evaluate()

    # Compute NDCG@10 from saved raw outputs
    gt_items   = [e["groundtruth"] for e in results.get("simulation_outputs", [])]
    pred_lists = [e["output"]      for e in results.get("simulation_outputs", [])]
    ndcg10     = compute_ndcg(pred_lists, gt_items, k=10)

    results["extra_metrics"] = {"ndcg_at_10": ndcg10}

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform",  choices=["yelp", "amazon", "goodreads", "all"], default="yelp")
    parser.add_argument("--tasks",     default="100", help="Number of tasks or 'all'")
    parser.add_argument("--data_dir",  default="./data")
    parser.add_argument("--output",    default=None)
    parser.add_argument("--workers",   type=int, default=5)
    parser.add_argument("--model",     default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--ablation",  choices=["full","cosine_only","single_llm","borda_only"],
                        default="full")
    parser.add_argument("--no-nigerian", action="store_true",
                        help="Disable Nigerian cultural context")
    args = parser.parse_args()

    api_key  = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL") or None
    if not api_key:
        logger.error("Set LLM_API_KEY before running.")
        sys.exit(1)

    n_tasks   = None if args.tasks == "all" else int(args.tasks)
    platforms = ["yelp", "amazon", "goodreads"] if args.platform == "all" else [args.platform]
    nigerian  = not args.no_nigerian
    llm       = OpenAICompatWrapper(api_key=api_key, model=args.model, base_url=base_url)

    suffix = f"_{args.ablation}" if args.ablation != "full" else ""
    if not nigerian:
        suffix += "_no_culture"

    # Load evaluation models once — reused across all platforms
    logger.info("Loading evaluation models (once for all platforms)...")
    shared_sim = ExtendedSimulator(data_dir=args.data_dir, device="cpu", cache=False)

    for platform in platforms:
        logger.info(f"\n{'='*55}")
        logger.info(f"Platform: {platform} | Tasks: {args.tasks} | Ablation: {args.ablation}")

        output = args.output or f"./results_task_b_{platform}{suffix}.json"
        results = run_platform(
            platform=platform,
            n_tasks=n_tasks,
            llm=llm,
            data_dir=args.data_dir,
            ablation_mode=args.ablation,
            nigerian_context=nigerian,
            output_path=output,
            args_workers=args.workers,
            shared_simulator=shared_sim,
        )

        metrics = results.get("metrics", {})
        extra   = results.get("extra_metrics", {})

        print(f"\n--- {platform.upper()} [{args.ablation}] ---")
        print(f"  HR@1              : {metrics.get('top_1_hit_rate', 'N/A'):.4f}")
        print(f"  HR@3              : {metrics.get('top_3_hit_rate', 'N/A'):.4f}")
        print(f"  HR@5              : {metrics.get('top_5_hit_rate', 'N/A'):.4f}")
        print(f"  Avg Hit Rate      : {metrics.get('average_hit_rate', 'N/A'):.4f}")
        print(f"  NDCG@10           : {extra.get('ndcg_at_10', 'N/A')}")
        logger.info(f"Results saved to {output}")


if __name__ == "__main__":
    main()
