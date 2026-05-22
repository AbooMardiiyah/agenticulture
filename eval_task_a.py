#!/usr/bin/env python3
"""
eval_task_a.py — Run Task A (user modeling) evaluation.

Usage:
    uv run python eval_task_a.py --platform yelp --tasks 100
    uv run python eval_task_a.py --platform all --tasks 100
    uv run python eval_task_a.py --platform yelp --tasks 100 --ablation cf_only
    uv run python eval_task_a.py --platform yelp --tasks 100 --no-nigerian

Ablation modes:
    full        Full system (default)
    llm_direct  No CF, no MDILU, no COTSC — single LLM call only
    cf_only     CF rating + single LLM call, no MDILU, no COTSC
    cf_mdilu    CF + MDILU, no COTSC
    cf_cotsc    CF + COTSC, no MDILU

Metrics reported:
    preference_estimation   1 - MAE/5 (framework default)
    rmse                    sqrt(mean((pred_star - actual_star)^2))
    review_generation       sentiment + emotion + topic cosine (framework default)
    rouge1 / rougeL         n-gram overlap between generated and ground-truth review
    overall_quality         (preference_estimation + review_generation) / 2
"""

import argparse
import json
import logging
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "evaluation", "AgentSocietyChallenge"))
sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase

from task_a.agent import ASCUserModelingAgent

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
        import time
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

class TaskASimWrapper(SimulationAgent):
    """
    Wraps ASCUserModelingAgent and selectively disables components
    to support ablation study configurations.
    """
    _shared_agent: ASCUserModelingAgent = None
    ablation_mode: str = "full"
    nigerian_context: bool = True

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        if TaskASimWrapper._shared_agent is None:
            TaskASimWrapper._shared_agent = ASCUserModelingAgent()
            logger.info("ASCUserModelingAgent initialised (shared instance).")
        self._agent = TaskASimWrapper._shared_agent
        outer_llm = llm

        def llm_fn(messages, temperature=0.1, max_tokens=1000, **_):
            return outer_llm(messages=messages, temperature=temperature, max_tokens=max_tokens)

        self._agent.llm = llm_fn

    def set_interaction_tool(self, tool):
        super().set_interaction_tool(tool)
        self._agent.set_interaction_tool(tool)

    def workflow(self) -> dict:
        user_id = self.task.get("user_id", "")
        item_id = self.task.get("item_id", "")
        mode    = TaskASimWrapper.ablation_mode
        nigerian = TaskASimWrapper.nigerian_context

        if mode == "llm_direct":
            result = self._agent._run_llm_direct(user_id, item_id, nigerian)
        elif mode == "cf_only":
            result = self._agent._run_cf_only(user_id, item_id, nigerian)
        elif mode == "cf_mdilu":
            result = self._agent._run_cf_mdilu(user_id, item_id, nigerian)
        elif mode == "cf_cotsc":
            result = self._agent._run_cf_cotsc_no_mdilu(user_id, item_id, nigerian)
        else:  # full
            result = self._agent.workflow(
                user_id=user_id,
                item_id=item_id,
                nigerian_context=nigerian,
            )

        return {"stars": result.get("stars", 3.0), "review": result.get("review", "")}


# ---------------------------------------------------------------------------
# Extra metrics: RMSE and ROUGE
# ---------------------------------------------------------------------------

def compute_rmse(result_file: str, groundtruth_dir: str) -> float:
    """
    Compute RMSE between predicted and actual star ratings.
    Reads simulation outputs paired with their groundtruth files.
    """
    with open(result_file) as f:
        data = json.load(f)

    sim_outputs = data.get("simulation_outputs", [])
    if not sim_outputs:
        logger.warning("No simulation_outputs in result file for RMSE calculation.")
        return float("nan")

    sq_errors = []
    for entry in sim_outputs:
        pred_stars  = entry.get("output", {}).get("stars", 3.0)
        actual_stars = entry.get("groundtruth", {}).get("stars")
        if actual_stars is not None:
            sq_errors.append((pred_stars - actual_stars) ** 2)

    return math.sqrt(sum(sq_errors) / len(sq_errors)) if sq_errors else float("nan")


def compute_rouge(result_file: str) -> dict:
    """
    Compute ROUGE-1 and ROUGE-L F1 between generated and ground-truth reviews.
    Requires: pip install rouge-score
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge-score not installed. Run: uv add rouge-score")
        return {"rouge1": float("nan"), "rougeL": float("nan")}

    with open(result_file) as f:
        data = json.load(f)

    sim_outputs = data.get("simulation_outputs", [])
    if not sim_outputs:
        return {"rouge1": float("nan"), "rougeL": float("nan")}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1_scores, rL_scores = [], []

    for entry in sim_outputs:
        pred_review   = entry.get("output", {}).get("review", "")
        actual_review = entry.get("groundtruth", {}).get("review", "")
        if pred_review and actual_review:
            scores = scorer.score(actual_review, pred_review)
            r1_scores.append(scores["rouge1"].fmeasure)
            rL_scores.append(scores["rougeL"].fmeasure)

    if not r1_scores:
        return {"rouge1": float("nan"), "rougeL": float("nan")}

    return {
        "rouge1": round(sum(r1_scores) / len(r1_scores), 4),
        "rougeL": round(sum(rL_scores) / len(rL_scores), 4),
    }


# ---------------------------------------------------------------------------
# Extended simulator that saves raw outputs for post-processing
# ---------------------------------------------------------------------------

class ExtendedSimulator(Simulator):
    """
    Subclass that patches evaluate() to also save raw (output, groundtruth)
    pairs into the JSON results file so RMSE and ROUGE can be computed later.
    """

    def evaluate(self):
        base = super().evaluate()

        paired = []
        gt_count = len(self.groundtruth_data)
        for i, out in enumerate(self.simulation_outputs):
            gt = self.groundtruth_data[i] if i < gt_count else {}
            paired.append({
                "task":        out.get("task", {}) if out else {},
                "output":      out.get("output", {}) if out else {},
                "groundtruth": gt,
            })
        base["simulation_outputs"] = paired
        return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_platform(platform, n_tasks, llm, data_dir, ablation_mode, nigerian_context, output_path, args_workers=3, shared_simulator=None):
    base = os.path.join(
        os.path.dirname(__file__),
        "evaluation", "AgentSocietyChallenge", "example", "track1", platform,
    )
    TaskASimWrapper._shared_agent = None
    TaskASimWrapper.ablation_mode  = ablation_mode
    TaskASimWrapper.nigerian_context = nigerian_context

    if shared_simulator is not None:
        simulator = shared_simulator
        # Reset outputs and tasks for the new platform
        simulator.simulation_outputs = []
        simulator.tasks = []
        simulator.groundtruth_data = []
    else:
        simulator = ExtendedSimulator(data_dir=data_dir, device="cpu", cache=False)
    simulator.set_task_and_groundtruth(
        task_dir=os.path.join(base, "tasks"),
        groundtruth_dir=os.path.join(base, "groundtruth"),
    )
    simulator.set_agent(TaskASimWrapper)
    simulator.set_llm(llm)
    simulator.run_simulation(number_of_tasks=n_tasks, enable_threading=True, max_workers=args_workers)

    # Save raw simulation outputs to disk immediately before evaluation
    # so a crash during the evaluation phase does not lose the API calls.
    raw_path = output_path.replace(".json", "_raw.json")
    raw_outputs = []
    gt_data = simulator.groundtruth_data
    for i, out in enumerate(simulator.simulation_outputs):
        gt = gt_data[i] if i < len(gt_data) else {}
        raw_outputs.append({
            "task":        out.get("task", {}) if out else {},
            "output":      out.get("output", {}) if out else {},
            "groundtruth": gt,
        })
    with open(raw_path, "w") as f:
        json.dump(raw_outputs, f, indent=2)
    logger.info(f"Raw outputs saved to {raw_path} ({len(raw_outputs)} tasks)")

    results = simulator.evaluate()

    # Compute extra metrics
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    rouge  = compute_rouge(output_path)
    rmse   = _compute_rmse_from_results(results)

    results["extra_metrics"] = {
        "rmse":   round(rmse, 4) if not math.isnan(rmse) else None,
        "rouge1": rouge["rouge1"],
        "rougeL": rouge["rougeL"],
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def _compute_rmse_from_results(results: dict) -> float:
    sq_errors = []
    for entry in results.get("simulation_outputs", []):
        pred   = entry.get("output", {}).get("stars", 3.0)
        actual = entry.get("groundtruth", {}).get("stars")
        if actual is not None:
            sq_errors.append((pred - actual) ** 2)
    return math.sqrt(sum(sq_errors) / len(sq_errors)) if sq_errors else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform",  choices=["yelp", "amazon", "goodreads", "all"], default="yelp")
    parser.add_argument("--tasks",     default="100", help="Number of tasks or 'all'")
    parser.add_argument("--data_dir",  default="./data")
    parser.add_argument("--output",    default=None)
    parser.add_argument("--workers",   type=int, default=5)
    parser.add_argument("--model",     default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--ablation",  choices=["full","llm_direct","cf_only","cf_mdilu","cf_cotsc"],
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

    # Create simulator once — loads emotion model (~953MB) only once
    # and reuses it across all platforms
    logger.info("Loading evaluation models (once for all platforms)...")
    shared_sim = ExtendedSimulator(data_dir=args.data_dir, device="cpu", cache=False)

    for platform in platforms:
        logger.info(f"\n{'='*55}")
        logger.info(f"Platform: {platform} | Tasks: {args.tasks} | Ablation: {args.ablation}")

        output = args.output or f"./results_task_a_{platform}{suffix}.json"
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
        print(f"  Preference Estimation : {metrics.get('preference_estimation', 'N/A'):.4f}")
        print(f"  RMSE                  : {extra.get('rmse', 'N/A')}")
        print(f"  Review Generation     : {metrics.get('review_generation', 'N/A'):.4f}")
        print(f"  ROUGE-1               : {extra.get('rouge1', 'N/A')}")
        print(f"  ROUGE-L               : {extra.get('rougeL', 'N/A')}")
        print(f"  Overall Quality       : {metrics.get('overall_quality', 'N/A'):.4f}")
        logger.info(f"Results saved to {output}")


if __name__ == "__main__":
    main()
