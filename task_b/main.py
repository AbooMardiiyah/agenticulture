import logging
import os
import time
import uvicorn
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from task_b.agent import Baseline666RecHackersAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="AgentiCulture — Task B: Recommendation",
    description=(
        "Platform-aware features · Self-consistency Borda voting · "
        "Cold-start · Cross-domain · Multi-turn · Nigerian cultural context"
    ),
    version="3.0.0",
)

agent = Baseline666RecHackersAgent()


def _init_llm():
    if agent.llm:
        return
    api_key  = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("No API key set — LLM calls will return stub responses.")
        return
    try:
        from openai import OpenAI
        base_url = os.environ.get("LLM_BASE_URL") or None
        model    = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct-Turbo")
        client   = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"LLM provider: {base_url or 'OpenAI'} | model: {model}")

        def llm_fn(messages, temperature=0.1, max_tokens=500, **_):
            last_exc = None
            for attempt in range(5):
                try:
                    resp = client.chat.completions.create(
                        model=model, messages=messages,
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    return resp.choices[0].message.content
                except Exception as exc:
                    last_exc = exc
                    wait = 10 * (2 ** attempt)
                    logger.warning(f"LLM call failed (attempt {attempt+1}/5): {exc} — retrying in {wait}s")
                    time.sleep(wait)
            raise last_exc

        agent.llm = llm_fn
        logger.info(f"LLM initialised: {model}")
    except Exception as exc:
        logger.warning(f"LLM init failed: {exc}")


def _init_interaction_tool():
    if agent.interaction_tool:
        return
    data_dir = os.environ.get("DATA_DIR", "")
    if data_dir and os.path.exists(data_dir):
        try:
            from websocietysimulator.tools import CacheInteractionTool
            agent.set_interaction_tool(CacheInteractionTool(data_dir))
            logger.info(f"Interaction tool loaded from {data_dir}")
        except Exception as exc:
            logger.warning(f"Interaction tool unavailable: {exc}")


class RecommendRequest(BaseModel):
    # Direct mode fields
    persona: str = Field(default="", description="User persona text — required for direct mode")
    context: str = Field(default="", description="Additional context (occasion, mood, etc.) — direct mode only")
    category: Optional[str] = Field(default=None, description="Item category filter — optional")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of items to return")

    # Benchmark mode fields
    user_id:        Optional[str]       = Field(default=None, description="Required for benchmark mode — looks up user history from dataset")
    candidate_list: Optional[List[str]] = Field(default=None, description="Required for both modes — list of item IDs to rank")

    # Optional item metadata for direct mode
    candidate_items: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Item metadata for direct mode (optional). Each entry must include 'item_id' plus descriptive fields (name, description, category). Enables semantic ranking without a dataset.",
    )

    nigerian_context: bool = Field(default=True, description="Enable Nigerian cultural context in ranking")
    session_id: Optional[str] = Field(default=None, description="Pass the same ID across turns for multi-turn session context")


class RecommendResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    reasoning: str = Field(default="")
    cold_start:  bool = Field(default=False)
    session_turn: int = Field(default=1)
    mode: str = Field(default="direct")


# Endpoints

@app.get("/")
async def root():
    return {
        "name":     "AgentiCulture — Task B: Recommendation",
        "version":  "3.0.0",
        "endpoint": "POST /recommend",
        "docs":     "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status":          "healthy",
        "llm_ready":       agent.llm is not None,
        "embedding_ready": agent.sentence_model is not None,
        "data_ready":      agent.interaction_tool is not None,
        "cached_users":    len(agent.preference_cache._cache) if agent.preference_cache else 0,
        "active_sessions": len(agent.sessions),
    }


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Generate personalised recommendations.

    Two modes:
    - **Benchmark mode**: provide `user_id` + `candidate_list`. Requires dataset
      mounted at DATA_DIR. Runs the full pipeline (cold-start detection,
      cross-domain, 3× LLM self-consistency + Borda, session state).
    - **Direct mode**: provide `persona` + `context`. No dataset needed.
      Returns a stub response (no ranking without candidate items).

    For multi-turn context, pass the same `session_id` string across requests.
    """
    _init_llm()
    _init_interaction_tool()

    try:
        # Benchmark mode
        if agent.interaction_tool and request.user_id and request.candidate_list:
            # Check cold-start flag for the response
            is_cold = False
            turn_num = 1
            try:
                reviews = agent.interaction_tool.get_reviews(user_id=request.user_id) or []
                is_cold = len(reviews) == 0
            except Exception:
                pass

            if request.session_id and request.session_id in agent.sessions:
                turn_num = len(agent.sessions[request.session_id].turns) + 1

            result = agent.workflow(
                user_id=request.user_id,
                candidate_list=request.candidate_list,
                nigerian_context=request.nigerian_context,
                session_id=request.session_id,
                persona=request.persona,
            )

            items = [
                {"product_id": pid, "rank": i + 1}
                for i, pid in enumerate(result[: request.top_k])
            ]
            reasoning = (
                "Cold-start: ranked by persona embedding similarity."
                if is_cold
                else f"Platform-aware · {agent.num_samples}× self-consistency · Borda count aggregation"
            )
            return RecommendResponse(
                recommendations=items,
                reasoning=reasoning,
                cold_start=is_cold,
                session_turn=turn_num,
                mode="benchmark",
            )

        # Direct mode — persona + candidate_list, no dataset
        # Uses cold-start persona-embedding ranking (no interaction tool needed)
        if request.candidate_list:
            item_features = (
                {c["item_id"]: c for c in request.candidate_items if "item_id" in c}
                if request.candidate_items
                else {}
            )
            result = agent._cold_start_rank(
                candidate_list=request.candidate_list,
                persona_text=request.persona or request.context,
                user_info={},
                item_features=item_features,
            )
            result = agent._fill_missing(result, request.candidate_list)
            items = [
                {"product_id": pid, "rank": i + 1}
                for i, pid in enumerate(result[: request.top_k])
            ]
            return RecommendResponse(
                recommendations=items,
                reasoning="Direct mode: ranked by persona-embedding cosine similarity.",
                cold_start=True,
                mode="direct",
            )

        return RecommendResponse(
            recommendations=[],
            reasoning="Provide candidate_list to rank items in direct mode.",
            mode="direct",
        )

    except Exception as exc:
        logger.error(f"/recommend error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8002)))
