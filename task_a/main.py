import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from task_a.agent import ASCUserModelingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AgentiCulture — Task A: User Modeling",
    description=(
        "Variance-adjusted CF · MDILU memory · Cold-start embedding · "
        "ReasoningCOTSC · Nigerian cultural context"
    ),
    version="3.0.0",
)

agent = ASCUserModelingAgent()

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

        def llm_fn(messages, temperature=0.1, max_tokens=1000, **_):
            import time
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


class ProductDetails(BaseModel):
    product_id:   str           = Field(..., description="Unique product or item identifier")
    product_name: str           = Field(default="")
    category:     Optional[str] = Field(default=None)
    description:  Optional[str] = Field(default=None)


class ReviewRequest(BaseModel):
    # Direct mode (persona + product details, no dataset needed)
    persona:         str           = Field(default="", description="User persona / profile text")
    product_details: ProductDetails

    # Benchmark mode (user_id + item_id → full pipeline via interaction tool)
    user_id: Optional[str] = Field(default=None)
    item_id: Optional[str] = Field(default=None)

    nigerian_context: bool = Field(default=True)


class ReviewResponse(BaseModel):
    stars:            float          = Field(..., description="Predicted star rating 1.0–5.0")
    review:           str            = Field(..., description="Generated review text")
    predicted_rating: Optional[float] = Field(default=None, description="CF-predicted rating before LLM")
    cold_start:       bool           = Field(default=False)
    mode:             str            = Field(default="direct")



@app.get("/")
async def root():
    return {
        "name":     "AgentiCulture — Task A: User Modeling",
        "version":  "3.0.0",
        "endpoint": "POST /generate-review",
        "docs":     "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status":          "healthy",
        "llm_ready":       agent.llm is not None,
        "embedding_ready": agent.sentence_model is not None,
        "data_ready":      agent.interaction_tool is not None,
    }


@app.post("/generate-review", response_model=ReviewResponse)
async def generate_review(request: ReviewRequest):
    """
    Generate a star rating and written review for a user+product pair.

    Two modes:
    - **Benchmark mode**: provide `user_id` + `item_id`. Requires dataset mounted
      at DATA_DIR. Runs the full MDILU → CF → COTSC pipeline.
    - **Direct mode**: provide `persona` + `product_details`. No dataset needed.
      Runs LLM generation with Nigerian cultural context.
    """
    _init_llm()
    _init_interaction_tool()

    try:
        if agent.interaction_tool and request.user_id and request.item_id:
            result = agent.workflow(
                user_id=request.user_id,
                item_id=request.item_id,
                nigerian_context=request.nigerian_context,
                persona=request.persona,
            )
            return ReviewResponse(
                stars=result["stars"],
                review=result["review"],
                predicted_rating=result.get("predicted_rating"),
                cold_start=result.get("cold_start", False),
                mode="benchmark",
            )

        result = agent.direct_generate(
            persona=request.persona,
            product_name=request.product_details.product_name,
            category=request.product_details.category or "general",
            description=request.product_details.description or "",
            nigerian_context=request.nigerian_context,
        )
        return ReviewResponse(stars=result["stars"], review=result["review"], mode="direct")

    except Exception as exc:
        logger.error(f"/generate-review error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
