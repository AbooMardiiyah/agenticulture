"""
task_a/agent.py — User Modeling Agent

All algorithm logic lives here. main.py only defines HTTP routes and
imports this class. Shared utilities come from core/.

Approach (ASC WWW'25 winning formula + enhancements):
  1. Multi-stage retrieve → plan → generate pipeline
  2. MDILU memory: cosine-similarity retrieval of similar past reviews
  3. Collaborative filtering core: variance-adjusted predicted rating
  4. Cold-start: persona-embedding guides item-review retrieval [NEW]
  5. ReasoningCOTSC: 3-sample majority-vote for RMSE stability [NEW]
  6. Nigerian cultural context throughout [bonus criterion]
"""
import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from core.embeddings import get_embedding_model, cosine_retrieve
from core.prompts import NIGERIAN_USER_MODELING_CONTEXT, REVIEW_OUTPUT_FORMAT
from core.utils import (
    detect_platform,
    compute_cf_statistics,
    variance_adjusted_rating,
    select_informative_reviews,
)

logger = logging.getLogger(__name__)


class ASCUserModelingAgent:
    """
    Team ASC's winning approach adapted for the DSN × BCT challenge.

    The agent is designed to be stateless per-request (no shared mutable
    state between HTTP calls) to allow safe concurrent use.
    """

    def __init__(self, llm=None):
        self.llm             = llm
        self.interaction_tool = None
        self.sentence_model  = get_embedding_model()

    def set_interaction_tool(self, tool):
        self.interaction_tool = tool


    def _mdilu_retrieve_similar(
        self, query: str, reviews: List[Dict], k: int = 3
    ) -> List[Dict]:
        """Find the k most similar historical reviews to the query text."""
        texts = [r.get("text", "") for r in reviews]
        idxs  = cosine_retrieve(query, texts, self.sentence_model, k=k)
        return [reviews[i] for i in idxs]

 

    def _cold_start_item_reviews(
        self, persona_text: str, item_reviews: List[Dict], k: int = 5
    ) -> List[Dict]:
        """
        When the user has no review history, embed their persona text and
        retrieve the k item reviews most similar to that persona.
        These act as style anchors so the LLM writes a contextually relevant review.
        """
        if not item_reviews:
            return []
        texts = [r.get("text", "") for r in item_reviews]
        idxs  = cosine_retrieve(persona_text, texts, self.sentence_model, k=k)
        return [item_reviews[i] for i in idxs] if idxs else item_reviews[:k]


    def _build_prompt(
        self,
        platform: str,
        user_info: Dict,
        item_info: Dict,
        selected_user_reviews: List[Dict],
        selected_item_reviews: List[Dict],
        cf_stats: Dict,
        predicted_rating: float,
        similar_reviews_text: str,
        cold_start_note: str,
        nigerian_context: bool,
    ) -> str:
        cultural = NIGERIAN_USER_MODELING_CONTEXT if nigerian_context else ""

        user_history = "".join(
            f"\n- ({r.get('stars','?')} stars): {r.get('text','')[:200]}"
            for r in selected_user_reviews
        ) or "[No prior review history — write as a new user based on your profile]"

        item_reviews = "".join(
            f"\n- ({r.get('stars','?')} stars): {r.get('text','')[:150]}"
            for r in selected_item_reviews
        )

        return f"""{cultural}
{cold_start_note}
You are simulating a real user review on {platform.upper()}.

YOUR PROFILE:
{json.dumps(user_info, indent=2, default=str)[:1500]}

YOUR HISTORICAL REVIEWS (learn your style from these):
{user_history}

STATISTICAL PROFILE:
- Your average rating: {cf_stats['user_mean']:.1f} stars
- Your rating consistency (variance): {cf_stats['user_variance']:.2f}
- This item's average rating: {cf_stats['item_mean']:.1f} stars
- Predicted rating for you: {predicted_rating:.1f} stars

ITEM TO REVIEW:
{json.dumps(item_info, indent=2, default=str)[:1200]}

OTHER USERS' REVIEWS FOR THIS ITEM:
{item_reviews}

{similar_reviews_text}

TASK:
1. Based on your statistical profile ({predicted_rating:.1f} stars predicted), select a rating.
2. Write a review that matches YOUR historical style — similar length, tone, and focus.
3. The review should feel authentically written by you.

{REVIEW_OUTPUT_FORMAT}"""


    def _cotsc_predict(self, prompt: str, n_samples: int = 3) -> Tuple[Optional[float], Optional[str]]:
        """
        Ask the LLM n_samples times, vote on the most consistent integer rating,
        then return the (stars, review_text) from the sample closest to the majority.

        Why: reduces random variance in the predicted star rating (lowers RMSE).
        """
        samples = []
        for i in range(n_samples):
            temp = 0.3 if i == 0 else 0.7
            raw  = self._call_llm(prompt, temperature=temp, max_tokens=400)
            sm   = re.search(r'stars:\s*([1-5](?:\.0|\.5)?)', raw, re.IGNORECASE)
            rm   = re.search(r'review:\s*(.+?)(?:\n\n|$)', raw, re.IGNORECASE | re.DOTALL)
            if sm:
                s = float(sm.group(1))
                r = rm.group(1).strip() if rm else raw.strip()[:512]
                samples.append((s, r))

        if not samples:
            return None, None

        majority_int = Counter(round(s) for s, _ in samples).most_common(1)[0][0]
        best_stars, best_review = min(samples, key=lambda x: abs(x[0] - majority_int))
        stars = max(1.0, min(5.0, round(best_stars * 2) / 2))

        logger.info(f"COTSC votes={[round(s) for s,_ in samples]}, majority={majority_int}, chosen={stars}")
        return stars, best_review

    def workflow(
        self,
        user_id: str,
        item_id: str,
        nigerian_context: bool = True,
        persona: str = "",
    ) -> Dict[str, Any]:
        """
        Full pipeline:
          1. Retrieve user + item data via interaction tool
          2. Detect platform (Yelp / Amazon / Goodreads)
          3. MDILU memory retrieval (or cold-start persona fallback)
          4. Collaborative filtering — variance-adjusted rating prediction
          5. Select the most informative reviews
          6. Build contextual prompt
          7. ReasoningCOTSC — 3-sample majority-vote generation
        """
        if not self.interaction_tool:
            return {"stars": 3.0, "review": "No interaction tool configured."}

        try:
            user_info    = self.interaction_tool.get_user(user_id=user_id)
            item_info    = self.interaction_tool.get_item(item_id=item_id)
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id) or []
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id) or []

            platform = detect_platform(item_info)

            similar_text = ""
            cold_note    = ""
            is_cold      = len(user_reviews) == 0

            if not is_cold and len(user_reviews) > 1:
                similar = self._mdilu_retrieve_similar(
                    query=user_reviews[0].get("text", ""),
                    reviews=user_reviews[1:],
                    k=3,
                )
                if similar:
                    similar_text = "\nMOST SIMILAR PAST REVIEWS:\n" + "".join(
                        f"- ({r.get('stars','?')} stars): {r.get('text','')[:150]}\n"
                        for r in similar
                    )

            if is_cold:
                cold_note = (
                    "\n[COLD-START USER: No prior review history. "
                    "Write as a brand-new user whose preferences match YOUR PROFILE above.]\n"
                )
                persona_text = persona or json.dumps(user_info, default=str)[:600]
                item_reviews = self._cold_start_item_reviews(persona_text, item_reviews, k=5)

            cf_stats = compute_cf_statistics(user_reviews, item_reviews)
            predicted = variance_adjusted_rating(
                cf_stats["user_mean"], cf_stats["item_mean"],
                cf_stats["user_variance"], cf_stats["item_variance"],
            )

            sel_user = select_informative_reviews(user_reviews, platform, max_reviews=5)
            sel_item = select_informative_reviews(item_reviews, platform, max_reviews=3)

            prompt = self._build_prompt(
                platform=platform,
                user_info=user_info,
                item_info=item_info,
                selected_user_reviews=sel_user,
                selected_item_reviews=sel_item,
                cf_stats=cf_stats,
                predicted_rating=predicted,
                similar_reviews_text=similar_text,
                cold_start_note=cold_note,
                nigerian_context=nigerian_context,
            )

            stars, review_text = self._cotsc_predict(prompt, n_samples=3)

            if stars is None:
                raw = self._call_llm(prompt, temperature=0.7, max_tokens=400)
                sm  = re.search(r'stars:\s*([1-5](?:\.0|\.5)?)', raw, re.IGNORECASE)
                rm  = re.search(r'review:\s*(.+?)(?:\n\n|$)', raw, re.IGNORECASE | re.DOTALL)
                stars       = max(1.0, min(5.0, round(float(sm.group(1)) * 2) / 2)) if sm else predicted
                review_text = rm.group(1).strip() if rm else raw.strip()[:512]

            return {
                "stars":            stars,
                "review":           (review_text or "")[:512],
                "predicted_rating": predicted,
                "cold_start":       is_cold,
            }

        except Exception as exc:
            logger.error(f"workflow error: {exc}", exc_info=True)
            return {"stars": 3.0, "review": f"An average experience. ({str(exc)[:60]})"}

    # ABLATION VARIANTS
    # Each method disables one or more components for comparison.
    
    def _run_llm_direct(self, user_id: str, item_id: str, nigerian_context: bool) -> Dict[str, Any]:
        """No CF, no MDILU, no COTSC — single LLM call with basic context only."""
        if not self.interaction_tool:
            return {"stars": 3.0, "review": "No interaction tool."}
        try:
            user_info = self.interaction_tool.get_user(user_id=user_id)
            item_info = self.interaction_tool.get_item(item_id=item_id)
            cultural  = NIGERIAN_USER_MODELING_CONTEXT if nigerian_context else ""
            prompt = f"""{cultural}
You are simulating a real user review.
USER PROFILE: {json.dumps(user_info, default=str)[:800]}
ITEM: {json.dumps(item_info, default=str)[:600]}
{REVIEW_OUTPUT_FORMAT}"""
            raw = self._call_llm(prompt, temperature=0.7, max_tokens=400)
            sm  = re.search(r'stars:\s*([1-5](?:\.0|\.5)?)', raw, re.IGNORECASE)
            rm  = re.search(r'review:\s*(.+?)(?:\n\n|$)', raw, re.IGNORECASE | re.DOTALL)
            stars  = max(1.0, min(5.0, float(sm.group(1)))) if sm else 3.0
            review = rm.group(1).strip() if rm else raw.strip()[:512]
            return {"stars": stars, "review": review, "cold_start": False}
        except Exception as exc:
            logger.error(f"_run_llm_direct error: {exc}", exc_info=True)
            return {"stars": 3.0, "review": "An average experience."}

    def _run_cf_only(self, user_id: str, item_id: str, nigerian_context: bool) -> Dict[str, Any]:
        """CF rating prediction + single LLM call — no MDILU retrieval, no COTSC."""
        if not self.interaction_tool:
            return {"stars": 3.0, "review": "No interaction tool."}
        try:
            user_info    = self.interaction_tool.get_user(user_id=user_id)
            item_info    = self.interaction_tool.get_item(item_id=item_id)
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id) or []
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id) or []
            platform     = detect_platform(item_info)
            cf_stats     = compute_cf_statistics(user_reviews, item_reviews)
            predicted    = variance_adjusted_rating(
                cf_stats["user_mean"], cf_stats["item_mean"],
                cf_stats["user_variance"], cf_stats["item_variance"],
            )
            cultural = NIGERIAN_USER_MODELING_CONTEXT if nigerian_context else ""
            prompt = f"""{cultural}
You are simulating a user review on {platform.upper()}.
USER PROFILE: {json.dumps(user_info, default=str)[:800]}
ITEM: {json.dumps(item_info, default=str)[:600]}
PREDICTED RATING: {predicted:.1f} stars (from collaborative filtering)
{REVIEW_OUTPUT_FORMAT}"""
            raw = self._call_llm(prompt, temperature=0.7, max_tokens=400)
            sm  = re.search(r'stars:\s*([1-5](?:\.0|\.5)?)', raw, re.IGNORECASE)
            rm  = re.search(r'review:\s*(.+?)(?:\n\n|$)', raw, re.IGNORECASE | re.DOTALL)
            stars  = max(1.0, min(5.0, float(sm.group(1)))) if sm else predicted
            review = rm.group(1).strip() if rm else raw.strip()[:512]
            return {"stars": stars, "review": review, "cold_start": len(user_reviews) == 0}
        except Exception as exc:
            logger.error(f"_run_cf_only error: {exc}", exc_info=True)
            return {"stars": 3.0, "review": "An average experience."}

    def _run_cf_mdilu(self, user_id: str, item_id: str, nigerian_context: bool) -> Dict[str, Any]:
        """CF + MDILU memory retrieval + single LLM call — no COTSC voting."""
        try:
            if not self.interaction_tool:
                return {"stars": 3.0, "review": "No interaction tool."}
            user_info    = self.interaction_tool.get_user(user_id=user_id)
            item_info    = self.interaction_tool.get_item(item_id=item_id)
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id) or []
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id) or []
            platform     = detect_platform(item_info)
            is_cold      = len(user_reviews) == 0
            similar_text, cold_note = "", ""
            if not is_cold and len(user_reviews) > 1:
                similar = self._mdilu_retrieve_similar(
                    query=user_reviews[0].get("text", ""), reviews=user_reviews[1:], k=3
                )
                if similar:
                    similar_text = "\nMOST SIMILAR PAST REVIEWS:\n" + "".join(
                        f"- ({r.get('stars','?')} stars): {r.get('text','')[:150]}\n" for r in similar
                    )
            if is_cold:
                cold_note    = "\n[COLD-START USER]\n"
                persona_text = json.dumps(user_info, default=str)[:600]
                item_reviews = self._cold_start_item_reviews(persona_text, item_reviews, k=5)
            cf_stats  = compute_cf_statistics(user_reviews, item_reviews)
            predicted = variance_adjusted_rating(
                cf_stats["user_mean"], cf_stats["item_mean"],
                cf_stats["user_variance"], cf_stats["item_variance"],
            )
            sel_user = select_informative_reviews(user_reviews, platform, max_reviews=5)
            sel_item = select_informative_reviews(item_reviews, platform, max_reviews=3)
            prompt   = self._build_prompt(
                platform=platform, user_info=user_info, item_info=item_info,
                selected_user_reviews=sel_user, selected_item_reviews=sel_item,
                cf_stats=cf_stats, predicted_rating=predicted,
                similar_reviews_text=similar_text, cold_start_note=cold_note,
                nigerian_context=nigerian_context,
            )
            raw = self._call_llm(prompt, temperature=0.7, max_tokens=400)
            sm  = re.search(r'stars:\s*([1-5](?:\.0|\.5)?)', raw, re.IGNORECASE)
            rm  = re.search(r'review:\s*(.+?)(?:\n\n|$)', raw, re.IGNORECASE | re.DOTALL)
            stars  = max(1.0, min(5.0, float(sm.group(1)))) if sm else predicted
            review = rm.group(1).strip() if rm else raw.strip()[:512]
            return {"stars": stars, "review": review, "cold_start": is_cold}
        except Exception as exc:
            logger.error(f"_run_cf_mdilu error: {exc}", exc_info=True)
            return {"stars": 3.0, "review": "An average experience."}

    def _run_cf_cotsc_no_mdilu(self, user_id: str, item_id: str, nigerian_context: bool) -> Dict[str, Any]:
        """CF + COTSC voting — no MDILU style anchor retrieval."""
        if not self.interaction_tool:
            return {"stars": 3.0, "review": "No interaction tool."}
        try:
            user_info    = self.interaction_tool.get_user(user_id=user_id)
            item_info    = self.interaction_tool.get_item(item_id=item_id)
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id) or []
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id) or []
            platform     = detect_platform(item_info)
            is_cold      = len(user_reviews) == 0
            cold_note    = "\n[COLD-START USER]\n" if is_cold else ""
            if is_cold:
                persona_text = json.dumps(user_info, default=str)[:600]
                item_reviews = self._cold_start_item_reviews(persona_text, item_reviews, k=5)
            cf_stats  = compute_cf_statistics(user_reviews, item_reviews)
            predicted = variance_adjusted_rating(
                cf_stats["user_mean"], cf_stats["item_mean"],
                cf_stats["user_variance"], cf_stats["item_variance"],
            )
            sel_user = select_informative_reviews(user_reviews, platform, max_reviews=5)
            sel_item = select_informative_reviews(item_reviews, platform, max_reviews=3)
            prompt = self._build_prompt(
                platform=platform, user_info=user_info, item_info=item_info,
                selected_user_reviews=sel_user, selected_item_reviews=sel_item,
                cf_stats=cf_stats, predicted_rating=predicted,
                similar_reviews_text="", cold_start_note=cold_note,
                nigerian_context=nigerian_context,
            )
            stars, review_text = self._cotsc_predict(prompt, n_samples=3)
            if stars is None:
                stars, review_text = predicted, "An average experience."
            return {"stars": stars, "review": (review_text or "")[:512], "cold_start": is_cold}
        except Exception as exc:
            logger.error(f"_run_cf_cotsc_no_mdilu error: {exc}", exc_info=True)
            return {"stars": 3.0, "review": "An average experience."}

    def direct_generate(
        self,
        persona: str,
        product_name: str,
        category: str = "general",
        description: str = "",
        nigerian_context: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a review without access to the Yelp/Amazon/Goodreads dataset.
        Used when the request includes persona + product details but no user_id/item_id.
        """
        cultural = NIGERIAN_USER_MODELING_CONTEXT if nigerian_context else ""
        persona_block = f"\nUSER PERSONA:\n{persona}\n" if persona else ""
        desc_block    = f"Description: {description[:250]}\n" if description else ""

        prompt = f"""{cultural}{persona_block}
Write an authentic review for '{product_name}'.
Category: {category}
{desc_block}
Base the review on the user persona above. If no persona is given, write as a typical Nigerian consumer.

{REVIEW_OUTPUT_FORMAT}"""

        raw = self._call_llm(prompt, temperature=0.7, max_tokens=300)
        sm  = re.search(r'stars:\s*([1-5](?:\.0|\.5)?)', raw, re.IGNORECASE)
        rm  = re.search(r'review:\s*(.+?)(?:\n\n|$)', raw, re.IGNORECASE | re.DOTALL)
        stars  = max(1.0, min(5.0, float(sm.group(1)))) if sm else 3.5
        review = rm.group(1).strip() if rm else raw.strip()[:512]
        return {"stars": stars, "review": review}

    def _call_llm(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1000) -> str:
        if self.llm:
            try:
                return self.llm(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                logger.error(f"LLM call failed: {exc}")
        return "stars: 3.0\nreview: An okay experience overall. Nothing too special but not bad either."
