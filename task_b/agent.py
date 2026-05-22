"""
task_b/agent.py — Recommendation Agent

All algorithm logic lives here. main.py only defines HTTP routes and
imports this class. Shared utilities come from core/.

Approach (baseline666 + RecHackers + sarvesh2003 adaptation):
  1. Platform-aware feature extraction            [baseline666]
  2. Self-consistency voting via Borda count      [RecHackers]
  3. Composite per-domain preference embeddings   [sarvesh2003 adaptation]
  4. Cold-start: persona-embedding cosine fallback [NEW — 25 pts]
  5. Cross-domain: collective_vec transfer         [NEW — 25 pts]
  6. Multi-turn session state                      [NEW]
  7. Nigerian cultural context                     [bonus criterion]
"""
import ast
import json
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from core.config import COLD_START_THRESHOLD, PLATFORM_FEATURES
from core.embeddings import get_embedding_model
from core.prompts import NIGERIAN_RECOMMENDATION_CONTEXT, RANKING_OUTPUT_FORMAT
from core.utils import (
    detect_platform,
    select_informative_reviews,
    borda_count_aggregation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Composite preference cache  (sarvesh2003 concept, in-memory)
# ---------------------------------------------------------------------------

class UserPreferenceCache:
    """
    Maintains per-user, per-platform embedding vectors.

    For each user we keep:
      {platform}_vec   — mean embedding of their reviews on that platform
      collective_vec   — count-weighted average across all platforms

    collective_vec enables cross-domain transfer: a user's Amazon taste
    informs Yelp recommendations when Yelp history is sparse.
    """

    def __init__(self, model):
        self.model  = model
        self._cache: Dict[str, Dict] = {}

    def _init_entry(self, user_id: str) -> Dict:
        entry = {
            "yelp_vec":       None,
            "amazon_vec":     None,
            "goodreads_vec":  None,
            "collective_vec": None,
            "domain_counts":  {"yelp": 0, "amazon": 0, "goodreads": 0},
        }
        self._cache[user_id] = entry
        return entry

    def update(self, user_id: str, reviews: List[Dict], platform: str) -> None:
        """Encode reviews for `platform`, update domain vec, rebuild collective_vec."""
        if not self.model or not reviews:
            return
        texts = [r.get("text", "") for r in reviews if r.get("text", "").strip()]
        if not texts:
            return
        try:
            embs       = self.model.encode(texts)
            domain_vec = np.mean(embs, axis=0)

            entry = self._cache.get(user_id) or self._init_entry(user_id)
            entry[f"{platform}_vec"] = domain_vec
            entry["domain_counts"][platform] = len(texts)

            # Rebuild collective_vec as count-weighted average
            vecs, weights = [], []
            for dom in ("yelp", "amazon", "goodreads"):
                if entry[f"{dom}_vec"] is not None:
                    vecs.append(entry[f"{dom}_vec"])
                    weights.append(float(entry["domain_counts"][dom]))
            if vecs:
                w = np.array(weights)
                entry["collective_vec"] = np.average(vecs, axis=0, weights=w / w.sum())
        except Exception as exc:
            logger.warning(f"UserPreferenceCache.update failed: {exc}")

    def get_preference_vec(self, user_id: str, platform: str) -> Optional[np.ndarray]:
        """
        Return the most useful preference vector for this user+platform.
        Falls back: domain_vec → collective_vec → None.
        """
        entry = self._cache.get(user_id)
        if not entry:
            return None
        domain_vec = entry.get(f"{platform}_vec")
        if domain_vec is not None and entry["domain_counts"].get(platform, 0) >= COLD_START_THRESHOLD:
            return domain_vec
        return entry.get("collective_vec")


# ---------------------------------------------------------------------------
# Multi-turn session state
# ---------------------------------------------------------------------------

class SessionState:
    """Accumulates per-session context for multi-turn recommendation."""

    def __init__(self):
        self.turns:      List[Dict]   = []
        self.platform:   Optional[str] = None
        self.created_at: float        = time.time()

    def record_turn(self, user_id: str, platform: str, result: List[str]) -> None:
        self.platform = platform
        self.turns.append({"user_id": user_id, "platform": platform, "top5": result[:5], "ts": time.time()})

    def context_hint(self) -> str:
        if not self.turns:
            return ""
        return (
            f"\n[SESSION CONTEXT: Turn {len(self.turns) + 1} of an ongoing "
            f"{self.platform or 'unknown'} recommendation session. "
            "Keep recommendations consistent with the user's browsing pattern.]\n"
        )


# ---------------------------------------------------------------------------
# Core agent
# ---------------------------------------------------------------------------

class Baseline666RecHackersAgent:
    """
    Combined:
      · baseline666  — platform-aware feature extraction
      · RecHackers   — 3 LLM samples + Borda count aggregation
      · sarvesh2003  — composite embedding preference cache (adapted, in-memory)
      · NEW          — cold-start, cross-domain boost, multi-turn sessions
    """

    def __init__(self, llm=None):
        self.llm              = llm
        self.interaction_tool = None
        self.sentence_model   = get_embedding_model()
        self.preference_cache = UserPreferenceCache(self.sentence_model) if self.sentence_model else None
        self.sessions: Dict[str, SessionState] = {}
        self.num_samples = 3

    def set_interaction_tool(self, tool):
        self.interaction_tool = tool

    # ------------------------------------------------------------------
    # Item feature extraction  (baseline666)
    # ------------------------------------------------------------------

    def _extract_item_features(self, item: Dict, platform: str) -> Dict:
        config    = PLATFORM_FEATURES.get(platform, PLATFORM_FEATURES["amazon"])
        extracted = {"item_id": item.get("item_id", "")}
        for field in config["item_fields"]:
            val = item.get(field)
            if val is not None:
                extracted[field] = str(val)[:200] if isinstance(val, str) else val
        return extracted

    # ------------------------------------------------------------------
    # Cold-start ranking  (25 pts)
    # ------------------------------------------------------------------

    def _cold_start_rank(
        self,
        candidate_list: List[str],
        persona_text:   str,
        user_info:      Dict,
        item_features:  Dict[str, Dict],
    ) -> List[str]:
        """
        No review history → embed persona text and rank candidates
        by cosine similarity. No LLM call — deterministic, fast, safe.
        """
        if not self.sentence_model:
            return candidate_list
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            query = (persona_text or "") + " " + json.dumps(user_info, default=str)[:400]
            q_emb = self.sentence_model.encode([query.strip()])

            ids, texts = [], []
            for iid in candidate_list:
                feat = item_features.get(iid, {"item_id": iid})
                text = " ".join(str(v) for k, v in feat.items() if k != "item_id" and v)[:300]
                ids.append(iid)
                texts.append(text or iid)

            i_embs = self.sentence_model.encode(texts)
            sims   = cosine_similarity(q_emb, i_embs)[0]
            return [iid for iid, _ in sorted(zip(ids, sims), key=lambda x: x[1], reverse=True)]
        except Exception as exc:
            logger.warning(f"Cold-start rank failed: {exc}")
            return candidate_list

    # ------------------------------------------------------------------
    # Cross-domain preference boost
    # ------------------------------------------------------------------

    def _cross_domain_scores(
        self,
        user_id:        str,
        platform:       str,
        candidate_list: List[str],
        item_features:  Dict[str, Dict],
    ) -> Dict[str, float]:
        """Cosine similarity between user's collective_vec and each candidate."""
        if not self.preference_cache or not self.sentence_model:
            return {}
        pref_vec = self.preference_cache.get_preference_vec(user_id, platform)
        if pref_vec is None:
            return {}
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            ids, texts = [], []
            for iid in candidate_list:
                feat = item_features.get(iid, {"item_id": iid})
                text = " ".join(str(v) for k, v in feat.items() if k != "item_id" and v)[:300]
                ids.append(iid)
                texts.append(text or iid)
            i_embs = self.sentence_model.encode(texts)
            sims   = cosine_similarity(pref_vec.reshape(1, -1), i_embs)[0]
            return {iid: float(s) for iid, s in zip(ids, sims)}
        except Exception as exc:
            logger.warning(f"Cross-domain scores failed: {exc}")
            return {}

    def _blend_cross_domain(
        self,
        borda_ranking:  List[str],
        cross_scores:   Dict[str, float],
        candidates:     List[str],
        alpha:          float = 0.3,
    ) -> List[str]:
        """70% Borda + 30% cross-domain embedding similarity."""
        n = len(candidates)
        borda = {iid: (n - rank) for rank, iid in enumerate(borda_ranking)}
        if cross_scores:
            lo, hi = min(cross_scores.values()), max(cross_scores.values())
            rng    = max(hi - lo, 1e-9)
            norm   = {iid: (s - lo) / rng * n for iid, s in cross_scores.items()}
        else:
            norm = {}
        blended = {
            iid: (1 - alpha) * borda.get(iid, 0) + alpha * norm.get(iid, 0)
            for iid in candidates
        }
        return sorted(candidates, key=lambda x: blended[x], reverse=True)

    # ------------------------------------------------------------------
    # LLM ranking — single sample  (RecHackers)
    # ------------------------------------------------------------------

    def _llm_rank_single(self, prompt: str, temperature: float = 0.1) -> Optional[List[str]]:
        if not self.llm:
            return None
        try:
            result = self.llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=800,
            )
            import re, json
            # Strip markdown code fences before parsing
            result = result.replace("```json", "").replace("```python", "").replace("```", "")
            match = re.search(r"\[.*?\]", result, re.DOTALL)
            if match:
                raw = match.group(0)
                # Try JSON first (double quotes), then Python literal (single quotes)
                try:
                    return json.loads(raw)
                except Exception:
                    return ast.literal_eval(raw)
        except Exception as exc:
            logger.warning(f"LLM rank sample failed: {exc}")
        return None

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_ranking_prompt(
        self,
        platform:            str,
        user_info:           Dict,
        user_reviews_text:   str,
        candidate_items_text: str,
        nigerian_context:    bool,
        session_context:     str = "",
        candidate_list:      List[str] = None,
    ) -> str:
        cultural = NIGERIAN_RECOMMENDATION_CONTEXT if nigerian_context else ""
        # Inject the actual candidate IDs so the model knows exactly what to return
        candidate_ids_str = json.dumps(candidate_list) if candidate_list else "[]"
        return f"""{cultural}{session_context}

You are a recommendation agent on {platform.upper()}.

USER PROFILE:
{json.dumps(user_info, indent=2, default=str)[:1200]}

USER'S HISTORICAL REVIEWS:
{user_reviews_text}

CANDIDATE ITEMS TO RANK:
{candidate_items_text}

INSTRUCTIONS:
1. Analyse the user's preferences from their review history.
2. Consider each candidate item's relevance to their taste.
3. Account for platform-specific signals that matter on {platform}.
4. Rank ALL items from MOST recommended (first) to LEAST (last).

Your final output should be ONLY a ranked list of these exact item IDs: {candidate_ids_str}
DO NOT output your analysis process!
Follow this format STRICTLY: {candidate_ids_str}
"""

    # ------------------------------------------------------------------
    # Main workflow
    # ------------------------------------------------------------------

    def workflow(
        self,
        user_id:        str,
        candidate_list: List[str],
        nigerian_context: bool = True,
        session_id:     Optional[str] = None,
        persona:        str = "",
        ablation_mode:  str = "full",
    ) -> List[str]:
        """
        Full pipeline:
          1.  Retrieve user + item data
          2.  Detect platform
          3.  Extract item features (baseline666)
          4.  Update preference cache (sarvesh2003)
          5.  Cold-start → persona-embedding fallback (NEW)
          6.  Cross-domain scoring for sparse history (NEW)
          7.  Multi-turn session context (NEW)
          8.  Review selection (baseline666)
          9.  Self-consistency voting — 3 samples + Borda (RecHackers)
          10. Blend with cross-domain signal (NEW)
          11. Record session turn (NEW)
        """
        if not self.interaction_tool or not candidate_list:
            return candidate_list

        try:
            # 1. Retrieve
            user_info    = self.interaction_tool.get_user(user_id=user_id)
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id) or []

            # 2. Platform
            first_item = self.interaction_tool.get_item(item_id=candidate_list[0])
            platform   = detect_platform(first_item)

            # 3. Item features
            item_features: Dict[str, Dict] = {}
            item_feats_list: List[Dict]    = []
            for iid in candidate_list:
                try:
                    item = self.interaction_tool.get_item(item_id=iid)
                    feat = self._extract_item_features(item, platform)
                except Exception:
                    feat = {"item_id": iid, "name": "Unknown"}
                item_features[iid] = feat
                item_feats_list.append(feat)

            # 4. Update preference cache
            if self.preference_cache and user_reviews:
                self.preference_cache.update(user_id, user_reviews, platform)

            # 5. Cold-start
            is_cold   = len(user_reviews) == 0
            is_sparse = 0 < len(user_reviews) < COLD_START_THRESHOLD

            # ablation: cosine_only forces the cold-start path for all users
            if is_cold or ablation_mode == "cosine_only":
                persona_text = persona or json.dumps(user_info, default=str)[:600]
                result = self._cold_start_rank(candidate_list, persona_text, user_info, item_features)
                if session_id:
                    self.sessions.setdefault(session_id, SessionState()).record_turn(user_id, platform, result)
                return self._fill_missing(result, candidate_list)

            # 6. Cross-domain scores (skipped in borda_only ablation)
            cross_scores: Dict[str, float] = {}
            if is_sparse and ablation_mode not in ("single_llm", "borda_only"):
                cross_scores = self._cross_domain_scores(user_id, platform, candidate_list, item_features)

            # 7. Session context
            session_context = ""
            if session_id:
                session_context = self.sessions.setdefault(session_id, SessionState()).context_hint()

            # 8. Review text for prompt
            sel_reviews      = select_informative_reviews(user_reviews, platform, max_reviews=5)
            user_reviews_txt = "".join(
                f"\n- ({r.get('stars','?')} stars) {r.get('text','')[:180]}" for r in sel_reviews
            )
            cand_txt = "".join(
                f"\n{json.dumps(feat, indent=2, default=str)[:250]}" for feat in item_feats_list
            )

            prompt = self._build_ranking_prompt(
                platform=platform,
                user_info=user_info,
                user_reviews_text=user_reviews_txt,
                candidate_items_text=cand_txt,
                nigerian_context=nigerian_context,
                session_context=session_context,
                candidate_list=candidate_list,
            )

            # 9. Self-consistency voting
            # ablation: single_llm uses only one sample; borda_only and full use 3
            n_samples = 1 if ablation_mode == "single_llm" else self.num_samples
            rankings: List[List[str]] = []
            for i in range(n_samples):
                temp = 0.1 if i == 0 else 0.7
                r = self._llm_rank_single(prompt, temperature=temp)
                if r and len(r) == len(candidate_list):
                    rankings.append(r)

            # 10. Aggregate
            if rankings:
                final = borda_count_aggregation(rankings, candidate_list)
                if cross_scores and ablation_mode == "full":
                    final = self._blend_cross_domain(final, cross_scores, candidate_list)
            else:
                # All LLM samples failed to parse — fall back to cosine ranking
                # which is always valid and deterministic
                logger.warning(f"All LLM samples failed for user {user_id}, falling back to cosine ranking")
                persona_text = persona or json.dumps(user_info, default=str)[:600]
                final = self._cold_start_rank(candidate_list, persona_text, user_info, item_features)

            final = self._fill_missing(final, candidate_list)

            # 11. Record session
            if session_id:
                self.sessions[session_id].record_turn(user_id, platform, final)

            return final

        except Exception as exc:
            logger.error(f"workflow error: {exc}", exc_info=True)
            return candidate_list

    def _fill_missing(self, ranking: List[str], candidates: List[str]) -> List[str]:
        ranked_set = set(ranking)
        return ranking + [iid for iid in candidates if iid not in ranked_set]
