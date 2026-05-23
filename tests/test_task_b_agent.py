import numpy as np
from unittest.mock import MagicMock
from task_b.agent import Baseline666RecHackersAgent, UserPreferenceCache, SessionState


CANDIDATES = ["item_1", "item_2", "item_3", "item_4", "item_5"]


def make_mock_llm(response=None):
    default = str(["item_1", "item_2", "item_3", "item_4", "item_5"])
    return MagicMock(return_value=response or default)


def make_mock_tool(user_reviews=None, items=None):
    tool = MagicMock()
    tool.get_user.return_value = {"user_id": "u1", "name": "Test User"}
    tool.get_reviews.return_value = user_reviews if user_reviews is not None else []

    default_item = {
        "item_id": "x",
        "categories": "Restaurants",
        "attributes": {"WiFi": "free"},
        "name": "Test Place",
    }
    if items:
        tool.get_item.side_effect = lambda item_id=None: items.get(item_id, default_item)
    else:
        tool.get_item.return_value = default_item
    return tool


def make_agent(llm_response=None, user_reviews=None, items=None):
    agent = Baseline666RecHackersAgent(llm=make_mock_llm(llm_response))
    agent.sentence_model = None
    agent.preference_cache = None
    agent.set_interaction_tool(make_mock_tool(user_reviews, items))
    return agent


class TestWorkflowOutputContract:
    def test_returns_list(self):
        agent = make_agent(user_reviews=[{"stars": 4, "text": "Good"}])
        result = agent.workflow(user_id="u1", candidate_list=CANDIDATES)
        assert isinstance(result, list)

    def test_returns_all_candidates(self):
        agent = make_agent(user_reviews=[{"stars": 4, "text": "Good"}])
        result = agent.workflow(user_id="u1", candidate_list=CANDIDATES)
        assert set(result) == set(CANDIDATES)

    def test_no_duplicates_in_result(self):
        agent = make_agent(user_reviews=[{"stars": 4, "text": "Good"}])
        result = agent.workflow(user_id="u1", candidate_list=CANDIDATES)
        assert len(result) == len(set(result))

    def test_result_length_equals_candidates(self):
        agent = make_agent(user_reviews=[{"stars": 4, "text": "Good"}])
        result = agent.workflow(user_id="u1", candidate_list=CANDIDATES)
        assert len(result) == len(CANDIDATES)

    def test_all_items_are_strings(self):
        agent = make_agent(user_reviews=[{"stars": 4, "text": "Good"}])
        result = agent.workflow(user_id="u1", candidate_list=CANDIDATES)
        assert all(isinstance(i, str) for i in result)


class TestWorkflowColdStart:
    def test_cold_start_returns_all_candidates(self):
        agent = make_agent(user_reviews=[])
        result = agent.workflow(user_id="u1", candidate_list=CANDIDATES)
        assert set(result) == set(CANDIDATES)

    def test_cold_start_does_not_call_llm(self):
        agent = make_agent(user_reviews=[])
        agent.llm = MagicMock()
        agent.workflow(user_id="u1", candidate_list=CANDIDATES)
        agent.llm.assert_not_called()

    def test_cold_start_with_embedding_model_uses_cosine_rank(self):
        agent = make_agent(user_reviews=[])
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(5, 64)
        agent.sentence_model = mock_model
        result = agent.workflow(user_id="u1", candidate_list=CANDIDATES, persona="tech lover")
        assert set(result) == set(CANDIDATES)


class TestWorkflowErrorHandling:
    def test_empty_candidate_list_returns_empty(self):
        agent = make_agent()
        result = agent.workflow(user_id="u1", candidate_list=[])
        assert result == []

    def test_no_interaction_tool_returns_candidates(self):
        agent = Baseline666RecHackersAgent(llm=make_mock_llm())
        agent.sentence_model = None
        agent.preference_cache = None
        result = agent.workflow(user_id="u1", candidate_list=CANDIDATES)
        assert result == CANDIDATES

    def test_llm_failure_still_returns_candidates(self):
        agent = make_agent(user_reviews=[{"stars": 4, "text": "Good"}])
        agent.llm = MagicMock(side_effect=Exception("API down"))
        result = agent.workflow(user_id="u1", candidate_list=CANDIDATES)
        assert set(result) == set(CANDIDATES)

    def test_malformed_llm_response_still_returns_candidates(self):
        agent = make_agent(
            llm_response="I cannot rank these items.",
            user_reviews=[{"stars": 4, "text": "Good"}]
        )
        result = agent.workflow(user_id="u1", candidate_list=CANDIDATES)
        assert set(result) == set(CANDIDATES)

    def test_item_fetch_failure_still_returns_candidates(self):
        agent = make_agent(user_reviews=[{"stars": 4, "text": "Good"}])
        agent.interaction_tool.get_item.side_effect = Exception("Item not found")
        result = agent.workflow(user_id="u1", candidate_list=CANDIDATES)
        assert set(result) == set(CANDIDATES)


class TestFillMissing:
    def setup_method(self):
        self.agent = Baseline666RecHackersAgent()

    def test_adds_missing_items(self):
        result = self.agent._fill_missing(["item_1", "item_2"], CANDIDATES)
        assert set(result) == set(CANDIDATES)

    def test_preserves_existing_order(self):
        partial = ["item_3", "item_1"]
        result  = self.agent._fill_missing(partial, CANDIDATES)
        assert result[0] == "item_3"
        assert result[1] == "item_1"

    def test_no_duplicates(self):
        result = self.agent._fill_missing(["item_1", "item_2"], CANDIDATES)
        assert len(result) == len(set(result))

    def test_full_ranking_unchanged(self):
        full = ["item_5", "item_4", "item_3", "item_2", "item_1"]
        result = self.agent._fill_missing(full, CANDIDATES)
        assert result == full

    def test_empty_ranking_returns_all_candidates(self):
        result = self.agent._fill_missing([], CANDIDATES)
        assert set(result) == set(CANDIDATES)


class TestBlendCrossDomain:
    def setup_method(self):
        self.agent = Baseline666RecHackersAgent()

    def test_returns_all_candidates(self):
        borda  = ["A", "B", "C"]
        cross  = {"A": 0.9, "B": 0.5, "C": 0.1}
        result = self.agent._blend_cross_domain(borda, cross, ["A", "B", "C"])
        assert set(result) == {"A", "B", "C"}

    def test_empty_cross_scores_uses_borda_order(self):
        borda  = ["A", "B", "C"]
        result = self.agent._blend_cross_domain(borda, {}, ["A", "B", "C"])
        assert result[0] == "A"

    def test_high_cross_score_can_promote_item(self):
        # B is last in Borda but has highest cosine score
        borda  = ["A", "C", "B"]
        cross  = {"A": 0.1, "B": 0.99, "C": 0.2}
        result = self.agent._blend_cross_domain(borda, cross, ["A", "B", "C"], alpha=0.9)
        assert result[0] == "B"

    def test_no_duplicates(self):
        borda  = ["A", "B", "C"]
        cross  = {"A": 0.5, "B": 0.3, "C": 0.8}
        result = self.agent._blend_cross_domain(borda, cross, ["A", "B", "C"])
        assert len(result) == len(set(result))


class TestUserPreferenceCache:
    def _make_cache(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 64)
        return UserPreferenceCache(model=mock_model)

    def test_update_stores_domain_vec(self):
        cache = self._make_cache()
        reviews = [{"text": "Great"}, {"text": "Good"}, {"text": "Ok"}]
        cache.update("u1", reviews, "yelp")
        assert cache._cache["u1"]["yelp_vec"] is not None

    def test_update_builds_collective_vec(self):
        cache = self._make_cache()
        reviews = [{"text": "Great"}, {"text": "Good"}]
        cache.update("u1", reviews, "yelp")
        assert cache._cache["u1"]["collective_vec"] is not None

    def test_get_preference_vec_returns_domain_vec_when_warm(self):
        cache = self._make_cache()
        reviews = [{"text": f"Review {i}"} for i in range(5)]
        cache.update("u1", reviews, "amazon")
        cache._cache["u1"]["domain_counts"]["amazon"] = 5
        vec = cache.get_preference_vec("u1", "amazon")
        assert vec is not None

    def test_get_preference_vec_returns_none_for_unknown_user(self):
        cache = self._make_cache()
        assert cache.get_preference_vec("unknown_user", "yelp") is None

    def test_falls_back_to_collective_vec_when_sparse(self):
        cache = self._make_cache()
        reviews = [{"text": "Review"}]
        cache.update("u1", reviews, "yelp")
        cache._cache["u1"]["domain_counts"]["yelp"] = 1
        vec = cache.get_preference_vec("u1", "yelp")
        assert vec is not None

    def test_multi_domain_update_all_tracked(self):
        cache = self._make_cache()
        reviews = [{"text": "Review"}]
        cache.update("u1", reviews, "yelp")
        cache.update("u1", reviews, "amazon")
        entry = cache._cache["u1"]
        assert entry["yelp_vec"] is not None
        assert entry["amazon_vec"] is not None

    def test_empty_reviews_skipped(self):
        cache = self._make_cache()
        cache.update("u1", [], "yelp")
        assert "u1" not in cache._cache

    def test_reviews_with_no_text_skipped(self):
        cache = self._make_cache()
        reviews = [{"stars": 4}, {"stars": 3}]   
        cache.update("u1", reviews, "yelp")
        assert "u1" not in cache._cache


class TestSessionState:
    def test_initial_state_empty(self):
        s = SessionState()
        assert s.turns == []
        assert s.platform is None

    def test_context_hint_empty_on_first_turn(self):
        s = SessionState()
        assert s.context_hint() == ""

    def test_record_turn_adds_entry(self):
        s = SessionState()
        s.record_turn("u1", "yelp", ["item_1", "item_2"])
        assert len(s.turns) == 1

    def test_context_hint_non_empty_after_turn(self):
        s = SessionState()
        s.record_turn("u1", "yelp", ["item_1"])
        hint = s.context_hint()
        assert len(hint) > 0
        assert "yelp" in hint.lower()

    def test_platform_set_after_turn(self):
        s = SessionState()
        s.record_turn("u1", "amazon", ["item_1"])
        assert s.platform == "amazon"

    def test_multiple_turns_tracked(self):
        s = SessionState()
        s.record_turn("u1", "yelp", ["item_1"])
        s.record_turn("u1", "yelp", ["item_2"])
        assert len(s.turns) == 2

    def test_top5_stored_per_turn(self):
        s = SessionState()
        items = ["a", "b", "c", "d", "e", "f", "g"]
        s.record_turn("u1", "yelp", items)
        assert s.turns[0]["top5"] == items[:5]
