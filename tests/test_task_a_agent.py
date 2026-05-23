from unittest.mock import MagicMock, patch
from task_a.agent import ASCUserModelingAgent


def make_mock_llm(response: str = "stars: 4.0\nreview: Great experience overall!"):
    """Returns a callable mock that simulates an LLM response."""
    return MagicMock(return_value=response)


def make_mock_tool(user_reviews=None, item_reviews=None, user=None, item=None):
    """Returns a mock interaction tool with configurable responses."""
    tool = MagicMock()
    tool.get_user.return_value   = user or {"user_id": "u1", "name": "Test User"}
    tool.get_item.return_value   = item or {
        "item_id": "i1", "name": "Test Restaurant",
        "categories": "Italian", "attributes": {"WiFi": "free"}
    }
    tool.get_reviews.side_effect = lambda user_id=None, item_id=None: (
        user_reviews if user_reviews is not None and user_id else
        item_reviews if item_reviews is not None and item_id else []
    )
    return tool


def make_agent(llm_response=None, user_reviews=None, item_reviews=None,
               user=None, item=None):
    """Convenience: build an agent with mock LLM + interaction tool."""
    agent = ASCUserModelingAgent(llm=make_mock_llm(llm_response or "stars: 4.0\nreview: Good."))
    agent.sentence_model = None  
    agent.set_interaction_tool(
        make_mock_tool(user_reviews, item_reviews, user, item)
    )
    return agent


class TestWorkflowOutputContract:
    def test_returns_dict(self):
        agent = make_agent(user_reviews=[{"stars": 4, "text": "Nice"}])
        result = agent.workflow(user_id="u1", item_id="i1")
        assert isinstance(result, dict)

    def test_has_stars_key(self):
        agent = make_agent()
        result = agent.workflow(user_id="u1", item_id="i1")
        assert "stars" in result

    def test_has_review_key(self):
        agent = make_agent()
        result = agent.workflow(user_id="u1", item_id="i1")
        assert "review" in result

    def test_stars_is_float(self):
        agent = make_agent()
        result = agent.workflow(user_id="u1", item_id="i1")
        assert isinstance(result["stars"], float)

    def test_stars_in_valid_range(self):
        agent = make_agent()
        result = agent.workflow(user_id="u1", item_id="i1")
        assert 1.0 <= result["stars"] <= 5.0

    def test_review_is_string(self):
        agent = make_agent()
        result = agent.workflow(user_id="u1", item_id="i1")
        assert isinstance(result["review"], str)

    def test_review_not_empty(self):
        agent = make_agent()
        result = agent.workflow(user_id="u1", item_id="i1")
        assert len(result["review"]) > 0

    def test_review_max_512_chars(self):
        long_response = "stars: 3.0\nreview: " + "x" * 1000
        agent = make_agent(llm_response=long_response)
        result = agent.workflow(user_id="u1", item_id="i1")
        assert len(result["review"]) <= 512

    def test_cold_start_flag_true_when_no_user_reviews(self):
        agent = make_agent(user_reviews=[])
        result = agent.workflow(user_id="u1", item_id="i1")
        assert result.get("cold_start") is True

    def test_cold_start_flag_false_when_reviews_exist(self):
        agent = make_agent(user_reviews=[{"stars": 4, "text": "Loved it"}])
        result = agent.workflow(user_id="u1", item_id="i1")
        assert result.get("cold_start") is False

    def test_has_predicted_rating_key(self):
        agent = make_agent(user_reviews=[{"stars": 4, "text": "Good"}])
        result = agent.workflow(user_id="u1", item_id="i1")
        assert "predicted_rating" in result


class TestWorkflowErrorHandling:
    def test_no_interaction_tool_returns_stub(self):
        agent = ASCUserModelingAgent(llm=make_mock_llm())
        agent.sentence_model = None
        result = agent.workflow(user_id="u1", item_id="i1")
        assert result["stars"] == 3.0

    def test_llm_failure_returns_fallback(self):
        agent = make_agent(user_reviews=[{"stars": 4, "text": "Great"}])
        agent.llm = MagicMock(side_effect=Exception("API down"))
        result = agent.workflow(user_id="u1", item_id="i1")
        assert isinstance(result["stars"], float)
        assert 1.0 <= result["stars"] <= 5.0

    def test_malformed_llm_response_uses_predicted_rating(self):
        agent = make_agent(
            llm_response="Sorry, I cannot help with that.",
            user_reviews=[{"stars": 4, "text": "Good"}]
        )
        result = agent.workflow(user_id="u1", item_id="i1")
        assert 1.0 <= result["stars"] <= 5.0

    def test_interaction_tool_exception_returns_fallback(self):
        agent = ASCUserModelingAgent(llm=make_mock_llm())
        agent.sentence_model = None
        tool = MagicMock()
        tool.get_user.side_effect = Exception("DB error")
        agent.set_interaction_tool(tool)
        result = agent.workflow(user_id="u1", item_id="i1")
        assert isinstance(result, dict)
        assert "stars" in result


class TestCOTSCPredict:
    def test_majority_vote_selects_most_common_rating(self):
        responses = [
            "stars: 4.0\nreview: Great!",
            "stars: 4.0\nreview: Loved it.",
            "stars: 2.0\nreview: Disappointing.",
        ]
        agent = ASCUserModelingAgent(llm=MagicMock(side_effect=responses))
        agent.sentence_model = None
        stars, review = agent._cotsc_predict("test prompt", n_samples=3)
        assert stars == 4.0

    def test_returns_none_none_when_all_samples_fail(self):
        agent = ASCUserModelingAgent(llm=MagicMock(return_value="no parseable output here"))
        agent.sentence_model = None
        stars, review = agent._cotsc_predict("test prompt", n_samples=3)
        assert stars is None
        assert review is None

    def test_single_valid_sample_returned(self):
        agent = ASCUserModelingAgent(llm=MagicMock(return_value="stars: 3.0\nreview: Okay."))
        agent.sentence_model = None
        stars, review = agent._cotsc_predict("test prompt", n_samples=1)
        assert stars == 3.0
        assert review == "Okay."

    def test_stars_clamped_in_range(self):
        agent = ASCUserModelingAgent(llm=MagicMock(return_value="stars: 5.0\nreview: Amazing!"))
        agent.sentence_model = None
        stars, _ = agent._cotsc_predict("test prompt", n_samples=1)
        assert 1.0 <= stars <= 5.0

    def test_stars_rounded_to_half(self):
        agent = ASCUserModelingAgent(llm=MagicMock(return_value="stars: 3.5\nreview: Decent."))
        agent.sentence_model = None
        stars, _ = agent._cotsc_predict("test prompt", n_samples=1)
        assert (stars * 2) == round(stars * 2)


class TestDirectGenerate:
    def test_returns_stars_and_review(self):
        agent = ASCUserModelingAgent(llm=make_mock_llm("stars: 4.0\nreview: Really good product."))
        agent.sentence_model = None
        result = agent.direct_generate(persona="Budget shopper", product_name="USB Hub")
        assert "stars" in result
        assert "review" in result

    def test_stars_in_range(self):
        agent = ASCUserModelingAgent(llm=make_mock_llm("stars: 5.0\nreview: Excellent!"))
        agent.sentence_model = None
        result = agent.direct_generate(persona="Tech enthusiast", product_name="Laptop Stand")
        assert 1.0 <= result["stars"] <= 5.0

    def test_works_without_persona(self):
        agent = ASCUserModelingAgent(llm=make_mock_llm("stars: 3.0\nreview: It is okay."))
        agent.sentence_model = None
        result = agent.direct_generate(persona="", product_name="Phone Case")
        assert isinstance(result["stars"], float)

    def test_nigerian_context_false_still_returns_result(self):
        agent = ASCUserModelingAgent(llm=make_mock_llm("stars: 2.0\nreview: Not great."))
        agent.sentence_model = None
        result = agent.direct_generate(
            persona="Regular user", product_name="Headphones", nigerian_context=False
        )
        assert result["stars"] == 2.0


class TestMDILURetrieve:
    def test_returns_k_reviews(self):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda x: (
            np.random.rand(len(x), 64) if isinstance(x, list) else np.random.rand(64)
        )
        agent = ASCUserModelingAgent()
        agent.sentence_model = mock_model
        reviews = [{"text": f"Review {i}", "stars": i % 5 + 1} for i in range(10)]
        result = agent._mdilu_retrieve_similar("query text", reviews, k=3)
        assert len(result) == 3

    def test_returns_subset_of_input_reviews(self):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda x: (
            np.random.rand(len(x), 64) if isinstance(x, list) else np.random.rand(64)
        )
        agent = ASCUserModelingAgent()
        agent.sentence_model = mock_model
        reviews = [{"text": f"Review {i}"} for i in range(5)]
        result = agent._mdilu_retrieve_similar("query", reviews, k=2)
        assert all(r in reviews for r in result)

    def test_returns_empty_when_no_reviews(self):
        agent = ASCUserModelingAgent()
        agent.sentence_model = MagicMock()
        with patch("task_a.agent.cosine_retrieve", return_value=[]):
            result = agent._mdilu_retrieve_similar("query", [], k=3)
        assert result == []
