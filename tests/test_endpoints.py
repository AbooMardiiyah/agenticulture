import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


class TestTaskAEndpoints:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Patch LLM and interaction tool init so no real calls are made."""
        with patch("task_a.main._init_llm"), patch("task_a.main._init_interaction_tool"):
            from task_a.main import app, agent
            agent.llm = MagicMock(return_value="stars: 4.0\nreview: Really enjoyed this place!")
            agent.sentence_model = None
            agent.interaction_tool = None  
            self.client = TestClient(app)
            yield

    def test_root_returns_200(self):
        r = self.client.get("/")
        assert r.status_code == 200

    def test_root_has_name_field(self):
        r = self.client.get("/")
        assert "name" in r.json()

    def test_health_returns_200(self):
        r = self.client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_field(self):
        r = self.client.get("/health")
        assert r.json()["status"] == "healthy"

    def test_health_shows_llm_ready(self):
        r = self.client.get("/health")
        assert r.json()["llm_ready"] is True

    def test_generate_review_direct_mode_200(self):
        r = self.client.post("/generate-review", json={
            "persona": "Budget-conscious Lagos student",
            "product_details": {
                "product_id": "item_1",
                "product_name": "Phone Case",
                "category": "electronics",
                "description": "Durable protective case"
            }
        })
        assert r.status_code == 200

    def test_generate_review_returns_stars(self):
        r = self.client.post("/generate-review", json={
            "persona": "Tech enthusiast",
            "product_details": {"product_id": "p1", "product_name": "Laptop Stand"}
        })
        assert "stars" in r.json()

    def test_generate_review_returns_review_text(self):
        r = self.client.post("/generate-review", json={
            "persona": "Foodie",
            "product_details": {"product_id": "p1", "product_name": "Air Fryer"}
        })
        assert "review" in r.json()
        assert len(r.json()["review"]) > 0

    def test_generate_review_stars_in_range(self):
        r = self.client.post("/generate-review", json={
            "persona": "Regular user",
            "product_details": {"product_id": "p1", "product_name": "USB Hub"}
        })
        stars = r.json()["stars"]
        assert 1.0 <= stars <= 5.0

    def test_generate_review_mode_is_direct(self):
        r = self.client.post("/generate-review", json={
            "persona": "Shopper",
            "product_details": {"product_id": "p1", "product_name": "Bag"}
        })
        assert r.json()["mode"] == "direct"

    def test_generate_review_nigerian_context_false(self):
        r = self.client.post("/generate-review", json={
            "persona": "User",
            "product_details": {"product_id": "p1", "product_name": "Shoes"},
            "nigerian_context": False
        })
        assert r.status_code == 200

    def test_generate_review_missing_product_id_422(self):
        r = self.client.post("/generate-review", json={
            "persona": "User",
            "product_details": {"product_name": "Shoes"} 
        })
        assert r.status_code == 422

    def test_generate_review_missing_product_details_returns_200(self):
        r = self.client.post("/generate-review", json={"persona": "User"})
        assert r.status_code == 200

class TestTaskBEndpoints:
    @pytest.fixture(autouse=True)
    def setup(self):
        with patch("task_b.main._init_llm"), patch("task_b.main._init_interaction_tool"):
            from task_b.main import app, agent
            agent.llm = MagicMock(return_value=str(["item_1", "item_2", "item_3"]))
            agent.sentence_model = None
            agent.interaction_tool = None
            self.client = TestClient(app)
            yield

    def test_root_returns_200(self):
        r = self.client.get("/")
        assert r.status_code == 200

    def test_health_returns_200(self):
        r = self.client.get("/health")
        assert r.status_code == 200

    def test_health_status_healthy(self):
        r = self.client.get("/health")
        assert r.json()["status"] == "healthy"

    def test_recommend_direct_mode_no_candidates_returns_empty(self):
        r = self.client.post("/recommend", json={
            "persona": "Tech lover",
            "context": "Looking for electronics"
        })
        assert r.status_code == 200
        assert r.json()["recommendations"] == []

    def test_recommend_direct_mode_with_candidates_returns_ranked(self):
        r = self.client.post("/recommend", json={
            "persona": "Book lover",
            "candidate_list": ["item_1", "item_2", "item_3"]
        })
        assert r.status_code == 200
        assert len(r.json()["recommendations"]) > 0

    def test_recommend_direct_with_candidates_returns_all(self):
        candidates = ["item_1", "item_2", "item_3"]
        r = self.client.post("/recommend", json={
            "persona": "Foodie",
            "candidate_list": candidates
        })
        returned_ids = [rec["product_id"] for rec in r.json()["recommendations"]]
        assert set(returned_ids) == set(candidates)

    def test_recommend_response_has_recommendations_key(self):
        r = self.client.post("/recommend", json={"persona": "User"})
        assert "recommendations" in r.json()

    def test_recommend_response_has_reasoning_key(self):
        r = self.client.post("/recommend", json={"persona": "User"})
        assert "reasoning" in r.json()

    def test_recommend_response_has_mode_key(self):
        r = self.client.post("/recommend", json={"persona": "User"})
        assert "mode" in r.json()

    def test_recommend_each_item_has_rank(self):
        r = self.client.post("/recommend", json={
            "persona": "User",
            "candidate_list": ["item_1", "item_2", "item_3"]
        })
        for rec in r.json()["recommendations"]:
            assert "rank" in rec

    def test_recommend_ranks_are_sequential(self):
        r = self.client.post("/recommend", json={
            "persona": "User",
            "candidate_list": ["item_1", "item_2", "item_3"]
        })
        ranks = [rec["rank"] for rec in r.json()["recommendations"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_recommend_top_k_limits_results(self):
        r = self.client.post("/recommend", json={
            "persona": "User",
            "candidate_list": ["item_1", "item_2", "item_3", "item_4", "item_5"],
            "top_k": 2
        })
        assert len(r.json()["recommendations"]) <= 2

    def test_recommend_cold_start_flag_present(self):
        r = self.client.post("/recommend", json={
            "persona": "User",
            "candidate_list": ["item_1", "item_2"]
        })
        assert "cold_start" in r.json()
