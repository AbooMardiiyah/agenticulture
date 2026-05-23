.PHONY: help up down restart logs health test eval-a eval-b

help:
	@echo "AgentiCulture — available commands:"
	@echo ""
	@echo "  make up        Build and start both services (Task A :8001, Task B :8002)"
	@echo "  make down      Stop and remove containers"
	@echo "  make restart   Stop, rebuild, and restart both services"
	@echo "  make logs      Follow logs from both services"
	@echo "  make health    Check health of both running services"
	@echo "  make test      Run unit tests"
	@echo "  make eval-a    Run Task A local evaluation (all platforms, 5 tasks)"
	@echo "  make eval-b    Run Task B local evaluation (all platforms, 5 tasks)"
	@echo ""
	@echo "  Requires: cp .env.example .env  and add LLM_API_KEY before running"

up:
	docker-compose up --build

down:
	docker-compose down

restart:
	docker-compose down && docker-compose up --build

logs:
	docker-compose logs -f

health:
	@curl -s http://localhost:8001/health | python3 -m json.tool && \
	 curl -s http://localhost:8002/health | python3 -m json.tool

test:
	uv run pytest tests/ -v

eval-a:
	uv run python eval_task_a.py --platform all --tasks 5 --data_dir ./data/eval --workers 1

eval-b:
	uv run python eval_task_b.py --platform all --tasks 5 --data_dir ./data/eval --workers 1
