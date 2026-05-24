# AgentiCulture

**DSN × BCT LLM Agent Challenge 3.0 submission**

Two containerised API services that model how Nigerian users behave on review platforms — predicting how they would rate and review an item (Task A) and ranking items they are most likely to enjoy (Task B).

---

## What this does

| | Task A — User Modeling | Task B — Recommendation |
|---|---|---|
| **Input** | User persona + product details | User persona + list of candidate items |
| **Output** | Predicted star rating + written review | Ranked list of items |
| **Port** | `8001` | `8002` |
| **Key endpoint** | `POST /generate-review` | `POST /recommend` |

Both services are culturally adapted to sound and reason like Nigerian users.

---

## Quick start

### Prerequisites
- Python 3.11+
- Docker Desktop (running)
- An LLM API key — Together AI (recommended), OpenAI, or any OpenAI-compatible provider

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd agenticulture

cp .env.example .env
# Open .env and add your API key
```

`.env` minimum:
```env
LLM_API_KEY=your-together-ai-key-here
LLM_BASE_URL=https://api.together.xyz/v1
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
```

### 2. Run with Docker (recommended)

```bash
make up
# or: docker-compose up --build
```

Both services start automatically. First run takes 3–5 minutes (downloads the embedding model once).

- Task A: http://localhost:8001/docs
- Task B: http://localhost:8002/docs

### 3. Run locally without Docker

```bash
uv sync

# Terminal 1 — Task A
uv run uvicorn task_a.main:app --port 8001 --reload

# Terminal 2 — Task B
uv run uvicorn task_b.main:app --port 8002 --reload
```

---

## Testing the APIs

### Task A — generate a review (direct/persona mode)

```bash
curl -X POST http://localhost:8001/generate-review \
  -H "Content-Type: application/json" \
  -d '{
    "persona": "Budget-conscious Lagos student, values delivery speed and durability.",
    "product_details": {
      "product_id": "item_1",
      "product_name": "Wireless Earbuds",
      "category": "electronics",
      "description": "Bluetooth earbuds with 24hr battery life"
    },
    "nigerian_context": true
  }'
```

Response:
```json
{
  "stars": 4.0,
  "review": "Honestly these earbuds surprised me. For the price ehn, the quality is decent...",
  "cold_start": false,
  "mode": "direct"
}
```

### Task B — rank recommendations (direct/persona mode)

```bash
curl -X POST http://localhost:8002/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "persona": "Young professional in Lagos who enjoys fine dining and values good service",
    "candidate_list": ["item_1", "item_2", "item_3", "item_4", "item_5"],
    "nigerian_context": true
  }'
```

Response:
```json
{
  "recommendations": [
    {"product_id": "item_3", "rank": 1},
    {"product_id": "item_1", "rank": 2},
    {"product_id": "item_5", "rank": 3}
  ],
  "cold_start": true,
  "session_turn": 1,
  "mode": "direct"
}
```

### Health checks

```bash
make health
# or individually:
curl http://localhost:8001/health
curl http://localhost:8002/health
```

---

## Makefile commands

```bash
make up        # Build and start both services
make down      # Stop containers
make logs      # Follow logs
make health    # Check both /health endpoints
make test      # Run unit tests
make eval-a    # Task A local eval (all platforms, 5 tasks)
make eval-b    # Task B local eval (all platforms, 5 tasks)
```

---

## Project structure

```
agenticulture/
│
├── task_a/                   ← Task A service (deliverable #1)
│   ├── main.py               ← FastAPI routes and request/response models
│   ├── agent.py              ← All Task A logic (CF, MDILU, COTSC)
│   ├── Dockerfile
│   └── requirements.txt
│
├── task_b/                   ← Task B service (deliverable #2)
│   ├── main.py               ← FastAPI routes and request/response models
│   ├── agent.py              ← All Task B logic (Borda, cross-domain, sessions)
│   ├── Dockerfile
│   └── requirements.txt
│
├── core/                     ← Shared utilities (embeddings, config, prompts, utils)
│
├── evaluation/               ← AgentSociety Challenge framework (local scoring only)
│
├── tests/                    ← Unit tests
├── eval_task_a.py            ← Local Task A evaluation script
├── eval_task_b.py            ← Local Task B evaluation script
├── print_results.py          ← Print metrics tables from saved results
│
├── eval_results/             ← Pre-computed results (full runs + ablations)
│
├── papers/                   ← Solution papers (LaTeX source + references)
│   ├── track_a_user_modeling_tiamiyu.pdf
│   ├── track_b_recommendation_tiamiyu.pdf
│   └── references.bib
│
├── Makefile                  ← Common commands
├── docker-compose.yml        ← Starts both services with one command
└── .env.example              ← Copy to .env and fill in your API key
```

---

## How the agents work

### Task A

1. Fetches the user's review history and the item's metadata (benchmark mode) or reads the persona text directly (direct mode).
2. Computes a **consistency-weighted** predicted rating — users with consistent rating patterns get higher weight because their mean is a reliable predictor; erratic raters are down-weighted toward the item and global means.
3. For users with no history (**cold-start**), falls back to embedding their persona text and finding the most similar item reviews.
4. Asks the LLM to write a review in the user's voice, using **Nigerian cultural prompts** (value-for-money framing, local expressions, communal tone).
5. Uses **COTSC** (asks the LLM 3 times, votes on the most consistent rating) to reduce random variance.

### Task B

1. Fetches the user's history and detects the platform (Yelp / Amazon / Goodreads).
2. Builds a **preference embedding vector** from the user's past reviews.
3. **Cold-start:** if fewer than 3 reviews exist, ranks candidates purely by cosine similarity to the persona text — no LLM needed.
4. **Warm path:** asks the LLM to rank candidates 3 times with different temperatures, then merges rankings using **Borda count voting**.
5. If the user has reviews on other platforms, blends in **cross-domain preferences** (70% platform-specific, 30% cross-domain).
6. Optional `session_id` tracks multi-turn context across requests.

---

## Local evaluation

### Dataset setup

The evaluation scripts need the  dataset placed at `data/eval/`. The folder must contain three files:

```
data/eval/
├── user.json      ← user profiles
├── item.json      ← item metadata
└── review.json    ← review history
```
These files are sourced from the DSN × BCT LLM Agent Challenge 3.0 dataset. Process the raw dataset using `stream_data.py` and place the output here before running evaluations.

---

### Flags reference

| Flag | Values | Description |
|------|--------|-------------|
| `--platform` | `yelp`, `amazon`, `goodreads`, `all` | Which platform to evaluate. Use `all` to run all three in sequence. |
| `--tasks` | any integer | Number of tasks to run per platform. Each task = one LLM API call. |
| `--data_dir` | path | Path to folder containing `user.json`, `item.json`, `review.json`. |
| `--workers` | integer | Parallel workers. Keep at `1` to avoid rate-limiting. |
| `--ablation` | see below | Run a specific ablation variant instead of the full system. |

---

### Quick smoke-test (2 tasks, ~2 min)

Load your `.env` first, then run:

```bash
set -a && source .env && set +a


# Task A — 2 tasks on Yelp (for quick test, you can use task 2 as increasing the tasks consumes api credits)
uv run python eval_task_a.py --platform yelp --tasks 2 --data_dir ./data/eval --workers 1

# Task B — 2 tasks on Yelp
uv run python eval_task_b.py --platform yelp --tasks 2 --data_dir ./data/eval --workers 1
```

To test a different platform, replace `yelp` with `amazon` or `goodreads`:
```bash
uv run python eval_task_a.py --platform amazon --tasks 2 --data_dir ./data/eval --workers 1
```

To run all three platforms at once for 2 tasks (3 × 2 tasks):
```bash
uv run python eval_task_a.py --platform all --tasks 2 --data_dir ./data/eval --workers 1
```

Results are printed to the terminal and saved to `eval_results/` automatically, e.g.:
- `eval_results/results_task_a_yelp.json`
- `eval_results/results_task_b_yelp.json`

---

### Full evaluation run (~1 hour per task, uses API credits)

```bash
# Task A — all platforms, 50 tasks each
uv run python eval_task_a.py --platform all --tasks 50 --data_dir ./data/eval --workers 1

# Task B — all platforms, 50 tasks each
uv run python eval_task_b.py --platform all --tasks 50 --data_dir ./data/eval --workers 1
```

Or use the Makefile shortcuts (5 tasks per platform):
```bash
make eval-a
make eval-b
```

Raw simulation outputs are saved as `*_raw.json` checkpoints before scoring, so no API credits are lost if evaluation crashes mid-run.

### Reproduce the paper tables from saved results

Pre-computed results are committed under `eval_results/`. No API key or dataset needed:

```bash
# Aggregate metrics tables (Task A and Task B)
python3 print_results.py --results_dir ./eval_results

# With per-task breakdown for all platforms
python3 print_results.py --results_dir ./eval_results --per-task

# Per-task breakdown for one platform only
python3 print_results.py --results_dir ./eval_results --per-task --platform yelp
```

To reproduce a specific ablation result:
```bash
# Example: Task A CF-only ablation on Yelp
uv run python eval_task_a.py --platform yelp --tasks 50 --ablation cf_only \
  --data_dir ./data/eval --workers 1
# Result saved to: eval_results/results_task_a_yelp_cf_only.json
```

Available Task A ablation modes: `llm_direct`, `cf_only`, `cf_mdilu`, `cf_cotsc`, `full`
Available Task B ablation modes: `cosine_only`, `single_llm`, `borda_only`, `full`

This uses the AgentSociety Challenge evaluation framework — the same metrics as WWW 2025.

---

## Deliverables checklist

| # | Deliverable | Location | Status |
|---|-------------|----------|--------|
| 1 | Task A containerised API | `task_a/` | Ready |
| 2 | Task B containerised API | `task_b/` | Ready |
| 3 | Solution papers (Track A + B) | `papers/track_a_user_modeling_tiamiyu.pdf`, `papers/track_b_recommendation_tiamiyu.pdf` | Ready |
| 4 | Code repository | This repo | Ready |

---

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_API_KEY` | Yes | — | Together AI / OpenAI / any compatible provider key |
| `LLM_BASE_URL` | No | OpenAI | Set to `https://api.together.xyz/v1` for Together AI |
| `LLM_MODEL` | No | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Any chat model from your provider |
| `DATA_DIR` | No | — | Path to dataset folder containing `user.json`, `item.json`, `review.json`. Defaults to `./data/eval` (the DSN × BCT dataset). |
| `ENABLE_NIGERIAN_CONTEXT` | No | `true` | Toggle Nigerian cultural adaptation |

---

## Tech stack

- **FastAPI** — web framework for both APIs
- **sentence-transformers** (`BAAI/bge-small-en-v1.5`) — local embeddings for similarity ranking
- **scikit-learn** — cosine similarity, statistical features
- **uv** — fast dependency installation in Docker
- **Docker Compose** — runs both services together
