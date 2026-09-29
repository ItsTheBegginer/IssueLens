# IssueLens

AI-powered GitHub Issue Localization — IBM Bob 2.0 Hackathon.

Given a GitHub repository URL and an issue number, IssueLens analyzes the issue
and pinpoints the files and code locations most likely responsible.

## Structure

```
issuelens/
├── frontend/        React + Vite UI
├── backend/         Python + FastAPI API
├── bob_sessions/    Bob AI session logs
├── docs/            Documentation
├── .env.example     Environment variable template
└── README.md
```

## Quick Start

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check: http://localhost:8000/health

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App: http://localhost:5173

### Environment

Copy `.env.example` to `.env` and fill in your tokens:

```bash
cp .env.example .env
```

Required variables:
- `GITHUB_TOKEN` — GitHub Personal Access Token (for repo + issue API access)
- `BOB_API_KEY` — IBM Bob Inference API key (from bob.ibm.com → API keys, type "Inference")
- `BOB_API_URL` — Base URL of the Bob LiteLLM proxy (provided in your hackathon portal)
- `BOB_MODEL_ID` — *(optional)* model name (default: `ibm/granite-3-3-8b-instruct`)
- `VITE_BACKEND_URL` — Backend URL seen by the browser (default: `http://localhost:8000`)

## Running Tests

```bash
cd backend
source .venv/bin/activate
pytest tests/
```
