# ConflictCast

**Agentic AI system for geopolitical conflict risk prediction and economic cascade modeling.**

ConflictCast uses a multi-agent pipeline (LangGraph + Groq / Llama 3.3) to ingest real-time geopolitical news and commodity market data, then generates structured conflict risk assessments with economic impact forecasts.

## How it works

```
[fetch_news] → [fetch_commodities] → [analyze_risk] → [format_output]
```

1. **fetch_news** — Queries the GDELT Project API for recent conflict-related news in the target region
2. **fetch_commodities** — Pulls live commodity prices (oil, gas, wheat, gold, copper) via Yahoo Finance
3. **analyze_risk** — Sends all data to Groq (Llama 3.3) for multi-step geopolitical reasoning
4. **format_output** — Packages the final structured report

## API

### `POST /predict`
Run a conflict risk prediction for any region.

**Request:**
```json
{ "region": "South China Sea" }
```

**Response:**
```json
{
  "region": "South China Sea",
  "generated_at": "2026-05-20T18:00:00Z",
  "data_sources": ["GDELT Project", "Yahoo Finance", "Groq / Llama 3.3"],
  "news_articles_analyzed": 18,
  "assessment": {
    "risk_score": 72,
    "risk_level": "high",
    "conflict_probability": "65%",
    "key_tension_signals": ["..."],
    "cascade_effects": { "..." },
    "early_interventions": ["..."],
    "analyst_summary": "..."
  }
}
```

### `GET /hotspots`
Returns risk predictions for the top 5 current global hotspot regions, ranked by risk score.

### `GET /health`
Health check.

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/conflictcast
cd conflictcast

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and add your Groq API key

# 4. Run locally
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for the interactive API documentation.

## Tech Stack

- **LangGraph** — Multi-agent graph orchestration
- **Claude API** — LLM reasoning (claude-sonnet-4-6)
- **FastAPI** — REST API framework
- **GDELT Project** — Real-time global event monitoring (free)
- **Yahoo Finance (yfinance)** — Commodity price data (free)
- **Render** — Cloud deployment

## Data Sources

| Source | Data | Cost |
|---|---|---|
| GDELT Project | Global news & conflict events | Free |
| Yahoo Finance | Commodity prices | Free |
| Groq API (Llama 3.3) | LLM reasoning | Free |
