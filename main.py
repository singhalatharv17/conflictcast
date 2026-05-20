"""
ConflictCast API
----------------
FastAPI backend exposing the ConflictCast agent as a REST API.

Endpoints:
  POST /predict          — Run a conflict risk prediction for a region
  GET  /hotspots         — Get predictions for top global hotspot regions
  GET  /health           — Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from agent import run_prediction

# Simple in-memory cache: { region -> (report, timestamp) }
# Prevents redundant Claude API calls for the same region within 1 hour
_cache: dict = {}
CACHE_TTL_MINUTES = 60


def get_cached_or_run(region: str) -> dict:
    """Return cached prediction if fresh, otherwise run the agent."""
    key = region.lower().strip()
    if key in _cache:
        report, cached_at = _cache[key]
        if datetime.utcnow() - cached_at < timedelta(minutes=CACHE_TTL_MINUTES):
            report["cached"] = True
            return report
    result = run_prediction(region)
    _cache[key] = (result, datetime.utcnow())
    return result

app = FastAPI(
    title="ConflictCast API",
    description="Agentic AI system for geopolitical conflict risk prediction and economic cascade modeling.",
    version="1.0.0",
)

# Allow requests from the React frontend (and any origin during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    region: str

    class Config:
        json_schema_extra = {
            "example": {"region": "South China Sea"}
        }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ConflictCast API"}


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Run the ConflictCast agent for a given region.
    Returns a structured conflict risk assessment with economic cascade modeling.
    """
    if not request.region.strip():
        raise HTTPException(status_code=400, detail="Region cannot be empty.")

    try:
        report = get_cached_or_run(request.region)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/hotspots")
def get_hotspots():
    """
    Run predictions for the top 5 current global hotspot regions.
    Returns ranked list by risk score.
    """
    hotspot_regions = [
        "South China Sea",
        "Eastern Ukraine",
        "Middle East",
        "Korean Peninsula",
        "Taiwan Strait",
    ]

    results = []
    for region in hotspot_regions:
        try:
            report = get_cached_or_run(region)
            results.append({
                "region": region,
                "risk_score": report["assessment"].get("risk_score", 0),
                "risk_level": report["assessment"].get("risk_level", "unknown"),
                "conflict_probability": report["assessment"].get("conflict_probability", "N/A"),
                "analyst_summary": report["assessment"].get("analyst_summary", ""),
            })
        except Exception as e:
            results.append({"region": region, "error": str(e)})

    # Sort by risk score descending
    results.sort(key=lambda x: x.get("risk_score", 0), reverse=True)
    return {"hotspots": results, "count": len(results)}
