"""
GDELT Fetcher
-------------
Pulls recent geopolitical news and conflict events for a given region
using the GDELT Document API v2 (free, no API key needed).
"""

import httpx
from datetime import datetime


def fetch_conflict_news(region: str, max_records: int = 20) -> list[dict]:
    """
    Search GDELT for recent conflict-related news articles about a region.
    Returns a list of article summaries for the agent to reason over.
    """

    # Keywords that signal geopolitical tension
    conflict_keywords = f'"{region}" (conflict OR tension OR sanctions OR military OR protest OR dispute OR crisis)'

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": conflict_keywords,
        "mode": "artlist",
        "maxrecords": max_records,
        "timespan": "30d",   # last 30 days
        "sort": "datescore", # most relevant + recent first
        "format": "json",
    }

    try:
        response = httpx.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        articles = data.get("articles", [])

        # Return only what the agent needs: title, source, date, url
        return [
            {
                "title": a.get("title", ""),
                "source": a.get("domain", ""),
                "date": a.get("seendate", ""),
                "url": a.get("url", ""),
            }
            for a in articles
            if a.get("title")
        ]

    except Exception as e:
        print(f"[GDELT] Error fetching news for '{region}': {e}")
        return []
