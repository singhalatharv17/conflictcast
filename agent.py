"""
ConflictCast Agent
------------------
A multi-agent pipeline built with LangGraph and Claude API.

Graph structure:
  [fetch_news] --> [fetch_commodities] --> [analyze_risk] --> [format_output]

The first two nodes run in parallel, then feed into analysis.
"""

import os
import json
from groq import Groq
from typing import TypedDict
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv

load_dotenv()

from fetchers.gdelt import fetch_conflict_news
from fetchers.commodities import fetch_commodity_prices


# ── State ────────────────────────────────────────────────────────────────────
# This is the shared data object that passes between all nodes in the graph.
# Each node reads from it and writes new fields into it.

class AgentState(TypedDict):
    region: str
    news_articles: list[dict]
    commodity_data: dict
    analysis: dict
    final_report: dict


# ── Nodes ─────────────────────────────────────────────────────────────────────

def fetch_news_node(state: AgentState) -> AgentState:
    """Node 1: Fetch recent conflict-related news from GDELT."""
    print(f"[Agent] Fetching news for: {state['region']}")
    articles = fetch_conflict_news(state["region"])
    return {**state, "news_articles": articles}


def fetch_commodities_node(state: AgentState) -> AgentState:
    """Node 2: Fetch current commodity prices from Yahoo Finance."""
    print("[Agent] Fetching commodity prices...")
    prices = fetch_commodity_prices()
    return {**state, "commodity_data": prices}


def analyze_risk_node(state: AgentState) -> AgentState:
    """
    Node 3: Send all collected data to Claude API for multi-step reasoning.
    Claude acts as a geopolitical analyst, scoring conflict risk and modeling
    economic cascade effects.
    """
    print("[Agent] Running Claude analysis...")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Format news headlines for the prompt
    headlines = "\n".join(
        [f"- [{a['date'][:8]}] {a['title']} ({a['source']})"
         for a in state["news_articles"][:15]]
    ) or "No recent news found."

    # Format commodity data
    commodity_summary = "\n".join(
        [f"- {k}: ${v.get('current_price', 'N/A')} ({v.get('30d_change_pct', 'N/A')}% 30d, {v.get('trend', 'N/A')})"
         for k, v in state["commodity_data"].items()
         if "error" not in v]
    )

    prompt = f"""You are a geopolitical risk analyst. Analyze the following data for the region: {state['region']}

RECENT NEWS HEADLINES (last 30 days):
{headlines}

COMMODITY MARKET DATA:
{commodity_summary}

Based on this data, provide a structured conflict risk assessment. Think step by step:
1. What are the key tension signals in the news?
2. Are commodity markets already pricing in risk (rising oil/gold = markets worried)?
3. What is the overall conflict escalation probability?
4. If conflict escalates, what are the likely economic cascade effects?
5. What early interventions could reduce risk?

Return your analysis as a JSON object with exactly this structure:
{{
  "risk_score": <number 0-100>,
  "risk_level": "<low|medium|high|critical>",
  "key_tension_signals": [<list of 3-5 specific signals from the news>],
  "market_interpretation": "<one sentence on what commodity trends signal>",
  "conflict_probability": "<percentage as string, e.g. 65%>",
  "predicted_timeline": "<e.g. next 3-6 months or near-term>",
  "cascade_effects": {{
    "commodity_impact": "<which commodities spike and by how much>",
    "inflation_forecast": "<estimated inflation impact by region>",
    "most_vulnerable_countries": [<list of 3-5 countries>]
  }},
  "early_interventions": [
    {{"action": "<intervention>", "cost": "<low|medium|high>", "impact": "<expected effect>"}}
  ],
  "analyst_summary": "<2-3 sentence plain English summary of the situation>"
}}

Return only the JSON object, no other text."""

    message = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.choices[0].message.content.strip()

    # Parse Claude's JSON response
    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if Claude returns extra text around the JSON
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        analysis = json.loads(match.group()) if match else {"error": "Failed to parse response"}

    return {**state, "analysis": analysis}


def format_output_node(state: AgentState) -> AgentState:
    """Node 4: Package the final report with metadata."""
    from datetime import datetime

    final_report = {
        "region": state["region"],
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_sources": ["GDELT Project", "Yahoo Finance", "Claude AI (claude-sonnet-4-6)"],
        "news_articles_analyzed": len(state["news_articles"]),
        "assessment": state["analysis"],
    }

    return {**state, "final_report": final_report}


# ── Build the Graph ───────────────────────────────────────────────────────────

def build_agent() -> StateGraph:
    """Assembles and compiles the LangGraph agent graph."""

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("fetch_news", fetch_news_node)
    graph.add_node("fetch_commodities", fetch_commodities_node)
    graph.add_node("analyze_risk", analyze_risk_node)
    graph.add_node("format_output", format_output_node)

    # Define edges (execution order)
    graph.set_entry_point("fetch_news")
    graph.add_edge("fetch_news", "fetch_commodities")
    graph.add_edge("fetch_commodities", "analyze_risk")
    graph.add_edge("analyze_risk", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()


# Compiled agent (imported by main.py)
agent = build_agent()


def run_prediction(region: str) -> dict:
    """Run the full agent pipeline for a given region."""
    initial_state: AgentState = {
        "region": region,
        "news_articles": [],
        "commodity_data": {},
        "analysis": {},
        "final_report": {},
    }
    result = agent.invoke(initial_state)
    return result["final_report"]
