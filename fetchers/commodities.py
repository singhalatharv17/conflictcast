"""
Commodities Fetcher
-------------------
Pulls current and recent commodity prices using yfinance (free, no API key needed).
Commodities tracked: Oil, Natural Gas, Wheat, Gold, Rare Earths proxy.
"""

import yfinance as yf


# Ticker symbols for key commodities
COMMODITIES = {
    "oil_wti":     "CL=F",   # WTI Crude Oil
    "natural_gas": "NG=F",   # Natural Gas
    "wheat":       "ZW=F",   # Wheat
    "gold":        "GC=F",   # Gold (safe-haven indicator)
    "copper":      "HG=F",   # Copper (industrial demand proxy)
}


def fetch_commodity_prices() -> dict:
    """
    Returns current price and 30-day % change for each key commodity.
    This data helps the agent assess whether markets are already pricing in risk.
    """
    results = {}

    for name, ticker in COMMODITIES.items():
        try:
            data = yf.download(ticker, period="1mo", interval="1d", progress=False)

            if data.empty:
                results[name] = {"error": "no data"}
                continue

            latest_price = float(data["Close"].iloc[-1])
            month_ago_price = float(data["Close"].iloc[0])
            pct_change = ((latest_price - month_ago_price) / month_ago_price) * 100

            results[name] = {
                "current_price": round(latest_price, 2),
                "30d_change_pct": round(pct_change, 2),
                "trend": "rising" if pct_change > 2 else "falling" if pct_change < -2 else "stable",
            }

        except Exception as e:
            results[name] = {"error": str(e)}

    return results
