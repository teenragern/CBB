import os
import requests
from typing import Any, Dict, List, Optional

BASE = "https://api.the-odds-api.com/v4"


class OddsAPI:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 20):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing ODDS_API_KEY (add it in Replit Secrets).")
        self.timeout = timeout

    def get_ncaab_odds(
        self,
        regions: str = "us",
        markets: str = "spreads,totals,h2h,alternate_spreads,alternate_totals",
        odds_format: str = "american",
    ) -> List[Dict[str, Any]]:
        url = f"{BASE}/sports/basketball_ncaab/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }
        r = requests.get(url, params=params, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Odds API error {r.status_code}: {r.text[:300]}")
        return r.json()