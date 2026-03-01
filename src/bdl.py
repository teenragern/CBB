import os
import time
import requests
from typing import Any, Dict, List, Optional, Tuple, Union

BASE = "https://api.balldontlie.io/ncaab/v1"


class BDLClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("BDL_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing BDL_API_KEY (add it in Replit Secrets).")
        self.timeout = timeout

    def _get(
        self,
        path: str,
        params: Optional[Union[Dict[str, Any], List[Tuple[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        url = f"{BASE}{path}"
        r = requests.get(
            url,
            headers={"Authorization": self.api_key},
            params=params,
            timeout=self.timeout,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"BDL error {r.status_code} for {url}: {r.text[:300]}")
        return r.json()

    def fetch_all(
        self,
        path: str,
        params: Optional[List[Tuple[str, Any]]] = None,
        per_page: int = 100,
        sleep_s: float = 0.05,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        cursor = None
        while True:
            p = list(params or [])
            p.append(("per_page", per_page))
            if cursor is not None:
                p.append(("cursor", cursor))
            data = self._get(path, params=p)
            out.extend(data.get("data", []))
            cursor = (data.get("meta") or {}).get("next_cursor")
            if not cursor:
                break
            time.sleep(sleep_s)
        return out


def arr(name: str, values: List[Any]) -> List[Tuple[str, Any]]:
    # BDL uses array-style query params like dates[]=YYYY-MM-DD, seasons[]=2025, game_ids[]=123
    return [(f"{name}[]", v) for v in values]
