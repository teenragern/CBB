import os
import time
import requests
from typing import Optional, Dict, Any

TELEGRAM_API = "https://api.telegram.org"
MAX_TELEGRAM_LEN = 3900  # keep under 4096


def _env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v.strip() if v else None


class TelegramNotifier:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or _env("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or _env("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False

        url = f"{TELEGRAM_API}/bot{self.token}/sendMessage"
        payload: Dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }

        # basic retries (handles rate limits / transient errors)
        for _ in range(3):
            r = requests.post(url, json=payload, timeout=20)
            if r.status_code == 200:
                return True
            if r.status_code == 429:
                try:
                    wait = r.json().get("parameters", {}).get("retry_after", 2)
                except Exception:
                    wait = 2
                time.sleep(wait)
                continue
            time.sleep(1)

        return False

    def send_long(self, text: str) -> bool:
        """
        Sends long messages by chunking into multiple Telegram messages.
        """
        if not self.enabled:
            return False

        text = (text or "").strip()
        if not text:
            return False

        ok_any = False
        while len(text) > MAX_TELEGRAM_LEN:
            chunk = text[:MAX_TELEGRAM_LEN]
            cut = chunk.rfind("\n")
            if cut < 1000:
                cut = MAX_TELEGRAM_LEN
            part = text[:cut].strip()
            text = text[cut:].strip()
            ok_any = self.send(part) or ok_any

        ok_any = self.send(text) or ok_any
        return ok_any


def format_recommendation(game: dict, r: dict) -> str:
    home = game["home_team"]["full_name"]
    away = game["visitor_team"]["full_name"]

    market = r["market"]
    side = r["side"]
    line = r["line"]
    odds = r["odds"]
    vendor = r.get("vendor", "")
    p_model = r.get("p_model", 0.0)
    p_mkt = r.get("p_market_devig", 0.0)
    edge = r.get("edge", 0.0)
    ev = r.get("ev_per_unit", 0.0)
    stake = r.get("stake_units", 0.0)

    return (
        f"{away} @ {home}\n"
        f"{market.upper()} — {side.upper()} | line {line} | odds {odds} | {vendor}\n"
        f"p_model {p_model:.3f} | p_mkt {p_mkt:.3f} | edge {edge:+.3f}\n"
        f"EV/1u {ev:+.3f} | stake {stake:.2f}u"
    )


def format_recommendation_line(game: dict, r: dict, decision: str = "BET", reason: str = "") -> str:
    home = game["home_team"]["full_name"]
    away = game["visitor_team"]["full_name"]

    tag = "BET" if decision.upper() == "BET" else "PASS"
    why = f" | {reason}" if reason else ""

    return (
        f"{tag}{why}\n"
        f"{away} @ {home} | {r['market']} {r['side']} line={r['line']} odds={r['odds']} "
        f"stake={r.get('stake_units',0):.2f}u edge={r.get('edge',0):+.3f} EV={r.get('ev_per_unit',0):+.3f} "
        f"({r.get('vendor','')})"
    )