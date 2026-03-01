import datetime as dt
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Tuple

from src.bdl import arr, BDLClient
from src.db import upsert_games, insert_odds_snapshot

NY = ZoneInfo("America/New_York")

def parse_utc_iso_z(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat((s or "").replace("Z", "+00:00"))

def snap_upcoming_odds(
    api: BDLClient,
    con,
    hours_ahead: int = 36,
    days_back: int = 1
) -> int:
    now_ny = dt.datetime.now(NY)
    start = (now_ny - dt.timedelta(days=days_back)).date()
    end = (now_ny + dt.timedelta(hours=hours_ahead)).date() + dt.timedelta(days=1)

    # Pull games in a window, then filter by start time
    raw_games = api.fetch_all("/games", params=[("start_date", start.isoformat()), ("end_date", end.isoformat())])

    upcoming = []
    for g in raw_games:
        try:
            start_utc = parse_utc_iso_z(g.get("date", ""))
            start_ny = start_utc.astimezone(NY)
        except Exception:
            continue
        if now_ny - dt.timedelta(hours=6) <= start_ny <= now_ny + dt.timedelta(hours=hours_ahead):
            upcoming.append(g)

    upsert_games(con, upcoming)

    game_ids = [int(g["id"]) for g in upcoming]
    if not game_ids:
        return 0

    odds_rows = api.fetch_all("/odds", params=arr("game_ids", game_ids))

    # Write a snapshot per odds row
    n = 0
    for o in odds_rows:
        gid = int(o["game_id"])
        insert_odds_snapshot(con, gid, (o.get("vendor") or ""), o)
        n += 1
    return n


def snap_close_odds(api, con, minutes_before: int = 5, window_min: int = 2) -> int:
    UTC = dt.timezone.utc
    now = dt.datetime.now(UTC)

    start = (now - dt.timedelta(hours=6)).date().isoformat()
    end = (now + dt.timedelta(hours=18)).date().isoformat()

    games = api.fetch_all("/games", params=[("start_date", start), ("end_date", end)])

    targets = []
    for g in games:
        try:
            tip = dt.datetime.fromisoformat(g["date"].replace("Z", "+00:00")).astimezone(UTC)
            mins = (tip - now).total_seconds() / 60.0
            if (minutes_before - window_min) <= mins <= (minutes_before + window_min):
                targets.append(int(g["id"]))
        except Exception:
            continue

    if not targets:
        return 0

    odds_rows = api.fetch_all("/odds", params=arr("game_ids", targets))
    n = 0
    for o in odds_rows:
        gid = int(o.get("game_id") or o.get("id") or 0)
        if not gid:
            continue
        insert_odds_snapshot(con, gid, (o.get("vendor") or ""), o, is_close=1)
        n += 1
    return n
