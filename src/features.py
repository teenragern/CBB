import datetime as dt
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import config as cfg

FTA_COEF = 0.44
UTC = dt.timezone.utc


def possessions_from_team_stat(ts: Dict) -> float:
    fga = float(ts.get("fga") or 0)
    oreb = float(ts.get("oreb") or 0)
    tov = float(ts.get("turnovers") or ts.get("turnover") or 0)
    fta = float(ts.get("fta") or 0)
    return fga - oreb + tov + FTA_COEF * fta


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def norm_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def compute_rest_days(games: List[Dict]) -> Dict:
    rows = []
    for g in games:
        gid = int(g["id"])
        d = g.get("date") or ""
        if not d:
            continue
        dtu = dt.datetime.fromisoformat(d.replace("Z", "+00:00"))
        hid = int(g["home_team"]["id"])
        aid = int(g["visitor_team"]["id"])
        rows.append((dtu, gid, hid, aid))

    rows.sort(key=lambda x: x[0])

    last_played: Dict[int, dt.datetime] = {}
    rest: Dict = {}
    for dtu, gid, hid, aid in rows:
        for tid in (hid, aid):
            prev = last_played.get(tid)
            if prev is None:
                rest_days = 7
            else:
                rest_days = max(0, int((dtu - prev).total_seconds() // 86400))
            rest[(gid, tid)] = rest_days
            last_played[tid] = dtu
    return rest


def _parse_game_date(game: Dict) -> Optional[dt.date]:
    s = (game.get("date") or "").replace("Z", "+00:00")
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s).date()
    except Exception:
        return None


def _selection_sunday_cutoff(year: int) -> dt.date:
    # Approximate cutoff per user spec.
    day = 15 if year >= 2025 else 16
    return dt.date(year, 3, day)


def infer_game_type(game: Dict) -> int:
    """
    Returns:
      0 = regular season
      1 = conference tournament
      2 = NCAA tournament

    Heuristic:
      - neutral-site only can be tourney-like
      - conference tournaments: after first week of March and before selection Sunday cutoff
      - NCAA tournament: on/after selection Sunday cutoff
    """
    is_neutral = int(game.get("is_neutral") or 0)
    if is_neutral != 1:
        return 0

    game_date = _parse_game_date(game)
    if game_date is None:
        return 0

    if game_date.month != 3:
        return 0

    cutoff = _selection_sunday_cutoff(game_date.year)

    if game_date >= cutoff:
        return 2

    if game_date > dt.date(game_date.year, 3, 7):
        return 1

    return 0


def compute_sos_ratings(games: List[Dict], team_stats: List[Dict], n_iter: int = 10) -> Dict[int, Dict[str, float]]:
    """
    Iterative SOS-adjusted offensive and defensive PPP ratings.

    Returns:
        dict[team_id] = {
            "adj_off": float,   # >1 is better offense
            "adj_def": float,   # >1 means allows more PPP than average (weaker defense)
        }
    """
    ts_map: Dict[Tuple[int, int], Dict] = {}
    for ts in team_stats:
        gid = int(ts["game"]["id"])
        tid = int(ts["team"]["id"])
        ts_map[(gid, tid)] = ts

    # team_games[team_id] -> list of (opponent_id, actual_ppp_off, actual_ppp_def)
    team_games: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    teams = set()

    for g in games:
        if g.get("status") != "post":
            continue

        gid = int(g["id"])
        hid = int(g["home_team"]["id"])
        aid = int(g["visitor_team"]["id"])

        if (gid, hid) not in ts_map or (gid, aid) not in ts_map:
            continue

        ts_home = ts_map[(gid, hid)]
        ts_away = ts_map[(gid, aid)]
        poss_h = possessions_from_team_stat(ts_home)
        poss_a = possessions_from_team_stat(ts_away)
        poss = 0.5 * (poss_h + poss_a)
        if poss <= 0:
            continue

        home_pts = float(g.get("home_score") or 0.0)
        away_pts = float(g.get("away_score") or 0.0)

        ppp_home = safe_div(home_pts, poss)
        ppp_away = safe_div(away_pts, poss)

        team_games[hid].append((aid, ppp_home, ppp_away))
        team_games[aid].append((hid, ppp_away, ppp_home))
        teams.add(hid)
        teams.add(aid)

    if not teams:
        return {}

    # Initialize from raw mean PPP offense/defense.
    league_vals = []
    for rows in team_games.values():
        for _opp_id, ppp_off, ppp_def in rows:
            league_vals.append(ppp_off)
            league_vals.append(ppp_def)
    league_ppp = sum(league_vals) / len(league_vals) if league_vals else 1.0
    if league_ppp <= 0:
        league_ppp = 1.0

    adj_off: Dict[int, float] = {}
    adj_def: Dict[int, float] = {}
    for tid in teams:
        rows = team_games.get(tid, [])
        if rows:
            adj_off[tid] = sum(r[1] for r in rows) / len(rows)
            adj_def[tid] = sum(r[2] for r in rows) / len(rows)
        else:
            adj_off[tid] = league_ppp
            adj_def[tid] = league_ppp

    # Normalize initial values so league average is 1.0.
    off_mean = sum(adj_off.values()) / len(adj_off)
    def_mean = sum(adj_def.values()) / len(adj_def)
    if off_mean > 0:
        adj_off = {tid: v / off_mean for tid, v in adj_off.items()}
    if def_mean > 0:
        adj_def = {tid: v / def_mean for tid, v in adj_def.items()}

    eps = 1e-6
    for _ in range(max(1, int(n_iter))):
        next_off: Dict[int, float] = {}
        for tid in teams:
            vals = []
            for opp_id, actual_ppp_off, _actual_ppp_def in team_games.get(tid, []):
                opp_adj_def = max(adj_def.get(opp_id, 1.0), eps)
                vals.append(actual_ppp_off / opp_adj_def)
            next_off[tid] = sum(vals) / len(vals) if vals else 1.0

        off_mean = sum(next_off.values()) / len(next_off)
        if off_mean > 0:
            next_off = {tid: v / off_mean for tid, v in next_off.items()}

        next_def: Dict[int, float] = {}
        for tid in teams:
            vals = []
            for opp_id, _actual_ppp_off, actual_ppp_def in team_games.get(tid, []):
                opp_adj_off = max(next_off.get(opp_id, 1.0), eps)
                vals.append(actual_ppp_def / opp_adj_off)
            next_def[tid] = sum(vals) / len(vals) if vals else 1.0

        def_mean = sum(next_def.values()) / len(next_def)
        if def_mean > 0:
            next_def = {tid: v / def_mean for tid, v in next_def.items()}

        adj_off, adj_def = next_off, next_def

    return {
        int(tid): {
            "adj_off": float(adj_off.get(tid, 1.0)),
            "adj_def": float(adj_def.get(tid, 1.0)),
        }
        for tid in sorted(teams)
    }


def load_injury_overrides() -> Dict[int, float]:
    try:
        with open(cfg.INJURY_OVERRIDES_PATH, "r", encoding="utf-8") as f:
            return {int(k): float(v) for k, v in json.load(f).items()}
    except Exception:
        return {}


def load_neutral_overrides() -> Dict[int, int]:
    try:
        with open(cfg.NEUTRAL_OVERRIDES_PATH, "r", encoding="utf-8") as f:
            return {int(k): int(v) for k, v in json.load(f).items()}
    except Exception:
        return {}