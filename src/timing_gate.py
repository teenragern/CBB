import datetime as dt
import sqlite3
from typing import Any, Dict, Optional, Tuple, List

UTC = dt.timezone.utc

LOOKBACK_HOURS_DEFAULT = int(__import__("os").getenv("TIMING_LOOKBACK_HOURS", "6"))
DRIFT_THRESHOLD_DEFAULT = float(__import__("os").getenv("TIMING_DRIFT_THRESHOLD", "0.5"))  # points
NEAR_TIP_MINUTES_DEFAULT = int(__import__("os").getenv("TIMING_NEAR_TIP_MINUTES", "30"))

MAJOR_BOOKS = {"draftkings", "fanduel", "caesars", "betmgm", "betrivers", "fanatics"}

def _parse_iso(s: str) -> dt.datetime:
    if not s:
        return dt.datetime.min.replace(tzinfo=UTC)
    ss = s.replace("Z", "+00:00")
    d = dt.datetime.fromisoformat(ss)
    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC)
    return d.astimezone(UTC)

def _is_bucket_tip_time(tip_dt_utc: dt.datetime) -> bool:
    return tip_dt_utc.minute == 0 and tip_dt_utc.second == 0 and tip_dt_utc.hour in (0, 2)

def _get_tip_dt(con: sqlite3.Connection, game_id: int) -> Optional[dt.datetime]:
    row = con.execute(
        "SELECT start_time FROM games WHERE game_id=?",
        (int(game_id),),
    ).fetchone()
    if not row or not row[0]:
        return None
    tip = _parse_iso(row[0])
    if _is_bucket_tip_time(tip):
        return None
    return tip

def _bet_perspective_line(market: str, side: str, spread_home: Optional[float], total: Optional[float]) -> Optional[float]:
    """
    Returns the line expressed from the BET's perspective:
      - spread home: use spread_home
      - spread away: use -spread_home
      - total: use total
    """
    if market == "spread":
        if spread_home is None:
            return None
        sh = float(spread_home)
        return sh if side == "home" else (-sh if side == "away" else None)

    if market == "total":
        if total is None:
            return None
        return float(total)

    return None

def _line_key_for_bet(market: str, side: str, line: float) -> float:
    """
    Higher key = better for our bet.
      - spread: higher line always better (e.g., -2.5 better than -3.5, +7.5 better than +6.5)
      - total under: higher total better
      - total over: lower total better => use -line
    """
    if market == "total" and side == "over":
        return -float(line)
    return float(line)

def _better_offer(market: str, side: str, a_line: float, a_odds: int, b_line: float, b_odds: int) -> bool:
    """
    True if A is better than B for our bet.
    Primary: best number (line_key)
    Secondary: best price (higher American odds is better, e.g. -105 > -115, +120 > +110)
    """
    ak = _line_key_for_bet(market, side, a_line)
    bk = _line_key_for_bet(market, side, b_line)
    if ak != bk:
        return ak > bk
    return int(a_odds) > int(b_odds)

def timing_gate_decision(
    con: sqlite3.Connection,
    game_id: int,
    rec: Dict[str, Any],
    lookback_hours: int = LOOKBACK_HOURS_DEFAULT,
    drift_threshold: float = DRIFT_THRESHOLD_DEFAULT,
    near_tip_minutes: int = NEAR_TIP_MINUTES_DEFAULT,
) -> Tuple[str, str]:
    """
    Returns:
      ("BET_NOW"|"WAIT", reason)
    """
    market = rec["market"]
    side = rec["side"]

    tip_dt = _get_tip_dt(con, game_id)
    if tip_dt is not None:
        mins_to_tip = (tip_dt - dt.datetime.now(UTC)).total_seconds() / 60.0
        if mins_to_tip <= near_tip_minutes:
            return "BET_NOW", f"near tip ({mins_to_tip:.0f}m)"

    since = (dt.datetime.now(UTC) - dt.timedelta(hours=int(lookback_hours))).isoformat()

    rows = con.execute(
        """
        SELECT ts_utc, vendor,
               spread_home, spread_home_odds, spread_away_odds,
               total, total_over_odds, total_under_odds
        FROM odds_snapshots
        WHERE game_id=? AND ts_utc >= ?
        """,
        (int(game_id), since),
    ).fetchall()

    if not rows:
        return "BET_NOW", "no snapshot history"

    # bucket into 10-minute windows
    buckets: Dict[int, Dict[str, Any]] = {}

    for (ts_utc, vendor, sh, sh_odds, sa_odds, tot, o_odds, u_odds) in rows:
        v = (vendor or "").lower()
        if v not in MAJOR_BOOKS:
            continue

        t = _parse_iso(ts_utc)
        bucket = int(t.timestamp() // 600)  # 10-min buckets

        line = _bet_perspective_line(market, side, sh, tot)
        if line is None:
            continue

        # pick the correct odds for side/market
        odds = 0
        if market == "spread":
            odds = int(sh_odds if side == "home" else sa_odds)
        elif market == "total":
            odds = int(o_odds if side == "over" else u_odds)

        if odds == 0:
            continue

        cur = buckets.get(bucket)
        if cur is None:
            buckets[bucket] = {"t": t, "line": float(line), "odds": int(odds), "vendor": v}
        else:
            if _better_offer(market, side, float(line), int(odds), float(cur["line"]), int(cur["odds"])):
                cur["line"] = float(line)
                cur["odds"] = int(odds)
                cur["vendor"] = v

    series = sorted(buckets.values(), key=lambda x: x["t"])
    if len(series) < 2:
        return "BET_NOW", "insufficient history"

    first = series[0]
    last = series[-1]

    drift_pts = _line_key_for_bet(market, side, last["line"]) - _line_key_for_bet(market, side, first["line"])

    # If line is improving for us => WAIT.
    if drift_pts >= drift_threshold:
        return "WAIT", f"drifting in favor (+{drift_pts:.1f} pts over ~{lookback_hours}h)"

    # If line is worsening => BET NOW.
    if drift_pts <= -drift_threshold:
        return "BET_NOW", f"drifting against (-{abs(drift_pts):.1f} pts over ~{lookback_hours}h)"

    return "BET_NOW", f"flat drift ({drift_pts:+.1f})"