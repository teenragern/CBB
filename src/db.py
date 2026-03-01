import os
import sqlite3
import datetime as dt
from typing import Any, Dict, List

DB_PATH = os.getenv("DB_PATH", "data/bot.db")


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS games (
            game_id INTEGER PRIMARY KEY,
            season INTEGER,
            date TEXT,
            start_time TEXT,
            status TEXT,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_team_name TEXT,
            away_team_name TEXT,
            home_score REAL,
            away_score REAL,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER,
            ts_utc TEXT,
            vendor TEXT,
            spread_home REAL,
            spread_home_odds INTEGER,
            spread_away_odds INTEGER,
            total REAL,
            total_over_odds INTEGER,
            total_under_odds INTEGER,
            raw_json TEXT
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER,
            ts_utc TEXT,
            mu_margin REAL,
            mu_total REAL,
            sigma_margin REAL,
            sigma_total REAL,
            fair_spread_home REAL,
            fair_total REAL,
            raw_json TEXT
        );

        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER,
            ts_utc TEXT,
            market TEXT,
            side TEXT,
            line REAL,
            odds INTEGER,
            p_model REAL,
            p_market_devig REAL,
            edge REAL,
            ev_per_unit REAL,
            kelly_f REAL,
            stake_units REAL,
            vendor TEXT,
            note TEXT
        );

        CREATE TABLE IF NOT EXISTS results (
            game_id INTEGER PRIMARY KEY,
            settled_ts_utc TEXT,
            home_score REAL,
            away_score REAL,
            final_margin REAL,
            final_total REAL
        );

        -- Dedupe key: one rec per (game, market, side, half-point line)
        CREATE UNIQUE INDEX IF NOT EXISTS rec_uniq2
        ON recommendations(game_id, market, side, CAST(ROUND(line*2) AS INTEGER));

        CREATE INDEX IF NOT EXISTS odds_game_ts
        ON odds_snapshots(game_id, ts_utc);

        CREATE INDEX IF NOT EXISTS rec_game_ts
        ON recommendations(game_id, ts_utc);
        """
    )
    con.commit()
    ensure_schema(con)


def ensure_schema(con: sqlite3.Connection) -> None:
    cols = [r[1] for r in con.execute("PRAGMA table_info(odds_snapshots)").fetchall()]
    if "is_close" not in cols:
        con.execute("ALTER TABLE odds_snapshots ADD COLUMN is_close INTEGER DEFAULT 0;")
        con.commit()


def upsert_games(con: sqlite3.Connection, games: List[Dict[str, Any]]) -> None:
    q = """
    INSERT INTO games (
        game_id, season, date, start_time, status,
        home_team_id, away_team_id, home_team_name, away_team_name,
        home_score, away_score, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(game_id) DO UPDATE SET
        season=excluded.season,
        date=excluded.date,
        start_time=excluded.start_time,
        status=excluded.status,
        home_team_id=excluded.home_team_id,
        away_team_id=excluded.away_team_id,
        home_team_name=excluded.home_team_name,
        away_team_name=excluded.away_team_name,
        home_score=excluded.home_score,
        away_score=excluded.away_score,
        updated_at=excluded.updated_at
    """
    rows = []
    ts = utc_now_iso()
    for g in games:
        gid = int(g["id"])
        season = int(g.get("season") or 0)
        date = g.get("date") or g.get("game_date") or ""
        start_time = g.get("start_time") or g.get("datetime") or g.get("date") or ""
        status = g.get("status") or ""
        home = g.get("home_team") or {}
        away = g.get("visitor_team") or {}
        rows.append(
            (
                gid,
                season,
                date,
                start_time,
                status,
                int(home.get("id") or 0),
                int(away.get("id") or 0),
                str(home.get("full_name") or home.get("name") or ""),
                str(away.get("full_name") or away.get("name") or ""),
                float(g.get("home_score") or 0.0),
                float(g.get("away_score") or 0.0),
                ts,
            )
        )
    con.executemany(q, rows)
    con.commit()


def insert_odds_snapshot(con: sqlite3.Connection, game_id: int, vendor: str, o: Dict[str, Any], is_close: int = 0) -> None:
    import json

    con.execute(
        """
        INSERT INTO odds_snapshots (
            game_id, ts_utc, vendor,
            spread_home, spread_home_odds, spread_away_odds,
            total, total_over_odds, total_under_odds,
            raw_json, is_close
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            game_id,
            utc_now_iso(),
            vendor,
            float(o.get("spread_home_value") or 0.0),
            int(o.get("spread_home_odds") or 0),
            int(o.get("spread_away_odds") or 0),
            float(o.get("total_value") or 0.0),
            int(o.get("total_over_odds") or 0),
            int(o.get("total_under_odds") or 0),
            json.dumps(o, separators=(",", ":"), ensure_ascii=False),
            int(is_close),
        ),
    )
    con.commit()


def insert_prediction(con: sqlite3.Connection, game_id: int, payload: Dict[str, Any]) -> None:
    import json

    con.execute(
        """
        INSERT INTO predictions (
            game_id, ts_utc,
            mu_margin, mu_total, sigma_margin, sigma_total,
            fair_spread_home, fair_total,
            raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            game_id,
            utc_now_iso(),
            float(payload["mu_margin"]),
            float(payload["mu_total"]),
            float(payload["sigma_margin"]),
            float(payload["sigma_total"]),
            float(payload["fair_spread_home"]),
            float(payload["fair_total"]),
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        ),
    )
    con.commit()


def insert_recommendation(con: sqlite3.Connection, rec: Dict[str, Any]) -> bool:
    """
    INSERT OR IGNORE so ANY UNIQUE index will dedupe.
    Returns True if inserted, False if skipped.
    """
    cur = con.execute(
        """
        INSERT OR IGNORE INTO recommendations (
            game_id, ts_utc, market, side, line, odds,
            p_model, p_market_devig, edge, ev_per_unit, kelly_f, stake_units,
            vendor, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(rec["game_id"]),
            utc_now_iso(),
            rec["market"],
            rec["side"],
            float(rec["line"]),
            int(rec["odds"]),
            float(rec["p_model"]),
            float(rec["p_market_devig"]),
            float(rec["edge"]),
            float(rec["ev_per_unit"]),
            float(rec["kelly_f"]),
            float(rec["stake_units"]),
            rec.get("vendor") or "",
            rec.get("note") or "",
        ),
    )
    con.commit()
    return cur.rowcount == 1


def settle_finished_games(con: sqlite3.Connection, games: List[Dict[str, Any]]) -> int:
    n = 0
    for g in games:
        if g.get("status") != "post":
            continue

        gid = int(g["id"])
        hs = float(g.get("home_score") or 0.0)
        ays = float(g.get("away_score") or 0.0)
        margin = hs - ays
        total = hs + ays

        con.execute(
            """
            INSERT INTO results (game_id, settled_ts_utc, home_score, away_score, final_margin, final_total)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO UPDATE SET
                settled_ts_utc=excluded.settled_ts_utc,
                home_score=excluded.home_score,
                away_score=excluded.away_score,
                final_margin=excluded.final_margin,
                final_total=excluded.final_total
            """,
            (gid, utc_now_iso(), hs, ays, margin, total),
        )
        n += 1

    con.commit()
    return n