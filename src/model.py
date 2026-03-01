from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import datetime as dt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import config as cfg
from src.features import (
    possessions_from_team_stat,
    safe_div,
    compute_rest_days,
    compute_sos_ratings,
    infer_game_type,
    load_injury_overrides,
    load_neutral_overrides,
)

UTC = dt.timezone.utc


@dataclass
class FittedModels:
    pace_model: Pipeline
    ppp_model: Pipeline
    sigma_margin: float
    sigma_total: float
    sigma_margin_by_abs: Optional[Dict[str, Any]] = field(default=None)
    sigma_total_by_mu: Optional[Dict[str, Any]] = field(default=None)
    sos_ratings: Dict[int, Dict[str, float]] = field(default_factory=dict)


def recency_weights(dates_utc, halflife_days: int) -> np.ndarray:
    max_dt = max(dates_utc)
    ages = np.array([(max_dt - d).total_seconds() / 86400.0 for d in dates_utc], dtype=float)
    w = 0.5 ** (ages / float(halflife_days))
    return w


def apply_row_weights(X, y: np.ndarray, w: np.ndarray):
    from scipy.sparse import issparse
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    sw = np.sqrt(np.clip(w, 1e-8, 1.0)).reshape(-1, 1)
    return X * sw, y * sw.ravel()


def sigma_buckets(x: np.ndarray, resid: np.ndarray, n_buckets: int) -> Dict[str, Any]:
    qs = np.quantile(x, np.linspace(0, 1, n_buckets + 1))
    qs[0] = -1e9
    qs[-1] = 1e9
    sigmas = []
    for i in range(n_buckets):
        m = (x >= qs[i]) & (x < qs[i + 1])
        sig = float(np.std(resid[m], ddof=1)) if m.sum() >= 10 else float(np.std(resid, ddof=1))
        sigmas.append(sig)
    return {"edges": qs.tolist(), "sigmas": sigmas}


def _parse_game_dt(g: Dict) -> dt.datetime:
    d = (g.get("date") or "").replace("Z", "+00:00")
    try:
        dtu = dt.datetime.fromisoformat(d)
        if dtu.tzinfo is None:
            dtu = dtu.replace(tzinfo=UTC)
        return dtu
    except Exception:
        return dt.datetime.min.replace(tzinfo=UTC)


def build_training_rows(
    games: List[Dict],
    team_stats: List[Dict],
    rest_map: Dict,
    inj: Dict[int, float],
    neutral_map: Dict[int, int],
    sos_ratings: Dict[int, Dict[str, float]],
) -> Tuple[List[Dict], List[Dict], List[dt.datetime]]:
    ts_map: Dict[Tuple[int, int], Dict] = {}
    for ts in team_stats:
        gid = int(ts["game"]["id"])
        tid = int(ts["team"]["id"])
        ts_map[(gid, tid)] = ts

    pace_rows: List[Dict] = []
    ppp_rows: List[Dict] = []
    pace_dates: List[dt.datetime] = []

    def sos_for(tid: int) -> Dict[str, float]:
        row = sos_ratings.get(tid) or {}
        return {
            "adj_off": float(row.get("adj_off", 1.0)),
            "adj_def": float(row.get("adj_def", 1.0)),
        }

    for g in games:
        if g.get("status") != "post":
            continue
        gid = int(g["id"])
        home = g["home_team"]
        away = g["visitor_team"]
        hid = int(home["id"])
        aid = int(away["id"])

        if (gid, hid) not in ts_map or (gid, aid) not in ts_map:
            continue

        poss_h = possessions_from_team_stat(ts_map[(gid, hid)])
        poss_a = possessions_from_team_stat(ts_map[(gid, aid)])
        poss = 0.5 * (poss_h + max(0, poss_a))  # Safeguard

        home_pts = float(g.get("home_score") or 0)
        away_pts = float(g.get("away_score") or 0)

        rest_home = rest_map.get((gid, hid), 7)
        rest_away = rest_map.get((gid, aid), 7)
        b2b_home = 1 if rest_home == 0 else 0
        b2b_away = 1 if rest_away == 0 else 0

        inj_home = inj.get(hid, 0.0)
        inj_away = inj.get(aid, 0.0)

        is_neutral = neutral_map.get(gid, 0)
        game_type = infer_game_type({"date": g.get("date"), "is_neutral": is_neutral})
        sos_home = sos_for(hid)
        sos_away = sos_for(aid)

        game_dt = _parse_game_dt(g)

        pace_rows.append(
            {
                "season": int(g["season"]),
                "game_type": game_type,
                "home_team_id": hid,
                "away_team_id": aid,
                "is_neutral": is_neutral,
                "rest_home": rest_home,
                "rest_away": rest_away,
                "b2b_home": b2b_home,
                "b2b_away": b2b_away,
                "inj_home": inj_home,
                "inj_away": inj_away,
                "poss": poss,
            }
        )
        pace_dates.append(game_dt)

        ppp_rows.append(
            {
                "season": int(g["season"]),
                "game_type": game_type,
                "off_team_id": hid,
                "def_team_id": aid,
                "is_home": 1,
                "rest": rest_home,
                "b2b": b2b_home,
                "inj_off": inj_home,
                "inj_def": inj_away,
                "adj_off_off": sos_home["adj_off"],
                "adj_def_off": sos_home["adj_def"],
                "adj_off_def": sos_away["adj_off"],
                "adj_def_def": sos_away["adj_def"],
                "ppp": safe_div(home_pts, poss),
            }
        )
        ppp_rows.append(
            {
                "season": int(g["season"]),
                "game_type": game_type,
                "off_team_id": aid,
                "def_team_id": hid,
                "is_home": 0,
                "rest": rest_away,
                "b2b": b2b_away,
                "inj_off": inj_away,
                "inj_def": inj_home,
                "adj_off_off": sos_away["adj_off"],
                "adj_def_off": sos_away["adj_def"],
                "adj_off_def": sos_home["adj_off"],
                "adj_def_def": sos_home["adj_def"],
                "ppp": safe_div(away_pts, poss),
            }
        )

    return pace_rows, ppp_rows, pace_dates

    return pace_rows, ppp_rows, pace_dates

    return pace_rows, ppp_rows, pace_dates


def fit_models(games: List[Dict], team_stats: List[Dict]) -> FittedModels:
    rest_map = compute_rest_days(games)
    inj = load_injury_overrides()
    neutral_map = load_neutral_overrides()
    sos_ratings = compute_sos_ratings(games, team_stats, n_iter=10)

    pace_rows, ppp_rows, pace_dates = build_training_rows(
        games, team_stats, rest_map, inj, neutral_map, sos_ratings
    )

    if len(pace_rows) < 200 or len(ppp_rows) < 400:
        raise RuntimeError(
            f"Not enough training data (pace_rows={len(pace_rows)}, ppp_rows={len(ppp_rows)}). "
            "Try a longer training window or include more seasons."
        )

    pace_df = pd.DataFrame(pace_rows)
    ppp_df = pd.DataFrame(ppp_rows)

    pace_X = pace_df[[
        "season", "game_type", "home_team_id", "away_team_id", "is_neutral",
        "rest_home", "rest_away", "b2b_home", "b2b_away",
        "inj_home", "inj_away",
    ]]
    pace_y = pace_df["poss"].values

    pace_pre = ColumnTransformer(
        [
            ("season", OneHotEncoder(handle_unknown="ignore"), ["season"]),
            ("game_type", OneHotEncoder(handle_unknown="ignore"), ["game_type"]),
            ("home", OneHotEncoder(handle_unknown="ignore"), ["home_team_id"]),
            ("away", OneHotEncoder(handle_unknown="ignore"), ["away_team_id"]),
            ("pass", "passthrough", [
                "is_neutral", "rest_home", "rest_away", "b2b_home", "b2b_away", "inj_home", "inj_away",
            ]),
        ]
    )

    pace_dates_arr = pace_dates
    w_pace = recency_weights(pace_dates_arr, cfg.RECENCY_HALFLIFE_DAYS)
    pace_X_transformed = pace_pre.fit_transform(pace_X)
    Xw_pace, yw_pace = apply_row_weights(pace_X_transformed, pace_y, w_pace)
    pace_ridge = Ridge(alpha=5.0)
    pace_ridge.fit(Xw_pace, yw_pace)

    pace_model = Pipeline([("pre", pace_pre), ("ridge", pace_ridge)])

    ppp_X = ppp_df[[
        "season", "game_type", "off_team_id", "def_team_id", "is_home",
        "rest", "b2b", "inj_off", "inj_def",
        "adj_off_off", "adj_def_off", "adj_off_def", "adj_def_def",
    ]]
    ppp_y = ppp_df["ppp"].values

    ppp_pre = ColumnTransformer(
        [
            ("season", OneHotEncoder(handle_unknown="ignore"), ["season"]),
            ("game_type", OneHotEncoder(handle_unknown="ignore"), ["game_type"]),
            ("off", OneHotEncoder(handle_unknown="ignore"), ["off_team_id"]),
            ("def", OneHotEncoder(handle_unknown="ignore"), ["def_team_id"]),
            ("pass", "passthrough", [
                "is_home", "rest", "b2b", "inj_off", "inj_def",
                "adj_off_off", "adj_def_off", "adj_off_def", "adj_def_def",
            ]),
        ]
    )

    ppp_dates = pace_dates_arr + pace_dates_arr
    w_ppp = recency_weights(ppp_dates, cfg.RECENCY_HALFLIFE_DAYS)
    ppp_X_transformed = ppp_pre.fit_transform(ppp_X)
    Xw_ppp, yw_ppp = apply_row_weights(ppp_X_transformed, ppp_y, w_ppp)
    ppp_ridge = Ridge(alpha=10.0)
    ppp_ridge.fit(Xw_ppp, yw_ppp)

    ppp_model = Pipeline([("pre", ppp_pre), ("ridge", ppp_ridge)])

    ts_map = {(int(ts["game"]["id"]), int(ts["team"]["id"])): ts for ts in team_stats}

    pred_m = []
    pred_t = []
    margins = []
    totals = []

    def sos_for(tid: int) -> Dict[str, float]:
        row = sos_ratings.get(tid) or {}
        return {
            "adj_off": float(row.get("adj_off", 1.0)),
            "adj_def": float(row.get("adj_def", 1.0)),
        }

    for g in games:
        if g.get("status") != "post":
            continue
        gid = int(g["id"])
        hid = int(g["home_team"]["id"])
        aid = int(g["visitor_team"]["id"])
        if (gid, hid) not in ts_map or (gid, aid) not in ts_map:
            continue

        home_pts = float(g.get("home_score") or 0)
        away_pts = float(g.get("away_score") or 0)

        rest_home = rest_map.get((gid, hid), 7)
        rest_away = rest_map.get((gid, aid), 7)
        b2b_home = 1 if rest_home == 0 else 0
        b2b_away = 1 if rest_away == 0 else 0
        inj_home = inj.get(hid, 0.0)
        inj_away = inj.get(aid, 0.0)
        is_neutral = neutral_map.get(gid, 0)
        game_type = infer_game_type({"date": g.get("date"), "is_neutral": is_neutral})

        sos_home = sos_for(hid)
        sos_away = sos_for(aid)

        P_hat = float(
            pace_model.predict(
                pd.DataFrame(
                    [{
                        "season": int(g["season"]),
                        "game_type": game_type,
                        "home_team_id": hid,
                        "away_team_id": aid,
                        "is_neutral": is_neutral,
                        "rest_home": rest_home,
                        "rest_away": rest_away,
                        "b2b_home": b2b_home,
                        "b2b_away": b2b_away,
                        "inj_home": inj_home,
                        "inj_away": inj_away,
                    }]
                )
            )[0]
        )

        ppp_h = float(
            ppp_model.predict(
                pd.DataFrame(
                    [{
                        "season": int(g["season"]),
                        "game_type": game_type,
                        "off_team_id": hid,
                        "def_team_id": aid,
                        "is_home": 1,
                        "rest": rest_home,
                        "b2b": b2b_home,
                        "inj_off": inj_home,
                        "inj_def": inj_away,
                        "adj_off_off": sos_home["adj_off"],
                        "adj_def_off": sos_home["adj_def"],
                        "adj_off_def": sos_away["adj_off"],
                        "adj_def_def": sos_away["adj_def"],
                    }]
                )
            )[0]
        )
        ppp_a = float(
            ppp_model.predict(
                pd.DataFrame(
                    [{
                        "season": int(g["season"]),
                        "game_type": game_type,
                        "off_team_id": aid,
                        "def_team_id": hid,
                        "is_home": 0,
                        "rest": rest_away,
                        "b2b": b2b_away,
                        "inj_off": inj_away,
                        "inj_def": inj_home,
                        "adj_off_off": sos_away["adj_off"],
                        "adj_def_off": sos_away["adj_def"],
                        "adj_off_def": sos_home["adj_off"],
                        "adj_def_def": sos_home["adj_def"],
                    }]
                )
            )[0]
        )

        mu_home = P_hat * ppp_h
        mu_away = P_hat * ppp_a

        margins.append(home_pts - away_pts)
        totals.append(home_pts + away_pts)
        pred_m.append(mu_home - mu_away)
        pred_t.append(mu_home + mu_away)

    resid_m = np.array(margins) - np.array(pred_m)
    resid_t = np.array(totals) - np.array(pred_t)

    sigma_margin = float(np.std(resid_m, ddof=1)) if len(resid_m) > 50 else 11.0
    sigma_total = float(np.std(resid_t, ddof=1)) if len(resid_t) > 50 else 14.0

    sigma_margin_by_abs = sigma_buckets(np.abs(np.array(pred_m)), resid_m, cfg.SIGMA_BUCKETS)
    sigma_total_by_mu = sigma_buckets(np.array(pred_t), resid_t, cfg.SIGMA_BUCKETS)

    return FittedModels(
        pace_model=pace_model,
        ppp_model=ppp_model,
        sigma_margin=sigma_margin,
        sigma_total=sigma_total,
        sigma_margin_by_abs=sigma_margin_by_abs,
        sigma_total_by_mu=sigma_total_by_mu,
        sos_ratings=sos_ratings,
    )