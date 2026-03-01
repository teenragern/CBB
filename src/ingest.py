from typing import Dict, List, Tuple
from src.bdl import BDLClient, arr


def fetch_games(api: BDLClient, start_date: str, end_date: str, seasons: List[int]) -> List[Dict]:
    params: List[Tuple[str, object]] = []
    params += arr("seasons", seasons)
    params += [("start_date", start_date), ("end_date", end_date)]
    return api.fetch_all("/games", params=params)


def fetch_team_stats(api: BDLClient, start_date: str, end_date: str, seasons: List[int]) -> List[Dict]:
    params: List[Tuple[str, object]] = []
    params += arr("seasons", seasons)
    params += [("start_date", start_date), ("end_date", end_date)]
    return api.fetch_all("/team_stats", params=params)


def fetch_odds(api: BDLClient, dates: List[str]) -> List[Dict]:
    # odds endpoint supports dates[] or game_ids[]
    params: List[Tuple[str, object]] = []
    params += arr("dates", dates)
    return api.fetch_all("/odds", params=params)
