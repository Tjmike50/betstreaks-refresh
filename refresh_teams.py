#!/usr/bin/env python3
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from supabase import create_client
from nba_api.stats.endpoints import leaguegamelog

# Robustly import NBAStatsHTTP from possible locations
NBAStatsHTTP = None
try:
    from nba_api.stats.library.http import NBAStatsHTTP
except Exception:
    try:
        from nba_api.library.http import NBAStatsHTTP
    except Exception:
        NBAStatsHTTP = None
        print("Warning: NBAStatsHTTP import failed; continuing without patching nba_api HTTP wrapper.")

# Configure timeout and session similarly to players file
NBA_TIMEOUT = int(os.environ.get("NBA_TIMEOUT", "60"))

_default_headers = {
    "User-Agent": os.environ.get("NBA_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                                 "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                 "Chrome/121.0.0.0 Safari/537.36"),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/plain, */*",
    "Connection": "keep-alive",
}

# Retry strategy (compat for different urllib3 versions)
try:
    _retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
        raise_on_status=False,
    )
except TypeError:
    _retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=frozenset(["HEAD", "GET", "OPTIONS"]),
        raise_on_status=False,
    )

_adapter = HTTPAdapter(max_retries=_retry_strategy)
_session = requests.Session()
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)
_session.headers.update(_default_headers)

_proxy = (os.environ.get("NBA_PROXY") or
          os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or
          os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"))
if _proxy:
    _session.proxies.update({"http": _proxy, "https": _proxy})

if NBAStatsHTTP is not None:
    try:
        NBAStatsHTTP.timeout = NBA_TIMEOUT
    except Exception:
        pass
    try:
        NBAStatsHTTP.session = _session
    except Exception:
        pass
    try:
        NBAStatsHTTP.headers.update(_default_headers)
    except Exception:
        pass
else:
    print("NBAStatsHTTP not available — configured requests.Session only.")

SEASON = os.environ.get("NBA_SEASON", "2025-26")
SEASON_TYPE = os.environ.get("NBA_SEASON_TYPE", "Regular Season")
PTS = [90, 95, 100, 105, 110, 115, 120, 125, 130]

def alen(v, t):
    m = v < t
    return int(m.argmax()) if m.any() else int(len(v))

def last_ge(v, t, n):
    n = min(n, len(v))
    x = v[:n]
    h = int((x >= t).sum())
    return h, n, round((h / n) * 100, 3) if n else 0.0

def last_le(v, t, n):
    n = min(n, len(v)); x = v[:n]; h = int((x <= t).sum())
    return h, n, round((h / n) * 100, 3) if n else 0.0

def _call_with_retry(func, *args, retries=5, base_wait=2, **kwargs):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            time.sleep(0.5)
            return func(*args, **kwargs)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            wait = base_wait * (2 ** (attempt - 1)) + random.uniform(0, 1.5)
            print(f"NBA request error: {e!r} — retry {attempt}/{retries} after {wait:.1f}s")
            time.sleep(wait)
    raise last_err

def get_league_gamelog_with_retry(season, season_type, retries=5):
    def _do():
        return leaguegamelog.LeagueGameLog(season=season, season_type_all_star=season_type,
                                           player_or_team_abbreviation="T").get_data_frames()[0]
    return _call_with_retry(_do, retries=retries)

def main():
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    lg = get_league_gamelog_with_retry(SEASON, SEASON_TYPE)
    lg["GAME_DATE"] = pd.to_datetime(lg["GAME_DATE"])
    now = datetime.utcnow().isoformat()

    rows = []
    for (abbr, tid), g in lg.groupby(["TEAM_ABBREVIATION", "TEAM_ID"]):
        g = g.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
        last = str(g.loc[0, "GAME_DATE"].date()); games = len(g)
        wl = g["WL"].astype(str).to_numpy(); s = 0
        for x in wl:
            if x == "W": s += 1
            else: break
        wins = int((g["WL"] == "W").sum())
        n10 = min(10, len(wl)); n5 = min(5, len(wl))
        rows.append({
            "sport": "NBA", "entity_type": "team", "player_id": int(tid), "player_name": abbr,
            "team_abbr": abbr, "stat": "ML", "threshold": 1, "streak_len": int(s),
            "streak_start": str(g.loc[s - 1, "GAME_DATE"].date()) if s > 0 else last,
            "last_game": last, "season_wins": wins, "season_games": games,
            "season_win_pct": round((wins / games) * 100, 3) if games else 0.0,
            "streak_win_pct": 100.0, "last10_hits": int((wl[:n10] == "W").sum()), "last10_games": n10,
            "last10_hit_pct": round(((wl[:n10] == "W").sum() / n10) * 100, 3) if n10 else 0.0,
            "last5_hits": int((wl[:n5] == "W").sum()), "last5_games": n5,
            "last5_hit_pct": round(((wl[:n5] == "W").sum() / n5) * 100, 3) if n5 else 0.0,
            "updated_at": now
        })

        pts = g["PTS"].astype(int).to_numpy()
        for t in PTS:
            so = alen(pts, t)
            if so >= 2:
                w = int((pts >= t).sum()); a10, b10, c10 = last_ge(pts, t, 10); a5, b5, c5 = last_ge(pts, t, 5)
                rows.append({
                    "sport": "NBA", "entity_type": "team", "player_id": int(tid), "player_name": abbr,
                    "team_abbr": abbr, "stat": "PTS", "threshold": int(t), "streak_len": int(so),
                    "streak_start": str(g.loc[so - 1, "GAME_DATE"].date()), "last_game": last,
                    "season_wins": w, "season_games": games, "season_win_pct": round((w / games) * 100, 3) if games else 0.0,
                    "streak_win_pct": 100.0, "last10_hits": a10, "last10_games": b10, "last10_hit_pct": c10,
                    "last5_hits": a5, "last5_games": b5,
                    "last5_hit_pct": c5, "updated_at": now
                })

        su = alen(-pts, -t)  # <= t
        if su >= 2:
            w = int((pts <= t).sum()); a10, b10, c10 = last_le(pts, t, 10); a5, b5, c5 = last_le(pts, t, 5)
            rows.append({
                "sport": "NBA", "entity_type": "team", "player_id": int(tid), "player_name": abbr,
                "team_abbr": abbr, "stat": "PTS_U", "threshold": int(t), "streak_len": int(su),
                "streak_start": str(g.loc[su - 1, "GAME_DATE"].date()), "last_game": last,
                "season_wins": w, "season_games": games, "season_win_pct": round((w / games) * 100, 3) if games else 0.0,
                "streak_win_pct": 100.0, "last10_hits": a10, "last10_games": b10, "last10_hit_pct": c10,
                "last5_hits": a5, "last5_games": b5,
                "last5_hit_pct": c5, "updated_at": now
            })

    sb.table("streaks").delete().eq("sport", "NBA").eq("entity_type", "team").execute()
    for i in range(0, len(rows), 500):
        sb.table("streaks").insert(rows[i:i + 500]).execute()
    print("Teams uploaded:", len(rows))

if __name__ == "__main__":
    main()