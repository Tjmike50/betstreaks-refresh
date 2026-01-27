#!/usr/bin/env python3
import os
import time
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from requests.exceptions import ReadTimeout, ConnectionError
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from supabase import create_client
from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.static import players
from nba_api.library.http import NBAStatsHTTP

# --- NBA HTTP/session configuration (robust but small changes)
# Configure sensible timeout and headers; allow override via env vars
NBAStatsHTTP.timeout = int(os.environ.get("NBA_TIMEOUT", "60"))  # seconds

# Default headers that mimic a modern browser (helps avoid simple bot blocking)
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

# Build a requests.Session with retries/backoff and optional proxy support
_retry_strategy = Retry(
    total=5,
    backoff_factor=1,  # base backoff; we also add jitter in our wrapper
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
    raise_on_status=False,
)

_adapter = HTTPAdapter(max_retries=_retry_strategy)
_session = requests.Session()
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

# Merge default headers into session headers (can be augmented via NBA_EXTRA_HEADERS env var)
_session.headers.update(_default_headers)
extra = os.environ.get("NBA_EXTRA_HEADERS")  # optional: "X-Foo:bar|X-Bar:baz"
if extra:
    for pair in extra.split("|"):
        if ":" in pair:
            k, v = pair.split(":", 1)
            _session.headers[k.strip()] = v.strip()

# Optional proxy env var (allows GitHub Actions to use a proxy)
_proxy = (os.environ.get("NBA_PROXY") or
          os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or
          os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"))
if _proxy:
    _session.proxies.update({"http": _proxy, "https": _proxy})

# Apply the session to nba_api HTTP wrapper
# nba_api expects NBAStatsHTTP.session and NBAStatsHTTP.headers / timeout to exist
NBAStatsHTTP.session = _session
NBAStatsHTTP.headers.update(_default_headers)

# --- App constants
SEASON = os.environ.get("NBA_SEASON", "2025-26")
SEASON_TYPE = os.environ.get("NBA_SEASON_TYPE", "Regular Season")
RECENT_DAYS = int(os.environ.get("RECENT_DAYS", "14"))
MIN_STREAK = int(os.environ.get("MIN_STREAK", "2"))
LOOKBACK = int(os.environ.get("LOOKBACK", "12"))
STATS = {"PTS": "PTS", "AST": "AST", "REB": "REB", "3PM": "FG3M"}

# --- Helpers
def alen(v, t):
    m = v < t
    return int(m.argmax()) if m.any() else int(len(v))

def last_ge(v, t, n):
    n = min(n, len(v))
    x = v[:n]
    h = int((x >= t).sum())
    return h, n, round((h / n) * 100, 3) if n else 0.0

def _call_with_retry(func, *args, retries=5, base_wait=2, **kwargs):
    """Call func with retries, exponential backoff and jitter on ReadTimeout/ConnectionError."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            # small pacing to avoid burst requests from CI
            time.sleep(0.5)
            return func(*args, **kwargs)
        except (ReadTimeout, ConnectionError) as e:
            last_err = e
            wait = base_wait * (2 ** (attempt - 1)) + random.uniform(0, 1.5)
            print(f"NBA request error: {e!r} — retry {attempt}/{retries} after {wait:.1f}s")
            time.sleep(wait)
    # if we exit loop without success, raise the last error
    raise last_err

def get_logs_with_retry(season, season_type, retries=5):
    """Get league/player game logs via playergamelogs.PlayerGameLogs with retries."""
    def _do():
        return playergamelogs.PlayerGameLogs(
            season=season,
            season_type_nullable=season_type
        ).get_data_frames()[0]
    return _call_with_retry(_do, retries=retries)

def get_player_gamelogs_with_retry(player_id, retries=5):
    """If you ever need per-player calls, do them through this wrapper (not used by default)."""
    from nba_api.stats.endpoints import playergamelogs as _pg
    def _do():
        # Example: instantiate endpoint for a player (if needed)
        return _pg.PlayerGameLogs(player_id_nullable=player_id).get_data_frames()[0]
    return _call_with_retry(_do, retries=retries)


# --- Main logic (kept largely as original)
def main():
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    # The repository previously attempted to call some "get_logs_with_retry" but had typos.
    lg = get_logs_with_retry(SEASON, SEASON_TYPE)
    lg["GAME_DATE"] = pd.to_datetime(lg["GAME_DATE"])

    act = {p["id"] for p in players.get_players() if p["is_active"]}
    lg = lg[lg["PLAYER_ID"].isin(act)].copy()
    today = pd.Timestamp.now(tz=timezone.utc).tz_localize(None)
    last = lg.groupby("PLAYER_ID")["GAME_DATE"].max()
    keep = set(last[last >= today - pd.Timedelta(days=RECENT_DAYS)].index)
    lg = lg[lg["PLAYER_ID"].isin(keep)].copy()
    now = datetime.now(timezone.utc).isoformat()

    res = []
    for (pid, pname), g in lg.groupby(["PLAYER_ID", "PLAYER_NAME"]):
        g = g.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
        team = g.loc[0, "TEAM_ABBREVIATION"]
        games = len(g)
        lastd = str(g.loc[0, "GAME_DATE"].date())
        for stat, col in STATS.items():
            v = pd.to_numeric(g[col], errors="coerce").fillna(0).astype(int).to_numpy()
            mx = int(v[: min(LOOKBACK, len(v))].max()) if len(v) else 0
            for t in range(1, mx + 1):
                s = alen(v, t)
                if s < MIN_STREAK:
                    continue
                wins = int((v >= t).sum())
                h10, g10, p10 = last_ge(v, t, 10)
                h5, g5, p5 = last_ge(v, t, 5)
                res.append({
                    "sport": "NBA", "entity_type": "player", "player_id": int(pid),
                    "player_name": pname, "team_abbr": team, "stat": stat,
                    "threshold": int(t), "streak_len": int(s),
                    "streak_start": str(g.loc[s - 1, "GAME_DATE"].date()),
                    "last_game": lastd, "season_wins": wins, "season_games": games,
                    "season_win_pct": round((wins / games) * 100, 3) if games else 0.0,
                    "streak_win_pct": 100.0,
                    "last10_hits": h10, "last10_games": g10, "last10_hit_pct": p10,
                    "last5_hits": h5, "last5_games": g5, "last5_hit_pct": p5,
                    "updated_at": now
                })

    df = pd.DataFrame(res)
    rows = df.to_dict("records")
    # batch insert to Supabase in chunks (existing logic)
    sb.table("streaks").delete().eq("sport", "NBA").eq("entity_type", "player").execute()
    for i in range(0, len(rows), 500):
        sb.table("streaks").insert(rows[i:i + 500]).execute()
    print("Players uploaded:", len(rows))


if __name__ == "__main__":
    main()
