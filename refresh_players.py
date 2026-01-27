#!/usr/bin/env python3
import os
import time
import random
import inspect
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

# --- NBA HTTP/session configuration (robust but small changes)
NBA_TIMEOUT = int(os.environ.get("NBA_TIMEOUT", "60"))  # seconds

_default_headers = {
    "User-Agent": os.environ.get(
        "NBA_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
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

# Allow optional extra headers via env var NBA_EXTRA_HEADERS, format: "X-Foo:bar|X-Bar:baz"
_extra = os.environ.get("NBA_EXTRA_HEADERS")
if _extra:
    for pair in _extra.split("|"):
        if ":" in pair:
            k, v = pair.split(":", 1)
            _session.headers[k.strip()] = v.strip()

# Optional proxy env var
_proxy = (os.environ.get("NBA_PROXY") or
          os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or
          os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"))
if _proxy:
    _session.proxies.update({"http": _proxy, "https": _proxy})

# Apply to nba_api wrapper if present
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
            time.sleep(0.5)  # pacing
            return func(*args, **kwargs)
        except (ReadTimeout, ConnectionError) as e:
            last_err = e
            wait = base_wait * (2 ** (attempt - 1)) + random.uniform(0, 1.5)
            print(f"NBA request error: {e!r} — retry {attempt}/{retries} after {wait:.1f}s")
            time.sleep(wait)
    raise last_err

def _try_playergamelogs_variants(season, season_type):
    """
    Try calling PlayerGameLogs with several keyword argument variants based on the
    installed nba_api implementation. Use inspect to detect constructor parameters.
    """
    ctor = playergamelogs.PlayerGameLogs
    sig = inspect.signature(ctor.__init__)
    pnames = sig.parameters.keys()

    # Build candidate kwargs dynamically
    candidates = []

    # Candidate 1: common older usage
    kwargs1 = {}
    if 'season' in pnames:
        kwargs1['season'] = season
    if 'season_type_nullable' in pnames:
        kwargs1['season_type_nullable'] = season_type
    if kwargs1:
        candidates.append(kwargs1)

    # Candidate 2: alternative names
    kwargs2 = {}
    if 'season_nullable' in pnames:
        kwargs2['season_nullable'] = season
    if 'season_type_nullable' in pnames:
        kwargs2['season_type_nullable'] = season_type
    if kwargs2:
        candidates.append(kwargs2)

    # Candidate 3: season + different season_type name
    kwargs3 = {}
    if 'season' in pnames:
        kwargs3['season'] = season
    if 'season_type_all_star' in pnames:
        kwargs3['season_type_all_star'] = season_type
    if 'season_type' in pnames:
        kwargs3['season_type'] = season_type
    if kwargs3:
        candidates.append(kwargs3)

    # Candidate 4: season_nullable + season_type_all_star
    kwargs4 = {}
    if 'season_nullable' in pnames:
        kwargs4['season_nullable'] = season
    if 'season_type_all_star' in pnames:
        kwargs4['season_type_all_star'] = season_type
    if kwargs4:
        candidates.append(kwargs4)

    # Candidate 5: no kwargs (let the library decide defaults) — last resort
    candidates.append({})

    last_err = None
    for kw in candidates:
        try:
            # construct and return dataframe if successful
            inst = ctor(**kw)
            dfs = inst.get_data_frames()
            if dfs:
                return dfs[0]
        except TypeError as e:
            last_err = e
            # try next candidate
            continue
        except Exception as e:
            # non-TypeErrors may be transient (network), raise to trigger retry wrapper
            raise

    # If none of the candidates worked, raise the last TypeError for visibility
    raise last_err or RuntimeError("PlayerGameLogs invocation failed without exception.")

def get_logs_with_retry(season, season_type, retries=5):
    """Get player game logs with retries and signature compatibility."""
    def _do():
        return _try_playergamelogs_variants(season, season_type)
    return _call_with_retry(_do, retries=retries)

def get_player_gamelogs_with_retry(player_id, retries=5):
    from nba_api.stats.endpoints import playergamelogs as _pg
    def _do():
        # Try a couple of common parameter names for a per-player call
        sig = inspect.signature(_pg.PlayerGameLogs.__init__)
        pnames = sig.parameters.keys()
        kwargs = {}
        if 'player_id_nullable' in pnames:
            kwargs['player_id_nullable'] = player_id
        elif 'player_id' in pnames:
            kwargs['player_id'] = player_id
        # Try season args too if available (best-effort)
        if 'season_nullable' in pnames:
            kwargs['season_nullable'] = SEASON
        elif 'season' in pnames:
            kwargs['season'] = SEASON
        return _pg.PlayerGameLogs(**kwargs).get_data_frames()[0]
    return _call_with_retry(_do, retries=retries)

# --- Main logic
def main():
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

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
    sb.table("streaks").delete().eq("sport", "NBA").eq("entity_type", "player").execute()
    for i in range(0, len(rows), 500):
        sb.table("streaks").insert(rows[i:i + 500]).execute()
    print("Players uploaded:", len(rows))

if __name__ == "__main__":
    main()
