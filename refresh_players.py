#!/usr/bin/env python3
import os
import time
import random
import inspect
import importlib
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
from requests.exceptions import ReadTimeout, ConnectionError as RequestsConnectionError
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from supabase import create_client
from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.static import players

NBAStatsHTTP = None
try:
    from nba_api.stats.library.http import NBAStatsHTTP
except Exception:
    try:
        from nba_api.library.http import NBAStatsHTTP
    except Exception:
        NBAStatsHTTP = None
        print("Warning: NBAStatsHTTP import failed; continuing without patching nba_api HTTP wrapper.", flush=True)

NBA_TIMEOUT = int(os.environ.get("NBA_TIMEOUT", "60"))
NBA_RETRIES = int(os.environ.get("NBA_RETRIES", "2"))
NBA_BASE_WAIT = float(os.environ.get("NBA_BASE_WAIT", "2"))
REQUEST_PACING_SECONDS = float(os.environ.get("REQUEST_PACING_SECONDS", "0.5"))

SEASON = os.environ.get("NBA_SEASON", "2025-26")
SEASON_TYPE = os.environ.get("NBA_SEASON_TYPE", "Regular Season")
RECENT_DAYS = int(os.environ.get("RECENT_DAYS", "21"))
MIN_STREAK = int(os.environ.get("MIN_STREAK", "2"))
LOOKBACK = int(os.environ.get("LOOKBACK", "12"))
DATE_BUFFER_DAYS = int(os.environ.get("DATE_BUFFER_DAYS", "7"))

STATS = {"PTS": "PTS", "AST": "AST", "REB": "REB", "3PM": "FG3M"}

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

try:
    _retry_strategy = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
        raise_on_status=False,
    )
except TypeError:
    _retry_strategy = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=frozenset(["HEAD", "GET", "OPTIONS"]),
        raise_on_status=False,
    )

_adapter = HTTPAdapter(max_retries=_retry_strategy)
_session = requests.Session()
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)
_session.headers.update(_default_headers)

_extra = os.environ.get("NBA_EXTRA_HEADERS")
if _extra:
    for pair in _extra.split("|"):
        if ":" in pair:
            k, v = pair.split(":", 1)
            _session.headers[k.strip()] = v.strip()

_proxy = (
    os.environ.get("NBA_PROXY")
    or os.environ.get("HTTPS_PROXY")
    or os.environ.get("https_proxy")
    or os.environ.get("HTTP_PROXY")
    or os.environ.get("http_proxy")
)
if _proxy:
    _session.proxies.update({"http": _proxy, "https": _proxy})

for mod_name in ("nba_api.stats.library.http", "nba_api.library.http"):
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        continue
    try:
        if hasattr(mod, "NBAStatsHTTP"):
            try:
                mod.NBAStatsHTTP.timeout = NBA_TIMEOUT
            except Exception:
                pass
            try:
                def _get_session_override(self):
                    return _session
                mod.NBAStatsHTTP.get_session = _get_session_override
            except Exception:
                pass
    except Exception:
        print(f"Could not monkeypatch {mod_name}, continuing.", flush=True)

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
        if hasattr(NBAStatsHTTP, "headers") and isinstance(NBAStatsHTTP.headers, dict):
            NBAStatsHTTP.headers.update(_default_headers)
    except Exception:
        pass

def alen(v, t):
    m = v < t
    return int(m.argmax()) if m.any() else int(len(v))

def last_ge(v, t, n):
    n = min(n, len(v))
    x = v[:n]
    h = int((x >= t).sum())
    return h, n, round((h / n) * 100, 3) if n else 0.0

def _call_with_retry(func, *args, retries=NBA_RETRIES, base_wait=NBA_BASE_WAIT, **kwargs):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            time.sleep(REQUEST_PACING_SECONDS)
            return func(*args, **kwargs)
        except (ReadTimeout, RequestsConnectionError, requests.exceptions.RequestException) as e:
            last_err = e
            wait = base_wait * (2 ** (attempt - 1)) + random.uniform(0, 2.0)
            print(f"NBA request error: {e!r} — retry {attempt}/{retries} after {wait:.1f}s", flush=True)
            time.sleep(wait)
    raise last_err

def _build_common_kwargs(param_names):
    kwargs = {}

    if "timeout" in param_names:
        kwargs["timeout"] = NBA_TIMEOUT

    if "headers" in param_names:
        kwargs["headers"] = dict(_session.headers)

    if "proxy" in param_names and _proxy:
        kwargs["proxy"] = _proxy

    return kwargs

def _recent_date_strings():
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=RECENT_DAYS + DATE_BUFFER_DAYS)
    return start_date.strftime("%m/%d/%Y"), end_date.strftime("%m/%d/%Y")

def _try_playergamelogs_variants(season, season_type):
    ctor = playergamelogs.PlayerGameLogs
    sig = inspect.signature(ctor.__init__)
    pnames = set(sig.parameters.keys())

    date_from, date_to = _recent_date_strings()
    common_kwargs = _build_common_kwargs(pnames)
    candidates = []

    def add_date_filters(kw):
        if "date_from_nullable" in pnames:
            kw["date_from_nullable"] = date_from
        if "date_to_nullable" in pnames:
            kw["date_to_nullable"] = date_to
        return kw

    kwargs1 = add_date_filters(dict(common_kwargs))
    if "season" in pnames:
        kwargs1["season"] = season
    if "season_type_nullable" in pnames:
        kwargs1["season_type_nullable"] = season_type
    if kwargs1:
        candidates.append(kwargs1)

    kwargs2 = add_date_filters(dict(common_kwargs))
    if "season_nullable" in pnames:
        kwargs2["season_nullable"] = season
    if "season_type_nullable" in pnames:
        kwargs2["season_type_nullable"] = season_type
    if kwargs2:
        candidates.append(kwargs2)

    kwargs3 = add_date_filters(dict(common_kwargs))
    if "season" in pnames:
        kwargs3["season"] = season
    if "season_type_all_star" in pnames:
        kwargs3["season_type_all_star"] = season_type
    if "season_type" in pnames:
        kwargs3["season_type"] = season_type
    if kwargs3:
        candidates.append(kwargs3)

    kwargs4 = add_date_filters(dict(common_kwargs))
    if "season_nullable" in pnames:
        kwargs4["season_nullable"] = season
    if "season_type_all_star" in pnames:
        kwargs4["season_type_all_star"] = season_type
    if kwargs4:
        candidates.append(kwargs4)

    last_err = None
    for kw in candidates:
        try:
            print(f"Trying PlayerGameLogs with kwargs keys: {sorted(kw.keys())}", flush=True)
            print(f"Using date window: {date_from} -> {date_to}", flush=True)
            inst = ctor(**kw)
            dfs = inst.get_data_frames()
            if dfs and len(dfs[0]) > 0:
                return dfs[0]
            if dfs:
                return dfs[0]
        except TypeError as e:
            last_err = e
            continue
        except Exception:
            raise

    raise last_err or RuntimeError("PlayerGameLogs invocation failed without exception.")

def get_logs_with_retry(season, season_type, retries=NBA_RETRIES):
    def _do():
        return _try_playergamelogs_variants(season, season_type)
    return _call_with_retry(_do, retries=retries)

def main():
    print("refresh_players.py started", flush=True)
    print(f"Season={SEASON} | SeasonType={SEASON_TYPE} | Timeout={NBA_TIMEOUT}", flush=True)
    print("Creating Supabase client...", flush=True)

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    print("Supabase client created", flush=True)
    print("Fetching recent player game logs...", flush=True)

    lg = get_logs_with_retry(SEASON, SEASON_TYPE)

    print(f"Fetched {len(lg)} player game log rows from nba_api", flush=True)

    if lg.empty:
        raise RuntimeError("Player game logs came back empty; aborting to avoid wiping valid data.")

    lg["GAME_DATE"] = pd.to_datetime(lg["GAME_DATE"])

    act = {p["id"] for p in players.get_players() if p["is_active"]}
    lg = lg[lg["PLAYER_ID"].isin(act)].copy()
    print(f"Rows after active-player filter: {len(lg)}", flush=True)

    today = pd.Timestamp.now(tz=timezone.utc).tz_localize(None)
    keep_cutoff = today - pd.Timedelta(days=RECENT_DAYS)
    last = lg.groupby("PLAYER_ID")["GAME_DATE"].max()
    keep = set(last[last >= keep_cutoff].index)
    lg = lg[lg["PLAYER_ID"].isin(keep)].copy()
    print(f"Rows after recent-player filter: {len(lg)}", flush=True)

    if lg.empty:
        raise RuntimeError("No recent active player logs remained after filtering; aborting to avoid wiping valid data.")

    now = datetime.now(timezone.utc).isoformat()
    res = []

    grouped = lg.groupby(["PLAYER_ID", "PLAYER_NAME"])
    total_players = len(grouped)
    print(f"Building streak rows for {total_players} players", flush=True)

    for idx, ((pid, pname), g) in enumerate(grouped, start=1):
        if idx % 25 == 0 or idx == total_players:
            print(f"Processed {idx}/{total_players} players", flush=True)

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
                    "sport": "NBA",
                    "entity_type": "player",
                    "player_id": int(pid),
                    "player_name": pname,
                    "team_abbr": team,
                    "stat": stat,
                    "threshold": int(t),
                    "streak_len": int(s),
                    "streak_start": str(g.loc[s - 1, "GAME_DATE"].date()),
                    "last_game": lastd,
                    "season_wins": wins,
                    "season_games": games,
                    "season_win_pct": round((wins / games) * 100, 3) if games else 0.0,
                    "streak_win_pct": 100.0,
                    "last10_hits": h10,
                    "last10_games": g10,
                    "last10_hit_pct": p10,
                    "last5_hits": h5,
                    "last5_games": g5,
                    "last5_hit_pct": p5,
                    "updated_at": now,
                })

    if not res:
        raise RuntimeError("No streak rows were generated; aborting to avoid wiping valid data.")

    df = pd.DataFrame(res)
    rows = df.to_dict("records")
    print(f"Generated {len(rows)} streak rows", flush=True)

    print("Deleting existing NBA player streak rows...", flush=True)
    sb.table("streaks").delete().eq("sport", "NBA").eq("entity_type", "player").execute()
    print("Deleted existing NBA player streak rows", flush=True)

    for i in range(0, len(rows), 500):
        chunk = rows[i:i + 500]
        sb.table("streaks").insert(chunk).execute()
        print(f"Inserted rows {i + 1}-{i + len(chunk)}", flush=True)

    print("Players uploaded:", len(rows), flush=True)

if __name__ == "__main__":
    main()
