import os, numpy as np, pandas as pd
from datetime import datetime, timezone
from supabase import create_client
from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.static import players

SEASON="2025-26"; SEASON_TYPE="Regular Season"
RECENT_DAYS=14; MIN_STREAK=2; LOOKBACK=12
STATS={"PTS":"PTS","AST":"AST","REB":"REB","3PM":"FG3M"}

def alen(v,t):
  m=v<t; return int(m.argmax()) if m.any() else int(len(v))

def last_ge(v,t,n):
  n=min(n,len(v)); x=v[:n]; h=int((x>=t).sum())
  return h,n, round((h/n)*100,3) if n else 0.0

def main():
  sb=create_client(os.environ["SUPABASE_URL"],os.environ["SUPABASE_SERVICE_ROLE_KEY"])
  lg = playergamelogs.PlayerGameLogs(
    season_nullable=SEASON,
    season_type_nullable=SEASON_TYPE
).get_data_frames()[0]

  lg["GAME_DATE"]=pd.to_datetime(lg["GAME_DATE"])
  act={p["id"] for p in players.get_players() if p["is_active"]}
  lg=lg[lg["PLAYER_ID"].isin(act)].copy()
  today=pd.Timestamp.now(tz=timezone.utc).tz_localize(None)
  last=lg.groupby("PLAYER_ID")["GAME_DATE"].max()
  keep=set(last[last>=today-pd.Timedelta(days=RECENT_DAYS)].index)
  lg=lg[lg["PLAYER_ID"].isin(keep)].copy()
  now=datetime.now(timezone.utc).isoformat()

  res=[]
  for (pid,pname),g in lg.groupby(["PLAYER_ID","PLAYER_NAME"]):
    g=g.sort_values("GAME_DATE",ascending=False).reset_index(drop=True)
    team=g.loc[0,"TEAM_ABBREVIATION"]; games=len(g); lastd=str(g.loc[0,"GAME_DATE"].date())
    for stat,col in STATS.items():
      v=pd.to_numeric(g[col],errors="coerce").fillna(0).astype(int).to_numpy()
      mx=int(v[:min(LOOKBACK,len(v))].max())
      for t in range(1,mx+1):
        s=alen(v,t)
        if s<MIN_STREAK: continue
        wins=int((v>=t).sum())
        h10,g10,p10=last_ge(v,t,10); h5,g5,p5=last_ge(v,t,5)
        res.append({"sport":"NBA","entity_type":"player","player_id":int(pid),
        "player_name":pname,"team_abbr":team,"stat":stat,"threshold":int(t),
        "streak_len":int(s),"streak_start":str(g.loc[s-1,"GAME_DATE"].date()),
        "last_game":lastd,"season_wins":wins,"season_games":games,
        "season_win_pct":round((wins/games)*100,3),"streak_win_pct":100.0,
        "last10_hits":h10,"last10_games":g10,"last10_hit_pct":p10,
        "last5_hits":h5,"last5_games":g5,"last5_hit_pct":p5,"updated_at":now})

  df=pd.DataFrame(res); rows=df.to_dict("records")
  sb.table("streaks").delete().eq("sport","NBA").eq("entity_type","player").execute()
  for i in range(0,len(rows),500): sb.table("streaks").insert(rows[i:i+500]).execute()
  print("Players uploaded:",len(rows))

if __name__=="__main__": main()
