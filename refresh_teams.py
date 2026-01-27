import os, numpy as np, pandas as pd
from datetime import datetime
from supabase import create_client
from nba_api.stats.endpoints import leaguegamelog

SEASON="2025-26"; SEASON_TYPE="Regular Season"
PTS=[90,95,100,105,110,115,120,125,130]

def alen(v,t):
  m=v<t; return int(m.argmax()) if m.any() else int(len(v))

def last_ge(v,t,n):
  n=min(n,len(v)); x=v[:n]; h=int((x>=t).sum())
  return h,n, round((h/n)*100,3) if n else 0.0

def last_le(v,t,n):
  n=min(n,len(v)); x=v[:n]; h=int((x<=t).sum())
  return h,n, round((h/n)*100,3) if n else 0.0

def main():
  sb=create_client(os.environ["SUPABASE_URL"],os.environ["SUPABASE_SERVICE_ROLE_KEY"])
  lg=leaguegamelog.LeagueGameLog(season=SEASON,season_type_all_star=SEASON_TYPE,
    player_or_team_abbreviation="T").get_data_frames()[0]
  lg["GAME_DATE"]=pd.to_datetime(lg["GAME_DATE"]); now=datetime.utcnow().isoformat()

  rows=[]
  for (abbr,tid),g in lg.groupby(["TEAM_ABBREVIATION","TEAM_ID"]):
    g=g.sort_values("GAME_DATE",ascending=False).reset_index(drop=True)
    last=str(g.loc[0,"GAME_DATE"].date()); games=len(g)
    wl=g["WL"].astype(str).to_numpy(); s=0
    for x in wl:
      if x=="W": s+=1
      else: break
    wins=int((g["WL"]=="W").sum())
    n10=min(10,len(wl)); n5=min(5,len(wl))
    ml10=int((wl[:n10]=="W").sum()); ml5=int((wl[:n5]=="W").sum())
    rows.append({"sport":"NBA","entity_type":"team","player_id":int(tid),"player_name":abbr,
    "team_abbr":abbr,"stat":"ML","threshold":1,"streak_len":int(s),
    "streak_start":str(g.loc[s-1,"GAME_DATE"].date()) if s>0 else last,"last_game":last,
    "season_wins":wins,"season_games":games,"season_win_pct":round((wins/games)*100,3),
    "streak_win_pct":100.0,"last10_hits":ml10,"last10_games":n10,
    "last10_hit_pct":round((ml10/n10)*100,3) if n10 else 0.0,
    "last5_hits":ml5,"last5_games":n5,
    "last5_hit_pct":round((ml5/n5)*100,3) if n5 else 0.0,"updated_at":now})

    pts=g["PTS"].astype(int).to_numpy()
    for t in PTS:
      so=alen(pts,t)
      if so>=2:
        w=int((pts>=t).sum()); a10,b10,c10=last_ge(pts,t,10); a5,b5,c5=last_ge(pts,t,5)
        rows.append({"sport":"NBA","entity_type":"team","player_id":int(tid),"player_name":abbr,
        "team_abbr":abbr,"stat":"PTS","threshold":int(t),"streak_len":int(so),
        "streak_start":str(g.loc[so-1,"GAME_DATE"].date()),"last_game":last,
        "season_wins":w,"season_games":games,"season_win_pct":round((w/games)*100,3),
        "streak_win_pct":100.0,"last10_hits":a10,"last10_games":b10,"last10_hit_pct":c10,
        "last5_hits":a5,"last5_games":b5,"last5_hit_pct":c5,"updated_at":now})
      su=alen(-pts,-t)  # <= t
      if su>=2:
        w=int((pts<=t).sum()); a10,b10,c10=last_le(pts,t,10); a5,b5,c5=last_le(pts,t,5)
        rows.append({"sport":"NBA","entity_type":"team","player_id":int(tid),"player_name":abbr,
        "team_abbr":abbr,"stat":"PTS_U","threshold":int(t),"streak_len":int(su),
        "streak_start":str(g.loc[su-1,"GAME_DATE"].date()),"last_game":last,
        "season_wins":w,"season_games":games,"season_win_pct":round((w/games)*100,3),
        "streak_win_pct":100.0,"last10_hits":a10,"last10_games":b10,"last10_hit_pct":c10,
        "last5_hits":a5,"last5_games":b5,"last5_hit_pct":c5,"updated_at":now})

  sb.table("streaks").delete().eq("sport","NBA").eq("entity_type","team").execute()
  for i in range(0,len(rows),500): sb.table("streaks").insert(rows[i:i+500]).execute()
  print("Teams uploaded:",len(rows))

if __name__=="__main__": main()
