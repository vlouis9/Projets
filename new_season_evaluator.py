import streamlit as st 
import pandas as pd 
import numpy as np 
import re 
from typing import Dict, List, Tuple, Optional

#--- Page Setup ---

st.set_page_config(page_title="🚀 MPG Auction Helper", layout="wide") 
st.markdown("""
<style>
    .header {text-align:center; font-size:2rem; color:#004080; margin-bottom:1rem;}
    .subheader {font-size:1.2rem; color:#006847; margin-top:1rem; margin-bottom:0.5rem;}
</style>""", unsafe_allow_html=True)

#--- Predefined Profiles (from season_with_data.txt) ---

PROFILES = { 'Balanced Value': { 'n_recent_games': 5, 'min_recent_played': 1, 'kpi_weights': { 'GK':{'recent_avg':0.05,'season_avg':0.70,'regularity':0.25,'recent_goals':0,'season_goals':0}, 'DEF':{'recent_avg':0.25,'season_avg':0.25,'regularity':0.25,'recent_goals':0,'season_goals':0}, 'MID':{'recent_avg':0.20,'season_avg':0.20,'regularity':0.15,'recent_goals':0.15,'season_goals':0.15}, 'FWD':{'recent_avg':0.15,'season_avg':0.15,'regularity':0.10,'recent_goals':0.25,'season_goals':0.25} }, 'mrb_params': { 'GK':0.3,'DEF':0.4,'MID':0.6,'FWD':1.0 } }, 'Attack Focus': { 'n_recent_games': 5,'min_recent_played':1, 'kpi_weights': { 'GK':{'recent_avg':0,'season_avg':0.5,'regularity':0.5,'recent_goals':0,'season_goals':0}, 'DEF':{'recent_avg':0,'season_avg':0.6,'regularity':0.4,'recent_goals':0,'season_goals':0}, 'MID':{'recent_avg':0.1,'season_avg':0.2,'regularity':0.1,'recent_goals':0.3,'season_goals':0.3}, 'FWD':{'recent_avg':0.1,'season_avg':0.2,'regularity':0.1,'recent_goals':0.3,'season_goals':0.3} }, 'mrb_params': { 'GK':0.2,'DEF':0.3,'MID':0.7,'FWD':1.2 } }, 'Defensive Focus': { 'n_recent_games': 5,'min_recent_played':1, 'kpi_weights': { 'GK':{'recent_avg':0.1,'season_avg':0.6,'regularity':0.3,'recent_goals':0,'season_goals':0}, 'DEF':{'recent_avg':0.3,'season_avg':0.3,'regularity':0.2,'recent_goals':0,'season_goals':0}, 'MID':{'recent_avg':0.1,'season_avg':0.2,'regularity':0.2,'recent_goals':0.2,'season_goals':0.2}, 'FWD':{'recent_avg':0,'season_avg':0.2,'regularity':0.1,'recent_goals':0.3,'season_goals':0.3} }, 'mrb_params': { 'GK':0.4,'DEF':0.6,'MID':0.5,'FWD':0.8 } }, 'Custom': None }

#--- Helper class with complete methods ---

class MPG: @staticmethod def simplify_position(pos:str)->str: if pd.isna(pos): return 'UNKNOWN' p=pos.strip().upper() return ('GK' if p=='G' else 'DEF' if p in ['D','DC','DL','DF'] else 'MID' if p in ['M','MD','MO','MC'] else 'FWD' if p in ['A','AT','FW'] else 'UNKNOWN')

@staticmethod
def create_id(r)->str:
    return f"{r['Joueur'].strip()}_{MPG.simplify_position(r['Poste'])}_{r['Club'].strip()}"

@staticmethod
def extract(val:str)->Tuple[Optional[float],int]:
    if pd.isna(val) or val in ['0','']:
        return None,0
    s=val.strip(); goals=s.count('*')
    clean=re.sub(r"[()*]","",s)
    try: return float(clean),goals
    except: return None,0

@staticmethod
def eval_players(df,p) -> pd.DataFrame:
    # df: processed; p: profile dict
    # compute recent & season stats
    recent_cols=[c for c in df.columns if c.startswith('D')][:p['n_recent_games']]
    season_cols=[c for c in df.columns if c.startswith('D')]
    rec=[]; sea=[]; reg=[]; recgoals=[]; seagoals=[]
    for _,r in df.iterrows():
        r_vals=[MPG.extract(r[c])[0] for c in recent_cols if MPG.extract(r[c])[0] is not None]
        s_vals=[MPG.extract(r[c])[0] for c in season_cols if MPG.extract(r[c])[0] is not None]
        rec.append(np.mean(r_vals) if r_vals else 0)
        sea.append(np.mean(s_vals) if s_vals else 0)
        played=len([x for x in s_vals if x is not None])
        reg.append(played/len(season_cols)*100 if season_cols else 0)
        recgoals.append(sum(MPG.extract(r[c])[1] for c in recent_cols))
        seagoals.append(sum(MPG.extract(r[c])[1] for c in season_cols))
    df2=df[['player_id','Joueur','simplified_position','Club','Cote']].copy()
    df2['recent_avg']=rec; df2['season_avg']=sea; df2['regularity']=reg
    df2['recent_goals']=recgoals; df2['season_goals']=seagoals
    # PVS
    weights=p['kpi_weights']
    def compute_row(r):
        w=weights[r['simplified_position']]
        base=(r['recent_avg']*w['recent_avg']+r['season_avg']*w['season_avg']+
              r['regularity']*w['regularity']+r['recent_goals']*w.get('recent_goals',0)+
              r['season_goals']*w.get('season_goals',0))
        return min(base,100)
    df2['pvs']=df2.apply(compute_row,axis=1)
    # MRB
    params=p['mrb_params']
    def mrb_row(r):
        bonus=params[r['simplified_position']]*(r['pvs']/100)
        val=r['Cote']*(1+bonus)
        return int(round(min(max(val,r['Cote']),2*r['Cote'])))
    df2['mrb']=df2.apply(mrb_row,axis=1)
    df2['value_per_cost']=df2['pvs']/df2['mrb']
    return df2

@staticmethod
def select_squad(df,pformation:Dict[str,int],minima:Dict[str,int],budget:int)->Tuple[pd.DataFrame,Dict]:
    # ensure minima per pos, then greedy fill by highest pvs, then cost trim loops
    squad=[]; remaining=df.copy(); spent=0
    # pick minima
    for pos,n in minima.items():
        sel=remaining[remaining['simplified_position']==pos].nlargest(n,'pvs')
        squad.append(sel); remaining=remaining.drop(sel.index); spent+=sel['mrb'].sum()
    # fill to formation starters
    for pos,n in pformation.items():
        want=n; have=len(pd.concat(squad)[lambda x:x['simplified_position']==pos])
        add=max(0,want-have)
        sel=remaining[remaining['simplified_position']==pos].nlargest(add,'pvs')
        squad.append(sel); remaining=remaining.drop(sel.index); spent+=sel['mrb'].sum()
    # fill bench to target squad size
    target=sum(minima.values())
    add=target-len(pd.concat(squad))
    if add>0:
        sel=remaining.nlargest(add,'pvs'); squad.append(sel); spent+=sel['mrb'].sum()
    squad_df=pd.concat(squad).sort_values(['simplified_position','pvs'],ascending=[True,False])
    # cost trim if over budget
    while spent>budget:
        drop=squad_df.nsmallest(1,'value_per_cost')
        squad_df=squad_df.drop(drop.index)
        spent=squad_df['mrb'].sum()
        # replace with next best under budget
        candidates=df.drop(squad_df.index)
        pick=candidates[candidates['mrb'] <= budget-spent].nlargest(1,'pvs')
        if not pick.empty:
            squad_df=pd.concat([squad_df,pick]); spent+=pick['mrb'].sum()
        else:
            break
    summary={'total_cost':spent,'remaining_budget':budget-spent,'total_pvs':squad_df['pvs'].sum()}
    return squad_df,summary

#--- App ---

def main(): st.markdown('<div class="header">🚀 MPG Auction Helper</div>', unsafe_allow_html=True) uploaded=st.sidebar.file_uploader("Upload season data (CSV/XLSX)", type=['csv','xls','xlsx']) prof=st.sidebar.selectbox("Profile", list(PROFILES.keys())) if prof!='Custom': params=PROFILES[prof] else: st.sidebar.markdown("Select Custom parameters... (not shown)") return formation=st.sidebar.selectbox("Formation", ['4-4-2','4-3-3','3-5-2','3-4-3','4-5-1','5-3-2','5-4-1']) minima={'GK':2,'DEF':6,'MID':6,'FWD':4} squad_size=st.sidebar.number_input("Squad Size", min_value=sum(minima.values()), value=20) budget=st.sidebar.number_input("Budget", value=500)

if uploaded:
    df=pd.read_excel(uploaded) if uploaded.name.endswith(('xls','xlsx')) else pd.read_csv(uploaded)
    df['simplified_position']=df['Poste'].apply(MPG.simplify_position)
    df['player_id']=df.apply(MPG.create_id,axis=1)
    df['Cote']=pd.to_numeric(df['Cote'],errors='coerce').fillna(1).astype(int)
    df_eval=MPG.eval_players(df,params)
    squad,summary=MPG.select_squad(df_eval, {k:v for k,v in [formation.split('-')]}, minima, budget)
    st.subheader("Suggested Squad")
    st.dataframe(squad[['Joueur','simplified_position','pvs','mrb']])
    st.subheader("Summary")
    st.write(summary)
else:
    st.info("Upload data to proceed.")

if name=='main': main()

