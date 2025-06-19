import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Tuple

# --- Page Config ---
st.set_page_config(page_title="MPG Auction Helper - New Season", layout="wide")
st.title("🚀 MPG Auction Helper (New Season)")

# --- Constants ---
DEFAULT_BUDGET = 500
DEFAULT_SQUAD_SIZE = 20
MINIMA = {'GK':2, 'DEF':6, 'MID':6, 'FWD':4}
FORMATIONS = {
    '4-4-2':{'GK':1,'DEF':4,'MID':4,'FWD':2}, '4-3-3':{'GK':1,'DEF':4,'MID':3,'FWD':3},
    '3-5-2':{'GK':1,'DEF':3,'MID':5,'FWD':2}, '3-4-3':{'GK':1,'DEF':3,'MID':4,'FWD':3},
    '4-5-1':{'GK':1,'DEF':4,'MID':5,'FWD':1}, '5-3-2':{'GK':1,'DEF':5,'MID':3,'FWD':2},
    '5-4-1':{'GK':1,'DEF':5,'MID':4,'FWD':1}
}
PROFILES = {
    'Balanced Value': {
        'n_recent_games': 5, 'min_recent_played': 1,
        'kpi_weights': {
            'GK': {'recent_avg':0.05,'season_avg':0.70,'regularity':0.25,'recent_goals':0.0,'season_goals':0.0},
            'DEF':{'recent_avg':0.25,'season_avg':0.25,'regularity':0.25,'recent_goals':0.0,'season_goals':0.0},
            'MID':{'recent_avg':0.20,'season_avg':0.20,'regularity':0.15,'recent_goals':0.15,'season_goals':0.15},
            'FWD':{'recent_avg':0.15,'season_avg':0.15,'regularity':0.10,'recent_goals':0.25,'season_goals':0.25}
        },
        'mrb_params': {'GK':0.3,'DEF':0.4,'MID':0.6,'FWD':1.0}
    },
    'Attack Focus': {
        'n_recent_games':5,'min_recent_played':1,
        'kpi_weights':{
            'GK':{'recent_avg':0.0,'season_avg':0.50,'regularity':0.50,'recent_goals':0.0,'season_goals':0.0},
            'DEF':{'recent_avg':0.0,'season_avg':0.60,'regularity':0.40,'recent_goals':0.0,'season_goals':0.0},
            'MID':{'recent_avg':0.10,'season_avg':0.20,'regularity':0.10,'recent_goals':0.30,'season_goals':0.30},
            'FWD':{'recent_avg':0.10,'season_avg':0.20,'regularity':0.10,'recent_goals':0.30,'season_goals':0.30}
        },
        'mrb_params':{'GK':0.2,'DEF':0.3,'MID':0.7,'FWD':1.2}
    },
    'Defensive Focus': {
        'n_recent_games':5,'min_recent_played':1,
        'kpi_weights':{
            'GK':{'recent_avg':0.10,'season_avg':0.60,'regularity':0.30,'recent_goals':0.0,'season_goals':0.0},
            'DEF':{'recent_avg':0.30,'season_avg':0.30,'regularity':0.20,'recent_goals':0.0,'season_goals':0.0},
            'MID':{'recent_avg':0.10,'season_avg':0.20,'regularity':0.20,'recent_goals':0.20,'season_goals':0.20},
            'FWD':{'recent_avg':0.0,'season_avg':0.20,'regularity':0.10,'recent_goals':0.30,'season_goals':0.30}
        },
        'mrb_params':{'GK':0.4,'DEF':0.6,'MID':0.5,'FWD':0.8}
    },
    'Custom': None
}

# --- Utility Functions ---
def simplify_position(pos):
    if pd.isna(pos): return 'UNKNOWN'
    p=pos.strip().upper()
    if p=='G': return 'GK'
    if p in ['D','DC','DL','DF']: return 'DEF'
    if p in ['M','MD','MO','MC']: return 'MID'
    if p in ['A','AT','FW']: return 'FWD'
    return 'UNKNOWN'

def create_player_id(r):
    return f"{r['Joueur'].strip()}_{simplify_position(r['Poste'])}_{r['Club'].strip()}"

def extract_rating(val):
    if pd.isna(val) or val in ['0','']:
        return None,0
    s=str(val).strip(); goals=s.count('*')
    clean=re.sub(r"[()*]", "", s)
    try: return float(clean), goals
    except: return None,0

# --- Core Class ---
class MPG:
    @staticmethod
    def compute_historical_kpis(df, cols):
        records=[]
        for _,r in df.iterrows():
            ratings=[]; goals=0; played=0
            for c in cols:
                rt,g=extract_rating(r[c])
                if rt is not None:
                    ratings.append(rt); goals+=g; played+=1
            avg=np.mean(ratings) if ratings else 0
            top5=sorted(ratings, reverse=True)[:5]
            pot=np.mean(top5) if top5 else 0
            reg=(played/len(cols))*100 if cols else 0
            records.append({'player_id':r['player_id'],'est_perf':avg*10,
                            'est_potential':pot*10,'est_regularity':reg,'est_goals':goals})
        return pd.DataFrame(records)

    @staticmethod
    def eval_players(df, profile):
        # recent and season columns
        rec_cols=[c for c in df.columns if c.startswith('D')][:profile['n_recent_games']]
        sea_cols=[c for c in df.columns if c.startswith('D')]
        out=[]
        for _,r in df.iterrows():
            rec_vals=[extract_rating(r[c])[0] for c in rec_cols if extract_rating(r[c])[0]!=None]
            sea_vals=[extract_rating(r[c])[0] for c in sea_cols if extract_rating(r[c])[0]!=None]
            recent_avg=np.mean(rec_vals) if rec_vals else 0
            season_avg=np.mean(sea_vals) if sea_vals else 0
            played=len(sea_vals)
            reg=played/len(sea_cols)*100 if sea_cols else 0
            rec_goals=sum(extract_rating(r[c])[1] for c in rec_cols)
            sea_goals=sum(extract_rating(r[c])[1] for c in sea_cols)
            out.append({**r[['player_id','Joueur','simplified_position','Club','Cote']].to_dict(),
                        'recent_avg':recent_avg,'season_avg':season_avg,
                        'regularity':reg,'recent_goals':rec_goals,'season_goals':sea_goals})
        df2=pd.DataFrame(out)
        # compute PVS & MRB
        df2['pvs']=df2.apply(lambda r: min(
            sum(r[k]*profile['kpi_weights'][r['simplified_position']][k] for k in profile['kpi_weights'][r['simplified_position']]),100), axis=1)
        def mrb(r):
            bonus=profile['mrb_params'][r['simplified_position']]*(r['pvs']/100)
            val=r['Cote']*(1+bonus)
            return int(round(min(max(val, r['Cote']), 2*r['Cote'])))
        df2['mrb']=df2.apply(mrb, axis=1)
        df2['value_per_cost']=df2['pvs']/df2['mrb']
        return df2

    @staticmethod
    def select_squad(df, formation, minima, budget):
        squad=[]; rem=df.copy(); spent=0
        # minima
        for pos,n in minima.items():
            sel=rem[rem['simplified_position']==pos].nlargest(n,'pvs')
            squad.append(sel); rem=rem.drop(sel.index); spent+=sel['mrb'].sum()
        # formation starters
        for pos,n in formation.items():
            have=len(pd.concat(squad)[lambda x:x['simplified_position']==pos])
            add=max(0,n-have)
            sel=rem[rem['simplified_position']==pos].nlargest(add,'pvs')
            squad.append(sel); rem=rem.drop(sel.index); spent+=sel['mrb'].sum()
        # bench
        total_slots=sum(minima.values())
        add=total_slots-len(pd.concat(squad))
        if add>0:
            sel=rem.nlargest(add,'pvs'); squad.append(sel); spent+=sel['mrb'].sum()
        squad_df=pd.concat(squad)
        # budget trim loop
        while spent>budget:
            drop=squad_df.nsmallest(1,'value_per_cost')
            squad_df=squad_df.drop(drop.index); spent=squad_df['mrb'].sum()
            cand=df.drop(squad_df.index)
            pick=cand[cand['mrb']<=budget-spent].nlargest(1,'pvs')
            if not pick.empty:
                squad_df=pd.concat([squad_df,pick]); spent+=pick['mrb'].sum()
            else: break
        summary={'total_cost':spent,'remaining_budget':budget-spent,'total_pvs':squad_df['pvs'].sum()}
        return squad_df, summary

# --- Data Load & Preprocessing ---
st.sidebar.header("📁 Upload Files")
f1=st.sidebar.file_uploader("Last Season File", type=['csv','xls','xlsx'])
f2=st.sidebar.file_uploader("New Season File", type=['csv','xls','xlsx'])
st.sidebar.header("⚙️ Settings")
profile_key=st.sidebar.selectbox("Profile", list(PROFILES.keys()))
formation_key=st.sidebar.selectbox("Formation", list(FORMATIONS.keys()))
budget=st.sidebar.number_input("Budget (€)", value=DEFAULT_BUDGET)

# Club tiers
st.sidebar.header("🏆 Club Tiers")
tier_map={}
if f2:
    df2=pd.read_excel(f2) if f2.name.endswith(('xls','xlsx')) else pd.read_csv(f2)
    df2['Club']=df2['Club'].astype(str)
    clubs=sorted(df2['Club'].unique()); TIERS={'Winner':100,'European':75,'Average':50,'Relegation':25}
    for c in clubs: tier_map[c]=TIERS[st.sidebar.selectbox(c,list(TIERS.keys()),index=2)]

if f1 and f2:
    df_last=pd.read_excel(f1) if f1.name.endswith(('xls','xlsx')) else pd.read_csv(f1)
    df_new=pd.read_excel(f2) if f2.name.endswith(('xls','xlsx')) else pd.read_csv(f2)
    for df in [df_last, df_new]:
        df['simplified_position']=df['Poste'].apply(simplify_position)
        df['player_id']=df.apply(create_player_id,axis=1)
        df['Club']=df['Club'].astype(str)
        df['Cote']=pd.to_numeric(df['Cote'],errors='coerce').fillna(1).astype(int)
    hist_ids=set(df_last['player_id']); new_ids=set(df_new['player_id'])
    known=hist_ids & new_ids; newonly=new_ids - hist_ids
    cols=[c for c in df_last.columns if re.fullmatch(r'D\d+',c)]
    df_kpi=MPG.compute_historical_kpis(df_last[df_last['player_id'].isin(known)],cols)
    maxs=df_kpi[['est_perf','est_potential','est_regularity','est_goals']].max()
    st.subheader("✍️ New Players Manual KPIs")
    manual_list=[]
    for _,r in df_new[df_new['player_id'].isin(newonly)].iterrows():
        st.markdown(f"**{r['Joueur']} ({r['simplified_position']}, {r['Club']})**")
        perf=st.slider("Performance %",0,100,50,key=r['player_id']+"_p")
        pot=st.slider("Potential %",0,100,50,key=r['player_id']+"_pot")
        reg=st.slider("Regularity %",0,100,50,key=r['player_id']+"_r")
        goals=st.slider("Goals %",0,100,0,key=r['player_id']+"_g")
        manual_list.append({'player_id':r['player_id'],'Joueur':r['Joueur'],'simplified_position':r['simplified_position'],
                            'Club':r['Club'],'Cote':r['Cote'],'est_perf':perf/100*maxs['est_perf'],
                            'est_potential':pot/100*maxs['est_potential'],'est_regularity':reg/100*maxs['est_regularity'],
                            'est_goals':goals/100*maxs['est_goals']})
    df_manual=pd.DataFrame(manual_list)
    df_known=df_new[df_new['player_id'].isin(known)][['player_id','Joueur','simplified_position','Club','Cote']]
    df_known=df_known.merge(df_kpi,on='player_id')
    df_eval=pd.concat([df_known,df_manual],ignore_index=True)
    df_eval['team_rank']=df_eval['Club'].map(tier_map)
    # Profile selection
    if profile_key!='Custom': prof=PROFILES[profile_key]
    else:
        st.subheader("🎛️ Custom Profile Settings")
        return
    # Evaluate
    df_evaluated=MPG.eval_players(df_eval, prof)
    st.subheader("📊 Evaluated Players")
    st.dataframe(df_evaluated)
    # Save/Load
    if st.button("💾 Save Eval"): st.download_button("Download JSON",data=df_evaluated.to_json(orient='records'),file_name='eval.json')
    up=st.file_uploader("Load Eval JSON",type=['json'])
    if up: df_evaluated=pd.read_json(up)
    # Squad Selection
    formation=FORMATIONS[information_key]
    squad,summary=MPG.select_squad(df_evaluated,formation,MINIMA,budget)
    st.subheader("🏆 Final Squad")
    st.dataframe(squad)
    st.subheader("📈 Summary")
    st.json(summary)
else:
    st.info("Upload both last and new season files to proceed.")
