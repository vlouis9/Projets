import streamlit as st
import pandas as pd
import numpy as np
import re

# --- Config ---
st.set_page_config(page_title="MPG Auction Helper – New Season", layout="wide")
st.title("🚀 MPG Auction Helper (New Season)")

# --- Constants ---
DEFAULT_BUDGET = 500
DEFAULT_SQUAD_SIZE = 20
MINIMA = {'GK':2, 'DEF':6, 'MID':6, 'FWD':4}
FORMATIONS = {
    '4-4-2':{'GK':1,'DEF':4,'MID':4,'FWD':2},
    '4-3-3':{'GK':1,'DEF':4,'MID':3,'FWD':3},
    '3-5-2':{'GK':1,'DEF':3,'MID':5,'FWD':2},
    '3-4-3':{'GK':1,'DEF':3,'MID':4,'FWD':3},
    '4-5-1':{'GK':1,'DEF':4,'MID':5,'FWD':1},
    '5-3-2':{'GK':1,'DEF':5,'MID':3,'FWD':2},
    '5-4-1':{'GK':1,'DEF':5,'MID':4,'FWD':1}
}
# KPI weights + MRB bonuses per profile
PROFILES = {
    'Balanced Value': {
        'weights': {'est_perf':0.20,'est_potential':0.20,'est_regularity':0.20,'est_goals':0.20,'team_rank':0.20},
        'mrb_bonus':{'GK':0.30,'DEF':0.40,'MID':0.60,'FWD':1.00}
    },
    'Attack Focus': {
        'weights': {'est_perf':0.10,'est_potential':0.30,'est_regularity':0.10,'est_goals':0.30,'team_rank':0.20},
        'mrb_bonus':{'GK':0.20,'DEF':0.30,'MID':0.70,'FWD':1.20}
    },
    'Defensive Focus': {
        'weights': {'est_perf':0.30,'est_potential':0.10,'est_regularity':0.30,'est_goals':0.10,'team_rank':0.20},
        'mrb_bonus':{'GK':0.40,'DEF':0.60,'MID':0.50,'FWD':0.80}
    },
    'Custom': None
}

# --- Helpers ---
def simp_pos(p):
    if pd.isna(p): return 'UNKNOWN'
    u = p.strip().upper()
    if u=='G': return 'GK'
    if u in ('D','DC','DL','DF'): return 'DEF'
    if u in ('M','MD','MO','MC'): return 'MID'
    if u in ('A','AT','FW'): return 'FWD'
    return 'UNKNOWN'

def make_id(r):
    return f"{r['Joueur'].strip()}_{simp_pos(r['Poste'])}_{r['Club'].strip()}"

def extract(val):
    if pd.isna(val) or val in ('','0'): return None,0
    s = str(val).strip()
    goals = s.count('*')
    clean = re.sub(r"[()*]","",s)
    try: return float(clean), goals
    except: return None,0

# --- Load & Prep ---
def load(f):
    df = pd.read_excel(f) if f.name.endswith(('xls','xlsx')) else pd.read_csv(f)
    df['simplified_position'] = df['Poste'].apply(simp_pos)
    df['player_id'] = df.apply(make_id,axis=1)
    df['Cote'] = pd.to_numeric(df['Cote'],errors='coerce').fillna(1).astype(int)
    df['Club'] = df['Club'].astype(str)
    return df

def hist_kpis(df,cols):
    out=[]
    for _,r in df.iterrows():
        rs,gl,pl=[],0,0
        for c in cols:
            v,g = extract(r[c])
            if v is not None:
                rs.append(v); gl+=g; pl+=1
        avg = np.mean(rs) if rs else 0
        pot = np.mean(sorted(rs,reverse=True)[:5]) if rs else 0
        reg = pl/len(cols)*100 if cols else 0
        out.append({'player_id':r['player_id'],'est_perf':avg*10,'est_potential':pot*10,'est_regularity':reg,'est_goals':gl})
    return pd.DataFrame(out)

def compute_pvs(df,weights):
    df['pvs']=df.apply(lambda r: sum(r[k]*weights[k] for k in weights),axis=1)
    return df

def compute_mrb(df,bonus):
    def f(r):
        b=bonus[r['simplified_position']]*r['pvs']/100
        val=r['Cote']*(1+b)
        return int(round(min(max(val,r['Cote']),2*r['Cote'])))
    df['mrb']=df.apply(f,axis=1)
    df['value_per_cost']=df['pvs']/df['mrb']
    return df

def select_squad(df,formation,minima,size,budget):
    rem=df.copy(); squad=[]
    # minima
    for pos,n in minima.items():
        sel=rem[rem['simplified_position']==pos].nlargest(n,'pvs')
        squad.append(sel); rem=rem.drop(sel.index)
    # formation starters
    curr=pd.concat(squad)
    for pos,n in formation.items():
        have=len(curr[curr['simplified_position']==pos])
        add=max(0,n-have)
        sel=rem[rem['simplified_position']==pos].nlargest(add,'pvs')
        squad.append(sel); rem=rem.drop(sel.index)
        curr=pd.concat(squad)
    # bench to size
    if len(curr)<size:
        sel=rem.nlargest(size-len(curr),'pvs')
        squad.append(sel); curr=pd.concat(squad)
    # budget loop
    while curr['mrb'].sum()>budget:
        drop=curr.nsmallest(1,'value_per_cost')
        curr=curr.drop(drop.index)
        cand=df.drop(curr.index)
        pick=cand[cand['mrb']<=budget-curr['mrb'].sum()].nlargest(1,'pvs')
        if pick.empty: break
        curr=pd.concat([curr,pick])
    return curr, {
        'total_cost':int(curr['mrb'].sum()),
        'remaining_budget':int(budget-curr['mrb'].sum()),
        'total_pvs':float(curr['pvs'].sum())
    }


# --- Sidebar Inputs ---
st.sidebar.header("📁 Upload Files")
hist = st.sidebar.file_uploader("Last Season File",type=['csv','xls','xlsx'])
new = st.sidebar.file_uploader("New Season File",type=['csv','xls','xlsx'])
profile = st.sidebar.selectbox("Profile",list(PROFILES.keys()))
formation = st.sidebar.selectbox("Formation",list(FORMATIONS.keys()))
budget = st.sidebar.number_input("Budget (€)",value=DEFAULT_BUDGET)
size   = st.sidebar.number_input("Squad Size",min_value=sum(MINIMA.values()),value=DEFAULT_SQUAD_SIZE)

# Club tiers
st.sidebar.header("🏆 Club Tiers")
tiers = {'Winner':100,'European':75,'Average':50,'Relegation':25}
tier_map={}
if new:
    df2 = load(new)
    for c in sorted(df2['Club'].unique()):
        tier_map[c] = tiers[st.sidebar.selectbox(c,list(tiers.keys()),index=2)]
else:
    tier_map={}

# --- Main Flow ---
if hist and new:
    df_hist = load(hist)
    df_new  = load(new)
    ids_hist = set(df_hist['player_id'])
    ids_new  = set(df_new['player_id'])
    known = ids_hist & ids_new
    newonly = ids_new - ids_hist

    # hist KPIs
    cols = [c for c in df_hist.columns if re.fullmatch(r'D\\d+',c)]
    df_hk = hist_kpis(df_hist[df_hist['player_id'].isin(known)],cols)
    maxs = df_hk[['est_perf','est_potential','est_regularity','est_goals']].max()

    # manual new
    st.subheader("✍️ Manual Input for New Players")
    manual=[]
    for _,r in df_new[df_new['player_id'].isin(newonly)].iterrows():
        st.markdown(f"**{r['Joueur']} ({r['simplified_position']}, {r['Club']})**")
        p = st.slider(f"Perf %",0,100,50,25,key=f"p_{r['player_id']}")
        u = st.slider(f"Pot  %",0,100,50,25,key=f"u_{r['player_id']}")
        g = st.slider(f"Reg  %",0,100,50,25,key=f"g_{r['player_id']}")
        q = st.slider(f"Goals%",0,100,0,25,  key=f"q_{r['player_id']}")
        manual.append({
            'player_id':r['player_id'],'Joueur':r['Joueur'],
            'simplified_position':r['simplified_position'],'Club':r['Club'],'Cote':r['Cote'],
            'est_perf':p/100*maxs['est_perf'],
            'est_potential':u/100*maxs['est_potential'],
            'est_regularity':g/100*maxs['est_regularity'],
            'est_goals':q/100*maxs['est_goals']
        })
    df_manual = pd.DataFrame(manual)

    # merge known
    df_known = df_new[df_new['player_id'].isin(known)][['player_id','Joueur','simplified_position','Club','Cote']]
    df_known = df_known.merge(df_hk,on='player_id')
    df_eval  = pd.concat([df_known,df_manual],ignore_index=True)
    df_eval['team_rank'] = df_eval['Club'].map(tier_map).fillna(50)

    # profile selection
    if profile=='Custom':
        st.subheader("🎛️ Custom Profile")
        weights,bonus = {},{}
        for k in ('est_perf','est_potential','est_regularity','est_goals','team_rank'):
            weights[k] = st.slider(f"{k}",0.0,1.0,0.2,0.05)
        for pos in MINIMA:
            bonus[pos] = st.slider(f"Bonus {pos}",0.0,2.0,1.0,0.1)
    else:
        prof = PROFILES[profile]
        weights=prof['weights']; bonus=prof['mrb_bonus']

    df_eval = compute_pvs(df_eval,weights)
    df_eval = compute_mrb(df_eval,bonus)

    # show eval
    st.subheader("📊 Evaluated Players")
    st.dataframe(df_eval)

    # save/load
    if st.button("💾 Save Eval"):
        st.download_button("Download JSON",data=df_eval.to_json(orient='records'),
                           file_name='eval.json')
    up = st.file_uploader("Load Eval JSON",type=['json'])
    if up:
        df_eval = pd.read_json(up)

    # select squad
    squad,summary = select_squad(df_eval,FORMATIONS[formation],MINIMA,size,budget)
    st.subheader("🏆 Final Squad")
    st.dataframe(squad)
    st.subheader("📈 Summary")
    st.json(summary)

else:
    st.info("Upload both Last Season and New Season files to proceed.")
