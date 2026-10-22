from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title='Singapore Sectoral Labour Market Stress Monitor', page_icon='📊', layout='wide')
ROOT = Path(__file__).resolve().parent
INDICATORS = {'employment_change': ('Employment Change', 'Thousands'), 'job_vacancy_rate': ('Job Vacancy Rate', '%'), 'recruitment_rate': ('Recruitment Rate', '%'), 'resignation_rate': ('Resignation Rate', '%'), 'retrench': ('Retrenchments', 'Persons')}
PALETTE = ['#0D9488','#F59E0B','#DC2626']

@st.cache_data
def load_data():
    h = pd.read_csv(ROOT/'historical_labour_data.csv')
    d = pd.read_csv(ROOT/'dashboard_data.csv')
    for frame in (h,d):
        frame['industry'] = frame['industry'].astype(str).str.strip().str.lower()
        frame['quarter'] = pd.PeriodIndex(frame['quarter'].astype(str).str.replace('-', '', regex=False).str.replace(' ', '', regex=False),freq='Q')
    h = h.sort_values(['industry','quarter']).reset_index(drop=True)
    return h,d

@st.cache_resource
def load_model():
    return joblib.load(ROOT/'labour_stress_model.joblib')

@st.cache_data
def build_clusters(history):
    # Reconstructed clustering, NOT the original capstone's saved cluster assignments.
    # Complete-case observations, five raw indicators standardized across industry-quarters.
    raw = history.dropna(subset=list(INDICATORS)).copy()
    x = StandardScaler().fit_transform(raw[list(INDICATORS)])
    xy = PCA(n_components=2,random_state=42).fit_transform(x)
    labels = KMeans(n_clusters=3,random_state=42,n_init=20).fit_predict(x)
    raw['Cluster ID'] = labels.astype(int)
    raw['PC1'],raw['PC2'] = xy[:,0],xy[:,1]
    raw['Industry'] = raw['industry'].str.title()
    raw['Quarter'] = raw['quarter'].astype(str)
    # Descriptive neutral names avoid misrepresenting recreated clusters as original states.
    raw['Cluster'] = raw['Cluster ID'].map(lambda v: f'Cluster {v+1}')
    return raw

def industry_filter(label, options, key, defaults=None):
    options = sorted(options)
    if defaults is None: defaults=options
    a,b = st.columns([4,1])
    with a:
        chosen=st.multiselect(label, options, default=defaults, key=key)
    with b:
        st.caption(f'{len(chosen)} of {len(options)} selected')
    return sorted(chosen)

def quarter_filter(data, key):
    qs=sorted(data['quarter'].unique())
    a,b=st.columns(2)
    with a: start=st.selectbox('From quarter',qs,index=0,format_func=str,key=f'{key}_from')
    with b: end=st.selectbox('To quarter',qs,index=len(qs)-1,format_func=str,key=f'{key}_to')
    if start>end:
        st.warning('The starting quarter must be earlier than or equal to the ending quarter.')
        return None,None
    return start,end

def line_plot(frame, value, unit, title):
    if frame.empty:
        st.info('No data for the selected industries and timeframe.')
        return
    base=alt.Chart(frame.assign(Quarter=frame['quarter'].astype(str))).mark_line(point=False,strokeWidth=2).encode(
        x=alt.X('Quarter:N',sort=sorted(frame['quarter'].astype(str).unique()),title='Quarter',axis=alt.Axis(labelAngle=-50,labelOverlap=True)),
        y=alt.Y(f'{value}:Q',title=f'{title} ({unit})'),
        color=alt.Color('Industry:N',legend=alt.Legend(title='Industry',orient='bottom')),
        tooltip=['Industry:N','Quarter:N',alt.Tooltip(f'{value}:Q',format=',.2f')]
    ).properties(height=310)
    st.altair_chart(base,use_container_width=True)

try:
    history,dashboard=load_data()
    model=load_model()
    features=list(model.feature_names_in_)
    current_q=dashboard['quarter'].max()
    latest=dashboard.loc[dashboard['quarter']==current_q].copy()
    if latest['industry'].duplicated().any() or latest.empty:
        raise ValueError('Latest prediction input has missing or duplicate industries.')
    if any(f not in latest.columns for f in features):
        raise ValueError('Missing trained-model features in dashboard_data.csv.')
    x=latest[features].apply(pd.to_numeric,errors='coerce')
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError('Prediction inputs contain invalid values.')
    latest['Score']=model.predict_proba(x)[:,1]*100
    latest['Flag']=model.predict(x).astype(int)
    latest['Status']=latest['Flag'].map({1:'Stress alert',0:'No stress alert'})
    latest['Industry']=latest['industry'].str.title()
    latest=latest.sort_values('Score',ascending=False)
    industries=sorted(history['industry'].str.title().unique())
except Exception as exc:
    st.error(f'Unable to initialise dashboard: {exc}')
    st.stop()

st.title('Singapore Sectoral Labour Market Stress Monitor')
st.caption(f'Latest source quarter: {current_q}  •  Next-quarter screening: {current_q+1}  •  Static MOM data snapshot')
st.info('Model stress scores are **uncalibrated classification scores**, not probabilities of actual economic distress. Alerts are exploratory screening signals.')
tab1,tab2,tab3=st.tabs(['① Stress Overview & Predictions','② Labour Market Indicators','③ K-Means Clustering'])

with tab1:
    st.subheader(f'{current_q+1} Predicted Labour Market Stress by Industry')
    a,b,c=st.columns(3)
    a.metric('Industries monitored',len(latest))
    b.metric('Flagged for next-quarter stress',int(latest['Flag'].sum()))
    c.metric('Not flagged',int((latest['Flag']==0).sum()))
    selected=industry_filter('Industries to display (A–Z)',industries,'overview_industries')
    view=latest.loc[latest['Industry'].isin(selected)].sort_values('Score',ascending=False)
    if not view.empty:
        order=view['Industry'].tolist()
        bars=alt.Chart(view).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y('Industry:N',sort=order,title=None,axis=alt.Axis(labelLimit=370)),
            x=alt.X('Score:Q',scale=alt.Scale(domain=[0,105]),title='Model stress score (%)'),
            color=alt.Color('Status:N',scale=alt.Scale(domain=['Stress alert','No stress alert'],range=['#DC2626','#0D9488']),legend=alt.Legend(orient='bottom')),
            tooltip=['Industry:N','Status:N',alt.Tooltip('Score:Q',format='.1f',title='Model stress score (%)')]
        )
        labels=alt.Chart(view).mark_text(align='left',dx=5,fontSize=12).encode(y=alt.Y('Industry:N',sort=order),x='Score:Q',text=alt.Text('Score:Q',format='.1f'))
        st.altair_chart((bars+labels).properties(height=max(300,len(view)*37)),use_container_width=True)
        st.caption('Bars are sorted from highest to lowest model stress score; labels show scores on a 0–100 scale (not calibrated risk probabilities).')
        st.dataframe(view[['Industry','Score','Status']].rename(columns={'Score':'Model stress score (%)'}).style.format({'Model stress score (%)':'{:.1f}'}),hide_index=True,use_container_width=True)
        st.download_button('Download latest prediction table',view[['Industry','Score','Status']].to_csv(index=False),file_name=f'predictions_{current_q+1}.csv',mime='text/csv')
    else: st.info('Select at least one industry.')

    st.divider()
    st.subheader('Historical Stress Signals')
    st.caption('Historical stress is derived from the same analyst-defined rule (at least two adverse rolling 12-quarter Z-scores). It is **not** a historical next-quarter model-score series.')
    hist_ind=industry_filter('Industries for historical trend (A–Z)',industries,'overview_history_industries',defaults=industries[:3])
    start,end=quarter_filter(history,'overview_history')
    # For each industry use the previous 12 quarters to standardise each indicator.
    zcols={}
    for col in ['employment_change','job_vacancy_rate','recruitment_rate','retrench']:
        grp=history.groupby('industry')[col]
        means=grp.transform(lambda s:s.shift(1).rolling(12,min_periods=12).mean())
        stds=grp.transform(lambda s:s.shift(1).rolling(12,min_periods=12).std())
        zcols[col]=(history[col]-means)/stds.replace(0,np.nan)
    z=pd.DataFrame(zcols)
    adverse=((z['employment_change']<=-1).astype(int)+(z['job_vacancy_rate']<=-1).astype(int)+(z['recruitment_rate']<=-1).astype(int)+(z['retrench']>=1).astype(int))
    hist=history[['quarter','industry']].copy()
    hist['Industry']=hist['industry'].str.title()
    hist['Adverse indicators']=adverse.where(z.notna().all(axis=1),np.nan)
    hist['Stress flag']=np.where(hist['Adverse indicators'].notna(),(hist['Adverse indicators']>=2).astype(int),np.nan)
    if start is not None:
        filtered=hist.loc[hist['Industry'].isin(hist_ind)&hist['quarter'].between(start,end)].dropna(subset=['Adverse indicators'])
        line_plot(filtered,'Adverse indicators','count (0–4)','Adverse indicators')
        st.caption('Stress criterion: 2 or more adverse indicators. Missing early quarters are excluded until a full 12-quarter lookback exists.')

with tab2:
    st.subheader('Historical Labour Market Indicators')
    st.caption('Choose any combination of industries and a timeframe. The same filters apply to all five charts.')
    selected=industry_filter('Industries (A–Z)',industries,'indicators_industries',defaults=industries[:3])
    start,end=quarter_filter(history,'indicators')
    if start is not None:
        frame=history.loc[history['industry'].str.title().isin(selected)&history['quarter'].between(start,end)].copy()
        frame['Industry']=frame['industry'].str.title()
        for field,(label,unit) in INDICATORS.items():
            st.markdown(f'#### {label}')
            line_plot(frame,field,unit,label)
        if not frame.empty:
            downloadable=frame[['quarter','Industry',*INDICATORS]].copy()
            downloadable['quarter']=downloadable['quarter'].astype(str)
            st.download_button('Download filtered indicator data',downloadable.to_csv(index=False),file_name='filtered_labour_indicators.csv',mime='text/csv')

with tab3:
    st.subheader('K-Means Clustering of Industry-Quarter Observations')
    st.markdown("### Understanding the Labour Market Clusters")
    st.write(
        "K-Means clustering groups industry-quarter observations "
        "based on similarities across five labour market indicators. "
        "The three labour market states identified in the original "
        "capstone analysis are described below."
    )

    cluster_definitions = pd.DataFrame({
        "Labour Market State": [
            "Strong / Active",
            "Cooling / Soft",
            "Severe Disruption"
        ],
        "Definition": [
            "Relatively stronger employment growth, job vacancies "
            "and recruitment activity.",
            "Weaker employment growth, vacancies and recruitment "
            "activity, without pronounced disruption.",
            "Pronounced adverse labour market movements, potentially "
            "including employment contraction and elevated retrenchments."
        ],
        "Interpretation": [
            "More active labour market with stronger hiring demand.",
            "Moderating labour demand or softer labour market conditions.",
            "Substantial disruption that may warrant closer examination."
        ]
    })

    st.dataframe(
        cluster_definitions,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "**How to interpret the charts:** Each dot represents one "
        "industry in one quarter. Dots with the same colour belong "
        "to the same reconstructed cluster. PCA reduces the five "
        "indicators to two dimensions for visualisation. "
        "The pie chart shows the number and percentage of "
        "industry-quarter observations in each cluster."
    )

    st.caption(
        "Note: The three named labour market states describe the "
        "original capstone findings. The current dashboard uses "
        "reconstructed clustering results, so Cluster 1, 2 and 3 "
        "have not yet been mapped to those states. "
        "These are not official MOM classifications."
    )

    st.divider()
