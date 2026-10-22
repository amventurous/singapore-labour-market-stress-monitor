from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Singapore Sectoral Labour Market Stress Monitor", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent

# ── Shared visual system: change colours here only ───────────────────────────
BG = "#0B111C"
SURFACE = "#172235"
TEXT = "#F2F6FC"
MUTED = "#A9B7C9"
GRID = "#344157"
ALERT = "#F07073"                 # Status only
NO_ALERT = "#8599B1"              # Status only
INDUSTRY_PALETTE = [
    "#D7BCE8", "#64C4D3", "#E7B96C", "#A7D69B",
    "#D6A6A1", "#93B5EA", "#E2D68A", "#B5A5E8",
    "#7FD3B6", "#E7A8CE", "#C4C9E3", "#B9C67C"
]  # categorical only; distinct from the two status colours
CLUSTER_COLORS = {
    "Strong / Active": "#59C7B2",
    "Cooling / Soft": "#C5A5ED",
    "Severe Disruption": "#E5B45E",
}
HEATMAP_SCALE = [
    [0.00, "#24364B"], [0.249, "#24364B"],
    [0.25, "#426582"], [0.499, "#426582"],
    [0.50, "#8273A4"], [0.749, "#8273A4"],
    [0.75, "#B88A60"], [0.999, "#B88A60"],
    [1.0, "#E5BD78"],
]
INDICATORS = {
    "employment_change": ("Employment change", "thousand persons"),
    "job_vacancy_rate": ("Job vacancy rate", "%"),
    "recruitment_rate": ("Recruitment rate", "%"),
    "resignation_rate": ("Resignation rate", "%"),
    "retrench": ("Retrenchments", "persons"),
}
ZCOLS = [f"{c}_zscore" for c in INDICATORS]
CLUSTER_NAMES = {0: "Severe Disruption", 1: "Strong / Active", 2: "Cooling / Soft"}
CLUSTER_ORDER = ["Strong / Active", "Cooling / Soft", "Severe Disruption"]

st.markdown(f"""<style>
.stApp {{background:{BG};color:{TEXT};}}
[data-testid="stSidebar"] {{background:{SURFACE};}}
[data-testid="stMetric"] {{background:{SURFACE};padding:1rem;border-radius:0.8rem;}}
[data-testid="stAlert"] {{background:{SURFACE};color:{TEXT};border:1px solid {GRID};border-radius:0.7rem;}}
div[data-testid="stMarkdownContainer"] p {{line-height:1.55;}}
</style>""", unsafe_allow_html=True)

def info(message):
    st.info(message, icon="ℹ️")

@st.cache_data
def load_data():
    hist = pd.read_csv(ROOT / "historical_labour_data.csv")
    dash = pd.read_csv(ROOT / "dashboard_data.csv")
    for df in (hist, dash):
        df["industry"] = df["industry"].astype(str).str.strip().str.lower()
        df["quarter"] = pd.PeriodIndex(
            df["quarter"].astype(str).str.replace("-", "", regex=False).str.replace(" ", "", regex=False), freq="Q"
        )
    return hist.sort_values(["industry", "quarter"]).reset_index(drop=True), dash

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "labour_stress_model.joblib")

@st.cache_data
def historical_features(hist):
    data = hist.copy()
    for col in INDICATORS:
        group = data.groupby("industry")[col]
        avg = group.transform(lambda s: s.shift(1).rolling(12, min_periods=12).mean())
        sd = group.transform(lambda s: s.shift(1).rolling(12, min_periods=12).std())
        data[col + "_zscore"] = (data[col] - avg) / sd.replace(0, np.nan)
    valid = data[["employment_change_zscore", "job_vacancy_rate_zscore", "recruitment_rate_zscore", "retrench_zscore"]].notna().all(axis=1)
    data["Adverse indicators"] = np.nan
    data.loc[valid, "Adverse indicators"] = (
        (data.loc[valid, "employment_change_zscore"] <= -1).astype(int)
        + (data.loc[valid, "job_vacancy_rate_zscore"] <= -1).astype(int)
        + (data.loc[valid, "recruitment_rate_zscore"] <= -1).astype(int)
        + (data.loc[valid, "retrench_zscore"] >= 1).astype(int)
    )
    data["Industry"] = data["industry"].str.title()
    return data

@st.cache_data
def notebook_clusters(features):
    # Original Capstone_Final.ipynb: five industry-relative 12-quarter z-scores;
    # StandardScaler -> full PCA -> first 3 PCs -> KMeans(k=3, seed=42, n_init=10).
    data = features.dropna(subset=ZCOLS).copy()
    pcs = PCA().fit_transform(StandardScaler().fit_transform(data[ZCOLS]))
    data["PC1"], data["PC2"], data["PC3"] = pcs[:, 0], pcs[:, 1], pcs[:, 2]
    data["Cluster ID"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(pcs[:, :3])
    data["Cluster"] = data["Cluster ID"].map(CLUSTER_NAMES)
    return data

def plot_style(fig, height=380, **kwargs):
    """Apply common chart styling without forwarding duplicate Plotly arguments."""
    fig.update_layout(template="plotly_dark")
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG)
    fig.update_layout(font=dict(color=TEXT, size=12))
    fig.update_layout(hoverlabel=dict(bgcolor=SURFACE, font_color=TEXT))
    fig.update_layout(height=height)
    fig.update_layout(margin=dict(l=12, r=18, t=30, b=35))
    fig.update_layout(legend=dict(font=dict(color=TEXT)))
    for name, value in kwargs.items():
        fig.update_layout(**{name: value})
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    return fig

def empty_notice():
    info("No observations match the selected industries and historical quarter range.")

try:
    history, dashboard = load_data()
    model = load_model()
    feature_names = list(model.feature_names_in_)
    current_q = dashboard["quarter"].max()
    latest = dashboard.loc[dashboard["quarter"] == current_q].copy()
    if latest.empty or latest["industry"].duplicated().any():
        raise ValueError("Latest prediction inputs are missing or duplicated.")
    X = latest[feature_names].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Invalid latest prediction features.")
    latest["Score"] = model.predict_proba(X)[:, 1] * 100
    latest["Flag"] = model.predict(X).astype(int)
    latest["Status"] = latest["Flag"].map({1: "Stress alert", 0: "No stress alert"})
    latest["Industry"] = latest["industry"].str.title()
    latest = latest.sort_values("Score", ascending=False)
    industries = sorted(history["industry"].str.title().unique())
    colors = dict(zip(industries, INDUSTRY_PALETTE))
    historical = historical_features(history)
    clusters = notebook_clusters(historical)
    assert clusters["Cluster ID"].value_counts().sort_index().to_dict() == {0: 42, 1: 268, 2: 386}, "K-Means output differs from original notebook."
except Exception as e:
    st.error(f"Unable to initialise dashboard: {e}")
    st.stop()

# ── One sidebar filter for all tabs ──────────────────────────────────────────
with st.sidebar:
    st.header("Dashboard filters")
    st.caption("Shared across all three tabs")
    for ind in industries:
        if f"industry_{ind}" not in st.session_state:
            st.session_state[f"industry_{ind}"] = True
    a, b = st.columns(2)
    def set_all(value):
        for name in industries:
            st.session_state[f"industry_{name}"] = value
    a.button("Select all", use_container_width=True, on_click=set_all, args=(True,))
    b.button("Clear all", use_container_width=True, on_click=set_all, args=(False,))
    chosen = [ind for ind in industries if st.checkbox(ind, key=f"industry_{ind}")]
    st.caption(f"**{len(chosen)} of {len(industries)} selected**")
    st.divider()
    qs = sorted(history["quarter"].unique())
    start = st.selectbox("From quarter", qs, index=0, format_func=str)
    end = st.selectbox("To quarter", qs, index=len(qs)-1, format_func=str)
    if start > end:
        st.error("From quarter must be before or equal to To quarter.")
    st.caption("Quarter filters apply to historical charts only. Predictions always use the latest available quarter.")

selected = chosen
valid_period = start <= end
def subset(data, time=True):
    mask = data["Industry"].isin(selected)
    if time:
        mask &= data["quarter"].between(start, end) if valid_period else False
    return data.loc[mask].copy()

st.title("Singapore Sectoral Labour Market Stress Monitor")
st.caption("Dashboard UI build: v7 — 23 October 2026")
st.caption(f"Latest source quarter: {current_q}  ·  Next-quarter screening: {current_q+1}  ·  Static MOM data snapshot")

tab1, tab2, tab3 = st.tabs(["① Stress Overview & Predictions", "② Labour Market Indicators", "③ K-Means Clustering"])

with tab1:
    st.subheader(f"Next-quarter stress screening · {current_q+1}")
    view = subset(latest, time=False).sort_values("Score", ascending=False)
    m1, m2, m3 = st.columns(3)
    m1.metric("Selected industries", len(view))
    m2.metric("Stress alerts", int(view["Flag"].sum()))
    m3.metric("No stress alert", int((view["Flag"] == 0).sum()))
    if not view.empty:
        fig = go.Figure()
        for status in ["Stress alert", "No stress alert"]:
            sub = view.loc[view["Status"] == status]
            fig.add_trace(go.Bar(
                x=sub["Score"], y=sub["Industry"], orientation="h",
                name=status, marker_color=ALERT if status == "Stress alert" else NO_ALERT,
                text=[f"{x:.1f}" for x in sub["Score"]], textposition="inside",
                insidetextanchor="end", textfont=dict(color="#FFFFFF", size=12),
                hovertemplate="%{y}<br>Model score: %{x:.1f}<br>" + status + "<extra></extra>",
            ))
        fig.update_layout(barmode="overlay", yaxis=dict(categoryorder="array", categoryarray=list(reversed(view["Industry"].tolist()))))
        fig.update_xaxes(range=[0, 100], title="Model stress score (0–100; uncalibrated)")
        plot_style(fig, height=max(330, 43 * len(view)))
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download latest prediction table", view[["Industry", "Score", "Status"]].to_csv(index=False),
                           file_name=f"predictions_{current_q+1}.csv", mime="text/csv")
    else:
        empty_notice()

    st.divider()
    st.subheader("Historical adverse-indicator heatmap")
    info("Each cell shows the number of adverse indicators (0–4). A white outline and ● mark identify observations meeting the analyst-defined stress criterion (2 or more). This is historical rule-based classification, not a series of model predictions.")
    h = subset(historical).dropna(subset=["Adverse indicators"])
    if not h.empty:
        quarters = sorted(h["quarter"].unique())
        ys = sorted(h["Industry"].unique())
        matrix = h.pivot(index="Industry", columns="quarter", values="Adverse indicators").reindex(index=ys, columns=quarters)
        z = matrix.to_numpy(dtype=float)
        fig = go.Figure(go.Heatmap(
            z=z, x=[str(q) for q in quarters], y=ys,
            colorscale=HEATMAP_SCALE, zmin=0, zmax=4, zsmooth=False,
            colorbar=dict(title="Adverse<br>indicators", tickvals=[0, 1, 2, 3, 4]),
            hovertemplate="%{y}<br>%{x}<br>Adverse indicators: %{z:.0f}<extra></extra>",
            xgap=1, ygap=3,
        ))
        xx, yy = np.where(np.isfinite(z) & (z >= 2))
        if len(xx):
            fig.add_trace(go.Scatter(
                x=[str(quarters[c]) for c in yy], y=[ys[r] for r in xx],
                mode="markers", marker=dict(symbol="circle-open", size=10, color="#FFFFFF", line=dict(width=1.5, color="#FFFFFF")),
                name="Stress criterion met (2+)", hoverinfo="skip",
            ))
        years = [i for i, q in enumerate(quarters) if q.quarter == 1]
        fig.update_xaxes(tickmode="array", tickvals=[str(quarters[i]) for i in years], ticktext=[str(quarters[i].year) for i in years], title="Year")
        fig.update_yaxes(autorange="reversed")
        plot_style(fig, height=max(380, 34 * len(ys) + 120))
        st.plotly_chart(fig, use_container_width=True)
    else:
        empty_notice()

with tab2:
    st.subheader("Historical labour market indicators")
    info("The figures above each chart show the latest available quarter within your selected timeframe, compared with the same quarter one year earlier. Values are shown separately by industry; no cross-industry averaging is performed.")
    frame = subset(historical)
    # One full-width chart per row. The sidebar is the only industry selector.
    for field, (label, unit) in INDICATORS.items():
        st.markdown(f"#### {label}")
        if frame.empty:
            empty_notice()
            continue
        latest_visible = frame["quarter"].max()
        prev_q = latest_visible - 4
        latest_rows = frame.loc[frame["quarter"] == latest_visible].sort_values("Industry")
        previous_rows = history.loc[history["quarter"] == prev_q]
        previous = previous_rows.set_index(previous_rows["industry"].str.title())[field]
        metrics = []
        for _, r in latest_rows.iterrows():
            now = r[field]
            old = previous.get(r["Industry"], np.nan)
            diff = now - old if pd.notna(now) and pd.notna(old) else np.nan
            delta = f"{diff:+,.2f} {unit} YoY" if pd.notna(diff) else "YoY unavailable"
            metrics.append(f"**{r['Industry']}**: {now:,.2f} {unit} · {delta}")
        if len(metrics) == 1:
            st.caption(metrics[0].replace("**", ""))
        elif metrics:
            st.caption(f"{latest_visible}: latest values and year-on-year changes for {len(metrics)} industries")
        fig = go.Figure()
        for ind in sorted(frame["Industry"].unique()):
            part = frame.loc[frame["Industry"] == ind].sort_values("quarter")
            fig.add_trace(go.Scatter(
                x=part["quarter"].dt.to_timestamp(), y=part[field], mode="lines",
                name=ind, line=dict(color=colors[ind], width=2),
                customdata=part["quarter"].astype(str),
                hovertemplate=f"{ind}<br>%{{customdata}}<br>{label}: %{{y:,.2f}} {unit}<extra></extra>",
            ))
        fig.update_xaxes(dtick="M12", tickformat="%Y", title="Year")
        fig.update_yaxes(title=f"{label} ({unit})")
        plot_style(fig, height=700, legend=dict(
            orientation="h", x=0, y=-0.17, xanchor="left", yanchor="top",
            font=dict(size=12, color=TEXT), tracegroupgap=5,
        ), margin=dict(l=75, r=25, t=25, b=190))
        st.plotly_chart(
            fig, use_container_width=True, key=f"indicator_{field}",
            config={"displayModeBar": False, "displaylogo": False}
        )
        st.divider()
    if not frame.empty:
        download = frame[["quarter", "Industry", *INDICATORS]].copy()
        download["quarter"] = download["quarter"].astype(str)
        st.download_button("Download filtered indicator data", download.to_csv(index=False),
                           file_name="filtered_labour_indicators.csv", mime="text/csv")

with tab3:
    st.subheader("Historical labour market states · PCA and K-Means")
    info("Descriptive historical analysis only. Cluster states show recurring patterns in industry-relative indicators; they do not predict next-quarter stress. Reproduced using the original capstone notebook's PCA and K-Means settings.")
    # Reference profiles are deliberately based on the full fitted historical clustering sample.
    profiles = clusters.groupby("Cluster")[ZCOLS].mean().reindex(CLUSTER_ORDER)
    names = {
        "employment_change_zscore": "Employment change (Z-score)",
        "job_vacancy_rate_zscore": "Job vacancy rate (Z-score)",
        "recruitment_rate_zscore": "Recruitment rate (Z-score)",
        "resignation_rate_zscore": "Resignation rate (Z-score)",
        "retrench_zscore": "Retrenchments (Z-score)",
    }
    st.markdown("#### Cluster definitions and average historical indicators")
    st.dataframe(profiles.rename(columns=names).round(2), use_container_width=True)
    st.caption("Averages are industry-relative historical Z-scores (not raw percentages or counts). Names follow the original notebook: Strong / Active, Cooling / Soft, and Severe Disruption.")
    st.markdown("#### Explore industry-quarter classifications")
    cluster_view = subset(clusters)
    if cluster_view.empty:
        empty_notice()
    else:
        left, right = st.columns([1.55, 1])
        with left:
            fig = go.Figure()
            for name in CLUSTER_ORDER:
                part = cluster_view.loc[cluster_view["Cluster"] == name]
                fig.add_trace(go.Scatter(
                    x=part["PC1"], y=part["PC2"], mode="markers", name=name,
                    marker=dict(color=CLUSTER_COLORS[name], size=7, opacity=0.72),
                    customdata=np.column_stack([part["Industry"], part["quarter"].astype(str)]),
                    hovertemplate="Industry: %{customdata[0]}<br>Quarter: %{customdata[1]}<br>State: " + name +
                                  "<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                ))
            fig.update_xaxes(title="Principal component 1")
            fig.update_yaxes(title="Principal component 2")
            plot_style(fig, height=440)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            counts = cluster_view.groupby("Cluster").size().reindex(CLUSTER_ORDER, fill_value=0)
            total = counts.sum()
            fig = go.Figure(go.Bar(
                x=counts.values, y=counts.index, orientation="h",
                marker_color=[CLUSTER_COLORS[n] for n in counts.index],
                text=[f"{v} ({v/total:.1%})" for v in counts.values],
                textposition="inside", textfont=dict(color="#FFFFFF", size=12),
                hovertemplate="%{y}: %{x} observations<extra></extra>",
            ))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title="Industry-quarter observations")
            plot_style(fig, height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{total:,} observations in the selected industries and quarters.")
        st.markdown("#### Industry-quarter observations")
        display_cols = ["Industry", "quarter", "Cluster", "PC1", "PC2", *INDICATORS]
        readable = {
            "quarter": "Quarter", "Cluster": "Labour market state",
            "PC1": "Principal component 1", "PC2": "Principal component 2",
            "employment_change": "Employment change (thousands)",
            "job_vacancy_rate": "Job vacancy rate (%)",
            "recruitment_rate": "Recruitment rate (%)",
            "resignation_rate": "Resignation rate (%)",
            "retrench": "Retrenchments (persons)",
        }
        table = cluster_view.sort_values(["Industry", "quarter"])[display_cols].rename(columns=readable)
        table["Quarter"] = table["Quarter"].astype(str)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button("Download selected cluster members", table.to_csv(index=False),
                           file_name="historical_cluster_members.csv", mime="text/csv")

st.divider()
st.caption("Source: MOM quarterly labour market statistics supplied with the capstone. Static snapshot; new MOM releases are not retrieved automatically.")
