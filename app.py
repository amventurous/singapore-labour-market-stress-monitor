import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from html import escape


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Singapore Labour Market Stress Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. VISUAL DESIGN
# ============================================================

NAVY = "#14253D"
BLUE = "#3266A8"
TEAL = "#087F8C"
RED = "#B84A4A"
GREY = "#657487"

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    h1, h2, h3 {
        color: #14253D;
        letter-spacing: -0.02em;
    }

    h1 {
        font-size: 2.1rem !important;
        font-weight: 750 !important;
    }

    div[data-testid="stMetric"] {
        background: #F4F7FB;
        border: 1px solid #E5EBF2;
        border-radius: 12px;
        padding: 16px 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #657487;
    }

    div[data-testid="stMetricValue"] {
        color: #14253D;
        font-weight: 700;
    }

    .hero {
        padding: 26px 30px;
        background: linear-gradient(
            110deg, #14253D, #244D78
        );
        border-radius: 15px;
        color: white;
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 29px;
        font-weight: 750;
        line-height: 1.3;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 15px;
        color: #D8E6F5;
        line-height: 1.65;
        max-width: 950px;
    }

    .eyebrow {
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #AFC9E8;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .insight {
        background: #F4F7FB;
        border-left: 4px solid #3266A8;
        padding: 18px 22px;
        border-radius: 0 10px 10px 0;
        line-height: 1.8;
        margin: 12px 0 20px 0;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. INDICATOR DEFINITIONS
# ============================================================

INDICATORS = [
    "employment_change",
    "job_vacancy_rate",
    "recruitment_rate",
    "resignation_rate",
    "retrench",
]

LABELS = {
    "employment_change": "Employment Change",
    "job_vacancy_rate": "Job Vacancy Rate",
    "recruitment_rate": "Recruitment Rate",
    "resignation_rate": "Resignation Rate",
    "retrench": "Retrenchments",
}

UNITS = {
    "employment_change": "Thousands",
    "job_vacancy_rate": "%",
    "recruitment_rate": "%",
    "resignation_rate": "%",
    "retrench": "Persons",
}

DESCRIPTIONS = {
    "employment_change":
        "Quarterly change in employment. Negative values "
        "indicate a contraction in employment.",

    "job_vacancy_rate":
        "The job vacancy rate indicates the relative "
        "level of unfilled positions.",

    "recruitment_rate":
        "The rate at which employers recruit workers.",

    "resignation_rate":
        "The rate at which employees resign from jobs.",

    "retrench":
        "The number of workers retrenched during "
        "the quarter.",
}

MOM_SOURCES = [
    {
        "indicator": "Employment Change",
        "dataset": "Quarterly Employment Change by Industry",
        "url": (
            "https://stats.mom.gov.sg/Pages/"
            "EmploymentTimeSeries.aspx"
        ),
    },
    {
        "indicator": "Job Vacancy Rate",
        "dataset": (
            "Job Vacancy Rate by Industry "
            "and Occupational Group"
        ),
        "url": (
            "https://stats.mom.gov.sg/Pages/"
            "JobVacancyTimeSeries.aspx"
        ),
    },
    {
        "indicator": "Recruitment Rate",
        "dataset": (
            "Average Monthly Recruitment Rate "
            "by Industry and Occupational Group"
        ),
        "url": (
            "https://stats.mom.gov.sg/Pages/"
            "LabourTurnoverTimeSeries.aspx"
        ),
    },
    {
        "indicator": "Resignation Rate",
        "dataset": (
            "Average Monthly Resignation Rate "
            "by Industry and Occupational Group"
        ),
        "url": (
            "https://stats.mom.gov.sg/Pages/"
            "LabourTurnoverTimeSeries.aspx"
        ),
    },
    {
        "indicator": "Retrenchments",
        "dataset": (
            "Number of Retrenched Employees "
            "by Industry and Occupational Group"
        ),
        "url": (
            "https://stats.mom.gov.sg/Pages/"
            "RetrenchmentTimeSeries.aspx"
        ),
    },
]


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def pretty_industry(name):
    return str(name).title()


def quarter_to_period(series):
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    return pd.PeriodIndex(cleaned, freq="Q")


def quarter_label(quarter):
    return f"{quarter.year} Q{quarter.quarter}"


def format_score(score):
    return f"{score:.1f}%"


def render_insight(text):
    st.markdown(
        '<div class="insight">'
        + escape(text)
        + "</div>",
        unsafe_allow_html=True,
    )


def make_bar_chart(data, height=None):

    source = data.copy()

    if height is None:
        height = max(300, len(source) * 38)

    chart = (
        alt.Chart(source)
        .mark_bar(
            cornerRadiusEnd=4,
            size=20,
        )
        .encode(
            x=alt.X(
                "Stress Score (%):Q",
                title="Model Stress Score (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y(
                "Industry:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=320),
            ),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(
                    domain=["Flagged", "Not Flagged"],
                    range=[RED, TEAL],
                ),
                legend=alt.Legend(
                    title=None,
                    orient="bottom",
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "Industry:N",
                    title="Industry",
                ),
                alt.Tooltip(
                    "Stress Score (%):Q",
                    title="Model Score (%)",
                    format=".1f",
                ),
                alt.Tooltip(
                    "Status:N",
                    title="Model Signal",
                ),
            ],
        )
        .properties(height=height)
    )

    return chart


def show_prediction_table(data):

    display = data[
        ["Industry", "Stress Score (%)", "Status"]
    ].copy()

    display = display.sort_values(
        "Stress Score (%)",
        ascending=False,
    )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Industry": st.column_config.TextColumn(
                "Industry",
                width="large",
            ),
            "Stress Score (%)":
                st.column_config.ProgressColumn(
                    "Model Stress Score",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            "Status": st.column_config.TextColumn(
                "Model Signal",
            ),
        },
    )


def generate_outlook(latest, reporting_text, forecast_text):

    total = len(latest)

    flagged = latest[
        latest["Predicted Stress"] == 1
    ].sort_values(
        "Stress Score (%)",
        ascending=False,
    )

    count = len(flagged)

    if count == 0:
        return (
            f"Based on {reporting_text} labour market "
            f"indicators, the model did not flag any "
            f"of the {total} monitored industries for "
            f"potential stress in {forecast_text}. "
            f"This does not rule out emerging risks "
            f"or sector-specific challenges."
        )

    names = flagged["Industry"].head(3).tolist()

    if len(names) == 1:
        names_text = names[0]

    elif len(names) == 2:
        names_text = " and ".join(names)

    else:
        names_text = (
            ", ".join(names[:-1])
            + ", and "
            + names[-1]
        )

    return (
        f"Based on {reporting_text} labour market "
        f"indicators, the model flagged {count} out "
        f"of {total} industries for potential stress "
        f"in {forecast_text}. "
        f"The flagged industries with the highest "
        f"model scores include {names_text}. "
        f"These signals identify areas that may "
        f"warrant closer monitoring and further "
        f"investigation."
    )


# ============================================================
# 5. LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(
        "labour_stress_model.joblib"
    )


@st.cache_data
def load_data():

    dashboard = pd.read_csv(
        "dashboard_data.csv"
    )

    historical = pd.read_csv(
        "historical_labour_data.csv"
    )

    required = (
        ["quarter", "industry"] + INDICATORS
    )

    missing = [
        column
        for column in required
        if column not in historical.columns
    ]

    if missing:
        raise ValueError(
            "Historical CSV missing columns: "
            + ", ".join(missing)
        )

    for data in [dashboard, historical]:

        data["quarter"] = quarter_to_period(
            data["quarter"]
        )

        data["industry"] = (
            data["industry"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    for indicator in INDICATORS:
        historical[indicator] = pd.to_numeric(
            historical[indicator],
            errors="coerce",
        )

    historical = historical.sort_values(
        ["industry", "quarter"]
    ).copy()

    if historical.duplicated(
        ["industry", "quarter"]
    ).any():
        raise ValueError(
            "Duplicate industry-quarter records "
            "found in historical_labour_data.csv."
        )

    return dashboard, historical


# ============================================================
# 6. GENERATE LATEST FORECAST
# ============================================================

try:

    model = load_model()

    dashboard, historical = load_data()

    features = list(model.feature_names_in_)

    missing_features = [
        column
        for column in features
        if column not in dashboard.columns
    ]

    if missing_features:
        raise ValueError(
            "Dashboard CSV missing model features: "
            + ", ".join(missing_features)
        )

    dashboard[features] = dashboard[
        features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    latest_reporting_quarter = (
        dashboard["quarter"].max()
    )

    latest = dashboard[
        dashboard["quarter"]
        == latest_reporting_quarter
    ].copy()

    if latest.empty:
        raise ValueError(
            "No records found for latest quarter."
        )

    if latest["industry"].duplicated().any():
        raise ValueError(
            "Duplicate industries found in "
            "the latest reporting quarter."
        )

    if not np.isfinite(
        latest[features].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Latest model features contain missing "
            "or invalid values."
        )

    latest["Stress Score (%)"] = (
        model.predict_proba(
            latest[features]
        )[:, 1] * 100
    )

    latest["Predicted Stress"] = (
        model.predict(
            latest[features]
        )
    )

    latest["Status"] = (
        latest["Predicted Stress"]
        .map({
            1: "Flagged",
            0: "Not Flagged",
        })
    )

    latest["Industry"] = (
        latest["industry"].apply(
            pretty_industry
        )
    )

    latest = latest.sort_values(
        "Stress Score (%)",
        ascending=False,
    )

except Exception as error:

    st.error(
        f"Unable to initialise dashboard: {error}"
    )

    st.stop()


# ============================================================
# 7. DASHBOARD VARIABLES
# ============================================================

forecast_quarter = (
    latest_reporting_quarter + 1
)

reporting_text = quarter_label(
    latest_reporting_quarter
)

forecast_text = quarter_label(
    forecast_quarter
)

n_industries = len(latest)

n_flagged = int(
    latest["Predicted Stress"].sum()
)

n_not_flagged = (
    n_industries - n_flagged
)

highest = latest.iloc[0]

flagged = latest[
    latest["Predicted Stress"] == 1
].copy()

industry_options = sorted(
    latest["Industry"].tolist()
)

outlook = generate_outlook(
    latest,
    reporting_text,
    forecast_text,
)


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## Labour Market Monitor"
    )

    st.caption(
        "Quarterly Industry-Level Early Warning"
    )

    st.divider()

    st.markdown("**Current Forecast**")

    st.markdown(
        f"### {forecast_text}"
    )

    st.caption(
        f"Based on data through {reporting_text}"
    )

    st.divider()

    st.markdown("**Dashboard Coverage**")

    st.write(
        f"{n_industries} industries monitored"
    )

    st.write(
        "5 labour market indicators"
    )

    st.write(
        "Balanced logistic regression"
    )

    st.divider()

    st.markdown("**Dashboard Guide**")

    st.caption(
        "1. Review the quarterly outlook.\n\n"
        "2. Compare industries.\n\n"
        "3. Explore historical indicators.\n\n"
        "4. Review methodology and data sources."
    )

    st.divider()

    st.markdown("**Official Data Portal**")

    st.link_button(
        "Visit MOM Statistics ↗",
        (
            "https://stats.mom.gov.sg/"
            "Statistics/Pages/Statistics.aspx"
        ),
    )

    st.caption(
        "Exploratory capstone project. "
        "Not an official MOM forecasting system."
    )


# ============================================================
# 9. DASHBOARD HEADER
# ============================================================

st.markdown(
    f"""
    <div class="hero">

        <div class="eyebrow">
            Labour Market Intelligence
        </div>

        <div class="hero-title">
            Singapore Labour Market Stress Monitor
        </div>

        <div class="hero-subtitle">

            An industry-level early-warning dashboard
            translating quarterly labour market
            indicators into accessible signals
            for preliminary monitoring and
            policy analysis.

            <br><br>

            <b>Reporting Period:</b>
            {reporting_text}

            &nbsp;&nbsp; | &nbsp;&nbsp;

            <b>Forecast Period:</b>
            {forecast_text}

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "This dashboard consolidates five key labour "
    "market indicators published by Singapore's "
    "Ministry of Manpower (MOM) to identify industries "
    "that may warrant closer attention in the "
    "subsequent quarter."
)

st.caption(
    "Predictions are exploratory early-warning "
    "signals based on an analyst-defined stress "
    "classification. They are not official MOM "
    "forecasts or calibrated probabilities."
)

st.divider()


# ============================================================
# 10. DASHBOARD TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Quarterly Overview",
    "Industry Comparison",
    "Labour Market Indicators",
    "Methodology & Limitations",
    "Data Sources & References",
])


# ============================================================
# TAB 1 — QUARTERLY OVERVIEW
# ============================================================

with tab1:

    st.header(
        f"Quarterly Outlook | {forecast_text}"
    )

    st.caption(
        f"Predictions based on labour market "
        f"data through {reporting_text}."
    )

    # HEADLINE METRICS

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Industries Monitored",
        n_industries,
    )

    c2.metric(
        "Industries Flagged",
        n_flagged,
    )

    c3.metric(
        "Industries Not Flagged",
        n_not_flagged,
    )

    c4.metric(
        "Highest Model Score",
        format_score(
            highest["Stress Score (%)"]
        ),
    )

    st.write("")

    # QUARTERLY NARRATIVE

    st.subheader(
        "Quarterly Labour Market Outlook"
    )

    render_insight(outlook)

    st.caption(
        "The narrative is generated automatically "
        "from the latest model predictions. "
        "It does not establish the causes of "
        "labour market developments."
    )

    # INDUSTRY RANKING

    st.subheader(
        "Industry-Level Stress Scores"
    )

    st.write(
        "Industries are ranked according to "
        "their model-generated stress scores."
    )

    st.altair_chart(
        make_bar_chart(latest),
        use_container_width=True,
    )

    st.caption(
        "Red indicates an industry flagged "
        "by the model; teal indicates an "
        "industry not flagged. Scores are "
        "not calibrated risk probabilities."
    )

    # FLAGGED INDUSTRIES

    st.subheader(
        "Industries Warranting Closer Monitoring"
    )

    if flagged.empty:

        st.success(
            "No industries were flagged "
            "for this forecast quarter."
        )

    else:

        show_prediction_table(flagged)

    st.divider()

    st.subheader(
        "How to Interpret These Results"
    )

    st.write(
        "A flagged industry should be treated "
        "as a prompt for further investigation, "
        "rather than confirmation of actual "
        "labour market distress. The model "
        "is designed for exploratory screening "
        "and can generate false alarms."
    )


# ============================================================
# TAB 2 — INDUSTRY COMPARISON
# ============================================================

with tab2:

    st.header("Industry Comparison")

    st.write(
        "Select and compare model-generated "
        "stress signals across industries."
    )

    selected_industries = st.multiselect(
        "Select Industries",
        options=industry_options,
        default=industry_options,
        key="comparison_industries",
    )

    if not selected_industries:

        st.info(
            "Select at least one industry "
            "to begin comparing."
        )

    else:

        comparison = latest[
            latest["Industry"].isin(
                selected_industries
            )
        ].copy()

        comparison = comparison.sort_values(
            "Stress Score (%)",
            ascending=False,
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Industries Selected",
            len(comparison),
        )

        c2.metric(
            "Flagged",
            int(
                comparison[
                    "Predicted Stress"
                ].sum()
            ),
        )

        c3.metric(
            "Not Flagged",
            int(
                (
                    comparison[
                        "Predicted Stress"
                    ] == 0
                ).sum()
            ),
        )

        st.subheader(
            "Comparative Stress Scores"
        )

        st.altair_chart(
            make_bar_chart(comparison),
            use_container_width=True,
        )

        st.subheader(
            "Industry Prediction Details"
        )

        show_prediction_table(comparison)

        export_data = comparison[
            [
                "Industry",
                "Stress Score (%)",
                "Status",
            ]
        ].copy()

        st.download_button(
            "Download Comparison (CSV)",
            data=export_data.to_csv(
                index=False
            ),
            file_name=(
                "labour_stress_comparison_"
                + str(forecast_quarter)
                + ".csv"
            ),
            mime="text/csv",
        )

        st.caption(
            "Model scores should be used "
            "for relative screening, not "
            "as definitive measurements "
            "of actual industry risk."
        )


# ============================================================
# TAB 3 — LABOUR MARKET INDICATORS
# ============================================================

with tab3:

    st.header(
        "Labour Market Indicators"
    )

    st.write(
        "Explore historical MOM labour market "
        "indicators to understand the context "
        "behind industry-level model signals."
    )

    # INDICATOR SELECTION

    indicator = st.selectbox(
        "Select Labour Market Indicator",
        INDICATORS,
        format_func=lambda x: LABELS[x],
    )

    # INDUSTRY SELECTION

    historical_industries = sorted(
        historical["industry"].unique()
    )

    selected_history_industries = st.multiselect(
        "Select Industries",
        options=historical_industries,
        default=historical_industries[:3],
        format_func=pretty_industry,
        key="history_industries",
    )

    st.info(
        DESCRIPTIONS[indicator]
    )

    if not selected_history_industries:

        st.warning(
            "Select at least one industry "
            "to view historical trends."
        )

    else:

        history_filtered = historical[
            historical["industry"].isin(
                selected_history_industries
            )
        ].copy()

        available_quarters = sorted(
            history_filtered[
                "quarter"
            ].unique()
        )

        if not available_quarters:

            st.warning(
                "No historical data available "
                "for this selection."
            )

        else:

            earliest = available_quarters[0]
            latest_history = available_quarters[-1]

            default_start = max(
                earliest,
                latest_history - 19,
            )

            quarter_options = [
                str(q)
                for q in available_quarters
            ]

            default_index = next(
                (
                    i
                    for i, q in enumerate(
                        available_quarters
                    )
                    if q >= default_start
                ),
                0,
            )

            start_quarter = st.selectbox(
                "Start Quarter",
                options=quarter_options,
                index=default_index,
            )

            start_period = pd.Period(
                start_quarter,
                freq="Q",
            )

            history_filtered = history_filtered[
                history_filtered["quarter"]
                >= start_period
            ].copy()

            history_filtered["Industry"] = (
                history_filtered["industry"]
                .apply(pretty_industry)
            )

            history_filtered["Quarter"] = (
                history_filtered["quarter"]
                .astype(str)
            )

            history_filtered = history_filtered[
                history_filtered[
                    indicator
                ].notna()
            ].copy()

            st.subheader(
                f"{LABELS[indicator]} Over Time"
            )

            st.caption(
                f"Unit: {UNITS[indicator]}"
            )

            if history_filtered.empty:

                st.info(
                    "No observations available "
                    "for this selection."
                )

            else:

                quarter_order = sorted(
                    history_filtered[
                        "Quarter"
                    ].unique()
                )

                trend = (
                    alt.Chart(
                        history_filtered
                    )
                    .mark_line(
                        point=True,
                        strokeWidth=2.5,
                    )
                    .encode(
                        x=alt.X(
                            "Quarter:O",
                            sort=quarter_order,
                            title="Reporting Quarter",
                            axis=alt.Axis(
                                labelAngle=-45
                            ),
                        ),
                        y=alt.Y(
                            f"{indicator}:Q",
                            title=(
                                f"{LABELS[indicator]} "
                                f"({UNITS[indicator]})"
                            ),
                            scale=alt.Scale(
                                zero=False
                            ),
                        ),
                        color=alt.Color(
                            "Industry:N",
                            title="Industry",
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "Industry:N"
                            ),
                            alt.Tooltip(
                                "Quarter:O"
                            ),
                            alt.Tooltip(
                                f"{indicator}:Q",
                                title=LABELS[indicator],
                                format=".2f",
                            ),
                        ],
                    )
                    .properties(height=420)
                    .interactive()
                )

                st.altair_chart(
                    trend,
                    use_container_width=True,
                )

                # LATEST OBSERVATIONS

                st.subheader(
                    "Latest Available Observations"
                )

                latest_observations = (
                    history_filtered
                    .sort_values("quarter")
                    .groupby(
                        "industry",
                        as_index=False,
                    )
                    .tail(1)
                )

                latest_observations = (
                    latest_observations[
                        [
                            "Industry",
                            "Quarter",
                            indicator,
                        ]
                    ]
                    .rename(
                        columns={
                            indicator:
                                LABELS[indicator]
                        }
                    )
                    .sort_values("Industry")
                )

                st.dataframe(
                    latest_observations.round(2),
                    hide_index=True,
                    use_container_width=True,
                )

                # DOWNLOAD INDICATOR DATA

                download_data = history_filtered[
                    [
                        "Industry",
                        "Quarter",
                        indicator,
                    ]
                ].copy()

                st.download_button(
                    "Download Indicator Data",
                    data=download_data.to_csv(
                        index=False
                    ),
                    file_name=(
                        f"{indicator}_history.csv"
                    ),
                    mime="text/csv",
                )

    st.divider()

    # INDICATOR DEFINITIONS

    st.subheader(
        "Understanding the Five Indicators"
    )

    with st.expander(
        "View Indicator Definitions"
    ):

        for key in INDICATORS:

            st.markdown(
                f"**{LABELS[key]}**"
            )

            st.write(
                DESCRIPTIONS[key]
            )

            st.caption(
                f"Unit: {UNITS[key]}"
            )


# ============================================================
# TAB 4 — METHODOLOGY AND LIMITATIONS
# ============================================================

with tab4:

    st.header(
        "Methodology & Limitations"
    )

    st.write(
        "This dashboard presents predictions "
        "from a supervised machine-learning "
        "model developed as part of an "
        "exploratory labour market stress "
        "analysis."
    )

    # DATA

    st.subheader(
        "Data and Indicators"
    )

    st.write(
        "The analysis uses historical "
        "quarterly industry-level data "
        "published by Singapore's "
        "Ministry of Manpower."
    )

    for indicator in INDICATORS:

        st.markdown(
            f"- {LABELS[indicator]}"
        )

    # FEATURE ENGINEERING

    st.subheader(
        "Feature Engineering"
    )

    st.write(
        "For each industry, the model uses "
        "quarter-on-quarter changes and "
        "rolling Z-scores calculated "
        "relative to the previous "
        "12 quarters."
    )

    st.write(
        "These features capture short-term "
        "movements and deviations from "
        "recent industry-specific patterns."
    )

    # STRESS LABEL

    st.subheader(
        "Definition of Labour Market Stress"
    )

    st.write(
        "The target variable is an "
        "analyst-defined stress classification. "
        "An industry-quarter is labelled "
        "stressed when at least two "
        "of four adverse conditions are met."
    )

    st.markdown(
        """
        - Employment-change Z-score ≤ −1
        - Job-vacancy-rate Z-score ≤ −1
        - Recruitment-rate Z-score ≤ −1
        - Retrenchment Z-score ≥ +1
        """
    )

    st.write(
        "The resignation rate is included "
        "as a predictive feature but is "
        "not used to define the stress label."
    )

    # MODEL

    st.subheader(
        "Predictive Model"
    )

    st.write(
        "The dashboard uses a balanced "
        "logistic regression model to predict "
        "whether an industry will meet the "
        "analyst-defined stress criteria "
        "in the following quarter."
    )

    st.write(
        "The model was evaluated using "
        "a chronological train-test split, "
        "rather than randomly mixing "
        "historical and future observations."
    )

    # MODEL PERFORMANCE

    st.subheader(
        "Model Evaluation"
    )

    st.caption(
        "Held-out test results from "
        "the capstone model evaluation."
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Precision",
        "0.487",
    )

    m2.metric(
        "Recall",
        "0.934",
    )

    m3.metric(
        "F1 Score",
        "0.640",
    )

    st.write(
        "The model achieved high recall, "
        "identifying most stressed "
        "industry-quarters in the test set. "
        "However, its lower precision "
        "indicates a substantial number "
        "of false alarms."
    )

    # LIMITATIONS

    st.subheader(
        "Key Limitations"
    )

    with st.expander(
        "1. Analyst-defined Stress Labels",
        expanded=True,
    ):

        st.write(
            "The stress classification was "
            "developed for this project. "
            "It is not an official MOM "
            "definition of labour market stress."
        )

    with st.expander(
        "2. False Positives"
    ):

        st.write(
            "The model prioritises sensitivity "
            "and may flag industries that do "
            "not subsequently meet the "
            "defined stress criteria."
        )

    with st.expander(
        "3. Uncalibrated Model Scores"
    ):

        st.write(
            "Model scores should not be "
            "interpreted as calibrated "
            "probabilities of actual "
            "labour market distress."
        )

    with st.expander(
        "4. Limited Explanatory Power"
    ):

        st.write(
            "The model identifies statistical "
            "patterns but does not establish "
            "the causes of labour market "
            "developments."
        )

    with st.expander(
        "5. Static Reporting Snapshot"
    ):

        st.write(
            "The dashboard reflects the "
            "latest dataset processed "
            "for this project. It does "
            "not automatically retrieve "
            "newly published MOM data."
        )

    with st.expander(
        "6. Not an Operational Forecasting System"
    ):

        st.write(
            "The results are intended for "
            "exploratory screening and "
            "further investigation, rather "
            "than standalone policy decisions."
        )

    st.divider()

    st.subheader(
        "Intended Use"
    )

    st.write(
        "The dashboard supports preliminary "
        "labour market monitoring, "
        "cross-industry comparisons and "
        "communication of model-generated "
        "early-warning signals."
    )


# ============================================================
# TAB 5 — DATA SOURCES & REFERENCES
# ============================================================

with tab5:

    st.header(
        "Data Sources & References"
    )

    st.write(
        "This dashboard uses publicly "
        "available quarterly labour market "
        "statistics published by Singapore's "
        "Ministry of Manpower (MOM). "
        "The five indicators were cleaned, "
        "standardised and consolidated "
        "for industry-level analysis."
    )

    st.subheader(
        "Official MOM Data Sources"
    )

    st.caption(
        "Select any link below to access "
        "the corresponding MOM statistical "
        "table and published datasets."
    )

    # OFFICIAL DATASET LINKS

    for source in MOM_SOURCES:

        with st.container(border=True):

            c1, c2 = st.columns(
                [4, 1],
                vertical_alignment="center",
            )

            with c1:

                st.markdown(
                    f"**{source['indicator']}**"
                )

                st.caption(
                    source["dataset"]
                )

            with c2:

                st.link_button(
                    "View MOM Data ↗",
                    source["url"],
                    use_container_width=True,
                )

    st.divider()

    # DATA PREPARATION

    st.subheader(
        "How the Data Was Prepared"
    )

    st.write(
        "The original MOM datasets were "
        "processed using Python in Google "
        "Colab to produce a consolidated "
        "industry-quarter dataset."
    )

    st.markdown(
        """
        The data preparation process involved:

        1. Extracting quarterly industry-level observations.
        2. Standardising industry classifications.
        3. Cleaning and merging the five indicators.
        4. Calculating quarter-on-quarter changes.
        5. Computing rolling 12-quarter Z-scores.
        6. Preparing features for next-quarter prediction.
        """
    )

    # DATA COVERAGE

    st.subheader(
        "Dataset Coverage"
    )

    historical_start = (
        historical["quarter"].min()
    )

    historical_end = (
        historical["quarter"].max()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Historical Data Starts",
        quarter_label(historical_start),
    )

    c2.metric(
        "Latest Model Input",
        reporting_text,
    )

    c3.metric(
        "Forecast Quarter",
        forecast_text,
    )

    st.write(
        f"**Industries monitored:** "
        f"{n_industries}"
    )

    st.write(
        "**Data source:** "
        "Ministry of Manpower, Singapore"
    )

    st.info(
        "The dashboard uses a prepared "
        "historical dataset and does not "
        "automatically retrieve new MOM "
        "releases. The official MOM source "
        "pages may contain more recent "
        "statistics than those processed "
        "for this capstone project."
    )

    st.divider()

    # REFERENCE PORTAL

    st.subheader(
        "Additional Official References"
    )

    st.link_button(
        "MOM Labour Market Statistics Portal ↗",
        (
            "https://stats.mom.gov.sg/"
            "Statistics/Pages/Statistics.aspx"
        ),
    )

    st.subheader(
        "Source Attribution"
    )

    st.caption(
        "Source: Ministry of Manpower, "
        "Singapore, Manpower Research and "
        "Statistics Department. Labour market "
        "statistical tables. Data have been "
        "processed and analysed for an "
        "independent capstone project."
    )


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.caption(
    f"Singapore Labour Market Stress Monitor | "
    f"Reporting Period: {reporting_text} | "
    f"Forecast Period: {forecast_text}"
)

st.caption(
    "Source: MOM quarterly labour market "
    "indicators. Exploratory machine-learning "
    "project. Not an official government forecast."
)
