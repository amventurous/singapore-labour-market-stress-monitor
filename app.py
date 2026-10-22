
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from pathlib import Path

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Singapore Labour Market Stress Monitor",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# INDICATORS AND OFFICIAL DATA SOURCES
# ============================================================

INDICATORS = {
    "employment_change": ("Employment Change", "Thousands"),
    "job_vacancy_rate": ("Job Vacancy Rate", "%"),
    "recruitment_rate": ("Recruitment Rate", "%"),
    "resignation_rate": ("Resignation Rate", "%"),
    "retrench": ("Retrenchments", "Persons"),
}

SOURCES = [
    (
        "Employment Change",
        "Quarterly Employment Change by Industry",
        "https://stats.mom.gov.sg/Pages/EmploymentTimeSeries.aspx"
    ),
    (
        "Job Vacancy Rate",
        "Job Vacancy Rate by Industry and Occupational Group",
        "https://stats.mom.gov.sg/Pages/JobVacancyTimeSeries.aspx"
    ),
    (
        "Recruitment Rate",
        "Average Monthly Recruitment Rate by Industry",
        "https://stats.mom.gov.sg/Pages/LabourTurnoverTimeSeries.aspx"
    ),
    (
        "Resignation Rate",
        "Average Monthly Resignation Rate by Industry",
        "https://stats.mom.gov.sg/Pages/LabourTurnoverTimeSeries.aspx"
    ),
    (
        "Retrenchments",
        "Number of Retrenched Employees by Industry",
        "https://stats.mom.gov.sg/Pages/RetrenchmentTimeSeries.aspx"
    ),
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def convert_quarters(values):
    cleaned = (
        values.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"[\s-]", "", regex=True)
    )
    return pd.PeriodIndex(cleaned, freq="Q")


def quarter_label(quarter):
    return f"{quarter.year} Q{quarter.quarter}"


def industry_label(name):
    return str(name).title()


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).resolve().parent / "labour_stress_model.joblib")


@st.cache_data
def load_data():
    dashboard = pd.read_csv(Path(__file__).resolve().parent / "dashboard_data.csv")
    history = pd.read_csv(Path(__file__).resolve().parent / "historical_labour_data.csv")

    for data in [dashboard, history]:
        data["quarter"] = convert_quarters(data["quarter"])
        data["industry"] = (
            data["industry"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    missing = [
        col for col in INDICATORS
        if col not in history.columns
    ]

    if missing:
        raise ValueError(
            "Missing historical columns: "
            + ", ".join(missing)
        )

    if history.duplicated(
        ["industry", "quarter"]
    ).any():
        raise ValueError(
            "Duplicate industry-quarter records found."
        )

    for indicator in INDICATORS:
        history[indicator] = pd.to_numeric(
            history[indicator],
            errors="coerce"
        )

    return dashboard, history


# ============================================================
# CHART AND TABLE FUNCTIONS
# ============================================================

def stress_chart(data):
    data = data.sort_values(
        "Stress Score (%)",
        ascending=False
    )

    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "Stress Score (%):Q",
                title="Model Stress Score (%)",
                scale=alt.Scale(domain=[0, 100])
            ),
            y=alt.Y(
                "Industry:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=340)
            ),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(
                    domain=["Flagged", "Not Flagged"],
                    range=["#BA4A4A", "#087F8C"]
                ),
                legend=alt.Legend(
                    title=None,
                    orient="bottom"
                )
            ),
            tooltip=[
                "Industry:N",
                alt.Tooltip(
                    "Stress Score (%):Q",
                    format=".1f"
                ),
                "Status:N"
            ]
        )
        .properties(
            height=max(280, len(data) * 38)
        )
    )

    return chart


def prediction_table(data):
    display = data[
        ["Industry", "Stress Score (%)", "Status"]
    ].copy()

    display = display.sort_values(
        "Stress Score (%)",
        ascending=False
    )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Industry": st.column_config.TextColumn(
                "Industry",
                width="large"
            ),
            "Stress Score (%)":
                st.column_config.ProgressColumn(
                    "Model Stress Score",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%"
                ),
            "Status": st.column_config.TextColumn(
                "Model Signal"
            )
        }
    )


# ============================================================
# INITIALISE MODEL AND PREDICTIONS
# ============================================================

try:
    model = load_model()
    dashboard, history = load_data()

    if not hasattr(model, "predict_proba"):
        raise ValueError("The exported model must support predict_proba().")

    features = list(model.feature_names_in_)

    missing_features = [
        col for col in features
        if col not in dashboard.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            + ", ".join(missing_features)
        )

    reporting_quarter = dashboard["quarter"].max()
    if reporting_quarter != history["quarter"].max():
        raise ValueError("Dashboard inputs and historical data have different latest quarters.")

    latest = dashboard[
        dashboard["quarter"] == reporting_quarter
    ].copy()

    if latest.empty:
        raise ValueError(
            "No records found for latest reporting quarter."
        )

    if latest["industry"].duplicated().any():
        raise ValueError(
            "Duplicate industries in latest reporting quarter."
        )

    X = latest[features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if not np.isfinite(
        X.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Model features contain missing or invalid values."
        )

    latest["Stress Score (%)"] = (
        model.predict_proba(X)[:, 1] * 100
    )

    latest["Predicted Stress"] = (
        model.predict(X).astype(int)
    )

    latest["Status"] = (
        latest["Predicted Stress"].map({
            1: "Flagged",
            0: "Not Flagged"
        })
    )

    latest["Industry"] = (
        latest["industry"].apply(industry_label)
    )

    latest = latest.sort_values(
        "Stress Score (%)",
        ascending=False
    )

except Exception as error:
    st.error(f"Unable to load dashboard: {error}")
    st.stop()


forecast_quarter = reporting_quarter + 1

reporting_text = quarter_label(reporting_quarter)
forecast_text = quarter_label(forecast_quarter)

total_industries = len(latest)

flagged = latest[
    latest["Predicted Stress"] == 1
].copy()

flagged_count = len(flagged)
not_flagged_count = total_industries - flagged_count

industry_options = sorted(
    latest["Industry"].tolist()
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Labour Market Monitor")

    st.caption("Quarterly Industry-Level Early Warning")

    st.divider()

    st.subheader(forecast_text)

    st.caption(
        f"Based on labour market data through {reporting_text}"
    )

    st.metric(
        "Industries Monitored",
        total_industries
    )

    st.metric(
        "Industries Flagged",
        flagged_count
    )

    st.divider()

    st.markdown("**Dashboard Coverage**")

    st.write("5 labour market indicators")
    st.write("Balanced logistic regression")
    st.write("Quarterly industry-level predictions")

    st.divider()

    st.markdown("**Official Data Portal**")

    st.link_button(
        "Visit MOM Statistics ↗",
        "https://stats.mom.gov.sg/Statistics/Pages/Statistics.aspx"
    )

    st.caption(
        "Independent capstone project. "
        "Not an official MOM forecasting system."
    )


# ============================================================
# DASHBOARD HEADER — NATIVE STREAMLIT
# ============================================================

st.caption(
    "LABOUR MARKET INTELLIGENCE  |  "
    "QUARTERLY EARLY-WARNING DASHBOARD"
)

st.title("Singapore Labour Market Stress Monitor")

st.write(
    "An industry-level early-warning dashboard that "
    "translates quarterly labour market indicators "
    "into accessible signals for preliminary "
    "monitoring and policy analysis."
)

header1, header2 = st.columns(2)

with header1:
    st.info(
        f"**Reporting Period:** {reporting_text}"
    )

with header2:
    st.info(
        f"**Forecast Period:** {forecast_text}"
    )

st.caption(
    "Predictions are exploratory model signals, "
    "not official MOM forecasts or calibrated "
    "probabilities of actual labour market stress."
)

st.divider()


# ============================================================
# DASHBOARD TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Quarterly Overview",
    "Industry Comparison",
    "Labour Market Indicators",
    "Methodology & Limitations",
    "Data Sources & References"
])


# ============================================================
# TAB 1 — QUARTERLY OVERVIEW
# ============================================================

with tab1:

    st.header(f"Quarterly Outlook | {forecast_text}")

    st.caption(
        f"Predictions based on {reporting_text} "
        "labour market indicators."
    )

    # HEADLINE METRICS

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Industries Monitored",
        total_industries
    )

    c2.metric(
        "Industries Flagged",
        flagged_count
    )

    c3.metric(
        "Not Flagged",
        not_flagged_count
    )

    c4.metric(
        "Highest Model Score",
        f"{latest['Stress Score (%)'].max():.1f}%"
    )

    st.write("")

    # AUTOMATIC QUARTERLY NARRATIVE

    st.subheader("Quarterly Labour Market Outlook")

    if flagged.empty:

        narrative = (
            f"Based on {reporting_text} labour market "
            f"indicators, the model did not flag any "
            f"of the {total_industries} industries "
            f"for potential stress in {forecast_text}. "
            "This does not rule out emerging risks "
            "or sector-specific challenges."
        )

    else:

        top_names = (
            flagged["Industry"]
            .head(3)
            .tolist()
        )

        if len(top_names) == 1:
            names_text = top_names[0]

        elif len(top_names) == 2:
            names_text = " and ".join(top_names)

        else:
            names_text = (
                ", ".join(top_names[:-1])
                + ", and "
                + top_names[-1]
            )

        narrative = (
            f"Based on {reporting_text} labour market "
            f"indicators, the model flagged "
            f"{flagged_count} out of {total_industries} "
            f"industries for potential stress in "
            f"{forecast_text}. "
            f"Flagged industries with the highest "
            f"model scores include {names_text}. "
            "These signals identify areas that may "
            "warrant closer monitoring and further "
            "investigation."
        )

    st.info(narrative)

    st.caption(
        "This summary is generated automatically "
        "from the latest model predictions. "
        "It does not establish the causes of "
        "labour market developments."
    )

    # INDUSTRY RANKING

    st.subheader("Industry-Level Stress Scores")

    st.write(
        "Industries are ranked by their "
        "model-generated stress scores."
    )

    st.altair_chart(
        stress_chart(latest),
        use_container_width=True
    )

    st.caption(
        "Red: Flagged by the model. "
        "Teal: Not flagged. "
        "Scores are uncalibrated model outputs, "
        "not real-world risk probabilities."
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

        prediction_table(flagged)

    st.divider()

    st.subheader("How to Interpret These Results")

    st.write(
        "A flagged industry should be treated as "
        "a prompt for further investigation, "
        "rather than confirmation of actual "
        "labour market distress. The model is "
        "designed for exploratory screening "
        "and may generate false alarms."
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
        default=industry_options
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
            ascending=False
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Industries Selected",
            len(comparison)
        )

        c2.metric(
            "Flagged",
            int(
                comparison[
                    "Predicted Stress"
                ].sum()
            )
        )

        c3.metric(
            "Not Flagged",
            int(
                (
                    comparison[
                        "Predicted Stress"
                    ] == 0
                ).sum()
            )
        )

        st.subheader("Comparative Stress Scores")

        st.altair_chart(
            stress_chart(comparison),
            use_container_width=True
        )

        st.subheader("Industry Prediction Details")

        prediction_table(comparison)

        export_data = comparison[
            [
                "Industry",
                "Stress Score (%)",
                "Status"
            ]
        ].copy()

        st.download_button(
            "Download Comparison CSV",
            data=export_data.to_csv(index=False),
            file_name=(
                f"labour_stress_comparison_"
                f"{forecast_quarter}.csv"
            ),
            mime="text/csv"
        )

        st.caption(
            "Differences in model scores should "
            "not be interpreted as equivalent "
            "differences in real-world risk."
        )


# ============================================================
# TAB 3 — LABOUR MARKET INDICATORS
# ============================================================

with tab3:

    st.header("Labour Market Indicators")

    st.write(
        "Explore historical MOM labour market "
        "indicators to understand the context "
        "behind industry-level model signals."
    )

    indicator = st.selectbox(
        "Select Labour Market Indicator",
        options=list(INDICATORS.keys()),
        format_func=lambda x: INDICATORS[x][0]
    )

    historical_industries = sorted(
        history["industry"].unique()
    )

    selected_history_industries = st.multiselect(
        "Select Industries",
        options=historical_industries,
        default=historical_industries[:3],
        format_func=industry_label
    )

    if not selected_history_industries:

        st.warning(
            "Select at least one industry "
            "to view historical trends."
        )

    else:

        filtered = history[
            history["industry"].isin(
                selected_history_industries
            )
        ].copy()

        available_quarters = sorted(
            filtered["quarter"].dropna().unique()
        )

        if not available_quarters:

            st.info(
                "No historical data available."
            )

        else:

            earliest = available_quarters[0]
            latest_history = available_quarters[-1]

            default_start = max(
                earliest,
                latest_history - 19
            )

            quarter_options = [
                str(q) for q in available_quarters
            ]

            default_index = next(
                (
                    i for i, q in enumerate(
                        available_quarters
                    )
                    if q >= default_start
                ),
                0
            )

            start_quarter = st.selectbox(
                "Start Quarter",
                options=quarter_options,
                index=default_index
            )

            start_period = pd.Period(
                start_quarter,
                freq="Q"
            )

            filtered = filtered[
                filtered["quarter"] >= start_period
            ].copy()

            filtered = filtered[
                filtered[indicator].notna()
            ].copy()

            filtered["Industry"] = (
                filtered["industry"].apply(
                    industry_label
                )
            )

            filtered["Quarter"] = (
                filtered["quarter"].astype(str)
            )

            indicator_name, indicator_unit = (
                INDICATORS[indicator]
            )

            st.subheader(
                f"{indicator_name} Over Time"
            )

            st.caption(
                f"Unit: {indicator_unit}"
            )

            if filtered.empty:

                st.info(
                    "No observations available "
                    "for this selection."
                )

            else:

                quarter_order = sorted(
                    filtered["Quarter"].unique()
                )

                trend = (
                    alt.Chart(filtered)
                    .mark_line(
                        point=True,
                        strokeWidth=2.5
                    )
                    .encode(
                        x=alt.X(
                            "Quarter:O",
                            sort=quarter_order,
                            title="Reporting Quarter",
                            axis=alt.Axis(
                                labelAngle=-45
                            )
                        ),
                        y=alt.Y(
                            f"{indicator}:Q",
                            title=(
                                f"{indicator_name} "
                                f"({indicator_unit})"
                            ),
                            scale=alt.Scale(zero=False)
                        ),
                        color=alt.Color(
                            "Industry:N",
                            title="Industry"
                        ),
                        tooltip=[
                            "Industry:N",
                            "Quarter:O",
                            alt.Tooltip(
                                f"{indicator}:Q",
                                format=".2f"
                            )
                        ]
                    )
                    .properties(height=420)
                )

                st.altair_chart(
                    trend,
                    use_container_width=True
                )

                st.subheader(
                    "Latest Available Observations"
                )

                latest_observations = (
                    filtered
                    .sort_values("quarter")
                    .groupby("industry")
                    .tail(1)
                )

                display = latest_observations[
                    [
                        "Industry",
                        "Quarter",
                        indicator
                    ]
                ].copy()

                display = display.rename(
                    columns={
                        indicator: indicator_name
                    }
                )

                st.dataframe(
                    display.round(2),
                    hide_index=True,
                    use_container_width=True
                )

                download_data = filtered[
                    [
                        "Industry",
                        "Quarter",
                        indicator
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
                    mime="text/csv"
                )

    st.divider()

    with st.expander("Understanding the Five Indicators"):

        st.markdown(
            "**Employment Change:** Quarterly "
            "employment movement, in thousands."
        )

        st.markdown(
            "**Job Vacancy Rate:** Indicator "
            "of unfilled jobs, expressed as a percentage."
        )

        st.markdown(
            "**Recruitment Rate:** Rate of "
            "employee recruitment."
        )

        st.markdown(
            "**Resignation Rate:** Rate of "
            "employee resignation."
        )

        st.markdown(
            "**Retrenchments:** Number of "
            "employees retrenched."
        )


# ============================================================
# TAB 4 — METHODOLOGY AND LIMITATIONS
# ============================================================

with tab4:

    st.header("Methodology & Limitations")

    st.subheader("Project Objective")

    st.write(
        "To predict whether an industry will meet "
        "the project's analyst-defined labour market "
        "stress criteria in the next quarter."
    )

    st.subheader("Data and Features")

    st.write(
        "The model uses five MOM indicators: "
        "employment change, job vacancy rate, "
        "recruitment rate, resignation rate "
        "and retrenchments."
    )

    st.write(
        "For each indicator, quarter-on-quarter "
        "changes and rolling Z-scores relative "
        "to the previous 12 quarters were "
        "calculated separately for each industry."
    )

    st.subheader("Stress Definition")

    st.write(
        "An industry-quarter is classified as "
        "stressed when at least two of the "
        "following four conditions are met:"
    )

    st.markdown(
        """
        - Employment-change Z-score ≤ −1
        - Job-vacancy-rate Z-score ≤ −1
        - Recruitment-rate Z-score ≤ −1
        - Retrenchment Z-score ≥ +1
        """
    )

    st.caption(
        "Resignation rate is used as a predictive feature but is not part "
        "of the stress label. The model uses ten features: the historical "
        "Z-score and quarter-on-quarter change for each of the five indicators."
    )

    st.subheader("Predictive Model")

    st.write(
        "Balanced logistic regression was selected based on the highest "
        "mean F1-score across four expanding-window validation folds "
        "(41.1%, versus 39.9% for the persistence baseline). "
        "It was subsequently evaluated on a chronological held-out "
        "test period (target quarters 2023Q1–2026Q2), then retrained "
        "on all labelled observations for the next-quarter forecast."
    )

    st.subheader("Model Evaluation")

    c1, c2, c3 = st.columns(3)

    c1.metric("Test Precision", "48.7%")
    c2.metric("Test Recall", "93.4%")
    c3.metric("Test F1 Score", "64.0%")

    st.write(
        "The model achieved high recall, meaning "
        "it identified most stressed industry-quarters "
        "in the test set. However, its modest precision "
        "indicates a substantial number of false alarms."
    )

    st.caption("Held-out balanced accuracy: 68.7%. Evaluation figures are from the final notebook, not calculated from the current forecast.")

    st.subheader("Key Limitations")

    with st.expander(
        "1. Analyst-Defined Stress Classification",
        expanded=True
    ):
        st.write(
            "The stress definition was developed "
            "for this project and is not an "
            "official MOM classification."
        )

    with st.expander("2. False Positives"):
        st.write(
            "The model prioritises sensitivity "
            "and may flag industries that do not "
            "subsequently meet the defined criteria."
        )

    with st.expander("3. Uncalibrated Model Scores"):
        st.write(
            "Model scores are not calibrated "
            "probabilities of actual labour "
            "market distress."
        )

    with st.expander("4. Limited Explanatory Power"):
        st.write(
            "The model identifies statistical "
            "patterns but does not establish "
            "the causes of labour market changes."
        )

    with st.expander("5. Static Data Snapshot"):
        st.write(
            "The dashboard uses prepared datasets "
            "and does not automatically retrieve "
            "new MOM releases."
        )

    with st.expander("6. Not an Operational System"):
        st.write(
            "The results are intended for "
            "exploratory screening and further "
            "investigation, not standalone "
            "policy decisions."
        )

    st.divider()

    st.subheader("Intended Use")

    st.write(
        "The dashboard supports preliminary "
        "labour market monitoring, cross-industry "
        "comparisons and communication of "
        "model-generated early-warning signals."
    )


# ============================================================
# TAB 5 — DATA SOURCES AND REFERENCES
# ============================================================

with tab5:

    st.header("Data Sources & References")

    st.write(
        "This dashboard uses publicly available "
        "labour market statistics published by "
        "Singapore's Ministry of Manpower (MOM)."
    )

    st.subheader("Official MOM Data Sources")

    st.caption(
        "Select a link below to access the "
        "corresponding MOM statistical tables."
    )

    for indicator_name, dataset, url in SOURCES:

        with st.container(border=True):

            left, right = st.columns(
                [4, 1],
                vertical_alignment="center"
            )

            with left:
                st.markdown(
                    f"**{indicator_name}**"
                )
                st.caption(dataset)

            with right:
                st.link_button(
                    "View MOM Data ↗",
                    url,
                    use_container_width=True
                )

    st.divider()

    st.subheader("How the Data Was Prepared")

    st.write(
        "The source datasets were cleaned, "
        "standardised and merged using Python "
        "in Google Colab to produce a consolidated "
        "historical industry-quarter dataset."
    )

    st.markdown(
        """
        The preparation process involved:

        1. Extracting quarterly industry-level observations.
        2. Standardising industry classifications.
        3. Cleaning and merging the five indicators.
        4. Calculating quarter-on-quarter changes.
        5. Computing rolling 12-quarter Z-scores.
        6. Preparing model inputs for next-quarter prediction.
        """
    )

    st.subheader("Dataset Coverage")

    historical_start = history["quarter"].min()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Historical Data Starts",
        quarter_label(historical_start)
    )

    c2.metric(
        "Latest Model Input",
        reporting_text
    )

    c3.metric(
        "Forecast Quarter",
        forecast_text
    )

    st.write(
        f"**Industries Monitored:** {total_industries}"
    )

    st.info(
        "This dashboard uses prepared historical "
        "data and does not automatically retrieve "
        "new MOM releases. Official MOM pages "
        "may contain more recent figures."
    )

    st.subheader("Additional References")

    st.link_button(
        "MOM Labour Market Statistics Portal ↗",
        "https://stats.mom.gov.sg/Statistics/Pages/Statistics.aspx"
    )

    st.subheader("Source Attribution")

    st.caption(
        "Source: Ministry of Manpower, Singapore, "
        "Manpower Research and Statistics Department. "
        "Data processed and analysed for an "
        "independent capstone project."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"Singapore Labour Market Stress Monitor | "
    f"Reporting: {reporting_text} | "
    f"Forecast: {forecast_text}"
)

st.caption(
    "Exploratory early-warning model. "
    "Not an official MOM forecast."
)
