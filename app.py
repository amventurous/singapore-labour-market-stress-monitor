import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Singapore Labour Market Stress Monitor",
    layout="wide"
)

st.title("Singapore Labour Market Stress Monitor")
st.caption(
    "Predicting next-quarter industry-level labour market stress"
)

INDICATORS = [
    "employment_change",
    "job_vacancy_rate",
    "recruitment_rate",
    "resignation_rate",
    "retrench"
]

LABELS = {
    "employment_change": "Employment Change ('000)",
    "job_vacancy_rate": "Job Vacancy Rate (%)",
    "recruitment_rate": "Recruitment Rate (%)",
    "resignation_rate": "Resignation Rate (%)",
    "retrench": "Retrenchments (Number)"
}


# =====================================================
# LOAD MODEL AND DATA
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load("labour_stress_model.joblib")


@st.cache_data
def load_data():
    dashboard = pd.read_csv("dashboard_data.csv")
    history = pd.read_csv("historical_labour_data.csv")

    required = ["quarter", "industry"] + INDICATORS

    missing = [c for c in required if c not in history.columns]
    if missing:
        raise ValueError(
            f"Historical CSV missing columns: {missing}"
        )

    history = history[required].copy()

    for data in [dashboard, history]:
        data["quarter"] = pd.PeriodIndex(
            data["quarter"].astype(str).str.replace(
                "-", "", regex=False
            ),
            freq="Q"
        )

        data["industry"] = (
            data["industry"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    for indicator in INDICATORS:
        history[indicator] = pd.to_numeric(
            history[indicator],
            errors="coerce"
        )

    if history.duplicated(
        ["industry", "quarter"]
    ).any():
        raise ValueError(
            "Historical data contain duplicate "
            "industry-quarter records."
        )

    return dashboard, history


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def calculate_features(data):

    result = data.sort_values(
        ["industry", "quarter"]
    ).copy()

    for indicator in INDICATORS:

        grouped = result.groupby("industry")[indicator]

        # Quarter-on-quarter changes
        result[f"{indicator}_qoq_change"] = grouped.diff()

        # Rolling mean of previous 12 quarters
        rolling_mean = grouped.transform(
            lambda s: s.shift(1).rolling(
                12, min_periods=12
            ).mean()
        )

        # Rolling standard deviation
        rolling_std = grouped.transform(
            lambda s: s.shift(1).rolling(
                12, min_periods=12
            ).std()
        )

        result[f"{indicator}_zscore"] = (
            (result[indicator] - rolling_mean)
            / rolling_std.replace(0, np.nan)
        )

    return result


def predict_stress(data, model, features):

    X = data[features].copy()

    if not np.isfinite(
        X.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Unable to calculate complete model features. "
            "Please check the historical data."
        )

    scores = model.predict_proba(X)[:, 1] * 100
    predictions = model.predict(X)

    return scores, predictions


def display_forecasts(data):

    if data.empty:
        st.info("No forecasts to display.")
        return

    selected = data.copy()

    selected["Stress Score (%)"] = (
        selected["Stress Score (%)"].astype(float)
    )

    selected = selected.sort_values(
        "Stress Score (%)",
        ascending=False
    )

    st.subheader("Predicted Stress Scores by Industry")

    st.bar_chart(
        selected.set_index("industry")["Stress Score (%)"],
        horizontal=True
    )

    st.subheader("Industry Stress Predictions")

    display = selected[
        ["industry", "Stress Score (%)", "Status"]
    ].copy()

    display["Stress Score (%)"] = (
        display["Stress Score (%)"].round(1)
    )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True
    )


# =====================================================
# INITIALISE APPLICATION
# =====================================================

try:
    model = load_model()
    dashboard, historical = load_data()
    features = list(model.feature_names_in_)

    dashboard["Stress Score (%)"], dashboard[
        "Predicted Stress"
    ] = predict_stress(
        dashboard, model, features
    )

except Exception as e:
    st.error(f"Unable to initialise dashboard: {e}")
    st.stop()

dashboard["Status"] = dashboard[
    "Predicted Stress"
].map({
    1: "Stressed",
    0: "Not Stressed"
})

if "manual_entries" not in st.session_state:
    st.session_state.manual_entries = []

if "manual_results" not in st.session_state:
    st.session_state.manual_results = []

st.info(
    "This dashboard provides exploratory early-warning "
    "signals. Stress scores are model outputs, not "
    "calibrated probabilities or official MOM forecasts."
)

tab1, tab2 = st.tabs([
    "Compare Industries",
    "Manual Forecast"
])


# =====================================================
# TAB 1: COMPARE INDUSTRIES
# =====================================================

with tab1:

    latest_quarter = dashboard["quarter"].max()

    current = dashboard[
        dashboard["quarter"] == latest_quarter
    ].copy()

    st.subheader(
        f"Labour Market Stress Forecast: "
        f"{latest_quarter + 1}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Industries Monitored",
        len(current)
    )

    col2.metric(
        "Industries Flagged",
        int(current["Predicted Stress"].sum())
    )

    col3.metric(
        "Industries Not Flagged",
        int((current["Predicted Stress"] == 0).sum())
    )

    st.subheader("Select Industries to Compare")

    industries = sorted(
        current["industry"].unique().tolist()
    )

    if "selected_industries" not in st.session_state:
        st.session_state.selected_industries = (
            industries.copy()
        )

    c1, c2 = st.columns(2)

    if c1.button("Select All"):
        st.session_state.selected_industries = (
            industries.copy()
        )

    if c2.button("Clear Selection"):
        st.session_state.selected_industries = []

    selected_industries = st.multiselect(
        "Tick industries to compare",
        industries,
        key="selected_industries",
        format_func=lambda x: x.title()
    )

    selected = current[
        current["industry"].isin(selected_industries)
    ].copy()

    if selected.empty:
        st.warning(
            "Please select at least one industry."
        )

    else:

        show_scores = st.checkbox(
            "Show Stress Score Chart",
            value=True
        )

        show_table = st.checkbox(
            "Show Prediction Table",
            value=True
        )

        show_indicators = st.checkbox(
            "Show Labour Market Indicator Charts",
            value=False
        )

        if show_scores or show_table:

            if show_scores and show_table:
                display_forecasts(selected)

            elif show_scores:
                st.subheader("Predicted Stress Scores")

                st.bar_chart(
                    selected.set_index("industry")[
                        "Stress Score (%)"
                    ],
                    horizontal=True
                )

            elif show_table:
                st.subheader("Industry Stress Predictions")

                st.dataframe(
                    selected[
                        ["industry", "Stress Score (%)", "Status"]
                    ].round(1),
                    hide_index=True,
                    use_container_width=True
                )

        if show_indicators:

            raw = historical[
                (historical["quarter"] == latest_quarter)
                & (
                    historical["industry"].isin(
                        selected_industries
                    )
                )
            ].copy()

            if raw.empty:
                st.warning(
                    "No indicator data available "
                    "for this quarter."
                )

            else:
                choices = st.multiselect(
                    "Select indicators to display",
                    INDICATORS,
                    default=INDICATORS,
                    format_func=lambda x: LABELS[x]
                )

                for i in range(0, len(choices), 2):

                    columns = st.columns(2)

                    for column, indicator in zip(
                        columns,
                        choices[i:i + 2]
                    ):

                        with column:
                            st.write(
                                f"**{LABELS[indicator]}**"
                            )

                            st.bar_chart(
                                raw.set_index("industry")[
                                    indicator
                                ],
                                horizontal=True
                            )


# =====================================================
# TAB 2: GUIDED MANUAL FORECAST
# =====================================================

with tab2:

    st.subheader("Manual Labour Market Forecast")

    st.write(
        "Enter new quarterly MOM figures to generate "
        "next-quarter labour market stress predictions."
    )

    st.info(
        "Quarters must be entered sequentially. "
        "The dashboard automatically identifies the "
        "next required quarter. You do not need to "
        "select or type the quarter yourself."
    )

    # STEP 1: SELECT INDUSTRY

    st.markdown("#### Step 1 — Select Industry")

    manual_industry = st.selectbox(
        "Industry",
        sorted(historical["industry"].unique()),
        format_func=lambda x: x.title(),
        key="manual_industry"
    )

    # Combine official history and session entries

    manual_entries_df = pd.DataFrame(
        st.session_state.manual_entries,
        columns=["quarter", "industry"] + INDICATORS
    )

    combined = pd.concat(
        [historical, manual_entries_df],
        ignore_index=True
    )

    combined = combined.sort_values(
        ["industry", "quarter"]
    )

    industry_data = combined[
        combined["industry"] == manual_industry
    ].copy()

    last_quarter = industry_data["quarter"].max()
    next_quarter = last_quarter + 1

    # STEP 2: AUTO-DETERMINE NEXT QUARTER

    st.markdown("#### Step 2 — Enter Quarterly Figures")

    col1, col2 = st.columns(2)

    col1.metric(
        "Latest Available Quarter",
        str(last_quarter)
    )

    col2.metric(
        "Next Required Quarter",
        str(next_quarter)
    )

    st.caption(
        f"Enter the five MOM indicators for "
        f"{next_quarter}. These will be used "
        f"to predict stress in {next_quarter + 1}."
    )

    with st.form(
        key=f"forecast_form_{manual_industry}_{next_quarter}"
    ):

        values = {}

        for indicator in INDICATORS:

            values[indicator] = st.number_input(
                LABELS[indicator],
                value=0.0,
                min_value=(
                    None if indicator == "employment_change"
                    else 0.0
                ),
                step=(
                    1.0 if indicator == "retrench"
                    else 0.1
                ),
                key=f"{manual_industry}_{next_quarter}_{indicator}"
            )

        submitted = st.form_submit_button(
            f"Save {next_quarter} & Generate Forecast",
            type="primary"
        )

    if submitted:

        try:

            # Check the previous 12 quarters
            previous_12 = industry_data.tail(12)[
                "quarter"
            ].tolist()

            required_12 = list(
                pd.period_range(
                    next_quarter - 12,
                    next_quarter - 1,
                    freq="Q"
                )
            )

            if previous_12 != required_12:
                raise ValueError(
                    "The previous 12 consecutive quarters "
                    "are required before forecasting."
                )

            new_row = {
                "quarter": next_quarter,
                "industry": manual_industry,
                **values
            }

            updated = pd.concat(
                [
                    combined,
                    pd.DataFrame([new_row])
                ],
                ignore_index=True
            )

            engineered = calculate_features(updated)

            target = engineered[
                (engineered["industry"] == manual_industry)
                & (engineered["quarter"] == next_quarter)
            ].copy()

            scores, predictions = predict_stress(
                target,
                model,
                features
            )

            status = (
                "Stressed"
                if predictions[0] == 1
                else "Not Stressed"
            )

            # Save successful input in the session
            st.session_state.manual_entries.append(
                new_row
            )

            # Save prediction result
            st.session_state.manual_results.append({
                "quarter": str(next_quarter),
                "forecast_quarter": str(next_quarter + 1),
                "industry": manual_industry,
                "Stress Score (%)": float(scores[0]),
                "Status": status
            })

            st.success(
                f"{next_quarter} saved successfully. "
                f"Forecast generated for "
                f"{next_quarter + 1}."
            )

            st.metric(
                "Stress Score",
                f"{scores[0]:.1f}%"
            )

            st.metric(
                "Predicted Status",
                status
            )

            st.caption(
                "The next required quarter will "
                "appear automatically after submission."
            )

            st.rerun()

        except Exception as e:
            st.error(str(e))

    # STEP 3: VIEW MANUAL FORECASTS

    st.divider()
    st.markdown("#### Step 3 — View Your Forecasts")

    results = pd.DataFrame(
        st.session_state.manual_results
    )

    if results.empty:

        st.info(
            "No manual forecasts generated yet. "
            "Enter the quarterly figures above to begin."
        )

    else:

        available_quarters = sorted(
            results["forecast_quarter"].unique(),
            reverse=True
        )

        selected_forecast_quarter = st.selectbox(
            "Forecast Quarter",
            available_quarters
        )

        quarter_results = results[
            results["forecast_quarter"]
            == selected_forecast_quarter
        ].copy()

        forecast_industries = sorted(
            quarter_results["industry"].unique()
        )

        selected_forecast_industries = st.multiselect(
            "Select Industries to Compare",
            forecast_industries,
            default=forecast_industries,
            format_func=lambda x: x.title(),
            key="manual_comparison"
        )

        comparison = quarter_results[
            quarter_results["industry"].isin(
                selected_forecast_industries
            )
        ].copy()

        display_forecasts(comparison)

        st.subheader("Saved Manual Forecasts")

        display_results = results.copy()

        st.dataframe(
            display_results.round(1),
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Download Manual Forecasts",
            data=display_results.to_csv(index=False),
            file_name="manual_forecasts.csv",
            mime="text/csv"
        )

        if st.button(
            "Clear All Manual Forecasts",
            type="secondary"
        ):

            st.session_state.manual_entries = []
            st.session_state.manual_results = []

            st.rerun()

st.divider()

st.caption(
    "Data source: MOM quarterly labour market indicators. "
    "Manual entries are retained only during the "
    "current Streamlit session. The trained model "
    "is not retrained when new quarters are entered."
)
