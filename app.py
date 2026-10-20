import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

st.set_page_config(
    page_title="Singapore Labour Market Stress Monitor",
    layout="wide"
)

st.title("Singapore Labour Market Stress Monitor")
st.write("Predicting next-quarter industry-level labour market stress")

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


@st.cache_resource
def load_model():
    return joblib.load("labour_stress_model.joblib")


@st.cache_data
def load_data():
    dashboard = pd.read_csv("dashboard_data.csv")
    historical = pd.read_csv("historical_labour_data.csv")

    required = ["quarter", "industry"] + INDICATORS
    missing = [c for c in required if c not in historical.columns]

    if missing:
        raise ValueError(
            f"Historical data missing columns: {missing}"
        )

    historical = historical[required].copy()

    for data in [dashboard, historical]:
        data["quarter"] = pd.PeriodIndex(
            data["quarter"].astype(str).str.replace(
                "-", "", regex=False
            ),
            freq="Q"
        )
        data["industry"] = (
            data["industry"].astype(str).str.strip().str.lower()
        )

    for c in INDICATORS:
        historical[c] = pd.to_numeric(
            historical[c], errors="coerce"
        )

    if historical.duplicated(["industry", "quarter"]).any():
        raise ValueError(
            "Historical data contain duplicate industry-quarter rows."
        )

    return dashboard, historical


def calculate_features(data):
    result = data.sort_values(
        ["industry", "quarter"]
    ).copy()

    for indicator in INDICATORS:
        grouped = result.groupby("industry")[indicator]

        # Quarter-on-quarter change
        result[f"{indicator}_qoq_change"] = grouped.diff()

        # Previous 12 quarters, excluding current quarter
        rolling_mean = grouped.transform(
            lambda s: s.shift(1).rolling(
                12, min_periods=12
            ).mean()
        )

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


def make_prediction(data, model, features):
    X = data[features]

    if X.isna().any().any():
        raise ValueError(
            "Insufficient historical data to calculate model features."
        )

    scores = model.predict_proba(X)[:, 1] * 100
    predictions = model.predict(X)

    return scores, predictions


def format_quarter(value):
    text = str(value).strip().upper().replace("-", "")
    text = text.replace(" ", "")

    if not re.fullmatch(r"\d{4}Q[1-4]", text):
        raise ValueError(
            "Enter a valid quarter, such as 2026Q3."
        )

    return pd.Period(text, freq="Q")


try:
    model = load_model()
    df, historical = load_data()
    features = list(model.feature_names_in_)
except Exception as e:
    st.error(f"Unable to load dashboard files: {e}")
    st.stop()

# Keep manual entries within the user's session.
if "manual_entries" not in st.session_state:
    st.session_state.manual_entries = pd.DataFrame(
        columns=["quarter", "industry"] + INDICATORS
    )

# Score existing dashboard data.
try:
    df["Stress Score (%)"], df["Predicted Stress"] = (
        make_prediction(df, model, features)
    )
except Exception as e:
    st.error(f"Unable to calculate dashboard predictions: {e}")
    st.stop()

df["Status"] = df["Predicted Stress"].map({
    1: "Stressed",
    0: "Not Stressed"
})

latest_quarter = df["quarter"].max()
prediction_quarter = latest_quarter + 1

st.caption(
    "Model scores are screening signals, not calibrated probabilities. "
    "Stress is based on an analyst-defined classification, not an "
    "official MOM stress definition."
)

tab1, tab2 = st.tabs([
    "📊 Compare Industries",
    "📝 Manual Forecast"
])

# --------------------------------------------------
# TAB 1: COMPARE INDUSTRIES
# --------------------------------------------------

with tab1:
    st.subheader(
        f"Labour Market Stress Forecast: {prediction_quarter}"
    )

    latest_df = df[
        df["quarter"] == latest_quarter
    ].copy()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Industries Monitored",
        len(latest_df)
    )
    col2.metric(
        "Industries Flagged",
        int(latest_df["Predicted Stress"].sum())
    )
    col3.metric(
        "Industries Not Flagged",
        int((latest_df["Predicted Stress"] == 0).sum())
    )

    st.subheader("Select Industries to Compare")

    industries = sorted(
        latest_df["industry"].unique().tolist()
    )

    if "selected_industries" not in st.session_state:
        st.session_state.selected_industries = industries.copy()

    left, right = st.columns(2)

    with left:
        if st.button("Select All Industries"):
            st.session_state.selected_industries = industries.copy()

    with right:
        if st.button("Clear Selection"):
            st.session_state.selected_industries = []

    selected_industries = st.multiselect(
        "Tick one or more industries",
        industries,
        key="selected_industries",
        format_func=lambda x: x.title()
    )

    selected = latest_df[
        latest_df["industry"].isin(selected_industries)
    ].copy()

    selected = selected.sort_values(
        "Stress Score (%)", ascending=False
    )

    if selected.empty:
        st.info("Select at least one industry to view predictions.")

    else:
        st.subheader("Choose What to Display")

        show_scores = st.checkbox(
            "Stress Score Bar Chart", value=True
        )
        show_table = st.checkbox(
            "Industry Prediction Table", value=True
        )
        show_indicators = st.checkbox(
            "Individual Labour Market Indicator Charts",
            value=False
        )

        if show_scores:
            st.subheader("Predicted Stress Scores by Industry")

            st.bar_chart(
                selected.set_index("industry")["Stress Score (%)"],
                horizontal=True
            )

        if show_table:
            st.subheader("Industry Stress Predictions")

            st.dataframe(
                selected[
                    ["industry", "Stress Score (%)", "Status"]
                ].round(1),
                hide_index=True,
                use_container_width=True
            )

        if show_indicators:
            st.subheader("Labour Market Indicators")

            # Historical raw indicators for selected quarter
            raw = historical[
                (historical["quarter"] == latest_quarter)
                & (historical["industry"].isin(selected_industries))
            ].copy()

            if raw.empty:
                st.warning(
                    "No raw indicator data found for this quarter."
                )
            else:
                chart_choices = st.multiselect(
                    "Choose indicators",
                    INDICATORS,
                    default=INDICATORS,
                    format_func=lambda x: LABELS[x]
                )

                for i in range(0, len(chart_choices), 2):
                    columns = st.columns(2)

                    for column, indicator in zip(
                        columns, chart_choices[i:i + 2]
                    ):
                        with column:
                            st.write(f"**{LABELS[indicator]}**")
                            st.bar_chart(
                                raw.set_index("industry")[indicator],
                                horizontal=True
                            )

# --------------------------------------------------
# TAB 2: MANUAL FORECAST
# --------------------------------------------------

with tab2:
    st.subheader("Manual Next-Quarter Forecast")

    st.write(
        "Enter the latest MOM quarterly figures for an industry. "
        "The dashboard will calculate the model features using "
        "the previous 12 quarters of data."
    )

    manual_industry = st.selectbox(
        "Select Industry",
        sorted(historical["industry"].unique()),
        format_func=lambda x: x.title(),
        key="manual_industry"
    )

    existing_manual = st.session_state.manual_entries.copy()

    combined_history = pd.concat(
        [historical, existing_manual],
        ignore_index=True
    )

    combined_history = combined_history.sort_values(
        ["industry", "quarter"]
    )

    industry_history = combined_history[
        combined_history["industry"] == manual_industry
    ].copy()

    last_quarter = industry_history["quarter"].max()
    next_quarter = last_quarter + 1

    st.info(
        f"Latest available quarter: {last_quarter}. "
        f"Next quarter to enter: {next_quarter}."
    )

    with st.form("manual_forecast_form"):
        quarter_input = st.text_input(
            "Reporting Quarter",
            value=str(next_quarter)
        )

        values = {}

        for indicator in INDICATORS:
            values[indicator] = st.number_input(
                LABELS[indicator],
                value=0.0,
                step=1.0 if indicator == "retrench" else 0.1
            )

        submit = st.form_submit_button(
            "Generate Forecast"
        )

    if submit:
        try:
            entered_quarter = format_quarter(quarter_input)

            if entered_quarter != next_quarter:
                raise ValueError(
                    f"Please enter {next_quarter}. "
                    "Quarters must be added consecutively."
                )

            if any(
                values[c] < 0
                for c in [
                    "job_vacancy_rate",
                    "recruitment_rate",
                    "resignation_rate",
                    "retrench"
                ]
            ):
                raise ValueError(
                    "Rates and retrenchment counts cannot be negative."
                )

            # Check the 12-quarter historical window.
            previous_12 = industry_history.tail(12)["quarter"].tolist()
            required_12 = list(
                pd.period_range(
                    entered_quarter - 12,
                    entered_quarter - 1,
                    freq="Q"
                )
            )

            if previous_12 != required_12:
                raise ValueError(
                    "The preceding 12 consecutive quarters "
                    "are required for this forecast."
                )

            new_row = {
                "quarter": entered_quarter,
                "industry": manual_industry,
                **values
            }

            updated = pd.concat(
                [
                    combined_history,
                    pd.DataFrame([new_row])
                ],
                ignore_index=True
            )

            calculated = calculate_features(updated)

            new_features = calculated[
                (calculated["industry"] == manual_industry)
                & (calculated["quarter"] == entered_quarter)
            ].copy()

            score, prediction = make_prediction(
                new_features, model, features
            )

            st.success(
                f"Forecast generated for {entered_quarter + 1}"
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Stress Score",
                f"{score[0]:.1f}%"
            )

            c2.metric(
                "Predicted Status",
                "Stressed" if prediction[0] == 1
                else "Not Stressed"
            )

            with st.expander("View Calculated Model Features"):
                st.dataframe(
                    new_features[features],
                    hide_index=True,
                    use_container_width=True
                )

            # Save entry only within this Streamlit session.
            st.session_state.manual_entries = pd.concat(
                [
                    st.session_state.manual_entries,
                    pd.DataFrame([new_row])
                ],
                ignore_index=True
            )

            st.caption(
                "Entry retained for this browser session. "
                "The model has not been retrained."
            )

        except Exception as e:
            st.error(str(e))

    if not st.session_state.manual_entries.empty:
        st.subheader("Manually Entered Quarters")

        st.dataframe(
            st.session_state.manual_entries,
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Download Manual Entries as CSV",
            data=st.session_state.manual_entries.to_csv(
                index=False
            ),
            file_name="manual_quarterly_entries.csv",
            mime="text/csv"
        )
