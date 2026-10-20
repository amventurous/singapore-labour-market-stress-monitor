import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Singapore Labour Market Stress Monitor",
    layout="wide"
)

st.title("Singapore Labour Market Stress Monitor")
st.write("Predicting next-quarter industry-level labour market stress")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("labour_stress_model.joblib")

model = load_model()

# Load dataset
df = pd.read_csv("dashboard_data.csv")

# Generate predictions
features = list(model.feature_names_in_)

df["Stress Score (%)"] = (
    model.predict_proba(df[features])[:, 1] * 100
)

df["Predicted Stress"] = model.predict(df[features])

df["Status"] = df["Predicted Stress"].map({
    1: "Stressed",
    0: "Not Stressed"
})

# Prediction period
latest_quarter = pd.Period(df["quarter"].iloc[0], freq="Q")
prediction_quarter = latest_quarter + 1

st.subheader(f"Labour Market Stress Forecast: {prediction_quarter}")

# Summary metrics
col1, col2, col3 = st.columns(3)

col1.metric("Industries Monitored", len(df))
col2.metric("Industries Flagged", int(df["Predicted Stress"].sum()))
col3.metric(
    "Industries Not Flagged",
    int((df["Predicted Stress"] == 0).sum())
)

# Industry selection
industry = st.selectbox(
    "Select Industry",
    ["All Industries"] + sorted(df["industry"].unique().tolist())
)

if industry == "All Industries":
    selected = df.copy()
else:
    selected = df[df["industry"] == industry].copy()

selected = selected.sort_values(
    "Stress Score (%)", ascending=False
)

# Visualisation
st.subheader("Predicted Stress Scores by Industry")

st.bar_chart(
    selected.set_index("industry")["Stress Score (%)"],
    horizontal=True
)

st.caption(
    "Stress scores are model outputs, not calibrated probabilities. "
    "Predictions are based on an analyst-defined stress classification "
    "and should be used for preliminary screening only."
)

# Prediction table
st.subheader("Industry Stress Predictions")

st.dataframe(
    selected[
        ["industry", "Stress Score (%)", "Status"]
    ].round(1),
    hide_index=True,
    use_container_width=True
)
