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
