Singapore Sectoral Labour Market Stress Monitor — original layout preserved — 31 Oct 2026

This package uses the exact app.py uploaded by the user (dashboard UI v9, 23 October 2026).
No app.py changes have been made. Thus the sidebar, original three tabs, dark theme,
three Overview metric cards, existing charts and download controls remain unchanged.

Files for your GitHub Streamlit repository (replace same-name files):
  app.py
  historical_labour_data.csv
  dashboard_data.csv
  labour_stress_model.joblib
  requirements.txt
  .streamlit/config.toml

This package's three data/model files are byte-identical to the three files from the earlier
Streamlit update ZIP and uploaded in this conversation. Re-uploading these alone will not
change charts, metrics or predictions when the existing app already has the same files.

Expected results when using all industries and the latest source quarter (2026Q2):
  12 industries monitored
  9 next-quarter stress alerts for 2026Q3
  696 clustering observations: cluster IDs 0=42, 1=268, 2=386
  (The original UI does not display a metric for the 8 current-quarter stressed industries;
   the rule-based stress heatmap still displays those historical classifications.)

Model saved with scikit-learn 1.6.1. Keep requirements.txt pinned as supplied.

Commit files to your existing GitHub repository. Your Streamlit Cloud app should redeploy
from the existing repository. If you only want to refresh data, app.py need not be replaced
if it matches this original file. Do not upload '(1)' or '(2)' suffixed filenames.
