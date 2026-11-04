# Singapore Sectoral Labour Market Stress Monitor

**An exploratory, data-driven approach to monitoring industry-level labour market conditions in Singapore.**

[Try the interactive Streamlit dashboard](https://singapore-labour-market-stress-monitor-ffzzdugeb2ekpzt7pomn9w.streamlit.app/)

## Why this project?

Headline labour market statistics can conceal divergent conditions across industries. A sector may experience falling vacancies and recruitment while other sectors continue expanding. This capstone explores whether public quarterly statistics can support two complementary policy-analysis questions:

1. **Historical understanding:** What recurring labour market conditions can be identified across industries?
2. **Next-quarter screening:** Can current indicators help flag industries that *may* meet an explicitly defined stress criterion in the following quarter?

The goal is to support **policy analysis and prioritisation**, not to replace official economic assessments.

## Dataset and scope

- **Source:** Public statistics from Singapore's Ministry of Manpower (MOM).
- **Coverage:** 12 broad industries, 2009 Q1–2026 Q2 (70 quarters; **840** industry-quarter records).
- **Five indicators:** Employment change, job vacancy rate, recruitment rate, resignation rate and retrenchments.
- **Engineered predictors:** Five quarter-on-quarter changes and five industry-specific historical Z-scores, calculated using preceding quarters to limit look-ahead bias.
- **Modelled data:** 696 observations with complete rolling-history features; 684 observations after constructing the next-quarter supervised target.

The data is aggregated by industry and quarter; it is not employee-level or firm-level data.

## Analytical approach

### 1. Historical patterns — PCA + K-Means

- Applied principal component analysis (PCA) to five industry-relative historical Z-scores.
- Retained three principal components representing **80.85%** of variance.
- Used K-Means to explore **three analyst-interpreted states**: **Strong / Active**, **Cooling / Soft** and **Severe Disruption**.
- Examined how industries transitioned between states in adjacent quarters.
- Compared alternative clustering approaches and tested the sensitivity of the interpretation.

**Important distinction:** These clusters describe historical data. They are neither official MOM categories nor the labels used to train the next-quarter classifier.

### 2. Next-quarter stress screening — supervised learning

Because there was no independently validated industry-quarter stress label, a transparent **analyst-defined screening rule** was created: an industry-quarter is flagged if at least two of these four conditions occur:

| Indicator | Adverse historical Z-score |
| --- | ---: |
| Employment change | ≤ −1 |
| Job vacancy rate | ≤ −1 |
| Recruitment rate | ≤ −1 |
| Retrenchments | ≥ +1 |

Resignation rate remains a predictor, but is not part of the rule. The target is shifted forward one quarter within each industry.

- Compared logistic regression, SVM, random forest and gradient boosting, including class-balancing configurations, against a persistence benchmark.
- Used **chronological training and testing**, with expanding-window validation to avoid treating future observations as training history.
- Selected **balanced logistic regression** provisionally, recognising its modest validation advantage and the trade-off between recall and false alarms.

### Held-out test results

Evaluation on target quarters **2023 Q1–2026 Q2** (168 observations):

| Metric | Result |
| --- | ---: |
| Recall | **93.4%** |
| Precision | **48.7%** |
| F1-score | **64.0%** |
| Balanced accuracy | **68.7%** |

The model identified most historical observations meeting the *project's* stress rule in this held-out period, but about half of its alerts were false positives. Performance varied across periods and requires further validation before operational use.

## Interactive dashboard

The [Streamlit Labour Market Stress Monitor](https://singapore-labour-market-stress-monitor-ffzzdugeb2ekpzt7pomn9w.streamlit.app/) provides:

- Industry comparisons and next-quarter screening alerts;
- Historical trends in five labour market indicators;
- PCA / K-Means exploration of historical states;
- Filters for industries and time periods, plus downloadable monitoring results.

Using the static dataset through **2026 Q2**, the model flagged **9 of 12 industries** for potential stress in **2026 Q3**, under the analyst-defined criterion. **This is not a verified forecast of actual economic distress** and is not a statement about present-day conditions.

## Policy relevance

The project illustrates how a transparent analytical pipeline can consolidate fragmented indicators, support exploratory comparisons and help analysts decide which sectors warrant closer investigation. Interpretability, historical validation and a clear account of uncertainty matter as much as predictive performance.

## Limitations

- The stress label reflects project-specific rules, **not official MOM classifications** or externally validated economic outcomes.
- Test-period recall is high but precision is low; false alerts could create unnecessary follow-up work.
- The cluster separation is moderate, and alternative clustering methods produce different groupings.
- Broad industry aggregates can hide differences between firms, occupations and subsectors.
- The dashboard uses a static snapshot and must be updated when newer statistics become available.
- This is an **educational capstone / research prototype**, not an operational government system.

## Tools

Python · pandas · scikit-learn · PCA · K-Means · Logistic Regression · Time-aware Model Evaluation · Streamlit · Data Visualisation

## Reproducibility

This README documents the analysis. To make the repository fully reproducible, include the **final verified notebook**, cleaned/publicly shareable input data or official source links, a `requirements.txt`, and the Streamlit app source. Keep the notebook's actual preprocessing and evaluation sequence intact. Do not upload credentials, confidential information or unofficially redistributed restricted datasets.

## Data attribution

Data sourced from Singapore's **Ministry of Manpower** public statistical time series and reports. Analytical states, stress rules and model outputs are independently developed for this educational project and must not be attributed to MOM.
