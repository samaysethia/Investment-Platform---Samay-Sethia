# InvestIQ — Platform Intelligence Dashboard

A full-stack data science application for analysing investor survey data, segmenting customers, predicting adoption intent, and generating prescriptive business strategies.

## Features
- **Descriptive Analysis** — Demographics, investment behaviour, feature demand
- **Diagnostic Analysis** — Correlation heatmaps, crosstab explorer, behavioural insights
- **Customer Segmentation** — K-Means clustering with elbow + silhouette method, persona cards
- **Classification Model** — Random Forest / Logistic Regression / Gradient Boosting to predict adoption (Likely / Neutral / Unlikely)
- **Association Rule Mining** — Apriori algorithm on investment products, features, goals
- **Regression Analysis** — Willingness to pay prediction with 5 model comparison
- **Prescriptive Intelligence** — Revenue opportunity, feature priority matrix, segment strategy
- **New Customer Predictor** — Upload new survey CSV → instant adoption + WTP predictions

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy — no environment variables needed

## Dataset
`investment_survey_data.csv` — 2,000 synthetic respondents, 89 columns, mimicking Indian fintech survey data with realistic correlations, demographics, and noise.

## Tech Stack
- `streamlit` — Dashboard framework
- `scikit-learn` — ML models
- `mlxtend` — Apriori / association rules
- `plotly` — Interactive charts
- `pandas`, `numpy` — Data processing
