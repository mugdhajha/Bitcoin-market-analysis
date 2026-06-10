# Bitcoin Market Analysis & Predictive Modeling

## Overview

This project performs an end-to-end data science analysis of Bitcoin's historical market behavior from September 2014 to June 2026 using daily OHLCV (Open, High, Low, Close, Volume) data.

The objective was not only to explore Bitcoin's market dynamics but also to investigate whether commonly used technical indicators can predict future price direction.

The project follows a complete data science workflow:

* Data Understanding
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Predictive Modeling
* Hypothesis Testing
* Insight Generation

---

## Dataset

**Period Covered:** 2014-09-17 to 2026-06-05

**Observations:** 4,280 daily records

### Features

| Feature   | Description            |
| --------- | ---------------------- |
| Open      | Opening price          |
| High      | Daily high price       |
| Low       | Daily low price        |
| Close     | Closing price          |
| Adj Close | Adjusted closing price |
| Volume    | Trading volume         |
| Date      | Trading date           |

---

## Project Structure

```text
bitcoin-market-analysis/
│
├── data/
│   ├── raw/
│   │   └── bitcoin.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
│
├── figures/
│
├── reports/
│   └── findings.md
│
├── src/
│
├── requirements.txt
│
└── README.md
```

---

## Exploratory Data Analysis

### Market Evolution

Bitcoin exhibited exponential long-term growth with multiple bull and bear market cycles across the 11-year period.

### Largest Daily Gain

* Date: 2017-12-07
* Return: +25.25%

### Largest Daily Loss

* Date: 2020-03-12
* Return: -37.17%

Interestingly, the largest daily decline occurred during the COVID-19 market panic rather than the 2018 bear market.

---

## Volatility Analysis

A 30-day rolling volatility measure was created to analyze market risk.

### Key Findings

* Peak volatility occurred during the COVID-19 crisis.
* Volatility clustered around major market events.
* Recent years showed lower average volatility compared to earlier speculative cycles.

This supports the hypothesis that Bitcoin may be gradually maturing as an asset class.

---

## Feature Engineering

The following technical indicators were engineered:

### Trend Features

* MA7 (7-Day Moving Average)
* MA30 (30-Day Moving Average)
* MA90 (90-Day Moving Average)

### Relative Strength Features

* RSI (14-Day Relative Strength Index)

### Momentum Features

* Daily Return

### Risk Features

* 30-Day Rolling Volatility

### Relative Trend Ratios

* Close_vs_MA30_ratio
* MA7_vs_MA30_ratio

---

## Hypothesis Testing

### Hypothesis 1

Higher trading volume should be associated with higher volatility.

**Result**

Correlation:

```text
-0.0245
```

**Conclusion**

Virtually no linear relationship exists between trading volume and volatility in this dataset.

The original hypothesis was rejected.

---

### Hypothesis 2

Bitcoin should be more volatile during major speculative cycles.

**Result**

Highest average volatility:

* 2017
* 2018
* 2021

Lower volatility observed in:

* 2023
* 2024
* 2025

**Conclusion**

The hypothesis was supported.

---

## Predictive Modeling

### Objective

Predict future Bitcoin direction using technical indicators.

### Features Used

* Daily Return
* Volatility
* RSI
* Close_vs_MA30_ratio
* MA7_vs_MA30_ratio

### Model 1

Logistic Regression

### Evaluation Strategy

Chronological Train-Test Split (80/20)

This avoids information leakage from future observations into the training set.

---

## Results

### Logistic Regression

Accuracy:

```text
~49%
```

### Key Insight

Despite extensive feature engineering, traditional technical indicators demonstrated limited predictive power for short-term Bitcoin direction.

Correlation analysis revealed extremely weak relationships between engineered features and future price movement.

This explains the model's near-random performance.

---

## Key Insights

* Bitcoin experiences strong volatility clustering around major market events.
* The COVID-19 crash produced the largest single-day loss and highest volatility regime.
* Trading volume showed almost no relationship with volatility.
* Bitcoin volatility appears lower in recent years than in earlier speculative cycles.
* Traditional technical indicators showed weak predictive power for future price direction.
* Predicting short-term Bitcoin movement remains extremely challenging.

---

## Future Work

* Random Forest Classification
* XGBoost Modeling
* Feature Importance Analysis
* MACD Indicator
* Bollinger Bands
* Market Regime Detection
* Volatility Forecasting
* Interactive Dashboard Development
* SHAP-Based Explainability

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebooks
* Git
* GitHub

---

## Author

Built as a hands-on Data Science project to explore financial time series analysis, feature engineering, hypothesis testing, and predictive modeling using Bitcoin market data.
