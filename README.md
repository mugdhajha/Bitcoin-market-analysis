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

## Feature Engineering Experiments

Beyond the baseline model, multiple technical indicators and momentum-based features were tested to evaluate whether they improved predictive performance.

### Baseline Random Forest

Features:

* Daily Return
* Volatility
* RSI
* Close_vs_MA30_ratio
* MA7_vs_MA30_ratio

Accuracy:

```text
52.09%
```

Feature Importance:

```text
Daily_return          22.8%
RSI                   20.9%
Volatility            20.6%
MA7_vs_MA30_ratio     18.8%
Close_vs_MA30_ratio   16.9%
```

---

### Experiment 1: MACD

#### Hypothesis

Since momentum-related indicators such as Daily Return and RSI were among the most important features, MACD was expected to provide additional predictive signal.

#### Implementation

MACD was calculated using the difference between the 12-day and 26-day exponential moving averages.

To account for Bitcoin's changing price scale over time, a normalized feature was created:

```text
MACD_ratio = MACD / Close
```

#### Results

Accuracy:

```text
51.25%
```

Feature Importance:

```text
Daily_return          19.2%
Volatility            18.0%
RSI                   17.3%
MACD_ratio            17.2%
Close_vs_MA30_ratio   14.4%
MA7_vs_MA30_ratio     13.9%
```

#### Findings

Although MACD_ratio became one of the most important features in the model, overall accuracy decreased.

This suggests that MACD contains information that the model actively uses, but the discovered patterns do not generalize well to unseen market conditions.

#### Conclusion

The hypothesis was partially supported.

* MACD captured meaningful information.
* The model assigned high importance to the feature.
* However, it did not improve out-of-sample predictive performance.

---

### Experiment 2: Multi-Day Momentum Features

#### Hypothesis

Longer-term momentum may contain additional predictive information beyond single-day returns.

Two new features were created:

```text
Return_7D
Return_30D
```

#### Results

Accuracy:

```text
51.49%
```

Feature Importance:

```text
Daily_return          18.4%
RSI                   15.4%
Volatility            15.0%
Return_7D             13.4%
MA7_vs_MA30_ratio     13.1%
Return_30D            12.5%
Close_vs_MA30_ratio   12.3%
```

#### Findings

The new momentum features received substantial feature importance, indicating that the model considered them useful.

However, overall predictive performance remained below the baseline Random Forest model.

#### Conclusion

The hypothesis was partially supported.

While the model utilized the momentum features, the additional information failed to improve generalization on unseen data.

---

### Experiment 3: Bollinger Bands

#### Hypothesis

Previous experiments indicated that Volatility was consistently among the most important features in the Random Forest model.

Since Bollinger Bands combine both price and volatility information, it was hypothesized that they could capture market conditions not represented by traditional moving-average ratios.

A normalized feature was created:

```text
BB_position
```

which measures the position of the current price within the Bollinger Band range.

Interpretation:

```text
0.0 → Price near lower band
0.5 → Price near middle band
1.0 → Price near upper band
```

#### Results

Accuracy:

```text
51.61%
```

Feature Importance:

```text
Daily_return          20.7%
RSI                   17.5%
BB_position           17.0%
Volatility            16.7%
MA7_vs_MA30_ratio     14.9%
Close_vs_MA30_ratio   13.2%
```

#### Findings

BB_position immediately became one of the most important features in the model, ranking third overall behind Daily Return and RSI.

This indicates that the model was able to extract useful information from the relationship between price and volatility.

However, despite its high importance, overall predictive performance remained below the baseline Random Forest model.

#### Conclusion

The hypothesis was partially supported.

* Bollinger Bands captured meaningful market information.
* The model assigned substantial importance to BB_position.
* However, the additional feature did not improve out-of-sample prediction accuracy.

This result reinforces a recurring theme observed throughout the project:

> A feature can contain useful information and still fail to improve predictive performance on unseen data.

#### Comparison with Previous Experiments

| Experiment               | Random Forest Accuracy |
| ------------------------ | ---------------------: |
| Baseline Random Forest   |             **52.09%** |
| + MACD_ratio             |                 51.25% |
| + Return_7D + Return_30D |                 51.49% |
| + BB_position            |                 51.61% |

#### Key Insight

Across all experiments, newly engineered features consistently received meaningful feature importance scores, indicating that they contained information relevant to market behavior.

However, none of the added features surpassed the baseline model, suggesting that much of the useful signal may already be captured by the original feature set:

```text
Daily_return
RSI
Volatility
Close_vs_MA30_ratio
MA7_vs_MA30_ratio
```

This suggests that short-term Bitcoin direction is difficult to predict using technical indicators derived solely from historical price and volume data.


## Experiment Summary

| Model Configuration                     | Accuracy   |
| --------------------------------------- | ---------- |
| Logistic Regression                     | 49.22%     |
| Logistic Regression + Momentum Features | 48.87%     |
| Logistic Regression (7-Day Target)      | 50.66%     |
| Random Forest (Baseline)                | **52.09%** |
| Random Forest + MACD_ratio              | 51.25%     |
| Random Forest + Return_7D + Return_30D  | 51.49%     |

---

## Key Takeaways

* Random Forest consistently outperformed Logistic Regression.
* Non-linear relationships exist within the technical indicators.
* Traditional technical indicators contain a small amount of predictive signal.
* Additional features such as MACD and multi-day momentum increased feature complexity but failed to improve out-of-sample accuracy.
* Short-term Bitcoin direction remains difficult to predict using technical indicators derived solely from historical OHLCV data.
* Model performance suggests that additional information sources or alternative feature engineering approaches may be required to achieve meaningful predictive gains.

## Conclusion

Multiple momentum and volatility-based indicators were evaluated. Although several engineered features achieved high model importance, none improved out-of-sample performance beyond the baseline Random Forest model (52.09%). This suggests that additional technical indicators largely capture information already present in the original feature set.



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

### Model 2

Random forest Classifier

### Motivation

Unlike Logistic Regression, Random Forest can capture non-linear relationships and interactions between technical indicators.

### Results

### Random Forest

Accuracy:

52.09%

### Classification Report:

Precision: 0.52
Recall: 0.52
F1-Score: 0.51

### Feature Importance Ranking

1. Daily Return           22.8%
2. RSI                    20.9%
3. Volatility             20.6%
4. MA7_vs_MA30_ratio      18.8%
5. Close_vs_MA30_ratio    16.9%

### Key Findings

* Random Forest outperformed Logistic Regression by approximately 3%.
* Non-linear relationships exist within the engineered technical indicators.
* Daily Return emerged as the strongest predictive feature.
* RSI and Volatility contributed significantly more information than suggested by simple correlation analysis.
* Despite improvement, predictive performance remained modest, indicating that short-term   Bitcoin direction is inherently difficult to forecast using traditional technical indicators alone.

### Conclusion

The results suggest that technical indicators contain a small amount of predictive signal, but not enough to reliably forecast short-term Bitcoin price direction. More advanced features such as MACD, Bollinger Bands, and longer-term momentum indicators may improve performance in future iterations of the project.

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
