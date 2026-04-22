# Census Income Intelligence Platform
### A Machine Learning Project for Retail Marketing Optimization

---

## Overview

This project was built for a retail business client who wants to identify high-income individuals for targeted marketing. Using the 1994–95 US Census Bureau dataset, I built two models:

1. **Income Classifier** - predicts whether an individual earns over or under $50K per year
2. **Customer Segmentation Model** - discovers natural marketing groups within the population

The final deliverable includes three Jupyter notebooks, a trained XGBoost model, a KMeans segmentation model, and an interactive Streamlit app where the client can enter any individual's details and instantly get a prediction and marketing recommendation.

---

## Project Structure

```
census-income-project/
├── data/
│   ├── census-bureau.data          # Raw dataset (original)
│   ├── census-bureau.columns       # Column names
│   ├── xgb_model_calibrated.pkl    # Final classification model
│   ├── xgb_model.pkl               # Uncalibrated XGBoost model
│   ├── scaler.pkl                  # StandardScaler for classification
│   ├── kmeans_model.pkl            # Final segmentation model
│   ├── scaler_seg.pkl              # StandardScaler for segmentation
│   ├── census_segmented.csv        # Dataset with cluster labels
│   ├── cluster_summary.csv         # Summary statistics per cluster
│   └── y_labels.csv                # Target labels
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_classification.ipynb     # Classification Model
│   └── 03_segmentation.ipynb       # Segmentation Model
├── plots/                          # All generated charts and visualizations
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Dataset

- **Source:** 1994–95 US Census Bureau
- **Size:** 199,523 individuals
- **Features:** 40 demographic and employment variables
- **Target:** Binary - income over $50K (6%) or under $50K (94%)
- **Key challenge:** Severe class imbalance (94/6 split)

> **Note:** Two large intermediate files (`census_cleaned.csv` and `X_preprocessed.csv`) are excluded from the repository due to GitHub file size limits. These are automatically regenerated when the notebooks are run in order.

---

## How to Run

### Prerequisites

- Python 3.9 or higher
- Git
- On Mac: `brew install libomp` (required for XGBoost)

### Step 1 - Clone the Repository

```bash
git clone https://github.com/Haritha1752/census-income-project
cd census-income-project
```

### Step 2 - Create a Virtual Environment

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3 - Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 - Run the Notebooks (in order)

Open Jupyter and run each notebook from top to bottom:

```bash
jupyter notebook
```

Run in this order:
1. `notebooks/01_eda.ipynb` - cleans the data and saves `census_cleaned.csv`
2. `notebooks/02_classification.ipynb` - trains the model and saves all pkl files
3. `notebooks/03_segmentation.ipynb` - builds segments and saves cluster files

> Each notebook must be run in order as each one depends on outputs from the previous.

### Step 5 - Launch the Streamlit App

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

---

## Approach

### Notebook 01 - Exploratory Data Analysis

The first thing I noticed was the severe class imbalance - 94% of the dataset earns under $50K. This immediately ruled out accuracy as a metric and shaped all modeling decisions downstream.

Key findings from EDA:
- **Education** is the strongest categorical predictor - PhD/professional degree holders earn over $50K more than 50% of the time vs 3.9% for high school graduates
- **Investment income** is a surprisingly strong signal - only 2.7% of low earners have any capital gains vs 19.5% of high earners (7x difference)
- **Age** peaks between 35–55 for high earners and drops off sharply after 60
- **Four migration columns** are missing for ~50% of records - dropped entirely
- **Self-employed incorporated** workers earn over $50K at 34.7% - more than 5x the average

### Notebook 02 - Classification

Three models were trained and compared:

| Model | AUC-ROC | F1 (Over $50K) |
|---|---|---|
| Logistic Regression | 0.9460 | 0.4358 |
| Random Forest | 0.9402 | 0.4299 |
| **XGBoost** | **0.9546** | **0.4911** |

**XGBoost** was selected as the final model. Key decisions made during modeling:

- `scale_pos_weight=15` to handle the 94/6 class imbalance
- Two rounds of RandomizedSearchCV for hyperparameter tuning - both confirmed default parameters were optimal
- **Platt scaling calibration** applied to correct XGBoost's probability overconfidence on imbalanced data
- 5-fold cross validation confirmed stability (mean AUC 0.9519, std 0.0018)
- SHAP analysis used for explainability
- Fairness check performed across sex, race, and education groups

**Business impact:** At threshold 0.5, the model delivers a **10.5x improvement** over random targeting. For every 10,000 people contacted, the model identifies 6,527 high-income individuals compared to just 620 through untargeted campaigns.

**Top predictors (SHAP):**
1. Age (mean |SHAP| = 1.85)
2. Weeks worked in year (1.03)
3. Tax filer status - Nonfiler (0.58)
4. Sex - Male (0.40)
5. Dividends from stocks (0.28)

### Notebook 03 - Customer Segmentation

KMeans clustering was applied to discover natural marketing groups. Feature selection was evidence-based - every dropped column has a documented reason (sparsity, too many unique values, ethical concerns, or overlap with a better feature).

**Key preprocessing decisions:**
- Ordinal encoding for education to preserve natural ordering (Children=0 to Professional degree=16)
- Weeks worked binned into 4 groups to handle bimodal distribution
- Race excluded to prevent ethnicity-defined marketing segments
- Binary flags created for capital gains and dividends (raw values were 96% and 89% zero)

**k=5 selected** based on combined analysis of elbow method and silhouette scores (0.2717). The difference between k=5 and k=8 was only 0.016 - not significant enough to justify 8 separate marketing campaigns.

**The 5 segments discovered:**

| Segment | Size | Income >$50K | Who They Are |
|---|---|---|---|
| 0 | 19.7% | 9.3% | Working adults, female, clerical, high school |
| 1 | 27.6% | 0.0% | Children and dependents |
| 2 | 25.6% | 11.4% | Working adults, male, professional specialty |
| 3 | 3.7% | **32.7%** | High earner investors - 100% have capital gains |
| 4 | 23.4% | 1.1% | Retired/elderly, not working |

**Segment 3 is the premium marketing target** - despite being only 3.7% of the population, they earn over $50K at 5x the average rate and every single person in this segment has investment income.

---

## Streamlit App

The interactive app has two pages:

**Income Predictor**
- Enter any individual's demographic and employment details
- Get an instant income prediction with probability score
- Executive summary tells the client exactly whether to target this person
- Marketing threshold guide shows decisions across different campaign strategies

**Customer Segments**
- Visual overview of all 5 segments
- Income rate by segment chart
- Detailed profile and marketing recommendation for each segment

---

## Key Technical Highlights

- **Calibration:** Platt scaling corrected XGBoost's probability overconfidence - essential for threshold-based business decisions
- **SHAP:** Model predictions are fully explainable at the individual level
- **Fairness:** AUC checked across sex, race, and education groups - Female AUC 0.9423, Male AUC 0.9472
- **Evidence-based feature selection:** Every preprocessing decision in the segmentation notebook is backed by data (sparsity analysis, unique value counts)
- **Business framing:** Results translated into ROI metrics, threshold tables, and plain English marketing recommendations

---

## Results Summary

| Metric | Value |
|---|---|
| Classification AUC-ROC | 0.9546 |
| 5-fold CV Mean AUC | 0.9519 (std 0.0018) |
| Marketing ROI lift | 10.5x at threshold 0.5 |
| Segmentation silhouette score | 0.2717 |
| Number of segments | 5 |
| Premium segment income rate | 32.7% (5x average) |

---

## Dependencies

See `requirements.txt` for full list. Key packages:

- `xgboost==2.1.4`
- `scikit-learn==1.6.1`
- `pandas==2.3.3`
- `numpy==2.0.2`
- `shap`
- `streamlit`
- `matplotlib==3.9.4`
- `seaborn==0.13.2`

---
## Demo Link: https://share.vidyard.com/watch/xaqYCu7ZxMJEu1MxdwXra3
