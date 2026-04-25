# Titanic Survival Prediction
### AI Assignment 2 — Data Cleaning, Feature Engineering & Feature Selection

---

## Project Overview

This project builds a predictive pipeline for the Kaggle Titanic dataset. The goal is to determine which passengers were likely to survive the disaster using machine learning techniques. The work covers data cleaning, feature engineering, and feature selection — organized as both an exploratory notebook and modular Python scripts.

---

## Project Structure

```
titanic_assignment/
│
├── data/
│   ├── train.csv                  # Raw training data
│   ├── test.csv                   # Raw test data
│   ├── train_cleaned.csv          # Output of data_cleaning.py
│   └── train_engineered.csv       # Output of feature_engineering.py
│
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb   # Full exploration and analysis
│
├── scripts/
│   ├── data_cleaning.py           # Part 1: Cleans raw data
│   ├── feature_engineering.py     # Part 2: Creates new features
│   └── feature_selection.py       # Part 3: Selects best features
│
├── README.md
└── requirements.txt
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline (scripts)

```bash
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/feature_selection.py
```

### 3. Or explore interactively (notebook)

```bash
jupyter notebook notebooks/Titanic_Feature_Engineering.ipynb
```

---

## Approach

### Part 1 — Data Cleaning

| Column | Issue | Decision |
|--------|-------|----------|
| `Age` | ~177 missing values | Imputed with median |
| `Embarked` | 2 missing values | Imputed with mode (most frequent port) |
| `Cabin` | ~687 missing (~77%) | Extracted `Deck` letter, then dropped |
| `Fare` | Extreme outliers | Capped at 99th percentile |
| Duplicates | Checked | None found |

### Part 2 — Feature Engineering

New features created from existing columns:

| Feature | Description |
|---------|-------------|
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | 1 if travelling alone, 0 otherwise |
| `Title` | Extracted from passenger name (Mr, Mrs, Miss, Rare, etc.) |
| `Deck` | First letter of Cabin (A–G, T) |
| `AgeGroup` | Binned into Child, Teen, Adult, Senior |
| `FarePerPerson` | `Fare / FamilySize` |
| `Fare_log` | Log-transformed Fare to reduce skew |
| `Age_log` | Log-transformed Age |

Categorical features (`Sex`, `Embarked`, `Title`, `Deck`, `AgeGroup`) were one-hot encoded. `Pclass` was kept as ordinal (1, 2, 3).

### Part 3 — Feature Selection

- Correlation matrix used to identify and remove highly correlated features (threshold > 0.9)
- Random Forest used to rank features by importance
- Features with importance below 0.01 were dropped
- Final selected features include: `Sex_male`, `Pclass`, `Fare_log`, `Age`, `Title_Mr`, `FamilySize`, `IsAlone`, and select `Deck_*` columns

---

## Key Findings

- **Sex** is the strongest predictor of survival — female passengers had a much higher survival rate
- **Pclass** is the second strongest — first class passengers survived at a higher rate
- **Title** captures social status and correlates strongly with survival (e.g., `Title_Mr` has low survival odds)
- Passengers **travelling alone** had slightly lower survival rates than those in small family groups
- **Fare** (log-transformed) is positively correlated with survival, closely tied to Pclass
- The **Deck** feature, while sparse, provides some signal for higher-class passengers

---

## Requirements

See `requirements.txt`. Key libraries:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## Dataset Source

[Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)