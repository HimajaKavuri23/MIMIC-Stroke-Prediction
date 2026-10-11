# Prediction of Postoperative Stroke in Elderly Patients with CATBoost

Binary classification of postoperative stroke risk in elderly SICU patients, trained on MIMIC-IV and externally validated on MIMIC-III. CATBoost achieved AUROC 0.92 on the independent test cohort, outperforming the XGBoost benchmark from Zhang et al. (2022).

> **Accepted for presentation and publication at the 3rd IEOM World Congress, 2026.**

---

## Overview

Postoperative stroke is a rare but life-threatening complication in elderly surgical patients. It arises from a complex interplay of comorbidities, metabolic changes, and perioperative factors that classical risk scores often fail to capture. This project builds a machine learning pipeline using structured ICU data — labs, vitals, comorbidities, and clinical scores — to flag high-risk patients early, without requiring imaging.

The pipeline covers full data extraction from raw MIMIC tables, preprocessing with clinical imputation, statistical validation, a 4-step feature selection process, cross-validated model comparison across 10 algorithms, and an ablation study to assess feature robustness.

---

## Dataset

| Split | Source | Patients | Stroke Cases | Non-Stroke |
|---|---|---|---|---|
| Training | MIMIC-IV v1.0 | 7,129 | 1,355 (19.0%) | 5,774 |
| Independent Test | MIMIC-III | 681 | 62 | 619 |

**Cohort criteria:**
- Inclusion: Patients from the Surgical ICU (SICU)
- Exclusion: Age < 55 years; individuals with excessive missing data (test set)

**MIMIC tables used:** PATIENTS, ADMISSIONS, ICUSTAYS, DIAGNOSES_ICD, LABEVENTS, CHARTEVENTS

**Data sources:**
- [MIMIC-IV v1.0](https://physionet.org/content/mimiciv/1.0/) — training cohort
- [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) — independent external validation

> Access to both datasets requires a PhysioNet credentialed account and completion of required training.

---

## Pipeline

```
MIMIC-IV (SICU, age ≥55)     MIMIC-III (SICU, age ≥55)
  ↓                                 ↓
Data Extraction               Data Extraction
  ├── Stroke labels (ICD-9/10)
  ├── Comorbidities (11 conditions, ICD-mapped)
  ├── Substance use flags (8 categories)
  ├── Drug flags (4 classes, pre-op prescriptions)
  ├── Lab values (23 items, first 24h of ICU stay)
  └── Vitals (5 items, first 24h of ICU stay)
  ↓
Data Preprocessing
  ├── Remove duplicates
  ├── Remove rows with >20% missingness
  ├── Drop columns with >30% missingness
  ├── SVD-based imputation (IterativeImputer)
  ├── Normalization (StandardScaler)
  ├── One-hot encoding (categorical features)
  ├── Elixhauser comorbidity score
  └── SOFA score
  ↓
Statistical Validation
  ├── t-tests: continuous features by stroke status (train vs. test)
  └── Chi-square: categorical features by stroke status
  ↓
Feature Selection (4 steps, 108 → 20 features)
  ├── Step 1: RFECV (XGBoost, 5-fold CV) → 80 features
  ├── Step 2: Permutation Importance (Random Forest) → top 25
  ├── Step 3: SHAP Importance (Random Forest) → top 30
  └── Step 4: Intersection → 20 final features
  ↓
Modeling (10 classifiers, 5-fold CV + SMOTE + GridSearchCV)
  └── CATBoost selected as final model
```

---

## Stroke Labeling

Stroke cases identified by ICD codes in DIAGNOSES_ICD:

| Standard | Prefixes |
|---|---|
| ICD-9 | 430, 431, 433, 434, 435, 436 |
| ICD-10 | I60, I61, I63, I64 |

---

## Feature Engineering

### Comorbidities (ICD-mapped binary flags)

pvd, htn, copd, diabetes, renal, liver, pud, sepsis, chf, cancer, rheumatic

### Substance Use Flags (binary)

Alcohol, Tobacco, Opioid, Cannabis, Cocaine, Sedative, Other, Smoking

### Drug Flags (pre-op prescriptions, binary)

Anticoagulant, Aspirin, Insulin, Statin

### Laboratory Values (first 24h of ICU stay, aggregated as mean/max/min)

Anion Gap, Albumin, Bilirubin, Creatinine, Hematocrit, Hemoglobin, WBC, Lactate, Potassium, PTT, PT, INR, Platelet Count, Urea Nitrogen, Sodium, Calcium, Magnesium, Phosphate, Chloride, AST, CRP, Troponin, eGFR

### Vitals (first 24h of ICU stay, aggregated as mean/max/min)

Heart Rate (220045), Systolic BP (220179), Diastolic BP (220180), Mean BP (220181), SpO2 (220277)

### Composite Clinical Scores

**Elixhauser Comorbidity Score** — weighted sum across 17 comorbidity categories (e.g., CHF=7, Cancer=14, Liver=11, Diabetes=5, Renal=5, COPD=3, PVD=2, HTN=−1)

**SOFA Score** — derived from SpO2 (respiratory), Platelet Count (coagulation), Mean Arterial Pressure (cardiovascular), Creatinine (renal)

---

## Preprocessing

| Step | Method |
|---|---|
| Duplicate removal | Drop duplicate patient records |
| Row filtering | Remove rows with >20% missing values |
| Column filtering | Drop columns with >30% missing (including Hemoglobin, Lactate, AST, Albumin, Troponin, CRP) |
| Imputation | IterativeImputer (SVD-based multivariate imputation) |
| Scaling | StandardScaler applied to continuous features |
| Encoding | OneHotEncoder for categorical features |
| Fit/transform split | All transformers fit on training set only; applied to both sets |

Final encoded dataset: 108 features per patient (before feature selection).

---

## Statistical Validation

**Continuous features — Welch's two-sample t-test (stroke vs. non-stroke):**
Most variables showed non-significant differences at α=0.05, confirming the classification challenge. `last_careunit` distribution and `gender` were significant (p < 0.01).

**Categorical features — Chi-square test:**
- `gender` (p = 0.0067): females more prevalent in stroke group (51.2% vs. 46.9%)
- `last_careunit` distributions (p < 0.0001): stroke patients discharged to neuro units at higher rates
- `insurance` type (p = 0.0719): not significant

**Train vs. test distributional comparison (t-tests):**
Significant differences found in `los`, substance use flags (Alcohol, Tobacco, Opioid), `sepsis`, `cancer`, and metabolic markers (`mean_Urea_Nitrogen`, `mean_chloride`), motivating external validation as a true generalizability check rather than a hold-out split.

---

## Feature Selection

Four-step funnel from 108 starting features to 20 final features:

| Step | Method | Output |
|---|---|---|
| 1 | RFECV with XGBClassifier (5-fold CV) | 80 features (rank=1) |
| 2 | Permutation Importance with Random Forest | Top 25 |
| 3 | SHAP TreeExplainer with Random Forest | Top 30 |
| 4 | Intersection of steps 2 & 3 | **20 final features** |

### Final 20 Features

| Category | Features |
|---|---|
| Demographics | age, los, gender |
| Comorbidities | cancer, liver, chf, htn |
| Lab values | Calcium, Potassium, Platelet Count, PTT, WBC, Phosphate, Chloride, Anion Gap, Urea Nitrogen |
| Vital signs | systolic_BP, SpO2, Heart_Rate |
| Composite scores | Elixhauser_score, SOFA_score |

**SHAP interpretation:** The strongest predictors are Elixhauser score, max_Calcium, max_Potassium, blood pressure extremes, and SOFA score. Higher values in these features increase model-predicted stroke risk.

---

## Modeling

10 classifiers trained with 5-fold cross-validation, SMOTE for class imbalance, and GridSearchCV for hyperparameter tuning:

Logistic Regression, Naive Bayes, SVM (RBF), KNN, MLP, GBDT, Decision Tree, Random Forest, XGBoost, LightGBM, CATBoost

### Results — Training Set (5-fold CV AUROC)

| Model | AUROC | Accuracy | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Random Forest | 0.82 (0.81–0.83) | 0.82 | 0.49 | 0.48 | 0.90 |
| XGBoost | 0.81 (0.80–0.83) | 0.53 | 0.42 | 0.94 | 0.44 |
| **CATBoost** | **0.824 (0.81–0.84)** | **0.822** | **0.48** | **0.44** | **0.91** |
| LightGBM | 0.82 (0.81–0.84) | 0.77 | 0.52 | 0.69 | 0.78 |

### Results — Independent Test Set (MIMIC-III)

| Model | AUROC | Accuracy | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Random Forest | 0.94 (0.933–0.941) | 0.89 | 0.71 | 0.65 | 0.95 |
| XGBoost | 0.89 (0.87–0.90) | 0.63 | 0.53 | 0.95 | 0.54 |
| **CATBoost** | **0.91 (0.89–0.92)** | **0.89** | **0.66** | **0.66** | **0.91** |
| LightGBM | 0.89 (0.87–0.90) | 0.80 | 0.65 | 0.86 | 0.79 |

---

## Proposed Model: CATBoost

CATBoost was selected as the final model for its best combination of discrimination and calibration on the independent test set:

| Comparison | AUROC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| **Our work (CATBoost)** | **0.92 (0.89–0.93)** | **0.89** | **0.66** | **0.91** |
| Zhang et al. 2022 (XGBoost) | 0.83 (0.79–0.87) | 0.83 | 0.97 | 0.30 |

CATBoost achieves a more calibrated decision boundary than Zhang et al.'s XGBoost, which flagged nearly every patient as high-risk (sensitivity 0.97, specificity 0.30). The improvement in specificity reduces false positives and makes the model more clinically applicable.

---

## Ablation Study

Leave-One-Feature-Out (LOFO) ablation on the 20-feature CATBoost model:
- **Baseline AUC: 0.904**
- Removing any individual feature caused minimal change in AUC
- No single feature is indispensable — the model distributes predictive signal across multiple correlated clinical variables
- Reflects the multifactorial nature of postoperative stroke risk

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | MIMIC-IV v1.0, MIMIC-III (PhysioNet) |
| EHR Processing | pandas, numpy |
| Imputation | scikit-learn IterativeImputer |
| Scaling / Encoding | scikit-learn StandardScaler, OneHotEncoder |
| Class Balancing | SMOTE (imbalanced-learn) |
| Feature Selection | RFECV (XGBClassifier), Permutation Importance, SHAP (TreeExplainer) |
| Hyperparameter Tuning | GridSearchCV |
| Statistical Tests | scipy.stats (ttest_ind, chi2_contingency) |
| Classification | CATBoost, XGBoost, LightGBM, scikit-learn |
| Environment | USC HPC cluster (CARC) |

---

## Repository Contents

```
mimic-stroke-prediction/
├── MIMIC_IV_data_extraction.ipynb      # MIMIC-IV cohort: ICD labeling, labs, vitals, comorbidities
├── MIMIC_III_data_extraction.ipynb     # MIMIC-III test set extraction
├── MIMIC_IV_data_preprocess.ipynb      # Imputation, scaling, encoding, Elixhauser/SOFA scores
├── Statistical_tests.ipynb             # t-tests and chi-square; train/test distributional shift
├── MIMIC_IV_feature_selection.ipynb    # RFECV → Permutation → SHAP → 20 final features
└── ISE_535_final_presentation.pdf      # Project presentation slides
```

> The MIMIC datasets are not included (PhysioNet credentialed access required). Notebooks reference data paths under `/project2/pishgar_1407/` and `/home1/bingi/` on the USC CARC HPC cluster.

---

## References

- Zhang, X., Fei, N., Zhang, X., Wang, Q., & Fang, Z. (2022). Machine Learning Prediction Models for Postoperative Stroke in Elderly Patients: Analyses of the MIMIC Database. *Frontiers in Aging Neuroscience*, 14, 897611. https://doi.org/10.3389/fnagi.2022.897611
- Li et al. (2025). Predicting Postoperative Stroke in Elderly SICU Patients: An Interpretable Machine Learning Model Using MIMIC Data. https://arxiv.org/abs/2506.03209

---

## Course

**ISE 535: Advanced Topics in Data Analytics** — Spring 2026  
USC Viterbi School of Engineering  
Instructor: Prof. Maryam Pishgar

---

## Authors

Harshitha Bingi · Abhijeet Nijjar · Himaja Kavuri  
USC Viterbi School of Engineering

Himaja Kavuri — MS Business Analytics, USC Marshall School of Business  
Portfolio: [himajakavuri23.github.io](https://himajakavuri23.github.io)  
LinkedIn: [linkedin.com/in/himaja-kavuri](https://linkedin.com/in/himaja-kavuri)
