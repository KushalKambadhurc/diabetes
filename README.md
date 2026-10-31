# Diabetes Risk Prediction Using Health Indicators
## UN Sustainable Development Goal 3: Good Health and Well-Being (Target 3.4)
### Methodological Triad: Statistical Hypothesis Testing, Multivariable Regression, and Discriminant Analysis

![Project Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![SDG Goal](https://img.shields.io/badge/UN%20SDG-3%20Good%20Health-red)
![Cohort Size](https://img.shields.io/badge/Sample%20Size-100%2C000%20Patients-purple)

---

## 📌 Executive Summary & SDG 3 Target 3.4 Alignment

Diabetes mellitus is a pervasive global health emergency. According to the International Diabetes Federation (IDF), over 537 million individuals worldwide suffer from diabetes, and nearly half remain undiagnosed until life-threatening vascular complications manifest.

This project directly addresses **United Nations Sustainable Development Goal 3 (Good Health and Well-being)**, specifically **Target 3.4**:
> *"By 2030, reduce by one third premature mortality from non-communicable diseases through prevention and treatment and promote mental health and well-being."*

By leveraging routine, non-invasive biometric and demographic health indicators, this project develops a robust statistical and predictive triage engine capable of early diabetes identification in low-resource and high-throughput clinical settings.

---

## 🔬 Core Methodological Components

### 1. Statistical Hypothesis Testing
- **Continuous Biometric Indicators** (`HbA1c_level`, `blood_glucose_level`, `bmi`, `age`):
  - Parametric **Welch's Two-Sample $t$-Test** (robust to variance heterogeneity).
  - Non-parametric **Mann-Whitney $U$ Test** (rank-based distributional invariance).
  - Standardized effect size estimation via **Cohen's $d$**.
- **Categorical Comorbidities & Lifestyle Factors** (`hypertension`, `heart_disease`, `smoking_history`, `gender`):
  - **Pearson's Chi-Square ($\chi^2$) Test of Independence**.
  - Association effect size quantification via **Cramér's $V$**.
- **Age Stratification Variance Analysis**:
  - **One-Way ANOVA ($F$-test)** and **Kruskal-Wallis ($H$-test)** across four age cohorts ($<30$, $30-45$, $45-60$, $60+$).

### 2. Regression Modeling
- **Multivariable Binary Logistic Regression (Logit)**:
  - Estimation of parameter coefficients ($\beta_j$), standard errors, and $z$-statistics.
  - Calculation of **Adjusted Odds Ratios ($\text{OR} = e^{\beta_j}$)** with **95% Confidence Intervals**.
  - Goodness-of-fit evaluation: **McFadden's Pseudo-$R^2$ (0.6109)**, **Nagelkerke Pseudo-$R^2$ (0.6781)**, **Cox & Snell Pseudo-$R^2$ (0.2991)**, Log-Likelihood, AIC, and BIC.
  - Multicollinearity diagnostics via **Variance Inflation Factor (VIF)**.
- **Multiple Linear Regression (OLS)**:
  - Modeling continuous fasting/casual blood glucose levels from routine indicators, testing residual normality, homoscedasticity, and explanatory power.

### 3. Discriminant Analysis (LDA & QDA)
- **Linear Discriminant Analysis (LDA)**:
  - Extraction of the canonical discriminant function ($\text{LD1}$) maximizing the between-class to within-class scatter ratio.
  - Formal testing of discriminatory significance via **Wilks' Lambda ($\Lambda$)** and **Bartlett's $\chi^2$ approximation**.
  - Computation of **Standardized Canonical Discriminant Coefficients** to rank feature contributions.
- **Quadratic Discriminant Analysis (QDA)**:
  - Evaluation of class-specific covariance matrix determinants ($\det(\mathbf{\Sigma}_1)/\det(\mathbf{\Sigma}_0) = 36.49$), formally proving covariance heterogeneity and justifying non-linear quadratic decision boundaries.

### 4. Comprehensive Machine Learning Evaluation & Medical Screening
- Comparison of **Logistic Regression**, **LDA**, **QDA**, **Random Forest (Balanced)**, and **Gradient Boosting**.
- Diagnostic evaluation tailored for clinical screening: **Sensitivity / Recall** (minimizing False Negatives), **Specificity**, **Precision**, **F1-Score**, **ROC-AUC**, and **PR-AUC (Average Precision)**.
- Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.

### 5. Multi-Cohort Validation on Benchmark Dataset
- Secondary external validation on the **Pima Indians Diabetes Dataset** ($N = 768$) to evaluate model transferability across physiological dimensions (Insulin, Skinfold Thickness, Diabetes Pedigree Function).

---

## 📊 Summary of Statistical & Empirical Findings

### Key Hypothesis Testing Results
| Health Indicator | Non-Diabetic Mean (SD) | Diabetic Mean (SD) | Difference | Welch $t$-Stat | $p$-value | Cohen's $d$ | Decision ($\alpha=0.01$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **HbA1c Level (%)** | 5.40 (0.87) | 6.93 (1.08) | **+1.53%** | 246.72 | $< 10^{-10}$ | **1.63** (Huge) | Reject $H_0$ |
| **Blood Glucose (mg/dL)** | 132.85 (34.08) | 194.10 (58.84) | **+61.25** | 94.64 | $< 10^{-10}$ | **1.61** (Huge) | Reject $H_0$ |
| **Body Mass Index (BMI)** | 26.88 (6.34) | 31.98 (7.62) | **+5.10** | 59.73 | $< 10^{-10}$ | **0.77** (Medium-Large) | Reject $H_0$ |
| **Age (Years)** | 40.11 (22.01) | 60.98 (14.56) | **+20.87** | 120.30 | $< 10^{-10}$ | **0.99** (Large) | Reject $H_0$ |

| Comorbid / Lifestyle Factor | Pearson $\chi^2$ Statistic | d.f. | $p$-value | Cramér's $V$ | Decision ($\alpha=0.01$) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Hypertension** | 3912.44 | 1 | $< 10^{-10}$ | 0.1978 | Reject $H_0$ (Significant Association) |
| **Heart Disease** | 2990.23 | 1 | $< 10^{-10}$ | 0.1729 | Reject $H_0$ (Significant Association) |
| **Smoking History** | 2174.41 | 5 | $< 10^{-10}$ | 0.1475 | Reject $H_0$ (Significant Association) |
| **Gender** | 186.29 | 1 | $< 10^{-10}$ | 0.0432 | Reject $H_0$ (Significant Association) |

---

### Multivariable Logistic Regression Odds Ratios (Adjusted)
| Predictor Variable | Adjusted Odds Ratio ($\text{OR}$) | 95% Confidence Interval | $z$-statistic | $p$-value | Clinical Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **HbA1c Level** | **2.378** | [2.317, 2.441] | 63.81 | $< 10^{-10}$ | +137.8% odds per 1.0% increase |
| **Hypertension** | **2.051** | [1.890, 2.226] | 16.78 | $< 10^{-10}$ | >2x odds in hypertensive individuals |
| **Heart Disease** | **1.708** | [1.530, 1.907] | 9.29 | $< 10^{-10}$ | +70.8% odds in patients with cardiac disease |
| **Body Mass Index (BMI)**| **1.093** | [1.088, 1.098] | 38.31 | $< 10^{-10}$ | +9.3% odds per unit BMI (+56% per 5 kg/m²) |
| **Blood Glucose** | **1.034** | [1.033, 1.035] | 67.43 | $< 10^{-10}$ | +3.4% odds per mg/dL (+39.6% per 10 mg/dL) |
| **Current Smoker** | **2.076** | [1.822, 2.365] | 10.92 | $< 10^{-10}$ | >2x odds compared to unrecorded baseline |
| **Former Smoker** | **1.863** | [1.657, 2.094] | 10.23 | $< 10^{-10}$ | +86.3% odds compared to baseline |

*Multicollinearity Assessment*: All Variance Inflation Factors (VIF) range between **1.02 and 1.37**, demonstrating total absence of collinear distortion.

---

### Canonical Discriminant Analysis Summary
- **Wilks' Lambda**: $\Lambda = 0.6528$
- **Bartlett's $\chi^2$ Test**: $\chi^2 = 34,114.03$ ($\text{df} = 6, p < 10^{-10}$)
- **Standardized Canonical Discriminant Coefficients (Weights on LD1)**:
  1. `blood_glucose_level`: $+0.6960$
  2. `HbA1c_level`: $+0.6598$
  3. `age`: $+0.2342$
  4. `bmi`: $+0.2136$
  5. `hypertension`: $+0.1909$
  6. `heart_disease`: $+0.1772$
- **Class Covariance Determinant Ratio**: $\det(\mathbf{\Sigma}_1) / \det(\mathbf{\Sigma}_0) = 36.49$, validating that healthy individuals exhibit tight glycemic clustering whereas diabetic patients exhibit wide dispersion, satisfying the prerequisite for Quadratic Discriminant Analysis.

---

### Model Performance Matrix (Primary Population: 100,000 Records)
| Model | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **97.21%** | 68.88% | **99.85%** | **97.66%** | **0.8079** | **0.9796** | **0.8868** |
| **Random Forest (Balanced)** | 91.21% | **90.82%** | 91.24% | 49.08% | 0.6372 | 0.9774 | 0.8809 |
| **Logistic Regression (Balanced)** | 88.82% | **88.94%** | 88.81% | 42.47% | 0.5749 | 0.9626 | 0.8231 |
| **Linear Discriminant Analysis (LDA)** | 95.73% | 59.88% | 99.07% | 85.62% | 0.7047 | 0.9534 | 0.7875 |
| **Quadratic Discriminant Analysis (QDA)** | 90.45% | 66.24% | 92.70% | 45.74% | 0.5411 | 0.9142 | 0.5187 |

---

### Cross-Cohort Benchmark Validation (Pima Indians Dataset: 768 Records)
| Model | Accuracy | Recall (Sensitivity) | Precision | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **75.32%** | 55.56% | **68.18%** | 0.6122 | **0.8300** |
| **Random Forest (Balanced)** | 72.73% | **75.93%** | 58.57% | **0.6613** | 0.8226 |
| **Logistic Regression (Balanced)** | 73.38% | 70.37% | 60.32% | 0.6496 | 0.8126 |
| **Linear Discriminant Analysis (LDA)**| 70.13% | 48.15% | 59.09% | 0.5306 | 0.8126 |
| **Quadratic Discriminant Analysis (QDA)**| 70.13% | 55.56% | 57.69% | 0.5660 | 0.7494 |

---

## 📈 Visualizations Catalog

All publication-quality figures are generated and stored in `output_visualizations/`:
1. `fig1_univariate_health_indicators.png`: Univariate biometric distributions (HbA1c, Glucose, BMI, Age) by diagnosis.
2. `fig2_categorical_risk_factors.png`: Proportionate stack charts for hypertension, heart disease, and smoking categories.
3. `fig3_discriminant_analysis.png`: Canonical Linear Discriminant (LD1) density projection and standardized feature weights.
4. `fig4_logistic_regression_forest_plot.png`: Forest plot of adjusted Odds Ratios with 95% Confidence Intervals.
5. `fig5_roc_and_pr_curves.png`: Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves across all models.
6. `fig6_cross_cohort_comparison.png`: ROC-AUC performance comparison across Primary Population and Pima Indians cohorts.

---

## 🏛️ Health Policy & SDG 3 Actionable Recommendations

1. **Two-Tier Community Screening Architecture**:
   - **Tier 1 (Non-Invasive Community Triage)**: Health posts and mobile clinics collect Age, BMI, Blood Pressure, and Smoking History.
   - **Tier 2 (Point-of-Care Confirmatory Testing)**: Patients categorized above the 75th percentile of risk receive rapid capillary HbA1c screening.
2. **Clinical Decision Thresholding**:
   - For population screening where missing an affected patient incurs severe health costs, deploy **Random Forest (Balanced)** or **Logistic Regression (Balanced)** to achieve **$\ge 90\%$ Sensitivity**.
   - For automated telemedicine triage with high caseloads, deploy **Gradient Boosting** to maintain **$99.8\%$ Specificity** and avoid overwhelming secondary diagnostic facilities.
3. **Preventive Lifestyle Interventions**:
   - Given that BMI elevates risk by $+56\%$ per 5 kg/m² and active smoking doubles risk ($\text{OR} = 2.076$), targeted behavioral and dietary interventions directly contribute to halting premature mortality under Target 3.4.

---

## 🚀 Execution Guide

### Environment Setup
```bash
# Clone or open repository
cd d:\diabetes

# Activate virtual environment
.\.venv\Scripts\activate

# Install dependencies (already pinned in requirements.txt)
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Execute the complete statistical pipeline and generate all figures
python diabetes_analytics.py

# Launch interactive Jupyter Notebook
jupyter notebook model.ipynb
```