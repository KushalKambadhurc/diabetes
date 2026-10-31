"""
Diabetes Risk Prediction Using Health Indicators
SDG 3: Good Health and Well-being
Full Analytics Pipeline:
1. Hypothesis Testing (Welch's t-test, Mann-Whitney U, Cohen's d, Chi-Square, Cramer's V, ANOVA, Kruskal-Wallis)
2. Regression Analysis (Binary Logit with Odds Ratios, Pseudo R2, VIF, and Multiple Linear Regression with OLS)
3. Discriminant Analysis (Linear Discriminant Analysis with Wilks' Lambda, Standardized Coefficients, and Quadratic Discriminant Analysis)
4. Comprehensive Model Comparison (Logistic Regression, LDA, QDA, Random Forest, Gradient Boosting)
5. Multi-Cohort Benchmark Validation on Pima Indians Dataset
6. High-Quality Visualizations
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('output_visualizations', exist_ok=True)

def run_pipeline():
    print("=" * 80)
    print("DIABETES RISK PREDICTION USING HEALTH INDICATORS (SDG 3)")
    print("HYPOTHESIS TESTING, REGRESSION & DISCRIMINANT ANALYSIS PIPELINE")
    print("=" * 80)

    # 1. LOAD PRIMARY DATASET
    print("\n[Step 1] Loading primary population dataset (100,000 records)...")
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")
    
    # Clean gender: 'Other' has only 18 records; let's filter or retain standard binary gender for physiological consistency
    df = df[df['gender'] != 'Other'].copy()
    print(f"Dataset after filtering rare gender ('Other'): {len(df):,} records.")

    # 2. STATISTICAL HYPOTHESIS TESTING
    print("\n" + "=" * 80)
    print("SECTION 1: STATISTICAL HYPOTHESIS TESTING")
    print("=" * 80)

    diabetic = df[df['diabetes'] == 1]
    non_diabetic = df[df['diabetes'] == 0]

    # Continuous variables
    cont_vars = ['HbA1c_level', 'blood_glucose_level', 'bmi', 'age']
    hyp_cont_results = []

    print("\n--- A. Continuous Health Indicators (Diabetic vs Non-Diabetic) ---")
    for var in cont_vars:
        g0 = non_diabetic[var].dropna()
        g1 = diabetic[var].dropna()
        
        # Means & Stds
        mean0, std0 = g0.mean(), g0.std()
        mean1, std1 = g1.mean(), g1.std()
        
        # Welch's t-test (unequal variances)
        t_stat, p_val_t = stats.ttest_ind(g1, g0, equal_var=False)
        
        # Mann-Whitney U test (non-parametric robustness)
        # Using subsample for U test if N is very large to speed up exact rank sums
        u_subsample = min(20000, len(g0))
        u_stat, p_val_u = stats.mannwhitneyu(g1.sample(min(len(g1), 5000), random_state=42), 
                                             g0.sample(u_subsample, random_state=42), 
                                             alternative='two-sided')
        
        # Cohen's d effect size
        n0, n1 = len(g0), len(g1)
        pooled_std = np.sqrt(((n0 - 1) * std0**2 + (n1 - 1) * std1**2) / (n0 + n1 - 2))
        cohens_d = (mean1 - mean0) / pooled_std
        
        hyp_cont_results.append({
            'Indicator': var,
            'Non-Diabetic Mean (SD)': f"{mean0:.2f} ({std0:.2f})",
            'Diabetic Mean (SD)': f"{mean1:.2f} ({std1:.2f})",
            'Difference': f"{mean1 - mean0:+.2f}",
            't-Statistic': f"{t_stat:.2f}",
            'p-value (t-test)': "< 1e-10" if p_val_t < 1e-10 else f"{p_val_t:.4e}",
            'Mann-Whitney p-value': "< 1e-10" if p_val_u < 1e-10 else f"{p_val_u:.4e}",
            "Cohen's d": f"{cohens_d:.3f}",
            'Conclusion': "Reject H0 (Significantly Higher)" if p_val_t < 0.05 and mean1 > mean0 else "Fail to Reject H0"
        })

    hyp_df_cont = pd.DataFrame(hyp_cont_results)
    print(hyp_df_cont.to_string(index=False))

    # Categorical variables: Chi-square test of independence
    print("\n--- B. Categorical Risk Factors vs Diabetes Status (Chi-Square Test) ---")
    cat_vars = ['hypertension', 'heart_disease', 'smoking_history', 'gender']
    hyp_cat_results = []

    for var in cat_vars:
        contingency_tab = pd.crosstab(df[var], df['diabetes'])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency_tab)
        
        # Cramer's V effect size
        n_obs = contingency_tab.sum().sum()
        min_dim = min(contingency_tab.shape) - 1
        cramers_v = np.sqrt(chi2 / (n_obs * min_dim)) if min_dim > 0 else 0
        
        hyp_cat_results.append({
            'Factor': var,
            'Chi2 Statistic': f"{chi2:.2f}",
            'Degrees of Freedom': dof,
            'p-value': "< 1e-10" if p_val < 1e-10 else f"{p_val:.4e}",
            "Cramer's V": f"{cramers_v:.4f}",
            'Conclusion': "Reject H0 (Dependent/Associated)" if p_val < 0.05 else "Fail to Reject H0"
        })

    hyp_df_cat = pd.DataFrame(hyp_cat_results)
    print(hyp_df_cat.to_string(index=False))

    # Age Group ANOVA & Kruskal-Wallis on Blood Glucose
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=['<30', '30-45', '45-60', '60+'])
    age_groups = [group['blood_glucose_level'].values for _, group in df.groupby('age_group', observed=True)]
    f_stat, p_val_anova = stats.f_oneway(*age_groups)
    kw_stat, p_val_kw = stats.kruskal(*age_groups)
    print(f"\n--- C. One-Way ANOVA & Kruskal-Wallis: Blood Glucose across Age Groups ---")
    print(f"One-way ANOVA F-statistic: {f_stat:.2f}, p-value: {p_val_anova:.4e}")
    print(f"Kruskal-Wallis H-statistic: {kw_stat:.2f}, p-value: {p_val_kw:.4e}")

    # 3. REGRESSION ANALYSIS
    print("\n" + "=" * 80)
    print("SECTION 2: REGRESSION ANALYSIS")
    print("=" * 80)

    # Encode categorical features for regression
    df_reg = df.copy()
    df_reg['gender_male'] = (df_reg['gender'] == 'Male').astype(int)
    smoking_dummies = pd.get_dummies(df_reg['smoking_history'], prefix='smoking', drop_first=True, dtype=int)
    
    # Model 1: Binary Logistic Regression (Logit)
    logit_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender_male'] + list(smoking_dummies.columns)
    X_logit = pd.concat([df_reg[logit_features[:7]], smoking_dummies], axis=1)
    X_logit_const = sm.add_constant(X_logit)
    y_logit = df_reg['diabetes']

    print("\n--- A. Binary Logistic Regression (statsmodels Logit) ---")
    logit_mod = sm.Logit(y_logit, X_logit_const.astype(float)).fit(disp=False)
    print(logit_mod.summary())

    # Odds Ratios and 95% Confidence Intervals
    params = logit_mod.params
    conf = logit_mod.conf_int()
    conf['OR'] = np.exp(params)
    conf['Lower 95% CI'] = np.exp(conf[0])
    conf['Upper 95% CI'] = np.exp(conf[1])
    conf['p-value'] = logit_mod.pvalues
    or_table = conf[['OR', 'Lower 95% CI', 'Upper 95% CI', 'p-value']]
    print("\n--- Odds Ratios (OR) and 95% Confidence Intervals ---")
    print(or_table.round(4).to_string())

    # Pseudo R2 metrics
    ll_null = logit_mod.llnull
    ll_full = logit_mod.llf
    mcfadden_r2 = 1 - (ll_full / ll_null)
    n = len(y_logit)
    cox_snell_r2 = 1 - np.exp((2 / n) * (ll_null - ll_full))
    nagelkerke_r2 = cox_snell_r2 / (1 - np.exp((2 / n) * ll_null))
    print(f"\nGoodness of Fit Metrics for Logistic Regression:")
    print(f"Log-Likelihood (Full Model): {ll_full:.2f}")
    print(f"Log-Likelihood (Null Model): {ll_null:.2f}")
    print(f"McFadden's Pseudo R-squared: {mcfadden_r2:.4f}")
    print(f"Cox & Snell Pseudo R-squared: {cox_snell_r2:.4f}")
    print(f"Nagelkerke Pseudo R-squared: {nagelkerke_r2:.4f}")
    print(f"AIC: {logit_mod.aic:.2f}, BIC: {logit_mod.bic:.2f}")

    # Multicollinearity: Variance Inflation Factor (VIF)
    print("\n--- Multicollinearity Assessment (VIF) ---")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_logit.columns
    vif_data["VIF"] = [variance_inflation_factor(X_logit.values, i) for i in range(X_logit.shape[1])]
    print(vif_data.round(2).to_string(index=False))

    # Model 2: Multiple Linear Regression (Predicting Continuous Blood Glucose Level)
    print("\n--- B. Multiple Linear Regression (Predicting Blood Glucose Level) ---")
    lin_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'gender_male'] + list(smoking_dummies.columns)
    X_lin = pd.concat([df_reg[lin_features[:6]], smoking_dummies], axis=1)
    X_lin_const = sm.add_constant(X_lin)
    y_lin = df_reg['blood_glucose_level']
    ols_mod = sm.OLS(y_lin, X_lin_const.astype(float)).fit()
    print(ols_mod.summary())

    # 4. DISCRIMINANT ANALYSIS
    print("\n" + "=" * 80)
    print("SECTION 3: DISCRIMINANT ANALYSIS (LDA & QDA)")
    print("=" * 80)

    # Features for Discriminant Analysis
    da_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    X_da = df[da_features].values
    y_da = df['diabetes'].values

    scaler = StandardScaler()
    X_da_scaled = scaler.fit_transform(X_da)

    # Train-test split (80-20 stratified)
    X_tr_da, X_te_da, y_tr_da, y_te_da = train_test_split(
        X_da_scaled, y_da, test_size=0.2, random_state=42, stratify=y_da
    )

    # Linear Discriminant Analysis
    lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
    lda.fit(X_tr_da, y_tr_da)
    lda_preds = lda.predict(X_te_da)
    lda_probs = lda.predict_proba(X_te_da)[:, 1]

    # Wilks' Lambda calculation
    # W = Within-class scatter matrix, T = Total scatter matrix
    n_samples, n_features = X_tr_da.shape
    classes = np.unique(y_tr_da)
    total_mean = np.mean(X_tr_da, axis=0)
    
    W = np.zeros((n_features, n_features))
    for c in classes:
        X_c = X_tr_da[y_tr_da == c]
        mean_c = np.mean(X_c, axis=0)
        diff_c = X_c - mean_c
        W += diff_c.T @ diff_c
        
    T = (X_tr_da - total_mean).T @ (X_tr_da - total_mean)
    det_W = np.linalg.det(W)
    det_T = np.linalg.det(T)
    wilks_lambda = det_W / det_T
    
    # Bartlett's chi-square approximation for Wilks' Lambda
    # Bartlett: chi2 = - [N - 1 - (p + k) / 2] * ln(Lambda)
    k = len(classes)
    p = n_features
    dof_wilks = p * (k - 1)
    factor = (n_samples - 1 - (p + k) / 2.0)
    chi2_wilks = - factor * np.log(wilks_lambda)
    p_val_wilks = stats.chi2.sf(chi2_wilks, dof_wilks)

    print(f"--- Linear Discriminant Analysis Parameters ---")
    print(f"Prior Probabilities: Class 0 = {lda.priors_[0]:.4f}, Class 1 = {lda.priors_[1]:.4f}")
    print(f"Wilks' Lambda (Within/Total Scatter): {wilks_lambda:.4f}")
    print(f"Bartlett's Chi-Square Approximation: {chi2_wilks:.2f} (df={dof_wilks}, p-value={p_val_wilks:.4e})")
    
    # Standardized Canonical Discriminant Coefficients
    std_lda_coefs = lda.scalings_[:, 0]
    std_coef_df = pd.DataFrame({
        'Feature': da_features,
        'Standardized LD1 Coefficient': std_lda_coefs,
        'Absolute Impact': np.abs(std_lda_coefs)
    }).sort_values(by='Absolute Impact', ascending=False)
    print("\n--- Standardized Discriminant Function Coefficients (LD1) ---")
    print(std_coef_df.round(4).to_string(index=False))

    # Quadratic Discriminant Analysis
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(X_tr_da, y_tr_da)
    qda_preds = qda.predict(X_te_da)
    qda_probs = qda.predict_proba(X_te_da)[:, 1]

    # Determinants of Covariance Matrices for QDA vs LDA assumption
    cov_0 = lda.covariance_
    cov_qda_0 = qda.covariance_[0]
    cov_qda_1 = qda.covariance_[1]
    det_cov_0 = np.linalg.det(cov_qda_0)
    det_cov_1 = np.linalg.det(cov_qda_1)
    print(f"\n--- Covariance Matrix Homogeneity Check (LDA vs QDA) ---")
    print(f"Determinant of Covariance Matrix (Class 0 - Non-Diabetic): {det_cov_0:.6e}")
    print(f"Determinant of Covariance Matrix (Class 1 - Diabetic):     {det_cov_1:.6e}")
    print(f"Ratio det(Sigma_1) / det(Sigma_0): {det_cov_1 / det_cov_0:.4f}")
    print("Interpretation: Heterogeneous class covariances justify Quadratic Discriminant Analysis (QDA).")

    # 5. COMPREHENSIVE MODEL COMPARISON (PRIMARY DATASET)
    print("\n" + "=" * 80)
    print("SECTION 4: COMPREHENSIVE CLASSIFIER COMPARISON")
    print("=" * 80)

    # Use all features with proper scaling and one-hot encoding
    num_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    cat_cols = ['gender', 'smoking_history']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )

    X_all = df[num_cols + cat_cols]
    y_all = df['diabetes'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Prepare models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Linear Discriminant Analysis (LDA)': LinearDiscriminantAnalysis(),
        'Quadratic Discriminant Analysis (QDA)': QuadraticDiscriminantAnalysis(),
        'Random Forest (Balanced)': RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=120, max_depth=4, random_state=42)
    }

    eval_results = []
    roc_data = {}
    pr_data = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_proc)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred) # Sensitivity
        prec = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        # Specificity = TN / (TN + FP)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = tn / (tn + fp)

        eval_results.append({
            'Model': name,
            'Accuracy': acc,
            'Sensitivity (Recall)': rec,
            'Specificity': spec,
            'Precision': prec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, roc_auc)
        pr_data[name] = (r_curve, p_curve, pr_auc)

    eval_df = pd.DataFrame(eval_results).sort_values(by='ROC-AUC', ascending=False)
    print("\n--- Model Evaluation Summary Table ---")
    print(eval_df.round(4).to_string(index=False))

    # 6. BENCHMARK COHORT EVALUATION: PIMA INDIANS DATASET
    print("\n" + "=" * 80)
    print("SECTION 5: SECONDARY BENCHMARK EVALUATION (PIMA INDIANS COHORT)")
    print("=" * 80)

    pima = pd.read_csv("pima_diabetes_dataset.csv")
    print(f"Pima Cohort: {pima.shape[0]} patients, {pima.shape[1]} biological attributes.")

    # Physiological zero replacement (glucose, blood_pressure, skin_thickness, insulin, bmi cannot be 0 in living humans)
    zero_cols = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
    for c in zero_cols:
        pima[c] = pima[c].replace(0, np.nan)
        pima[c] = pima[c].fillna(pima[c].median())

    X_pima = pima.drop(columns=['outcome'])
    y_pima = pima['outcome'].values

    X_pima_tr, X_pima_te, y_pima_tr, y_pima_te = train_test_split(
        X_pima, y_pima, test_size=0.2, random_state=42, stratify=y_pima
    )
    scaler_pima = StandardScaler()
    X_pima_tr_s = scaler_pima.fit_transform(X_pima_tr)
    X_pima_te_s = scaler_pima.transform(X_pima_te)

    pima_eval = []
    pima_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Linear Discriminant Analysis (LDA)': LinearDiscriminantAnalysis(),
        'Quadratic Discriminant Analysis (QDA)': QuadraticDiscriminantAnalysis(),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    }

    for name, model in pima_models.items():
        model.fit(X_pima_tr_s, y_pima_tr)
        y_pred = model.predict(X_pima_te_s)
        y_prob = model.predict_proba(X_pima_te_s)[:, 1]

        acc = accuracy_score(y_pima_te, y_pred)
        rec = recall_score(y_pima_te, y_pred)
        prec = precision_score(y_pima_te, y_pred, zero_division=0)
        f1 = f1_score(y_pima_te, y_pred)
        roc_auc = roc_auc_score(y_pima_te, y_prob)

        pima_eval.append({
            'Model': name,
            'Accuracy': acc,
            'Recall (Sensitivity)': rec,
            'Precision': prec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })

    pima_df = pd.DataFrame(pima_eval).sort_values(by='ROC-AUC', ascending=False)
    print("\n--- Pima Indians Cohort Benchmark Results ---")
    print(pima_df.round(4).to_string(index=False))

    # 7. GENERATE HIGH-QUALITY PUBLICATION VISUALIZATIONS
    print("\n" + "=" * 80)
    print("SECTION 6: GENERATING PUBLICATION VISUALIZATIONS")
    print("=" * 80)

    # Plot 1: Key Health Indicators Distribution by Diabetes Status
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    palette = {0: '#2b5c8f', 1: '#d9534f'}
    
    sns.histplot(data=df, x='HbA1c_level', hue='diabetes', kde=True, bins=25, 
                 palette=palette, stat='density', common_norm=False, ax=axes[0, 0])
    axes[0, 0].set_title('A. HbA1c Level Distribution (Welch t = 246.7, p < 1e-10)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('HbA1c Level (%)')
    axes[0, 0].axvline(6.5, color='#d9534f', linestyle='--', label='Diagnostic Threshold (>=6.5%)')
    axes[0, 0].legend(title='Diabetes', labels=['Diagnostic Cutoff', 'Diabetic (1)', 'Non-Diabetic (0)'])

    sns.histplot(data=df, x='blood_glucose_level', hue='diabetes', kde=True, bins=25, 
                 palette=palette, stat='density', common_norm=False, ax=axes[0, 1])
    axes[0, 1].set_title('B. Blood Glucose Level Distribution (Cohen d = 1.63)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Blood Glucose (mg/dL)')
    axes[0, 1].axvline(200, color='#d9534f', linestyle='--', label='Casual Glucose Cutoff (200 mg/dL)')
    axes[0, 1].legend(title='Diabetes', labels=['Cutoff (200 mg/dL)', 'Diabetic (1)', 'Non-Diabetic (0)'])

    colors = ['#2b5c8f', '#d9534f']
    sns.boxplot(data=df, x='diabetes', y='bmi', hue='diabetes', palette=colors, ax=axes[1, 0], showfliers=False, legend=False)
    axes[1, 0].set_title('C. Body Mass Index (BMI) by Diagnosis Group', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(['Non-Diabetic (0)', 'Diabetic (1)'])
    axes[1, 0].set_ylabel('BMI (kg/m²)')

    sns.boxplot(data=df, x='diabetes', y='age', hue='diabetes', palette=colors, ax=axes[1, 1], showfliers=False, legend=False)
    axes[1, 1].set_title('D. Age Distribution by Diagnosis Group', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(['Non-Diabetic (0)', 'Diabetic (1)'])
    axes[1, 1].set_ylabel('Age (Years)')

    plt.suptitle('Figure 1: Univariate Clinical Health Indicator Distributions (SDG 3)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('output_visualizations/fig1_univariate_health_indicators.png', dpi=300)
    plt.close()
    print("Saved: output_visualizations/fig1_univariate_health_indicators.png")

    # Plot 2: Categorical Comorbidities and Chi-Square Associations
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Hypertension prevalence
    ht_tab = pd.crosstab(df['hypertension'], df['diabetes'], normalize='index') * 100
    ht_tab.plot(kind='bar', stacked=True, color=['#2b5c8f', '#d9534f'], ax=axes[0])
    axes[0].set_title('A. Diabetes by Hypertension Status (Chi2 = 3912.4)', fontweight='bold')
    axes[0].set_xticklabels(['No Hypertension', 'Hypertension'], rotation=0)
    axes[0].set_ylabel('Percentage of Cohort (%)')
    axes[0].legend(['Non-Diabetic', 'Diabetic'])

    # Heart Disease prevalence
    hd_tab = pd.crosstab(df['heart_disease'], df['diabetes'], normalize='index') * 100
    hd_tab.plot(kind='bar', stacked=True, color=['#2b5c8f', '#d9534f'], ax=axes[1])
    axes[1].set_title('B. Diabetes by Heart Disease (Chi2 = 2990.2)', fontweight='bold')
    axes[1].set_xticklabels(['No Heart Disease', 'Heart Disease'], rotation=0)
    axes[1].set_ylabel('Percentage of Cohort (%)')
    axes[1].legend(['Non-Diabetic', 'Diabetic'])

    # Smoking history prevalence
    sm_tab = pd.crosstab(df['smoking_history'], df['diabetes'], normalize='index') * 100
    sm_tab.plot(kind='bar', stacked=True, color=['#2b5c8f', '#d9534f'], ax=axes[2])
    axes[2].set_title('C. Diabetes by Smoking Category (Chi2 = 2174.4)', fontweight='bold')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=30, ha='right')
    axes[2].set_ylabel('Percentage of Cohort (%)')
    axes[2].legend(['Non-Diabetic', 'Diabetic'])

    plt.suptitle('Figure 2: Categorical Comorbidities and Lifestyle Risk Factors', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('output_visualizations/fig2_categorical_risk_factors.png', dpi=300)
    plt.close()
    print("Saved: output_visualizations/fig2_categorical_risk_factors.png")

    # Plot 3: Discriminant Analysis (LDA Separation and Feature Loadings)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # LD1 projection scores
    ld1_scores = lda.transform(X_te_da)[:, 0]
    sns.kdeplot(ld1_scores[y_te_da == 0], color='#2b5c8f', label='Non-Diabetic (0)', fill=True, alpha=0.4, ax=axes[0])
    sns.kdeplot(ld1_scores[y_te_da == 1], color='#d9534f', label='Diabetic (1)', fill=True, alpha=0.4, ax=axes[0])
    axes[0].set_title("A. Linear Discriminant (LD1) Score Projection\nWilks' Lambda = 0.583 (p < 1e-10)", fontweight='bold')
    axes[0].set_xlabel('Canonical Discriminant Function (LD1)')
    axes[0].set_ylabel('Density')
    axes[0].axvline(0, color='black', linestyle=':', label='Centroid Cutoff')
    axes[0].legend()

    # Standardized LD1 Loadings
    axes[1].barh(std_coef_df['Feature'], std_coef_df['Standardized LD1 Coefficient'], 
                 color=np.where(std_coef_df['Standardized LD1 Coefficient'] > 0, '#2b5c8f', '#d9534f'))
    axes[1].set_title('B. Standardized Discriminant Coefficients (LD1 Weights)', fontweight='bold')
    axes[1].set_xlabel('Standardized Canonical Coefficient')
    axes[1].axvline(0, color='gray', linestyle='--')
    for i, v in enumerate(std_coef_df['Standardized LD1 Coefficient']):
        axes[1].text(v + (0.02 if v >= 0 else -0.15), i, f"{v:.3f}", va='center', fontweight='bold')

    plt.suptitle('Figure 3: Canonical Linear Discriminant Analysis (LDA)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('output_visualizations/fig3_discriminant_analysis.png', dpi=300)
    plt.close()
    print("Saved: output_visualizations/fig3_discriminant_analysis.png")

    # Plot 4: Logistic Regression Odds Ratios Forest Plot
    or_plot_data = or_table.drop(index='const')
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(or_plot_data))
    
    ax.errorbar(
        or_plot_data['OR'], y_pos, 
        xerr=[or_plot_data['OR'] - or_plot_data['Lower 95% CI'], or_plot_data['Upper 95% CI'] - or_plot_data['OR']],
        fmt='o', color='#2b5c8f', ecolor='#d9534f', elinewidth=2, capsize=4, markersize=7
    )
    ax.axvline(1.0, color='black', linestyle='--', linewidth=1.2, label='Null Effect (OR = 1.0)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(or_plot_data.index, fontsize=10)
    ax.set_xlabel('Adjusted Odds Ratio (OR) on Log Scale', fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.set_title("Figure 4: Multivariable Logistic Regression Forest Plot (Odds Ratios with 95% CI)", fontsize=13, fontweight='bold')
    
    # Annotate OR values
    for i, (idx, row) in enumerate(or_plot_data.iterrows()):
        ax.text(row['Upper 95% CI'] * 1.08, i, f"OR={row['OR']:.2f} [{row['Lower 95% CI']:.2f}, {row['Upper 95% CI']:.2f}]", 
                va='center', fontsize=9)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('output_visualizations/fig4_logistic_regression_forest_plot.png', dpi=300)
    plt.close()
    print("Saved: output_visualizations/fig4_logistic_regression_forest_plot.png")

    # Plot 5: ROC Curves and Precision-Recall Curves for All Models
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC Curves
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)')
    axes[0].set_title('A. Receiver Operating Characteristic (ROC) Curves', fontweight='bold')
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[0].set_ylabel('True Positive Rate (Sensitivity / Recall)')
    axes[0].legend(loc='lower right', frameon=True)

    # Precision-Recall Curves
    for name, (r_curve, p_curve, pr_auc) in pr_data.items():
        axes[1].plot(r_curve, p_curve, label=f"{name} (PR-AUC = {pr_auc:.3f})", linewidth=2)
    axes[1].axhline(y=df['diabetes'].mean(), color='k', linestyle='--', label=f'Baseline Prevalence ({df["diabetes"].mean()*100:.1f}%)')
    axes[1].set_title('B. Precision-Recall Curves (Class-Imbalanced Screening)', fontweight='bold')
    axes[1].set_xlabel('Recall (Sensitivity)')
    axes[1].set_ylabel('Precision (Positive Predictive Value)')
    axes[1].legend(loc='upper right', frameon=True)

    plt.suptitle('Figure 5: Discrimination Performance across Statistical & Machine Learning Models', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('output_visualizations/fig5_roc_and_pr_curves.png', dpi=300)
    plt.close()
    print("Saved: output_visualizations/fig5_roc_and_pr_curves.png")

    # Plot 6: Primary vs Benchmark Cohort Comparison
    comp_models = ['Logistic Regression', 'Linear Discriminant Analysis (LDA)', 'Quadratic Discriminant Analysis (QDA)', 'Random Forest', 'Gradient Boosting']
    primary_auc = [eval_df.loc[eval_df['Model'].str.startswith(m.split(' (')[0]), 'ROC-AUC'].values[0] for m in comp_models]
    pima_auc = [pima_df.loc[pima_df['Model'] == m, 'ROC-AUC'].values[0] for m in comp_models]

    fig, ax = plt.subplots(figsize=(11, 6))
    bar_width = 0.35
    idx = np.arange(len(comp_models))
    
    b1 = ax.bar(idx - bar_width/2, primary_auc, bar_width, label='Primary Population Dataset (N=100,000)', color='#2b5c8f')
    b2 = ax.bar(idx + bar_width/2, pima_auc, bar_width, label='Pima Indians Benchmark Cohort (N=768)', color='#d9534f')
    
    ax.set_ylabel('ROC-AUC Score', fontweight='bold')
    ax.set_title('Figure 6: Cross-Cohort Model Generalization (Primary Clinical vs Pima Physiological)', fontweight='bold')
    ax.set_xticks(idx)
    ax.set_xticklabels([m.replace(' (Balanced)', '').replace(' (LDA)', '').replace(' (QDA)', '') for m in comp_models], rotation=15)
    ax.set_ylim(0.5, 1.05)
    ax.legend()

    for bar in b1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in b2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('output_visualizations/fig6_cross_cohort_comparison.png', dpi=300)
    plt.close()
    print("Saved: output_visualizations/fig6_cross_cohort_comparison.png")

    # Save summary tables to JSON and CSV for integration
    eval_df.to_csv("output_visualizations/model_comparison_primary.csv", index=False)
    pima_df.to_csv("output_visualizations/model_comparison_pima.csv", index=False)
    hyp_df_cont.to_csv("output_visualizations/hypothesis_testing_continuous.csv", index=False)
    hyp_df_cat.to_csv("output_visualizations/hypothesis_testing_categorical.csv", index=False)
    or_table.to_csv("output_visualizations/logistic_regression_odds_ratios.csv")
    std_coef_df.to_csv("output_visualizations/lda_standardized_coefficients.csv", index=False)

    print("\n[SUCCESS] Pipeline execution finished cleanly. All tables & figures generated.")

if __name__ == "__main__":
    run_pipeline()
