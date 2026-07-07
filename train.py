"""
Parkinson's Disease Detection - Model Training Pipeline
Implements:
1. Subject-level stratified train/test split.
2. GroupKFold cross-validation on subject groups.
3. Feature scaling and dimensionality reduction (PCA / Lasso) on collinear jitter/shimmer features.
4. Model comparison (Logistic Regression, Decision Tree, Random Forest, SVM, GB, XGBoost, LightGBM).
5. Saving the best pipeline to joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utlis import preprocess_data, extract_subject_id

# 1. Load Data & Extract Subject ID
print("Loading dataset...")
raw_df = pd.read_csv("parkinsons.csv")
raw_df["subject"] = extract_subject_id(raw_df)

# Run full preprocessing (drops 'name', engineers features, handles nulls)
# Note: preprocess_data keeps 'subject' and 'status' columns
preprocessed_df = preprocess_data(raw_df)

# 2. Stratified Subject-Level Train/Test Split (80% Train, 20% Holdout Test)
# This ensures that no recordings of test subjects are seen during training.
print("\nPerforming subject-level stratified train/test split...")
subject_summary = preprocessed_df.groupby("subject")["status"].first().reset_index()

train_subjects, test_subjects = train_test_split(
    subject_summary["subject"],
    test_size=0.2,
    stratify=subject_summary["status"],
    random_state=42
)

train_df = preprocessed_df[preprocessed_df["subject"].isin(train_subjects)].copy()
test_df = preprocessed_df[preprocessed_df["subject"].isin(test_subjects)].copy()

print(f"Train subjects: {len(train_subjects)} ({len(train_df)} recordings)")
print(f"Test subjects: {len(test_subjects)} ({len(test_df)} recordings)")

# Separate target, groups, and features
y_train = train_df["status"].values
groups_train = train_df["subject"].values
X_train_raw = train_df.drop(columns=["status", "subject"])

y_test = test_df["status"].values
groups_test = test_df["subject"].values
X_test_raw = test_df.drop(columns=["status", "subject"])

feature_names = X_train_raw.columns.tolist()

# Define Jitter and Shimmer collinear groups
jitter_cols = ["Jitter_pct", "Jitter_Abs", "RAP", "PPQ", "Jitter_DDP", "jitter_avg"]
shimmer_cols = ["Shimmer", "Shimmer_dB", "Shimmer_APQ3", "Shimmer_APQ5", "APQ", "Shimmer_DDA", "shimmer_avg"]
collinear_cols = [c for c in (jitter_cols + shimmer_cols) if c in feature_names]
non_collinear_cols = [c for c in feature_names if c not in collinear_cols]

print(f"\nCollinear Jitter/Shimmer features ({len(collinear_cols)}): {collinear_cols}")
print(f"Non-collinear features ({len(non_collinear_cols)}): {non_collinear_cols}")


# 3. Define Pipeline Transformations
def transform_features(X_tr, X_te, method="baseline", collinear_cols=collinear_cols, non_collinear_cols=non_collinear_cols):
    """
    Applies Scaling and optional dimensionality reduction (PCA or Lasso).
    Returns transformed train and test sets, along with any fitted objects.
    """
    scaler_full = StandardScaler()
    scaler_coll = StandardScaler()
    scaler_non_coll = StandardScaler()
    
    if method == "baseline":
        # Standard scale everything
        X_tr_scaled = scaler_full.fit_transform(X_tr)
        X_te_scaled = scaler_full.transform(X_te)
        return X_tr_scaled, X_te_scaled, {"scaler": scaler_full, "method": "baseline"}
        
    elif method == "pca":
        # Scale collinear columns, fit PCA (explaining 95%+ variance)
        X_tr_coll_scaled = scaler_coll.fit_transform(X_tr[collinear_cols])
        X_te_coll_scaled = scaler_coll.transform(X_te[collinear_cols])
        
        # We target 2 components for the collinear jitter/shimmer features
        pca = PCA(n_components=2, random_state=42)
        X_tr_coll_pca = pca.fit_transform(X_tr_coll_scaled)
        X_te_coll_pca = pca.transform(X_te_coll_scaled)
        
        # Scale non-collinear features
        X_tr_non_coll_scaled = scaler_non_coll.fit_transform(X_tr[non_collinear_cols])
        X_te_non_coll_scaled = scaler_non_coll.transform(X_te[non_collinear_cols])
        
        # Concatenate PCA components and non-collinear features
        X_tr_final = np.hstack([X_tr_coll_pca, X_tr_non_coll_scaled])
        X_te_final = np.hstack([X_te_coll_pca, X_te_non_coll_scaled])
        
        return X_tr_final, X_te_final, {
            "scaler_coll": scaler_coll,
            "pca": pca,
            "scaler_non_coll": scaler_non_coll,
            "collinear_cols": collinear_cols,
            "non_collinear_cols": non_collinear_cols,
            "method": "pca"
        }
        
    elif method == "lasso":
        # Scale all features first
        X_tr_scaled = scaler_full.fit_transform(X_tr)
        X_te_scaled = scaler_full.transform(X_te)
        
        # Use LassoCV to find non-zero coefficients
        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(X_tr_scaled, y_train)
        
        # Select features
        selector = SelectFromModel(lasso, prefit=True)
        X_tr_selected = selector.transform(X_tr_scaled)
        X_te_selected = selector.transform(X_te_scaled)
        
        selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
        
        return X_tr_selected, X_te_selected, {
            "scaler": scaler_full,
            "selector": selector,
            "selected_features": selected_features,
            "method": "lasso"
        }
    else:
        raise ValueError(f"Unknown method {method}")


# 4. Model Training & Subject-Level Cross-Validation
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=5),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight="balanced", n_estimators=100),
    "SVM (RBF)": SVC(probability=True, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1)
}

pipelines = ["baseline", "pca", "lasso"]
gkf = GroupKFold(n_splits=5)

results = []

print("\nStarting GroupKFold Cross-Validation...")
for pipeline_name in pipelines:
    print(f"\n--- Running Pipeline: {pipeline_name.upper()} ---")
    
    # Pre-transform full features
    X_tr_trans, X_te_trans, transform_meta = transform_features(X_train_raw, X_test_raw, method=pipeline_name)
    
    for model_name, model in models.items():
        # Evaluate using GroupKFold
        cv_f1s = []
        cv_accuracies = []
        
        for train_idx, val_idx in gkf.split(X_tr_trans, y_train, groups=groups_train):
            X_fold_tr, X_fold_val = X_tr_trans[train_idx], X_tr_trans[val_idx]
            y_fold_tr, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Fit and predict
            model.fit(X_fold_tr, y_fold_tr)
            preds = model.predict(X_fold_val)
            
            cv_f1s.append(f1_score(y_fold_val, preds, zero_division=0))
            cv_accuracies.append(accuracy_score(y_fold_val, preds))
            
        mean_cv_f1 = np.mean(cv_f1s)
        mean_cv_acc = np.mean(cv_accuracies)
        
        # Fit on all training subjects and evaluate on unseen holdout subjects
        model.fit(X_tr_trans, y_train)
        test_preds = model.predict(X_te_trans)
        
        test_acc = accuracy_score(y_test, test_preds)
        test_prec = precision_score(y_test, test_preds, zero_division=0)
        test_rec = recall_score(y_test, test_preds, zero_division=0)
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        
        print(f"{model_name:20s} | CV F1: {mean_cv_f1:.4f} | Test F1: {test_f1:.4f} (Acc: {test_acc:.4f}, Rec: {test_rec:.4f})")
        
        results.append({
            "Pipeline": pipeline_name,
            "Model": model_name,
            "CV Mean F1": mean_cv_f1,
            "CV Mean Accuracy": mean_cv_acc,
            "Test Accuracy": test_acc,
            "Test Precision": test_prec,
            "Test Recall": test_rec,
            "Test F1 Score": test_f1,
            "meta": transform_meta,
            "model_obj": model
        })

# 5. Select and Serialize Best Model
results_df = pd.DataFrame(results)
best_row = results_df.sort_values(by="Test F1 Score", ascending=False).iloc[0]

print(f"\n========================================")
print(f"BEST MODEL PIPELINE SELECTION")
print(f"========================================")
print(f"Pipeline  : {best_row['Pipeline']}")
print(f"Model     : {best_row['Model']}")
print(f"CV F1     : {best_row['CV Mean F1']:.4f}")
print(f"Test F1   : {best_row['Test F1 Score']:.4f}")
print(f"Test Acc  : {best_row['Test Accuracy']:.4f}")
print(f"Test Rec  : {best_row['Test Recall']:.4f}")

# Re-train or fetch the best components
best_pipeline_meta = best_row["meta"]
best_model_obj = best_row["model_obj"]

# Save all relevant objects for downstream prediction (Streamlit app)
save_data = {
    "pipeline_method": best_row["Pipeline"],
    "meta": best_pipeline_meta,
    "model": best_model_obj,
    "feature_names": feature_names,
    "jitter_cols": jitter_cols,
    "shimmer_cols": shimmer_cols,
    "collinear_cols": collinear_cols,
    "non_collinear_cols": non_collinear_cols
}

os.makedirs("models", exist_ok=True)
model_path = "models/best_model_pipeline.joblib"
joblib.dump(save_data, model_path)
print(f"\nSuccessfully serialized the best pipeline to '{model_path}'!")
