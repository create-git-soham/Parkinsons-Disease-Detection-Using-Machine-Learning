"""
Parkinson's Disease Detection - Utility Functions
Reusable helper functions for data loading, preprocessing, and model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.model_selection import cross_val_score


# 1. Load Data

def load_data(
    filepath="parkinsons.csv"
):
    return pd.read_csv(filepath)


# 2. Data Cleaning

def drop_id(df):
    """Drop the recording-name column -- It identifies recordings, not patients."""
    return df.drop(columns=["name"], errors="ignore")


# 3. Feature Engineering

def create_features(df):
    df_feat = df.copy()

    if {"Fhi_Hz", "Flo_Hz"}.issubset(df_feat.columns):
        df_feat["frequency_range"] = (
            df_feat["Fhi_Hz"] - df_feat["Flo_Hz"]
        )

    jitter_cols = [
        c for c in df_feat.columns
        if c.startswith(("Jitter", "RAP", "PPQ"))
    ]
    if jitter_cols:
        df_feat["jitter_avg"] = df_feat[jitter_cols].mean(axis=1)

    shimmer_cols = [
        c for c in df_feat.columns
        if c.startswith("Shimmer")
    ]
    if shimmer_cols:
        df_feat["shimmer_avg"] = df_feat[shimmer_cols].mean(axis=1)

    if {"HNR", "NHR"}.issubset(df_feat.columns):
        df_feat["voice_to_noise"] = (
            df_feat["HNR"] / (df_feat["NHR"] + 1e-9)
        )

    return df_feat


def preprocess_data(df):
    df_clean = drop_id(df)
    df_clean = create_features(df_clean)
    df_clean = df_clean.fillna(
        df_clean.median(numeric_only=True)
    )
    return df_clean


# 4. Model Evaluation

def evaluate_model(model_name, y_true, y_pred):
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }

    print(f"\n{'=' * 40}")
    print(f"{model_name}")
    print(f"{'=' * 40}")

    for k, v in metrics.items():
        if k != "Model":
            print(f"{k:12s}: {v:.4f}")

    return metrics


def plot_confusion_matrix(
    y_true, y_pred, model_name, ax=None
):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Parkinson's"],
        yticklabels=["Healthy", "Parkinson's"],
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")

    return ax


def plot_roc_curves(
    models_dict, X_test, y_test
):
    plt.figure(figsize=(10, 7))

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def cross_validate_model(
    model, X, y, cv=5
):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1"
    )

    print(f"CV F1 Scores : {scores.round(4)}")
    print(
        f"Mean F1      : "
        f"{scores.mean():.4f} "
        f"(+/- {scores.std():.4f})"
    )

    return scores


def compare_models(results_list):
    df_results = pd.DataFrame(results_list)
    df_results = (
        df_results
        .sort_values(
            "F1 Score",
            ascending=False
        )
        .reset_index(drop=True)
    )
    return df_results


def extract_subject_id(df):
    """Extract subject ID (e.g. '01') from the 'name' column."""
    if "name" in df.columns:
        return df["name"].str.extract(r"_S(\d+)_")[0]
    return None