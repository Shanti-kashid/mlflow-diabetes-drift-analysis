"""
MLflow Experiment Tracking and Data Drift Analysis
Dataset: scikit-learn Diabetes regression dataset
Python: 3.12+
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
EXPERIMENT_NAME = "diabetes-drift-monitoring"
RUN_NAME = "ridge-baseline-and-drift"
RANDOM_STATE = 42
DRIFT_THRESHOLD_PSI = 0.2
DRIFT_THRESHOLD_P = 0.05


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Calculate PSI using quantile edges from reference data."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0
    # Infinite outer edges ensure values outside the reference range are counted.
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def regression_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def create_plots(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                 report: pd.DataFrame, reference_predictions: np.ndarray,
                 current_predictions: np.ndarray) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Feature distribution comparison
    n_features = len(reference_df.columns)
    cols = 2
    rows = int(np.ceil(n_features / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    axes = np.asarray(axes).reshape(-1)
    for i, feature in enumerate(reference_df.columns):
        axes[i].hist(reference_df[feature], bins=20, alpha=0.65, density=True, label="Reference")
        axes[i].hist(current_df[feature], bins=20, alpha=0.55, density=True, label="Current")
        axes[i].set_title(feature)
        axes[i].legend()
    for ax in axes[n_features:]:
        ax.axis("off")
    fig.suptitle("Reference vs Current Feature Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # PSI chart
    sorted_report = report.sort_values("psi", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(sorted_report["feature"], sorted_report["psi"])
    ax.axvline(DRIFT_THRESHOLD_PSI, linestyle="--", linewidth=1.5,
               label=f"PSI threshold = {DRIFT_THRESHOLD_PSI}")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("PSI per Feature")
    ax.legend()
    for bar, value in zip(bars, sorted_report["psi"]):
        ax.text(value + max(sorted_report["psi"].max() * 0.012, 0.01),
                bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "psi_per_feature.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Prediction distributions
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(reference_predictions, bins=20, alpha=0.65, density=True, label="Reference predictions")
    ax.hist(current_predictions, bins=20, alpha=0.55, density=True, label="Current predictions")
    ax.set_title("Prediction Distribution Drift")
    ax.set_xlabel("Predicted disease progression")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "prediction_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_analysis(tracking_uri: str | None = None) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Local SQLite backend; paths are relative to the project folder.
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")

    dataset = load_diabetes(as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.copy()
    X_reference, X_current, y_reference, y_current = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE
    )

    # Simulate a production-like population shift in selected numeric features.
    X_current = X_current.copy()
    X_current["bmi"] = X_current["bmi"] + 0.06
    X_current["bp"] = X_current["bp"] + 0.05
    X_current["s5"] = X_current["s5"] * 1.5
    rng = np.random.default_rng(RANDOM_STATE)
    X_current["age"] = X_current["age"] + rng.normal(0, 0.0005, len(X_current))

    model = Ridge(alpha=1.0)
    model.fit(X_reference, y_reference)
    pred_reference = model.predict(X_reference)
    pred_current = model.predict(X_current)
    metrics_reference = regression_metrics(y_reference.to_numpy(), pred_reference)
    metrics_current = regression_metrics(y_current.to_numpy(), pred_current)

    rows = []
    for feature in X_reference.columns:
        ks_stat, ks_pvalue = ks_2samp(X_reference[feature].to_numpy(), X_current[feature].to_numpy())
        psi = population_stability_index(X_reference[feature].to_numpy(), X_current[feature].to_numpy())
        drifted = bool((ks_pvalue < DRIFT_THRESHOLD_P) or (psi > DRIFT_THRESHOLD_PSI))
        rows.append({
            "feature": feature,
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "psi": float(psi),
            "drifted": drifted,
        })
    report = pd.DataFrame(rows)
    prediction_ks_stat, prediction_ks_pvalue = ks_2samp(pred_reference, pred_current)
    drift_share = float(report["drifted"].mean())

    report.to_csv(REPORTS / "drift_report.csv", index=False)
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "run_name": RUN_NAME,
        "dataset": "sklearn.datasets.load_diabetes",
        "records_total": int(len(X)),
        "reference_records": int(len(X_reference)),
        "current_records": int(len(X_current)),
        "features": int(X.shape[1]),
        "drifted_features": int(report["drifted"].sum()),
        "drift_share": drift_share,
        "prediction_ks_stat": float(prediction_ks_stat),
        "prediction_ks_pvalue": float(prediction_ks_pvalue),
        "reference_metrics": metrics_reference,
        "current_metrics": metrics_current,
        "thresholds": {"ks_pvalue": DRIFT_THRESHOLD_P, "psi": DRIFT_THRESHOLD_PSI},
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    create_plots(X_reference, X_current, report, pred_reference, pred_current)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME) as active_run:
        mlflow.log_param("model", "Ridge")
        mlflow.log_param("alpha", 1.0)
        mlflow.log_param("dataset", "sklearn.datasets.load_diabetes")
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("split", "60:40")
        mlflow.log_param("drift_simulation", "bmi +0.06; bp +0.05; s5 x1.5; age negligible noise")
        mlflow.log_param("psi_bins", 10)
        mlflow.log_param("ks_pvalue_threshold", DRIFT_THRESHOLD_P)
        mlflow.log_param("psi_threshold", DRIFT_THRESHOLD_PSI)

        for name, value in metrics_reference.items():
            mlflow.log_metric(f"reference_{name}", value)
        for name, value in metrics_current.items():
            mlflow.log_metric(f"current_{name}", value)
        mlflow.log_metric("drifted_features", int(report["drifted"].sum()))
        mlflow.log_metric("drift_share", drift_share)
        mlflow.log_metric("prediction_ks_stat", float(prediction_ks_stat))
        mlflow.log_metric("prediction_ks_pvalue", float(prediction_ks_pvalue))

        mlflow.sklearn.log_model(model, name="ridge_diabetes_model")
        mlflow.log_artifact(str(REPORTS / "drift_report.csv"), artifact_path="reports")
        mlflow.log_artifact(str(REPORTS / "summary.json"), artifact_path="reports")
        mlflow.log_artifact(str(REPORTS / "feature_distributions.png"), artifact_path="plots")
        mlflow.log_artifact(str(REPORTS / "psi_per_feature.png"), artifact_path="plots")
        mlflow.log_artifact(str(REPORTS / "prediction_distributions.png"), artifact_path="plots")

        print("\n=== MLflow Diabetes Drift Analysis Completed ===")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Run name:   {RUN_NAME}")
        print(f"Run ID:     {active_run.info.run_id}")
        print(f"Reference records: {len(X_reference)} | Current records: {len(X_current)}")
        print(f"Drifted features: {int(report['drifted'].sum())}/{len(report)} ({drift_share:.0%})")
        print(f"Prediction KS: statistic={prediction_ks_stat:.4f}, p-value={prediction_ks_pvalue:.6g}")
        print("\nFeature drift report:")
        print(report.to_string(index=False, formatters={
            "ks_stat": "{:.3f}".format,
            "ks_pvalue": "{:.4g}".format,
            "psi": "{:.3f}".format,
        }))
        print("\nReference metrics:", metrics_reference)
        print("Current metrics:  ", metrics_current)
        print(f"\nReports saved to: {REPORTS}")
        print("Open MLflow UI at http://127.0.0.1:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLflow experiment and drift analysis.")
    parser.add_argument("--tracking-uri", default=None, help="Optional MLflow tracking URI.")
    args = parser.parse_args()
    run_analysis(args.tracking_uri)
