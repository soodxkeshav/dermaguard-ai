"""Evaluation, fairness analysis, and publication-quality reporting utilities."""

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
) -> dict:
    """Return aggregate, per-class, and confusion-matrix metrics."""
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=list(range(len(class_names))),
            average="macro",
            zero_division=0,
        )
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=list(range(len(class_names))),
        ).tolist(),
    }


def fairness_metrics(
    metadata: pd.DataFrame,
    labels: Sequence[int],
    predictions: Sequence[int],
    group_column: str = "fitzpatrick_scale",
) -> dict:
    """Calculate performance by Fitzpatrick group and worst-group gaps.

    Unknown metadata values (negative values, blank values, or ``unknown``)
    are excluded from group comparisons and reported separately in the output.
    """
    if group_column not in metadata.columns:
        raise ValueError(f"Missing fairness column: {group_column}")
    if len(metadata) != len(labels) or len(labels) != len(predictions):
        raise ValueError("Metadata, labels, and predictions must have equal length")

    frame = metadata.reset_index(drop=True).copy()
    frame["label"] = np.asarray(labels)
    frame["prediction"] = np.asarray(predictions)
    raw_groups = frame[group_column].astype(str).str.strip()
    known = ~raw_groups.isin({"", "-1", "nan", "None", "unknown"})
    frame["group"] = raw_groups

    by_group = {}
    for group, group_frame in frame.loc[known].groupby("group", sort=True):
        group_metrics = classification_metrics(
            group_frame["label"].tolist(),
            group_frame["prediction"].tolist(),
            [str(index) for index in range(int(max(frame["label"].max(), frame["prediction"].max())) + 1)],
        )
        by_group[group] = {
            "support": int(len(group_frame)),
            "accuracy": group_metrics["accuracy"],
            "balanced_accuracy": group_metrics["balanced_accuracy"],
            "macro_precision": group_metrics["macro_precision"],
            "macro_recall": group_metrics["macro_recall"],
            "macro_f1": group_metrics["macro_f1"],
        }

    group_names = list(by_group)
    fairness = {"groups": by_group, "unknown_support": int((~known).sum())}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        values = [by_group[group][metric] for group in group_names]
        fairness[f"worst_group_{metric}"] = min(values) if values else None
        fairness[f"best_group_{metric}"] = max(values) if values else None
        fairness[f"{metric}_gap"] = max(values) - min(values) if values else None
    return fairness


def save_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    """Save a high-resolution confusion-matrix figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6), dpi=180)
    sns.heatmap(
        np.asarray(matrix),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_training_curves(history: Iterable[dict], output_path: Path) -> None:
    """Save loss, accuracy, and macro-F1 training curves."""
    history_frame = pd.DataFrame(history)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=180)
    plots = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
    ]
    for axis, (metric, label) in zip(axes, plots):
        axis.plot(history_frame["epoch"], history_frame[f"train_{metric}"], label="Train", linewidth=2)
        axis.plot(history_frame["epoch"], history_frame[f"val_{metric}"], label="Validation", linewidth=2)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_fairness_plot(fairness: dict, output_path: Path) -> None:
    """Save group accuracy and macro-F1 comparison."""
    group_frame = pd.DataFrame.from_dict(fairness["groups"], orient="index")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5), dpi=180)
    group_frame[["accuracy", "macro_f1"]].plot.bar(ax=axis, color=["#0f766e", "#f97316"])
    axis.set_xlabel("Fitzpatrick skin-tone group")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_comparison_table(
    result_paths: dict[str, Path],
    output_path: Path,
) -> pd.DataFrame:
    """Combine baseline, V1, and V2 JSON results into CSV and Markdown."""
    rows = []
    for model_name, result_path in result_paths.items():
        result = pd.read_json(result_path, typ="series")
        test = result.get("test", {})
        if hasattr(test, "to_dict"):
            test = test.to_dict()
        rows.append({
            "model": model_name,
            "accuracy": test.get("accuracy", result.get("test_accuracy")),
            "macro_precision": test.get("macro_precision", result.get("test_precision_weighted")),
            "macro_recall": test.get("macro_recall", result.get("test_recall_weighted")),
            "macro_f1": test.get("macro_f1", result.get("test_f1_weighted")),
        })
    table = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    markdown_rows = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for row in table.itertuples(index=False):
        values = [
            f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        markdown_rows.append("| " + " | ".join(values) + " |")
    output_path.with_suffix(".md").write_text("\n".join(markdown_rows) + "\n", encoding="utf-8")
    return table
