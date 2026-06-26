"""Plotting and evaluation utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix


def plot_curves(results, save_path: str | Path | None = None):
    """Plot training vs validation loss and accuracy curves."""
    epochs = range(1, len(results["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, results["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, results["val_loss"], label="Val Loss", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, results["train_acc"], label="Train Accuracy", marker="o", markersize=3)
    axes[1].plot(epochs, results["val_acc"], label="Val Accuracy", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return save_path


def evaluate_and_plot_confusion_matrix(
    model,
    dataloader,
    class_names,
    device,
    save_path: str | Path | None = None,
    max_labels: int = 20,
):
    """Evaluate model and plot a confusion matrix (subset of classes if many)."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            preds = model(X).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    unique_classes = np.unique(np.concatenate([all_labels, all_preds]))
    if len(unique_classes) > max_labels:
        top_classes = np.unique(all_labels)[:max_labels]
        mask = np.isin(all_labels, top_classes)
        plot_labels = all_labels[mask]
        plot_preds = all_preds[mask]
        plot_names = [class_names[i] for i in top_classes]
        title_suffix = f" (first {max_labels} classes)"
    else:
        plot_labels = all_labels
        plot_preds = all_preds
        plot_names = class_names
        title_suffix = ""

    cm = confusion_matrix(plot_labels, plot_preds, labels=range(len(plot_names)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[n[:15] for n in plot_names],
        yticklabels=[n[:15] for n in plot_names],
        ax=ax,
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(f"Confusion Matrix{title_suffix}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    accuracy = (all_preds == all_labels).mean()

    return save_path, report, accuracy
