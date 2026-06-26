#!/usr/bin/env python3
"""Run the full transfer learning pipeline (used to generate README figures)."""

from pathlib import Path

import torch
from torch import nn

from src.datasets import create_dataloaders
from src.engine import evaluate, train
from src.model_builder import create_efficientnet_b0, get_parameter_groups, unfreeze_top_layers
from src.utils import evaluate_and_plot_confusion_matrix, plot_curves

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "notebooks"
DATA_DIR = PROJECT_ROOT / "data"

EPOCHS = 10
FINE_TUNE_EPOCHS = 5
LR = 1e-3
FINE_TUNE_LR = 1e-4
BATCH_SIZE = 32

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  train_loader, val_loader, test_loader, num_classes, class_names = create_dataloaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    val_ratio=0.15,
  )

  model = create_efficientnet_b0(num_classes=num_classes, freeze_features=True).to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)

  print("=== Phase 1: Train classifier head ===")
  results = train(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=EPOCHS,
    device=device,
  )

  print("=== Phase 2: Fine-tune top layers ===")
  model = unfreeze_top_layers(model, num_layers=4)
  optimizer = torch.optim.Adam(
    get_parameter_groups(model, classifier_lr=LR, backbone_lr=FINE_TUNE_LR)
  )
  fine_tune_results = train(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=FINE_TUNE_EPOCHS,
    device=device,
  )

  for key in results:
    results[key].extend(fine_tune_results[key])

  curves_path = plot_curves(results, save_path=OUTPUT_DIR / "training_curves.png")
  print(f"Saved training curves: {curves_path}")

  test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
  print(f"Test Loss: {test_loss:.4f}")
  print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

  cm_path, report, _ = evaluate_and_plot_confusion_matrix(
    model=model,
    dataloader=test_loader,
    class_names=class_names,
    device=device,
    save_path=OUTPUT_DIR / "confusion_matrix.png",
  )
  print(f"Saved confusion matrix: {cm_path}")
  print(report)

  checkpoint_path = OUTPUT_DIR / "efficientnet_b0_flowers102.pt"
  torch.save({"model_state_dict": model.state_dict(), "num_classes": num_classes}, checkpoint_path)
  print(f"Model checkpoint: {checkpoint_path}")

  metrics_path = OUTPUT_DIR / "metrics.txt"
  metrics_path.write_text(
    f"test_loss={test_loss:.6f}\ntest_accuracy={test_acc:.6f}\n",
    encoding="utf-8",
  )
  print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
  main()
