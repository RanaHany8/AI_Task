"""Training and evaluation loops."""

import torch
from tqdm.auto import tqdm


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (y_pred.argmax(dim=1) == y).float().mean().item()

    return train_loss / len(dataloader), train_acc / len(dataloader)

def val_step(model, dataloader, loss_fn, device):
    model.eval()
    val_loss, val_acc = 0.0, 0.0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            val_pred = model(X)
            loss = loss_fn(val_pred, y)

            val_loss += loss.item()
            val_acc += (val_pred.argmax(dim=1) == y).float().mean().item()

    return val_loss / len(dataloader), val_acc / len(dataloader)


def evaluate(model, dataloader, loss_fn, device):
    """Evaluate loss and accuracy on a held-out dataset."""
    return val_step(model, dataloader, loss_fn, device)


def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, device):
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, device
        )
        val_loss, val_acc = val_step(model, val_dataloader, loss_fn, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    return results
