"""Train the CNN and select the checkpoint using validation accuracy."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.model import CNN


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
            total_predictions += batch_size

    return (
        total_loss / total_predictions,
        100.0 * correct_predictions / total_predictions,
    )


def train_model(
    train_loader,
    validation_loader,
    epochs=20,
    learning_rate=0.001,
    model_path="models/cnn_model.pth",
):
    """Train with AdamW, a validation-driven scheduler, and best-model saving."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    best_validation_accuracy = 0.0

    for epoch in range(epochs):
        train_loss, train_accuracy = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy = _run_epoch(
            model, validation_loader, criterion, device
        )
        scheduler.step(validation_accuracy)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), model_path)
            best_marker = " *"
        else:
            best_marker = ""

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.2f}% | "
            f"LR: {current_lr:.6f}{best_marker}",
            flush=True,
        )

    # Evaluate the best validation checkpoint, not merely the final epoch.
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(
        f"Best validation accuracy: {best_validation_accuracy:.2f}%"
    )
    print(f"Best model weights saved to {model_path}")
    return model, device
