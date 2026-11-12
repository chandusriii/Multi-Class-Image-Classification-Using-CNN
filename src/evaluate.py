"""Evaluate the selected CNN checkpoint on the held-out test dataset."""

import torch
import torch.nn as nn


def evaluate_model(model, test_loader, device):
    """Report loss and accuracy on the untouched CIFAR-10 test split."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
            total_predictions += batch_size

    average_loss = total_loss / total_predictions
    accuracy = 100.0 * correct_predictions / total_predictions
    print(f"Test Loss: {average_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
