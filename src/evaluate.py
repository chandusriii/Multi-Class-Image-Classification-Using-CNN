"""Evaluate a trained CNN on a multi-class image test dataset."""

import torch


def evaluate_model(model, test_loader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted_classes = outputs.argmax(dim=1)

            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = 100.0 * correct_predictions / total_predictions
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
