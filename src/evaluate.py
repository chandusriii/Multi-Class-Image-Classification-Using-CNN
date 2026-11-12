"""Evaluate the best checkpoint and export class-level diagnostics."""

from pathlib import Path

import torch
import torch.nn as nn


CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]


def evaluate_model(model, test_loader, device, report_dir="reports"):
    """Report held-out test metrics and save confusion-matrix CSV files."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    confusion = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += batch_size

            for actual, predicted in zip(labels.cpu(), predictions.cpu()):
                confusion[actual, predicted] += 1

    average_loss = total_loss / total_predictions
    accuracy = 100.0 * correct_predictions / total_predictions
    class_totals = confusion.sum(dim=1)
    class_correct = confusion.diag()
    class_accuracy = 100.0 * class_correct.float() / class_totals.clamp(min=1)

    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / "confusion_matrix.csv"
    with matrix_path.open("w", encoding="utf-8") as file:
        file.write("actual/predicted," + ",".join(CLASS_NAMES) + "\n")
        for class_name, row in zip(CLASS_NAMES, confusion.tolist()):
            file.write(class_name + "," + ",".join(map(str, row)) + "\n")

    accuracy_path = output_dir / "per_class_accuracy.csv"
    with accuracy_path.open("w", encoding="utf-8") as file:
        file.write("class,correct,total,accuracy_percent\n")
        for index, class_name in enumerate(CLASS_NAMES):
            file.write(
                f"{class_name},{class_correct[index].item()},"
                f"{class_totals[index].item()},{class_accuracy[index].item():.2f}\n"
            )

    print(f"Test Loss: {average_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("Per-class accuracy:")
    for class_name, value in zip(CLASS_NAMES, class_accuracy.tolist()):
        print(f"  {class_name}: {value:.2f}%")
    print(f"Confusion matrix saved to {matrix_path}")
    print(f"Per-class accuracy saved to {accuracy_path}")
    return accuracy
