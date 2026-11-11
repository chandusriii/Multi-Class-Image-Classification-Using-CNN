"""Run CIFAR-10 CNN training and evaluation."""

from pathlib import Path

import torch

from src.data_loader import get_data_loaders
from src.evaluate import evaluate_model
from src.train import train_model


def main():
    Path("models").mkdir(exist_ok=True)

    train_loader, test_loader = get_data_loaders()
    model, device = train_model(train_loader)
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
