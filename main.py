"""Run the CIFAR-10 training, validation, and final test pipeline."""

from src.data_loader import get_data_loaders
from src.evaluate import evaluate_model
from src.train import train_model


def main():
    train_loader, validation_loader, test_loader = get_data_loaders()
    model, device = train_model(
        train_loader,
        validation_loader,
        epochs=20,
    )
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
