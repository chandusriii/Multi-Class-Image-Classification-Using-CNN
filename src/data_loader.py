"""Load and preprocess the CIFAR-10 benchmark dataset for image classification."""

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms


def get_data_loaders(data_dir="./data", batch_size=64):
    """Download CIFAR-10 and return training and testing DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    if len(train_dataset) != 50_000 or len(test_dataset) != 10_000:
        raise RuntimeError(
            "Incomplete CIFAR-10 dataset detected. Expected 50,000 training "
            f"and 10,000 test images, but found {len(train_dataset)} training "
            f"and {len(test_dataset)} test images. Check the dataset files "
            "or remove the incomplete local CIFAR-10 folder and rerun to "
            "allow Torchvision to download the complete dataset."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
