"""Load CIFAR-10 with augmentation and a reproducible validation split."""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


# Use a checksum-verified mirror if the original Toronto host is unavailable.
CIFAR10.url = (
    "https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/"
    "resolve/c6e562a/cifar-10-python.tar.gz"
)

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_data_loaders(data_dir="./data", batch_size=64, seed=42):
    """Return augmented training, clean validation, and clean test loaders."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    evaluation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Separate dataset instances ensure validation never receives augmentation.
    train_dataset = CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    validation_dataset = CIFAR10(
        root=data_dir, train=True, download=True, transform=evaluation_transform
    )
    test_dataset = CIFAR10(
        root=data_dir, train=False, download=True, transform=evaluation_transform
    )

    if len(train_dataset) != 50_000 or len(test_dataset) != 10_000:
        raise RuntimeError(
            "Incomplete CIFAR-10 dataset. Expected 50,000 training and "
            f"10,000 test images; found {len(train_dataset)} and "
            f"{len(test_dataset)}."
        )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    train_indices = indices[:45_000]
    validation_indices = indices[45_000:]

    train_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(validation_dataset, validation_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_subset, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        "Dataset split: "
        f"{len(train_subset):,} training / "
        f"{len(validation_subset):,} validation / "
        f"{len(test_dataset):,} test images"
    )
    return train_loader, validation_loader, test_loader
