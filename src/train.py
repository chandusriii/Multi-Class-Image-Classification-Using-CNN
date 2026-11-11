"""Training loop for the CIFAR-10 CNN."""

import torch
import torch.nn as nn
import torch.optim as optim

from src.model import CNN


def train_model(train_loader, epochs=10, learning_rate=0.001, model_path="models/cifar10_cnn.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {average_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")
    return model, device
