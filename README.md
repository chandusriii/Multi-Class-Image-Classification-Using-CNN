# CIFAR-10 Multi-Class Image Classification Using CNN

## Overview
This project demonstrates the practical implementation of a **Convolutional Neural Network (CNN)** using PyTorch for multi-class image classification.

The model is trained on the CIFAR-10 dataset to recognize images belonging to 10 different categories.

## Objectives
- Understand the fundamentals of CNN architecture.
- Explore how convolution and pooling layers extract visual features.
- Build and train a CNN model using PyTorch.
- Evaluate a deep learning model on a multi-class image classification task.

## Dataset: CIFAR-10
CIFAR-10 contains 60,000 colour images across 10 classes.

| Dataset detail | Value |
|---|---:|
| Training images | 50,000 |
| Test images | 10,000 |
| Image dimensions | 32 × 32 pixels |
| Channels | 3 (RGB) |
| Number of classes | 10 |

**Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.

## CNN Architecture
The notebook implements a custom CNN with three convolutional layers, ReLU activations, max-pooling, and fully connected layers.

| Stage | Configuration |
|---|---|
| Input | 3 × 32 × 32 |
| Convolution 1 | 3 → 32 channels |
| Convolution 2 | 32 → 64 channels |
| Convolution 3 | 64 → 128 channels |
| Activation | ReLU |
| Downsampling | Max Pooling |
| Flatten | 128 × 4 × 4 features |
| Fully connected | 2048 → 256 |
| Output | 256 → 10 classes |

## Implementation Workflow
The Jupyter notebook walks through the complete experiment:

1. Importing the required libraries.
2. Loading the CIFAR-10 dataset.
3. Applying image transformations and normalization.
4. Creating training and testing DataLoaders.
5. Defining the CNN architecture.
6. Setting up the loss function and optimizer.
7. Training the model with backpropagation.
8. Evaluating classification performance on the test set.

## Technologies
- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

## Training and Results
The model was trained for **10 epochs**.

| Metric | Result |
|---|---:|
| Test accuracy | **74.71%** |
| Training loss (start) | 0.7556 |
| Training loss (end) | 0.1135 |

These values describe the recorded run in the notebook. Results may vary with changes to the environment, random seed, or training configuration.

## Project Structure
```text
CIFAR-10-Multi-Class-Image-Classification-Using-CNN/
├── README.md
├── requirements.txt
└── notebooks/
    └── CNN_for_CIFAR10.ipynb
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/chandusriii/CIFAR-10-Multi-Class-Image-Classification-Using-CNN.git
cd CIFAR-10-Multi-Class-Image-Classification-Using-CNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `notebooks/CNN_for_CIFAR10.ipynb` and run the cells.

## Learning Outcomes
- Understand the flow of a CNN for image classification.
- Work with image datasets and preprocessing pipelines in PyTorch.
- Implement convolutional, pooling, and fully connected layers.
- Train a model and interpret its loss and test accuracy.
