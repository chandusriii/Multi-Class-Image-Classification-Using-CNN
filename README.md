# Multi-Class Image Classification Using CNN

A PyTorch-based deep learning project that uses a **Convolutional Neural Network (CNN)** to classify images into multiple categories.

The current implementation uses the **CIFAR-10 dataset** as the image-classification benchmark. The repository name describes the task and model rather than tying the project title to one dataset.

## Project Overview

Image classification is a computer vision task in which a model learns visual patterns from images and predicts the category each image belongs to. This project explores the end-to-end CNN workflow, including image preprocessing, model architecture, training, and evaluation.

## Dataset

This implementation uses **CIFAR-10**, a dataset containing 60,000 colour images across 10 classes:

- 50,000 training images
- 10,000 test images
- Image dimensions: 32 × 32 pixels
- Channels: 3 (RGB)

**Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.

The modular data-loading pipeline uses Torchvision's CIFAR-10 dataset utility and downloads the dataset automatically when it is not already available locally.

## CNN Architecture

The custom CNN contains three convolutional blocks followed by fully connected layers.

| Stage | Configuration |
|---|---|
| Input | 3 × 32 × 32 RGB image |
| Convolution channels | 3 → 32 → 64 → 128 |
| Activation | ReLU |
| Pooling | 2 × 2 Max Pooling after each convolution |
| Flatten | 128 × 4 × 4 = 2,048 features |
| Fully connected layers | 2,048 → 256 → 10 classes |

## Workflow

1. Load the image dataset.
2. Convert images to tensors and normalize the RGB channels.
3. Create training and test data loaders.
4. Define the CNN architecture.
5. Train the model using forward propagation and backpropagation.
6. Optimize model parameters using Adam and Cross-Entropy Loss.
7. Evaluate the trained model on the test dataset.
8. Save the trained model weights.

## Recorded Notebook Result

The step-by-step notebook contains a recorded 10-epoch experiment with the following results:

| Metric | Recorded value |
|---|---:|
| Test accuracy | **74.71%** |
| First epoch training loss | 0.7556 |
| Final epoch training loss | 0.1135 |

These values are from the recorded notebook experiment. They should not be interpreted as a verified result from every run of the modular training pipeline; results can vary with initialization, data order, and runtime environment.

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

## Project Structure

```text
Multi-Class-Image-Classification-Using-CNN/
├── notebooks/
│   └── Multi_Class_Image_Classification.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `models/` directory and trained checkpoint are generated when the training script runs. Dataset files are stored locally under `data/`; these generated files are not committed to the repository.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/chandusriii/Multi-Class-Image-Classification-Using-CNN.git
cd Multi-Class-Image-Classification-Using-CNN
```

### 2. Install dependencies

It is recommended to use a virtual environment.

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.venv\\Scripts\\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

### 3. Train and evaluate

```bash
python main.py
```

On the first run, Torchvision attempts to download CIFAR-10 into the local `data/` directory. An internet connection is required for this initial download. The trained weights are saved under `models/`.

### 4. Explore the notebook

```bash
jupyter notebook
```

Open `notebooks/CNN_for_CIFAR10.ipynb` to explore the step-by-step implementation.

## Learning Outcomes

- Understand convolutional layers, feature maps, activation functions, and pooling.
- Prepare image data using PyTorch and Torchvision.
- Design a CNN for multi-class image classification.
- Apply Cross-Entropy Loss and the Adam optimizer.
- Understand forward propagation, backpropagation, and model evaluation.
- Structure an experimental notebook into reusable Python modules.

## Current Status

The notebook experiment and modular training pipeline are both included. The notebook's recorded metrics are documented above; a fresh, complete run of the modular pipeline should be performed in the local environment before reporting its results as the project's final benchmark.

## Author

**Chandu Sri**

- GitHub: https://github.com/chandusriii
- Portfolio: https://chandusriii.github.io/chandusriii/
