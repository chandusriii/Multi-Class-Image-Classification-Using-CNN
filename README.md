# CIFAR-10 Multi-Class Image Classification Using CNN

## Overview
A practical implementation of a **Convolutional Neural Network (CNN)** using PyTorch to classify CIFAR-10 images into 10 categories.

## Dataset
CIFAR-10 contains 60,000 colour images: 50,000 training images and 10,000 test images. Each image is 32 × 32 pixels with 3 RGB channels.

**Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.

## CNN Architecture
The custom CNN uses three convolutional layers with ReLU activation and max pooling, followed by fully connected layers.

| Stage | Configuration |
|---|---|
| Input | 3 × 32 × 32 |
| Convolution blocks | 3 → 32 → 64 → 128 channels |
| Activation | ReLU |
| Pooling | 2 × 2 Max Pooling after each convolution |
| Flatten | 128 × 4 × 4 = 2048 features |
| Fully connected | 2048 → 256 → 10 classes |

## Implementation
The project demonstrates:
1. CIFAR-10 dataset loading and normalization.
2. DataLoader creation for batches.
3. CNN architecture implementation.
4. Multi-class loss calculation with CrossEntropyLoss.
5. Model optimization using Adam.
6. Forward propagation and backpropagation.
7. Model evaluation using test accuracy.

## Results
The recorded notebook experiment was trained for 10 epochs.

| Metric | Recorded value |
|---|---:|
| Test accuracy | **74.71%** |
| First epoch training loss | 0.7556 |
| Final epoch training loss | 0.1135 |

Results from a new run may differ due to random initialization and data shuffling.

## Tech Stack
Python · PyTorch · Torchvision · NumPy · Matplotlib · Jupyter Notebook

## Project Structure
```text
CIFAR-10-Multi-Class-Image-Classification-Using-CNN/
├── notebooks/
│   └── CNN_for_CIFAR10.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   └── (saved model weights; generated during training)
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run the Project

Clone the repository:

```bash
git clone https://github.com/chandusriii/CIFAR-10-Multi-Class-Image-Classification-Using-CNN.git
cd CIFAR-10-Multi-Class-Image-Classification-Using-CNN
```

Create and activate a virtual environment (optional but recommended), then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the modular training and evaluation pipeline:

```bash
python main.py
```

The CIFAR-10 dataset is downloaded into `data/` on the first run. Trained weights are saved to `models/cifar10_cnn.pth`.

To explore the step-by-step implementation, launch Jupyter:

```bash
jupyter notebook
```

Then open `notebooks/CNN_for_CIFAR10.ipynb`.

## Learning Outcomes
- Understand convolution, activation, pooling, and flattening.
- Prepare image data with PyTorch and Torchvision.
- Build a CNN for multi-class image classification.
- Implement training, backpropagation, and evaluation.
- Organize a notebook experiment into reusable Python modules.
