# Multi-Class Image Classification Using CNN

A modular **PyTorch** computer-vision project that trains a custom Convolutional Neural Network (CNN) to classify images from the **CIFAR-10** dataset. The project includes reproducible data splitting, image augmentation, validation-based checkpoint selection, held-out test evaluation, and class-level diagnostics.

## Results

The improved training pipeline was evaluated in GitHub Actions on the full CIFAR-10 dataset.

| Metric | Original pipeline | Improved pipeline |
|---|---:|---:|
| Test accuracy | 75.17% | **82.96%** |
| Training epochs | 10 | 20 |
| Validation-based checkpoint selection | No | Yes |
| Training augmentation | Basic | Random crop + horizontal flip |
| Regularization | Basic | Batch normalization + dropout + AdamW weight decay |

**Absolute test-accuracy gain: +7.79 percentage points.**

Improved run details:

- Best validation accuracy: **83.32%** (epoch 19)
- Final logged training accuracy: 82.89%
- Final logged validation accuracy: 83.26%
- Held-out test loss: 0.4944
- Held-out test accuracy: **82.96%**

The baseline and improved values above are from separate successful GitHub Actions runs, so they are experimental run results rather than a guarantee of identical results on every machine. The notebook also contains a separate historical experiment (74.71% test accuracy); it is not the source of the 75.17% baseline figure.

**Latest improved run:** [GitHub Actions run #9](https://github.com/chandusriii/Multi-Class-Image-Classification-Using-CNN/actions/runs/43146861139)

## Dataset

CIFAR-10 contains 60,000 RGB images (32 × 32 pixels) in 10 classes:

Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.

- 45,000 images for training
- 5,000 images for validation
- 10,000 images for final testing

The validation split is reproducible (seed 42) and uses evaluation transforms rather than training augmentation. The test split is kept separate from training and validation.

The data loader uses a checksum-verified Hugging Face mirror if the original Toronto download host is unavailable.

## Model architecture

The CNN uses three convolutional blocks:

| Stage | Configuration |
|---|---|
| Input | 3 × 32 × 32 |
| Convolution channels | 3 → 32 → 64 → 128 |
| Each block | 3×3 convolution, batch normalization, ReLU, 2×2 max pooling |
| Flattened features | 2,048 |
| Classifier | 2,048 → 256 → 10 |
| Regularization | Batch normalization and 0.4 dropout |

## Training approach

- Random crop with padding and random horizontal flip on training images
- CIFAR-10 channel normalization
- AdamW optimizer with weight decay
- ReduceLROnPlateau learning-rate scheduler driven by validation accuracy
- Best checkpoint selected using validation accuracy
- Final evaluation on the held-out test split

## Evaluation outputs

After training, the evaluation step reports overall test loss and accuracy, then exports:

- `reports/confusion_matrix.csv` — counts by actual and predicted class
- `reports/per_class_accuracy.csv` — correct predictions, class totals, and accuracy for each class

These reports are generated locally and uploaded with the GitHub Actions run artifacts.

## Project structure

```text
Multi-Class-Image-Classification-Using-CNN/
├── notebooks/
│   └── Multi_Class_Image_Classification.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── main.py
├── reports/                         # Generated evaluation CSVs
├── models/                          # Generated model checkpoint
├── .github/workflows/cnn-training.yml
├── requirements.txt
└── README.md
```

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/chandusriii/Multi-Class-Image-Classification-Using-CNN.git
cd Multi-Class-Image-Classification-Using-CNN
```

### 2. Create an environment and install dependencies

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\\Scripts\\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. Train and evaluate

```bash
python main.py
```

The first run downloads CIFAR-10. The best checkpoint is saved to `models/cnn_model.pth`; evaluation reports are written to `reports/`.

## Automated training

The GitHub Actions workflow trains and evaluates the model on pushes to `main` and `improve-cifar10-generalization`, or when manually triggered. It uploads the training log, model checkpoint, and evaluation CSVs as a run artifact.

## Tech stack

Python · PyTorch · Torchvision · NumPy · Matplotlib · Jupyter Notebook · GitHub Actions

## Author

**Chandu Sri**

- GitHub: [chandusriii](https://github.com/chandusriii)
- Portfolio: [chandusriii.github.io](https://chandusriii.github.io/chandusriii/)
