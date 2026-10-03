# MNIST Neural Network Classification

A deep-learning portfolio project comparing **Multilayer Perceptrons (MLPs)** and **Convolutional Neural Networks (CNNs)** for handwritten-digit classification on MNIST.

This repository is a cleaned and reproducible refactor of recovered university coursework.

## Project goals

- Compare dense MLPs with CNNs for image classification.
- Examine how additional model capacity affects performance.
- Compare Adam with SGD + momentum at different learning rates.
- Evaluate models using a clean train/validation/test workflow.
- Demonstrate reproducible deep-learning experimentation.

## Dataset

The project uses the MNIST handwritten-digit dataset through:

```python
keras.datasets.mnist
```

MNIST contains:
- 60,000 training images
- 10,000 test images
- 28×28 grayscale images
- 10 classes: digits 0–9

The dataset is downloaded automatically by Keras and is not stored in this repository.

## Verified portfolio results

The refactored workflow was run locally on Windows using Python 3.12.6 and TensorFlow 2.18.1.

| Model | Test accuracy | Test loss |
|---|---:|---:|
| **CNN + Adam** | **99.14%** | **0.0376** |
| CNN Improved | 99.09% | 0.0350 |
| CNN + SGD (lr=0.01) | 98.83% | 0.0384 |
| CNN + SGD (lr=0.005) | 98.68% | 0.0390 |
| CNN Baseline | 98.54% | 0.0520 |
| MLP Improved | 97.63% | 0.1098 |
| MLP Baseline | 97.58% | 0.0779 |

The best-performing model in the full portfolio run was **CNN + Adam**, achieving **99.14% test accuracy**.

## Model architectures

### MLP baseline

```text
Input(784)
→ Dense(128, ReLU)
→ Dense(10, Softmax)
```

### MLP improved

```text
Input(784)
→ Dense(256, ReLU)
→ Dense(128, ReLU)
→ Dense(10, Softmax)
```

### CNN baseline

```text
Input(28,28,1)
→ Conv2D(32, 3×3, ReLU)
→ MaxPool(2×2)
→ Flatten
→ Dense(128, ReLU)
→ Dense(10, Softmax)
```

### CNN improved

```text
Input(28,28,1)
→ Conv2D(32, 3×3, ReLU)
→ Conv2D(64, 3×3, ReLU)
→ MaxPool(2×2)
→ Flatten
→ Dense(128, ReLU)
→ Dropout(0.3)
→ Dense(10, Softmax)
```

## Evaluation design

The portfolio version improves on the original coursework workflow.

Instead of using the official MNIST test set for validation during training, it:

1. creates a validation split from the original training set;
2. trains models only on the training subset;
3. uses the validation subset for model-development feedback;
4. keeps the official 10,000-image test set untouched until final evaluation.

This provides a cleaner estimate of generalisation performance.

## Installation

```bash
pip install -r requirements.txt
```

## Quick smoke test

```bash
python src/train_models.py --quick
```

## Full experiment

```bash
python src/train_models.py
```

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook

## Skills demonstrated

- Deep learning
- Multilayer Perceptrons
- Convolutional Neural Networks
- Image classification
- Multi-class classification
- Dropout regularisation
- Adam optimisation
- SGD with momentum
- Learning-rate comparison
- Train/validation/test separation
- Confusion-matrix analysis
- Reproducible experiment design

## Original coursework context

The recovered coursework recorded a best test accuracy of **99.18%** for CNN + Adam.

The results above are from the **refactored portfolio workflow**, which keeps the official test set separate from model development.

## Portfolio note

Original assessment reports containing personal/student information and conflicting draft notebooks are intentionally excluded from the public repository.
