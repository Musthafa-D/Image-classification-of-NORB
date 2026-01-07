# Image Classification on NORB (PyTorch)

Image classification on the (small) NORB dataset using PyTorch.  
This repo includes:
- dataset preparation (convert to `.jpg`, organize into class folders)
- a custom PyTorch Dataset/DataLoader
- training scripts for a CNN model and a linear baseline model
- a short comparison of results between CNN vs Linear model

> Note: The dataset is prepared into image folders before training.

---

## Repository Structure

- `DatasetNorb.py`  
  PyTorch Dataset / DataLoader utilities for loading NORB images.  
- `main_CNN.py`  
  Training script for the CNN-based classifier.  
- `main_LNN.py`  
  Training script for the linear / fully-connected baseline classifier.  
- `File separator.ipynb`  
  Notebook used to reorganize / rename extracted NORB images into class-wise folders.  
- `README.md`  
  Project documentation.

---

## What You’ll Learn From This Project

- Building a custom dataset pipeline in PyTorch (Dataset + DataLoader)
- Training and evaluating image classifiers
- Understanding why CNNs usually outperform linear models for vision tasks
- Practical data preparation: file conversion, renaming, folder organization

---

## Setup

### 1) Create environment (recommended)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
