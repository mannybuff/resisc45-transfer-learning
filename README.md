# Remote Sensing Scene Classification – NWPU-RESISC45

This project is a cleaned, portfolio-ready version of a graduate computer vision homework (HW5).  
It focuses on **remote sensing scene classification** on the **NWPU-RESISC45** dataset using a mix of:

- Fisher Vector feature representations
- Subspace / dimensionality reduction methods (PCA, LPP, LDA)
- A multilayer perceptron (MLP) classifier
- Transfer learning with pretrained CNN features

The goal of this repo is to showcase a **classical CV + transfer learning pipeline** with clear experiments and saved results, not to ship a production model.

---

## 1. Problem & Dataset

**Task:** Classify overhead / remote-sensing images into 45 scene categories (airports, harbors, commercial areas, etc.).

**Dataset:** [NWPU-RESISC45]

- 45 classes
- 700 RGB images per class
- Resolution 256×256 pixels

(Images themselves are *not* included here; this repo stores **results and models**, not the raw dataset.)

---

## 2. Project Structure

Suggested repo layout:

```text
resisc45-transfer-learning/
├─ README.md                 # This file – portfolio-facing overview
├─ requirements.txt          # Python dependencies
├─ .gitignore
├─ notebooks/
│  └─ HW5-Transfer-Learning.ipynb   # Main analysis & training notebook
├─ figures/
│  ├─ hw5_global_best_accuracy.png
│  ├─ part1_confusion/
│  │  ├─ cm_test_kd16_nc64.png
│  │  ├─ cm_test_kd16_nc128.png
│  │  ├─ cm_test_kd24_nc64.png
│  │  └─ cm_test_kd24_nc128.png
│  ├─ part2_pca_eigenspectrum.png
│  ├─ part2_subspace_test_accuracy.png
│  ├─ part2_confusion/
│  │  ├─ cm_lda.png
│  │  ├─ cm_lpp_d32.png
│  │  ├─ cm_lpp_d64.png
│  │  └─ cm_lpp_d128.png
│  ├─ part3_mlp_test_accuracy.png
│  ├─ part3_confusion/
│  │  ├─ cm_mlp_*.png
│  └─ part3_curves/
│     ├─ mlp_learning_curve_*.png
├─ models/
│  └─ part1_fv/
│     ├─ gmm_kd16_nc64.joblib
│     ├─ gmm_kd16_nc128.joblib
│     ├─ gmm_kd24_nc64.joblib
│     ├─ gmm_kd24_nc128.joblib
│     ├─ pca_kd16.joblib
│     └─ pca_kd24.joblib
└─ Report.md                 # Original, detailed homework-style writeup (optional to keep)
```

> You already have most of these folders and files from the original `cv-hw5` project; this README just re-frames it as a portfolio project.

---

## 3. Methods & Experiments

The notebook is organized in three main parts:

### 3.1 Part 1 – Fisher Vectors + Classical Classifier

- Build **Fisher Vector (FV)** representations over local image descriptors.
- Use **GMM-based** encoding with different:
  - Descriptor dimensionalities (`kd16`, `kd24`)
  - Numbers of components (`nc64`, `nc128`)
- Reduce dimensionality with **PCA** on the FVs.
- Train a simple classifier in FV space.
- Evaluate with:
  - Test accuracy per configuration
  - Confusion matrices saved under `figures/part1_confusion/`.

This part highlights the tradeoffs between descriptor dimension, codebook size, and downstream accuracy.

---

### 3.2 Part 2 – Subspace Methods (PCA, LPP, LDA)

Given high-dimensional features, this part investigates several subspace methods:

- **PCA** – energy-preserving, unsupervised dimensionality reduction.
- **LPP (Locality Preserving Projections)** – graph-based; preserves local neighborhood structure.
- **LDA (Linear Discriminant Analysis)** – supervised; maximizes between-class variance and minimizes within-class variance.

Saved artifacts include:

- PCA eigenspectrum plots (`figures/part2_pca_eigenspectrum.png`)
- Test accuracy across subspace dimensions (`figures/part2_subspace_test_accuracy.png`)
- Confusion matrices for each subspace method (`figures/part2_confusion/*.png`)

This part looks at how different subspaces affect class separability and robustness.

---

### 3.3 Part 3 – MLP Classifier on Feature Representations

A **multilayer perceptron (MLP)** classifier is trained on top of feature representations (e.g., pooled CNN features or FV/PCA features).

- Hidden layer sizes such as 1024 → 512 (exact configurations documented in the notebook and plots).
- Nonlinear activations, dropout, and standard optimization (SGD/Adam).
- Training curves:
  - Accuracy vs. epoch (`figures/part3_curves/mlp_learning_curve_*.png`)
- Final test accuracy and confusion matrices:
  - `figures/part3_confusion/cm_mlp_*.png`
- `figures/hw5_global_best_accuracy.png` summarizes which configuration achieved the best test performance.

This part illustrates how far a relatively simple neural classifier can go when paired with good feature representations.

---

## 4. Technologies & Libraries

Key Python libraries used (see `requirements.txt` for exact list):

- **NumPy**, **Pandas** – numeric and tabular processing
- **Matplotlib** – plotting curves and confusion matrices
- **scikit-learn** – PCA, LDA, metrics, train/test splits, etc.
- **SciPy** – sparse matrices and linear algebra utilities
- **Joblib** – saving/loading trained models (GMMs, PCA)
- **PyTorch** – deep learning and pretrained CNN features
- **Torchvision** – pretrained models and image transforms
- **Pillow (PIL)** – image loading and basic preprocessing
- **IPython.display** – inline result visualization in the notebook

The code is written for a standard Python 3.x environment and a GPU is helpful but not strictly required (transfer learning is faster with one).

---

## 5. How to Run the Notebook

### 5.1 Install dependencies

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate     # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### 5.2 Prepare the dataset

1. Download the **NWPU-RESISC45** dataset from its official source.
2. Organize it on disk in the structure expected by the notebook (e.g., one folder per class).
3. Update any dataset path variables in `notebooks/HW5-Transfer-Learning.ipynb` to point to your local copy.

> The dataset itself is not distributed in this repository.

### 5.3 Start Jupyter

```bash
jupyter notebook
```

Then open:

```text
notebooks/HW5-Transfer-Learning.ipynb
```

Run the cells for the parts you want to reproduce. Be aware that some stages (e.g., Fisher Vector construction and training) can be computationally intensive.

---

## 6. What This Repo Demonstrates

This project demonstrates:

- End-to-end handling of a **non-trivial, multi-class vision dataset**.
- Practical use of **Fisher Vectors** as a bridge between local descriptors and global image features.
- Comparison of **subspace learning methods** (PCA, LPP, LDA) for dimensionality reduction and classification.
- Application of **MLP on top of learned features**.
- Careful **result tracking** using saved models, confusion matrices, and learning curves.

It is a good portfolio example of:

> “Given a reasonably complex dataset, can I design and evaluate a range of classical + deep models, analyze errors, and keep my results organized?”

---

## 7. Original Homework Context

This work was originally completed as part of a graduate-level Computer Vision course assignment  
(*HW5 – Transfer Learning on NWPU-RESISC45*).  

The file `Report.md` preserves a more formal, assignment-style writeup of the experiments and conclusions.

---

## 8. License

This repo is intended for educational and portfolio purposes.

If this project is hosted on GitHub with a **MIT License** selected at repo creation time, the canonical license text will live in the `LICENSE` file generated by GitHub.
