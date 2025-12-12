# Remote Sensing Scene Classification on NWPU-RESISC45

This document summarizes a set of experiments in **remote sensing scene classification** using the **NWPU‑RESISC45** dataset.  
The work was originally completed as a graduate computer vision assignment and has been re‑framed here as a **portfolio project** that highlights:

- Practical experience with medium‑scale image datasets  
- Classical feature pipelines (Fisher Vectors + SVM)  
- Subspace learning methods (PCA, LPP, LDA)  
- Supervised neural networks (MLP) on top of learned features  

The focus of this write‑up is on design decisions, comparative results, and lessons learned, rather than on step‑by‑step homework instructions.

---

## 1. Problem & Dataset

**Task.** Given a remote‑sensing image, predict which scene type it belongs to (e.g., airport, harbor, commercial area).

**Dataset.** NWPU‑RESISC45:

- 45 scene classes  
- 700 RGB images per class  
- Image resolution: 256 × 256  
- Strong intra‑class variation (illumination, season, viewpoint) and inter‑class similarity  

A standard split was used throughout all experiments:

- 500 training images per class  
- 100 validation images per class  
- 100 test images per class  

This yields **22,500 train**, **4,500 validation**, and **4,500 test images**.

---

## 2. Overall Pipeline

Rather than committing to a single model, the project explored a **family of pipelines** that build on top of a fixed CNN feature extractor:

1. Extract convolutional feature maps from a pretrained CNN (e.g., VGG16).  
2. Convert local descriptors into global image‑level features using **Fisher Vectors (FV)**.  
3. Optionally apply **dimensionality reduction** or **discriminative subspace learning**.  
4. Train either a **linear SVM** or a **multilayer perceptron (MLP)** classifier.  

This structure allows direct comparison of:

- Unsupervised vs supervised dimensionality reduction  
- Linear vs nonlinear classifiers  
- Capacity vs generalization

---

## 3. Part 1 – Fisher Vectors + Linear SVM

### 3.1 Method

For each image:

1. Extract convolutional feature maps from an intermediate VGG16 layer.  
2. Treat each spatial location as a local descriptor.  
3. Fit a **Gaussian Mixture Model (GMM)** to a subset of descriptors.  
4. Encode each image as a **Fisher Vector (FV)** with respect to the GMM parameters.  
5. Optionally apply **PCA** to the FV for dimensionality reduction and whitening.  
6. Train a **linear SVM** on the FVs.

Hyperparameters that were explored:

- Descriptor dimensionality after PCA on local descriptors:  
  - `kd ∈ {16, 24}`  
- Number of GMM components (visual words / mixture components):  
  - `nc ∈ {64, 128}`  

These directly control the FV dimensionality (`2 * kd * nc`) and therefore the memory and compute cost of the classifier.

### 3.2 Results & Observations

Key observations:

- Increasing `kd` from 16 → 24 generally improves representation power but also increases dimensionality.  
- Increasing `nc` from 64 → 128 further increases FV dimensionality and can improve accuracy up to a point.  
- There is a clear **diminishing returns** effect: beyond a certain FV size the gains are small and the cost grows rapidly.  

Confusion matrices for the main configurations are stored under:

- `figures/part1_confusion/`

These help identify classes that are systematically confused (e.g., similar urban patterns, water‑related scenes, etc.).

Overall, the FV + SVM pipeline provides a **strong classical baseline** that is competitive with many off‑the‑shelf CNN classifiers, especially when hyperparameters are tuned.

---

## 4. Part 2 – Subspace Methods: PCA, LPP, LDA

### 4.1 Motivation

Fisher Vectors are very high‑dimensional and can be noisy.  
Subspace methods can:

- Reduce computation  
- Improve numerical conditioning  
- Emphasize discriminative directions  

This project compared three approaches:

- **PCA** (Principal Component Analysis): unsupervised, variance‑preserving.  
- **LPP** (Locality Preserving Projections): graph‑based, preserves local neighborhood structure.  
- **LDA** (Linear Discriminant Analysis): supervised, maximizes between‑class variance while minimizing within‑class variance.

### 4.2 Experiments

Subspaces of various dimensionalities were learned:

- PCA: retain a fixed percentage of variance.  
- LPP: low‑dimensional embeddings with different target dimensions (e.g., 32, 64, 128).  
- LDA: dimensionality limited by the number of classes (45) but fully supervised.

On top of each subspace, the same **linear SVM** was trained.

### 4.3 Results & Takeaways

High‑level findings:

- **PCA** is effective at compressing FVs while preserving most of the classification performance.  
  It also tends to stabilize optimization by filtering out very small‑variance directions.  
- **LPP** can give small gains in some regimes, but it is sensitive to graph construction and hyperparameters, and more expensive to compute.  
- **LDA** provides the most discriminative linear subspace in theory, but in practice can overfit when the number of classes is large and feature dimensionality is high.

Plots in:

- `figures/part2_pca_eigenspectrum.png`  
- `figures/part2_subspace_test_accuracy.png`  
- `figures/part2_confusion/`

illustrate how accuracy changes with dimensionality and which class confusions are resolved or remain.

Overall, subspace methods were most useful for **dimensionality reduction and interpretability**, with modest impact on top‑line accuracy compared to carefully tuned FV + SVM.

---

## 5. Part 3 – MLP Classifier on Top of FVs

### 5.1 Architecture

To explore a more expressive classifier, a **multilayer perceptron (MLP)** was trained directly on the Fisher Vectors (or their PCA‑reduced versions).

A typical architecture:

- Input: FV or FV‑PCA feature  
- Hidden layer(s) with ReLU  
- Dropout for regularization  
- Output layer: 45‑way softmax  

Multiple configurations were tested, including:

- Single hidden layer vs two hidden layers  
- Different hidden sizes (e.g., 512, 1024)  
- With and without additional regularization

### 5.2 Results

Findings:

- A well‑regularized MLP **outperformed** the linear SVM, delivering the best overall test accuracy of the project.  
- Overly deep or wide MLPs did **not** help; they tended to overfit and reduce test accuracy despite higher training accuracy.  
- Careful monitoring of validation curves (accuracy and loss) was essential to avoid over‑capacity models.

Key plots:

- `figures/part3_mlp_test_accuracy.png` – comparison across MLP variants  
- `figures/part3_curves/` – learning curves for different settings  
- `figures/part3_confusion/` – confusion matrices of the best models

The conclusion is that **label‑aware nonlinear models** can unlock additional performance, but only when paired with proper regularization and early stopping.

---

## 6. Lessons Learned

Some broader takeaways from this project:

1. **Strong baselines matter.**  
   FV + linear SVM is a robust starting point that is surprisingly competitive, especially when hyperparameters are tuned and dimensionality is managed.

2. **Dimensionality reduction is a tool, not a goal.**  
   PCA / LPP / LDA are most valuable when they simplify models and improve conditioning, but they are not guaranteed to improve accuracy.

3. **Model capacity must match the problem.**  
   Larger MLPs are not automatically better. Without enough data or regularization, they overfit and hurt performance.

4. **Visualization and diagnostics are essential.**  
   Confusion matrices and learning curves quickly reveal whether the model is underfitting, overfitting, or confusing specific classes.

5. **Reusable pipelines are powerful.**  
   The FV + subspace + MLP pipeline can be adapted to many other datasets that offer local descriptors or CNN feature maps.

---

## 7. How This Fits in a Portfolio

From a portfolio perspective, this project demonstrates:

- End‑to‑end management of a **medium‑scale vision dataset**  
- Use of both **classical computer vision** and **neural methods**  
- Systematic experimental design with documented results  
- Proficiency with libraries such as NumPy, scikit‑learn, PyTorch, and Matplotlib  

The corresponding repository includes:

- A Jupyter notebook with implementation details (`notebooks/HW5-Transfer-Learning.ipynb`)  
- Saved model artifacts (`models/`)  
- Figures for all main experiments (`figures/`)  
- This concise, professional‑style report (`Report.md`)
