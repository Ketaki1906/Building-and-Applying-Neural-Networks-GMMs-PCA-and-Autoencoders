# Building-and-Applying-Neural-Networks-GMMs-PCA-and-Autoencoders

## Project Overview

Complete implementation of **core machine learning models from scratch**, addressing challenges through:

* **Multi-Layer Perceptrons (MLPs)** for classification and regression
* **Gaussian Mixture Models (GMMs)** for image segmentation
* **Principal Component Analysis (PCA)** for dimensionality reduction and analysis
* **Autoencoders & Variational Autoencoders (VAEs)** for anomaly detection and data generation

---

## Project Details

### 1. **MLP Models**

* **Multi-Class Symbol Classification** using handwritten historical symbols
* **House Price Prediction** using real estate data from Bangalore
* **Multi-Label News Article Classification** with TF-IDF features

> Implemented from scratch: Forward & backpropagation, activation functions (Sigmoid, Tanh, ReLU), and optimizers (SGD, Batch GD, Mini-Batch GD)

---

### 2. **Gaussian Mixture Model (GMM)**

* Custom GMM for medical image segmentation (gray matter, white matter, CSF)
* Visualization and analysis of intensity distributions and misclassifications

---

### 3. **Principal Component Analysis (PCA)**

* Dimensionality reduction on MNIST dataset
* Visualization of explained variance and lossy reconstructions
* Classification before and after PCA

---

### 4. **AutoEncoder & Variational AutoEncoder**

* **Autoencoder** for anomaly detection based on digit reconstruction
* **Variational Autoencoder (VAE)** with BCE and MSE losses
* Latent space visualization and grid sampling experiments

---

## How to Run

1. Open the individual notebooks in Jupyter:

   * `MLP_Classification.ipynb`
   * `MLP_Regression.ipynb`
   * `MLP_Multilabel_News.ipynb`
   * `GMM_Segmentation.ipynb`
   * `PCA_Analysis.ipynb`
   * `Autoencoder_MNIST.ipynb`
   * `VAE_MNIST.ipynb`

2. ```./Graph``` contains the all the graphs obtained to visualise the training and outputs
---

## Evaluation Metrics Used

* Classification: Accuracy, Precision, Recall, Hamming Loss
* Regression: MSE, RMSE, R²
* Anomaly Detection: F1-score, AUC-ROC
* PCA: Explained Variance
* GMM: Segmentation Accuracy

---

## Notes

* All visualizations, graphs, and observations are included as markdown cells within notebooks.
* The code is modular and well-documented for clarity and reusability.

---
