# Quantum Optics Subdiffraction Image Classifier

This repository implements a hybrid physics-deep learning framework for subdiffraction image classification in the single-photon regime. By leveraging quantum optics theory—specifically Hermite-Gauss basis functions and continuous spatial point-spread functions—this project models the transformation of discrete light sources and utilizes a parameterized unitary matrix combined with a Feed-Forward Neural Network (FFNN) to classify subdiffraction images (e.g., MNIST).

## Features

* **Physics-Informed Data Transformation**: Analytically computes the measurement matrix $U$ derived from continuous-space quantum state overlaps.
* **Learnable Unitary Transformations**: Uses skew-Hermitian generators and matrix exponentials in PyTorch to ensure the transformation matrix $V(\theta)$ remains strictly unitary during gradient descent.
* **Parallel Hyperparameter Sweeps**: Fully integrated with **Hydra** to easily sweep parameters (like the point-spread function variance, $\sigma$) across multiple parallel jobs.
* **Automated Artifact Generation**: Automatically generates and saves training metrics (CSV), test accuracies, and loss profile plots (PDF) isolated per execution.

## Requirements

This project requires **Python 3.8+**. Install the dependencies using:

```bash
pip install -r requirements.txt
