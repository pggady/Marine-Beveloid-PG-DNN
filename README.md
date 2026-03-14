# Marine-Beveloid-PG-DNN

This repository contains the dataset and source code for the paper: 
**"Geometric design and hard-constrained optimization of marine beveloid gearboxes with coupled intersecting and crossed shafts based on physics-guided deep neural network"**.

## Contents
* `dataset.csv`: The training dataset generated via Latin Hypercube Sampling (LHS).
* `pg_dnn_model.py`: The PyTorch implementation of the Physics-Guided Deep Neural Network, including the custom loss functions embedded with Hertzian contact mechanics.
* `tpe_optimization.py`: The script for the Tree-structured Parzen Estimator (TPE) algorithm.

## Requirements
To run this code, please install the following dependencies:
* Python 3.8+
* PyTorch
* Optuna (for TPE)
* Pandas & NumPy

## Usage
1. Run `pg_dnn_model.py` to train the surrogate model using the provided dataset.
2. Run `tpe_optimization.py` to execute the system-level geometric parameter optimization.
