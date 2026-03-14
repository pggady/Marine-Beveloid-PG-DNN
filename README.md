# Marine-Beveloid-PG-DNN-Optimization

This repository contains the dataset and source code for the research paper: 
[cite_start]**"Geometric design and hard-constrained optimization of marine beveloid gearboxes with coupled intersecting and crossed shafts based on physics-guided deep neural network"**[cite: 3374, 3375, 3376].

## Repository Structure

The codebase is highly modularized, separating the mathematical system modeling, the surrogate evaluation environment, and the optimization algorithm:

* `pgnn_training_data.csv`: The comprehensive training dataset generated via Latin Hypercube Sampling (LHS) for the surrogate model.
* `pgnn_contact_stress.py`: The PyTorch implementation of the Physics-Guided Deep Neural Network (PG-DNN ). It includes the custom loss functions embedded with Hertzian contact mechanics for predicting maximum contact stress and the semi-major axis of the contact ellipse.
* `system.py`: The core mathematical solver establishing the system-level geometric and constraint model for both intersected and crossed beveloid gear pairs.
* `gear_env_v2.py`: The evaluation environment wrapper (built with Gymnasium interfaces) that formulates the complex multi-constraint design space and seamlessly connects the mathematical solver/surrogate with the optimizer.
* `tpe.py`: The execution script for the Tree-structured Parzen Estimator (TPE) algorithm, which navigates the non-convex design space to find the optimal geometric parameters.
* `requirements.txt`: The list of Python dependencies required to run the framework.

## Dependencies

To run this code, please ensure you have Python installed, and then install the required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
