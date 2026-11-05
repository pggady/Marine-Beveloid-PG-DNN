# Marine-Beveloid-PG-DNN

This repository contains the dataset, the trained surrogate model and the source code for the research paper
**"Geometric design and hard-constrained optimization of marine beveloid gearboxes based on physics-guided deep neural network"**.

## Repository structure

### Surrogate model

* `contact_stress.py`: Hertzian contact solver (principal curvatures, contact ellipse and maximum contact stress).
* `generate_lhs_dataset.py`, `regenerate_training_data.py`: Latin hypercube sampling (LHS) of the 12 inputs and evaluation by the Hertzian solver.
* `pgnn_contact_stress.py`: PyTorch implementation of the physics-guided deep neural network (PG-DNN), with the loss terms derived from Hertzian contact mechanics, the selection of the loss weights, training and evaluation.
* `boundary_subset_accuracy.py`: accuracy of the PG-DNN on the test samples near the boundaries of the input space.

### System model and optimization

* `system.py`: system-level model of the intersecting and crossed beveloid gear pairs of the gearbox.
* `gear_env_v2.py`: evaluation environment (Gymnasium interface) with the decision variables, the derived quantities, the hard constraints and the contact evaluation by the solver or the PG-DNN.
* `tpe.py`: constrained tree-structured Parzen estimator (TPE) with geometric pre-check, PG-DNN screening and verification by the Hertzian solver.
* `ga_baseline.py`: genetic algorithm with a penalty function, used as a baseline.
* `robustness_mc.py`: Monte Carlo robustness analysis of the optimized design.
* `export_design.py`, `analysis_fpd.py`: export of the optimized design and analysis of the first principal direction (FPD) angle.

### Figures

* `plot_fig11.py`: parametric validation of the PG-DNN against the Hertzian solver (Fig. 11), computed from the models.
* `plot_fig11_from_stats.py`: Fig. 11 redrawn from `figures/Fig11_stats.json`.
* `plot_fig12.py`: convergence, distribution of the trials and pass rates of the TPE optimization (Fig. 12).

### Data, model and results

* `data/pgnn_training_data.csv`: 10466 valid LHS samples with the 12 inputs and the outputs `Pmax` (maximum contact stress) and `a_len` (semi-major axis of the contact ellipse). The filter in `pgnn_contact_stress.py` keeps 9087 samples, which are split into training, validation and test sets of 70%, 15% and 15%.
* `pgnn_model/`: trained PG-DNN (`model.pth`, `scaler.pkl`), data split, training history, weight grid and test metrics.
* `pgnn_model_no_physics/`: network trained without the physics-guided loss terms (ablation).
* `results_tpe/`: trials (`optuna_results.csv`) and summaries of the TPE runs with random seeds 42 and 1 to 4, width factors 0.70 to 0.90, TPE with the Hertzian solver, timing runs and robustness analysis.
* `results_ga/`: trials and summaries of the genetic algorithm.
* `figures/Fig11_stats.json`: data of Fig. 11.

## Usage

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train the PG-DNN (weight grid, full training and ablation):

```bash
python pgnn_contact_stress.py --data data/pgnn_training_data.csv --grid --grid-epochs 300 --ablation
```

Run the optimization and draw the figures:

```bash
python tpe.py --seed 42 --out results_tpe/pinn_w0.8_seed42
python plot_fig11_from_stats.py figures/Fig11_stats.json Fig11.pdf
python plot_fig12.py --csv results_tpe/pinn_w0.8_seed42/optuna_results.csv --out Fig12.pdf
```

## Note

The geometric calculation of the beveloid gear pairs (pitch cone relations, profile shift and safety factors
against undercutting and pointing, module `gear_calculations`) is not included in this repository. The training and
evaluation of the PG-DNN and the plotting scripts run without it. `tpe.py`, `ga_baseline.py`, `robustness_mc.py` and
the scripts that build the gearbox model require it; the results of these runs are provided in `results_tpe/` and
`results_ga/`.

## License

This project is licensed under the MIT License (see `LICENSE`).
