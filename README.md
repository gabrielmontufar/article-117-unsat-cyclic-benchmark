# Supplementary benchmark for article 117

This folder contains a reproducible material-point benchmark for the hydro-mechanical model of unsaturated soils under hydraulic and seismic cycles.

Run `python scripts/run_unsat_cyclic_benchmark.py` to regenerate the CSV files and figures.

Install dependencies with `python -m pip install -r requirements.txt`.

Models compared: constant suction baseline, no-hysteresis suction model, hysteresis-only model, and full hysteretic-damage model.

The file `data/external_validation_trend_checks.csv` records the external trend-validation sources used in the manuscript. It is a qualitative validation table, not a calibrated experimental data set.

Additional QA files include `data/timestep_convergence.csv` and `data/model_parameters.csv`.

Persistent archive for all versions: https://doi.org/10.5281/zenodo.20100858.

Licensing: code is MIT; generated data and figures are CC BY 4.0. See `LICENSE`.
