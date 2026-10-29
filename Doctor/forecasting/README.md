# Unified forecasting experiments

## Layout
- `core/`: shared scalar-series training/evaluation and exactly nine model implementations.
- `seven_datasets/`: seven mechanism experiment matrix.
- `blood_glucose/`: glucose experiment matrix.
- `pneumonia/`: original autoencoder and joint pixel-loss runners, adapted to shared models.
- `source_manifest.json`: original source paths and SHA-256 checksums before packaging edits.
- `PACKAGING_NOTES.md`: changes and verification limits.

## Scalar-series entry point
Run `python run_experiments.py --help` from this directory. The launcher changes the child
working directory to each experiment's output folder; input and output paths are resolved before launching.
Use `--dry-run` to inspect commands without importing torch or training.
Both series scenarios cover 15→1, 24→3, 24→6, 24→12 and nine models.
Require the actual independent sample length explicitly; do not infer it from a filename.
CSV must contain `date`, covariates, and the selected target. Rows for each independent
sample must be contiguous, exactly sample_len rows; no partial samples are discarded.
Split by sample (70/10/20), then create windows inside samples only. StandardScaler fits
training samples only; `--scale 0` disables it. MSE/MAE default to standardized units.
`--inverse` selects original units; additional original-unit metrics are saved by default.
MAPE on centered standardized values is not a meaningful percentage metric.
Checkpoints save preprocessing statistics, sample split indices and experiment arguments.

## Dependencies
`requirements.txt` lists curated direct dependencies, not a verified historical version lock.
No environment setup is performed by this package. Use an existing suitable environment.
Mamba and real xLSTM need separate optional packages and compatible runtimes; no fallback
is allowed. Their availability must not be inferred from the presence of a model file.
Python 3.10+ is needed by the image runners.

## Research status
These files package existing experiment code. Historical per-run hyperparameters, seeds,
dataset variants and correspondence to all paper table cells have not been established.
The JSON matrices are execution templates, NOT certified reconstructions of paper results.
No historical metrics are invented or copied without provenance. Read each scenario README.
Original TSLib MIT license is retained in LICENSE. Patient data generation remains in Patient.

## Examples (run from this directory)
Inspect glucose commands, assuming the supplied file really uses 48-step independent samples:
```powershell
python run_experiments.py --scenario blood_glucose --dataset blood_glucose --csv PATH_TO_GLUCOSE.csv --sample-len 48 --freq t --models all --dry-run
```
Inspect one simulated mechanism; 200 must match the actual generator's sample length:
```powershell
python run_experiments.py --scenario seven_datasets --dataset delayed_response --csv PATH_TO_DELAYED.csv --sample-len 200 --freq d --models all --dry-run
```
Remove `--dry-run` to train. Add `--windows 15:1` to select one setting, or
`--extra --train_epochs 100 --patience 10` to specify reviewed training parameters.
These are usage examples, not historical paper hyperparameter claims.
The scalar launcher expects four numerical columns for simulated data (three interventions
and obs), and three for glucose (insulin, carbohydrate, glucose). Adapt explicit channel
counts if using a different schema; the examples do not silently infer a new prediction task.

Run `python verify_package.py` for syntax, source preservation (in the original workspace),
and synthetic sample-boundary/scaling checks. This does not train neural models.
With an existing suitable torch interpreter, `python smoke_models.py` checks seven models and
`python smoke_training.py` runs small synthetic one-epoch training checks in temporary folders.
Neither installs packages. See VALIDATION.md for completed checks and remaining limits.
