# Packaging validation

Completed without installing packages or changing the original experiment environment:

- Parsed packaged Python files and checked local `from` imports resolve.
- Verified SHA-256 of all 32 recorded original source files remains unchanged.
- Exercised the actual packaged scalar loader with synthetic 48-step independent samples
  for 15→1, 24→3, 24→6 and 24→12.
- Checked train/validation/test sample IDs are disjoint and cover the complete dataset.
- Checked every input/target window against original sample indices: no boundary crossing.
- Checked training-only standardization, shared validation/test statistics and inverse transform.
- Checked incomplete trailing samples are rejected rather than silently discarded.
- Generated 288 dry-run commands: seven simulated mechanisms plus glucose, nine models,
  four window settings. Dry runs do not import neural models or launch training.

Additional runtime checks used an existing PyTorch interpreter, with no installation:
- 56 scalar forward/backward cases: seven available models × two input schemas × four windows.
- Seven pneumonia temporal adapters passed forward/backward checks.
- All seven available scalar models completed one synthetic-data training epoch, checkpoint save,
  test evaluation, and original-scale artifact consistency checks using the 24→12 setting.
- Inference-only checkpoint reload and original-scale evaluation passed.
- Pneumonia completed one AE epoch and one DLinear joint epoch using existing images and two
  windows per split; decoding and pixel MSE/RMSE/MAE JSON and CSV output checks passed.

Review caught and fixed pandas 't' alias compatibility, unused attention imports, pixel RMSE
CSV field registration, checkpoint path handling and long Windows artifact names.
The optional real xLSTM CPU backend name was corrected to vanilla based on the local upstream
source; the package is not installed in the validation interpreter, so it was not executed.

Not validated: real xLSTM/Mamba execution, GPU kernels, full-dataset convergence or paper numerical
reproduction. Those optional dependencies were not installed. Synthetic smoke tests verify code
paths and interfaces, not scientific predictive performance.
