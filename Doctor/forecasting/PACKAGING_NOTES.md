# Packaging changes and limitations

Original files remain untouched. Changes apply only in this staging copy:
- Nine-model lazy registry; removed unrelated tasks, datasets and dependencies from entry points.
- Included local layers/utils reached by those models and the forecasting experiment imports.
- Independent sample length is required; reject partial samples and empty partitions.
- Training-only standardization; local RNG preserves source sample permutation without resetting global RNG.
- Weighted validation loss; save arguments, sample indices and scaler alongside checkpoints.
- TiDE uses c_out=1 because this implementation forecasts each variable separately.
- Real xLSTM only; removed ordinary-LSTM fallback and broken local xlstm path injection.
- Image runner imports shared model modules directly and reports pixel RMSE=sqrt(pixel MSE).

Scientific checks still needed before claiming exact paper reproduction:
- Source sample_len was 200 at packaging time; earlier glucose experiments used 48. Select explicitly.
- Glucose file containing 'noisy_24000' has values near 3 in its first rows, while another glucose
  file has values near 150. Confirm units and whether preprocessing already occurred upstream.
- Confirm mechanism filename prefixes, sample lengths and dataset versions against Patient generation.
- Image joint runner appends future intervention tokens: this is conditional forecasting with
  known/planned future interventions, not prediction from historical observations alone.
- Image code expects 200 rows per sequence and splits 70/15/15; scalar code uses 70/10/20.
- Image DLinear baseline runner and joint runner have different training objectives/defaults.
  Select the runner corresponding to the paper rather than combining their results.
- Validation averaging and true-xLSTM behavior differ from historical code. New runs must not
  be paired silently with old paper metrics.
- Small runtime smoke checks are recorded in VALIDATION.md; full paper experiments were not rerun.
- No original environment was changed or dependency installed during packaging or review.

Review fixes: minute-frequency compatibility with modern pandas; removed unused attention classes
and their unrelated dependencies; baseline and joint pixel RMSE including CSV column registration;
short hashed checkpoint names for Windows; different data/training flags get distinct output paths;
checkpoint loading respects --checkpoints and device; NaN/infinite inputs and channel mismatches fail
explicitly; label_len is restricted to zero for this independent-sample loader. Real xLSTM defaults
to its vanilla torch backend; --xlstm_backend cuda requests compiled kernels explicitly.
