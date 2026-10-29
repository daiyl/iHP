# blood_glucose

Use the shared launcher with explicit CSV, sample length and sampling frequency.
Configurations enumerate the requested paper experiment scope; historical hyperparameters are not verified.
Metrics: MSE/MAE with the scale stated explicitly. Generated datasets remain in Patient.

Dataset key: `blood_glucose`. Schema: date, insulin, carbohydrate, glucose.
Use `--freq t` for the existing half-hourly records (the minute-level calendar feature set).
Specify sample length from the actual generation configuration; do not mix 48-step and
200-step datasets. Check upstream units/preprocessing before interpreting original-unit errors.
