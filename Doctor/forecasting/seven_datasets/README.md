# seven_datasets

Use the shared launcher with explicit CSV, sample length and sampling frequency.
Configurations enumerate the requested paper experiment scope; historical hyperparameters are not verified.
Metrics: MSE/MAE with the scale stated explicitly. Generated datasets remain in Patient.

Dataset keys: `delayed_response`, `periodicity`, `synergistic_effect`, `tolerance`,
`dependency_induced_deterioration`, `allergic_response`, `antagonistic_effect`.
Supply each dataset CSV explicitly. Local filename prefixes such as de/pe/sy/to/dep/al/an
must be verified against generation code before assigning paper labels or noisy variants.
The loader expects date, three intervention columns and obs; use features MS to predict obs.
No generator or Patient code is copied into this module.
