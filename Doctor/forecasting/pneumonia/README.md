# Pneumonia: Section 5.3.2

Entry points: `run_pixel_autoencoder_dlinear.py` (baseline) and
`run_pixel_autoencoder_joint_models.py` (nine temporal models with joint pixel loss).
Run `python pneumonia/run_pixel_autoencoder_joint_models.py --source-dir PATH --models all --seq-len 15 --pred-len 1`.
These preserve separate original training pipelines; paper correspondence must be verified.
Use --output-dir to choose an artifact location; the default remains beside the runner.

Source structure: SOURCE/<group>/<sample>/observation_sequence.csv (200 rows; a1,a2,a3),
and <sample>/{pneumonia,fundus,oct}/frame_0000.png through frame_0199.png.
Images are RGB resized to 64×64 by default, scaled to [0,1] and stacked as nine channels.
Evaluate pixel MAE and RMSE (sqrt of pixel MSE), aggregated across channels/modalities.
Split is by independent sequences (70/15/15). Windows stay in a sequence.
Future intervention tokens are supplied: document this known-future-input assumption.

Patient's pre-trained observation generators remain in Patient and are needed for generation,
not for loading already generated images here. The forecasting convolutional autoencoder
and its training code are included in the baseline runner and reused by the joint runner.
No image data or pretrained weights are bundled. The model backend defaults to ../core.
