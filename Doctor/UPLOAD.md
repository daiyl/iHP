# Uploading this staging folder

This folder contains publication source only; no environment, generated dataset, image
archive, checkpoint or unverified paper metrics is bundled. Original files remain in place.

1. Clone daiyl/iHP into a separate directory and create a branch.
2. Copy the **contents** of this `doctor/` folder into the clone's existing `Doctor/` directory.
   Preserve other existing Doctor files and review the replacement README.
3. From the clone root run `git add Doctor`, then `git diff --cached --stat` and
   `git diff --cached`. Check the staged files contain only intended source/docs.
4. Commit, push the branch, and open a pull request targeting main.

No GitHub upload or git initialization was performed while assembling this folder.
Before claiming exact paper reproducibility, reconcile dataset versions, seeds, metric
scales and historical parameters described in forecasting/PACKAGING_NOTES.md.
