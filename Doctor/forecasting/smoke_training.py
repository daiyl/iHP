"""One-epoch synthetic end-to-end checks; requires existing torch dependencies, installs nothing."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory(prefix='ihp_check_') as folder:
        folder = Path(folder)
        index = np.arange(20 * 48)
        pd.DataFrame({'date': pd.date_range('2000-01-01', periods=len(index), freq='30min'),
                      'insulin': np.sin(index / 7) + 1,
                      'carbohydrate': np.cos(index / 9) + 1,
                      'glucose': 120 + 10 * np.sin(index / 12)}).to_csv(folder / 'glucose.csv', index=False)
        command = [sys.executable, str(ROOT / 'run_experiments.py'),
                   '--scenario', 'blood_glucose', '--dataset', 'blood_glucose',
                   '--csv', str(folder / 'glucose.csv'), '--sample-len', '48', '--freq', 't',
                   '--models', 'LSTM', 'iTransformer', 'PatchTST', 'DLinear', 'TiDE', 'SCINet', 'TimesNet',
                   '--windows', '24:12', '--output-dir', str(folder / 'outputs'),
                   '--extra', '--train_epochs', '1', '--batch_size', '256', '--num_workers', '0',
                   '--use_gpu', 'false', '--d_model', '16', '--d_ff', '32', '--e_layers', '1',
                   '--n_heads', '4', '--num_kernels', '2']
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout[-10000:] + '\n' + result.stderr[-6000:])
        metrics = list((folder / 'outputs').rglob('metrics.npy'))
        assert len(metrics) == 7, len(metrics)
        for path in metrics:
            values = np.load(path)
            assert np.isfinite(values[:3]).all(), path
            run_dir = path.parents[2]
            preprocess = next(run_dir.rglob('preprocessing.npz'))
            with np.load(preprocess) as data:
                expected = np.load(path.parent / 'pred.npy') * data['scale'][-1] + data['mean'][-1]
            assert np.allclose(expected, np.load(path.parent / 'pred_original.npy'))
            print('PASS one-epoch training/test/artifacts:', run_dir.name, flush=True)
        # Check inference-only path loads the same checkpoint and supports original-unit evaluation.
        launch = next((folder / 'outputs').rglob('launch.json'))
        inference = json.loads(launch.read_text())['command'] + ['--is_training', '0', '--inverse']
        result = subprocess.run(inference, cwd=launch.parent, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout[-4000:] + '\n' + result.stderr[-4000:])
        assert 'Primary metrics scale: original' in result.stdout
        print('PASS inference-only checkpoint reload and original-scale evaluation')


if __name__ == '__main__':
    main()
