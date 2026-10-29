"""Dependency-light checks: syntax, local imports, source preservation and sample isolation."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parent


def main():
    files = list(ROOT.rglob('*.py'))
    for file in files:
        tree = ast.parse(file.read_text(encoding='utf-8-sig'), filename=str(file))
        if ROOT / 'core' not in file.parents:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in {
                'data_provider', 'exp', 'models', 'layers', 'utils'
            }:
                module = ROOT / 'core' / node.module.replace('.', '/')
                assert module.with_suffix('.py').exists() or (module / '__init__.py').exists(), (file, node.module)
    # Original source preservation is checked when this folder is still in the original workspace.
    manifest = json.loads((ROOT / 'source_manifest.json').read_text(encoding='utf-8'))
    original_root = ROOT.parents[1]
    checked = 0
    for entry in manifest:
        original = original_root / entry['source']
        if original.is_file():
            assert hashlib.sha256(original.read_bytes()).hexdigest() == entry['source_sha256'], original
            checked += 1
    print(f'PASS: syntax for {len(files)} files, local imports, {checked} original source checksums')

    # Exercise the actual packaged loader, replacing only torch's Dataset base if unavailable.
    import numpy as np
    import pandas as pd
    try:
        import torch
    except ImportError:
        torch_module = types.ModuleType('torch')
        utils_module = types.ModuleType('torch.utils')
        data_module = types.ModuleType('torch.utils.data')
        data_module.Dataset = object
        sys.modules.update({'torch': torch_module, 'torch.utils': utils_module, 'torch.utils.data': data_module})
    sys.path.insert(0, str(ROOT / 'core'))
    from data_provider.data_loader import Dataset_Custom
    with tempfile.TemporaryDirectory() as folder:
        values = np.arange(20 * 48, dtype=float)
        frame = pd.DataFrame({'date': pd.date_range('2000-01-01', periods=len(values), freq='30min'),
                              'covariate': values % 11, 'target': values})
        frame.to_csv(Path(folder) / 'samples.csv', index=False)
        args = types.SimpleNamespace(sample_len=48, seed=7, scale=1)
        for seq_len, pred_len in [(15, 1), (24, 3), (24, 6), (24, 12)]:
            sets = [Dataset_Custom(args, folder, flag=flag, size=[seq_len, 0, pred_len],
                    features='MS', data_path='samples.csv', target='target', timeenc=0)
                    for flag in ['train', 'val', 'test']]
            train, val, test = sets
            ids = [set(ds.sample_indices) for ds in sets]
            assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
            assert len(set.union(*ids)) == 20
            assert np.allclose(train.blocks.reshape(-1, 2).mean(0), 0, atol=1e-10)
            for ds in sets:
                assert np.allclose(ds.scaler.mean_, train.scaler.mean_)
                for index in range(len(ds)):
                    x, y, _, _ = ds[index]
                    sid, offset = divmod(index, ds.windows_per_sample)
                    original = ds.inverse_transform(np.concatenate([x, y]))
                    first = ds.sample_indices[sid] * 48 + offset
                    assert np.allclose(original[:, -1], np.arange(first, first + seq_len + pred_len))
        frame.iloc[:-1].to_csv(Path(folder) / 'partial.csv', index=False)
        try:
            Dataset_Custom(args, folder, size=[15, 0, 1], data_path='partial.csv', target='target')
        except ValueError:
            pass
        else:
            raise AssertionError('Partial samples were not rejected')
    print('PASS: four window settings, independent sample splits, train-only scaling, inverse transform, partial sample rejection')


if __name__ == '__main__':
    main()
