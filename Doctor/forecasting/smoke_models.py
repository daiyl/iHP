"""Forward/backward interface checks using existing dependencies, with no package installation."""
import argparse
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


def main():
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / 'core'))
    models = ['LSTM', 'iTransformer', 'PatchTST', 'DLinear', 'TiDE', 'SCINet', 'TimesNet']
    count = 0
    for name in models:
        module = importlib.import_module('models.' + name)
        for channels in [3, 4]:
            for seq, pred in [(15, 1), (24, 3), (24, 6), (24, 12)]:
                config = SimpleNamespace(task_name='long_term_forecast', seq_len=seq, label_len=0,
                    pred_len=pred, enc_in=channels, dec_in=channels,
                    c_out=1 if name == 'TiDE' else channels, d_model=16, d_ff=32,
                    n_heads=4, e_layers=1, d_layers=1, dropout=0.0, factor=1,
                    embed='timeF', freq='d', activation='gelu', moving_avg=25,
                    individual=False, top_k=5, num_kernels=2)
                model = module.Model(config)
                x = torch.randn(2, seq, channels)
                output = model(x, torch.zeros(2, seq, 3), torch.zeros(2, pred, channels),
                               torch.zeros(2, pred, 3))
                assert output.shape[0] == 2 and output.shape[1] >= pred and output.shape[2] == channels, (name, output.shape)
                assert torch.isfinite(output).all(), name
                loss = output[:, -pred:, -1:].square().mean()
                loss.backward()
                assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), name
                count += 1
        print('PASS scalar forward/backward:', name, flush=True)
    sys.path.insert(0, str(ROOT / 'pneumonia'))
    import run_pixel_autoencoder_joint_models as joint
    parser_args = sys.argv
    sys.argv = [parser_args[0]]
    try:
        args = joint.parse_args()
    finally:
        sys.argv = parser_args
    args.tsl_root = ROOT / 'core'
    args.latent_dim = 4
    args.d_model = 16
    args.d_ff = 32
    args.n_heads = 4
    for name in models:
        adapter = joint.TSLForecastAdapter(name, args)
        output = adapter(torch.randn(2, args.seq_len + args.pred_len, args.latent_dim + 3))
        assert output.shape == (2, 1, 4), (name, output.shape)
        output.square().mean().backward()
        assert torch.isfinite(output).all(), name
        print('PASS image temporal adapter:', name, flush=True)
    print(f'PASS: {count} scalar checks and {len(models)} pneumonia adapter checks; xLSTM/Mamba require optional dependencies')


if __name__ == '__main__':
    main()
