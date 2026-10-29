"""Launch the four scalar forecasting settings without duplicating the framework."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['seven_datasets', 'blood_glucose'], required=True)
    parser.add_argument('--dataset', required=True, help='Dataset key in the scenario JSON')
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--sample-len', type=int, required=True)
    parser.add_argument('--freq', choices=['s', 't', 'h', 'd', 'b', 'w', 'm', 'a'], required=True,
                        help='Use t for minute-level data, including half-hourly glucose records')
    parser.add_argument('--models', nargs='+', default=['all'])
    parser.add_argument('--windows', nargs='+', choices=['15:1', '24:3', '24:6', '24:12'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scale', type=int, choices=[0, 1], default=1)
    parser.add_argument('--inverse', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                        help='Additional core flags, e.g. --extra --train_epochs 100 --patience 10')
    args = parser.parse_args()
    config = json.loads((ROOT / args.scenario / 'configs/experiments.json').read_text(encoding='utf-8'))
    if args.dataset not in config['datasets']:
        parser.error('Choose dataset from: ' + ', '.join(config['datasets']))
    models = config['models'] if args.models == ['all'] else args.models
    if any(model not in config['models'] for model in models):
        parser.error('Unknown model; choose from ' + ', '.join(config['models']))
    windows = [list(map(int, window.split(':'))) for window in args.windows] if args.windows else config['windows']
    if args.sample_len < max(seq + pred for seq, pred in windows):
        parser.error('sample-len must cover each complete input + prediction window')
    protected = {'--model', '--model_id', '--data', '--root_path', '--data_path', '--target',
                 '--features', '--sample_len', '--seq_len', '--pred_len', '--seed', '--scale',
                 '--freq', '--inverse', '--checkpoints', '--label_len', '--des', '--enc_in', '--dec_in'}
    protected.add('--c_out')
    if any(token.split('=')[0] in protected for token in args.extra):
        parser.error('--extra cannot override dataset, identity, window, scale or split flags')
    csv_path = args.csv.resolve()
    data_hash = None
    channels = 4 if args.scenario == 'seven_datasets' else 3
    if not args.dry_run:
        if not csv_path.is_file():
            parser.error('CSV does not exist: ' + str(csv_path))
        data_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    for model in models:
        for seq_len, pred_len in windows:
            tag = f'{args.dataset}_{model}_{seq_len}to{pred_len}_seed{args.seed}_scale{args.scale}_inverse{int(args.inverse)}'
            command = [sys.executable, str(ROOT / 'core/run.py'),
                       '--model', model, '--model_id', tag, '--data', 'custom',
                       '--root_path', str(csv_path.parent), '--data_path', csv_path.name,
                       '--features', 'MS', '--target', config['datasets'][args.dataset]['target'],
                       '--sample_len', str(args.sample_len), '--seq_len', str(seq_len),
                       '--label_len', '0', '--pred_len', str(pred_len), '--freq', args.freq,
                       '--enc_in', str(channels), '--dec_in', str(channels),
                       '--c_out', str(channels), '--seed', str(args.seed), '--scale', str(args.scale)]
            if args.inverse:
                command.append('--inverse')
            command.extend(args.extra)
            # Different data, sampling periods or training flags must not share checkpoints.
            identity = json.dumps({'arguments': command[2:], 'csv_sha256': data_hash}, sort_keys=True)
            signature = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:10]
            run_dir = args.output_dir.resolve() / f'{tag}_{signature}'
            print(subprocess.list2cmdline(command), flush=True)
            if not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / 'launch.json').write_text(json.dumps({
                    'command': command, 'csv_sha256': data_hash, 'configuration_status': config['status']
                }, indent=2), encoding='utf-8')
                subprocess.run(command, cwd=run_dir, check=True)


if __name__ == '__main__':
    main()
