"""Run an isolated, small end-to-end trial (not a paper-quality training run)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learned-reference', action='store_true',
                        help='Also train and exercise the optional offline model')
    args = parser.parse_args()
    run_dir = ROOT / 'outputs' / 'trials' / time.strftime('%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    # Write independent configs so repeated trials do not overwrite each other.
    import yaml
    for stage in ['reference', 'train_wcet', 'train_scheduler', 'deployment']:
        config = yaml.safe_load((ROOT / f'configs/quick_{stage}.yaml').read_text())
        config['config']['wcet_schedule_path'] = str(run_dir / 'reference.yaml')
        config['config']['runs_path'] = str(run_dir / ('wcet' if stage == 'train_wcet' else 'scheduler'))
        (run_dir / f'{stage}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    results = []

    def run(stage, *flags):
        command = [sys.executable, '-m', 'retnas.retnas', str(run_dir / f'{stage}.yaml'), *flags]
        print('Running', stage, *flags, flush=True)
        with (run_dir / f'{stage}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        results.append({'stage': stage, 'exit_code': result.returncode})
        (run_dir / 'status.json').write_text(json.dumps(results, indent=2) + '\n')
        if result.returncode:
            raise SystemExit(f'{stage} failed; inspect {run_dir / (stage + ".log")}')

    if args.learned_reference:
        run('train_wcet')
        wcet = next((run_dir / 'wcet').glob('*/actor_last.pth'))
        run('reference', '--compute-wcet', '--reference-method', 'model', '--wcet-model', str(wcet))
    else:
        run('reference', '--compute-wcet', '--reference-method', 'edf')
    run('train_scheduler')
    model = next((run_dir / 'scheduler').glob('*/actor_last.pth'))
    run('deployment', '--test', '--model', str(model))
    print(f'Trial complete. Logs, models, configurations and reference: {run_dir}')


if __name__ == '__main__':
    main()
