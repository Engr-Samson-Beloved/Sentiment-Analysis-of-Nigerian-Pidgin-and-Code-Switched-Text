"""Install project dependencies and verify the synthetic workflow; Python 3.11+."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheelhouse', type=Path, help='Optional local directory of dependency wheels for offline installation')
    parser.add_argument('--skip-install', action='store_true')
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error('Python 3.11 or newer is required')
    if sys.prefix == sys.base_prefix:
        parser.error('Activate the project virtual environment first')
    output = ROOT / 'results' / 'metrics' / ('smoke-' + time.strftime('%Y%m%d-%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    completed = []
    def run(*arguments: str) -> None:
        command = [sys.executable, *map(str, arguments)]
        print('Running:', ' '.join(command), flush=True)
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout, end='', flush=True)
        (output / f'step-{len(completed):02d}.log').write_text(result.stdout, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Command failed with exit {result.returncode}: {command}')
        completed.append(command)
    def cli(command: str, *arguments: str) -> None:
        run('-m', 'naija_sentiment.cli', command, *arguments)
    try:
        if not args.skip_install:
            flags = ['--no-index', '--find-links', str(args.wheelhouse.resolve())] if args.wheelhouse else []
            run('-m', 'pip', 'install', '--retries', '0', *flags, '-r', 'requirements.txt')
        run('-m', 'pytest', '-q')
        fixture = ROOT / 'data/samples/synthetic.csv'
        annotations = ROOT / 'data/samples/synthetic_annotations.csv'
        cli('init-templates', '--output', output / 'templates')
        cli('validate', '--input', fixture, '--output', output / 'validation')
        cli('audit-permissions', '--input', fixture, '--permissions', ROOT / 'data/samples/synthetic_permissions.csv', '--output', output / 'permissions')
        cli('audit-privacy', '--input', fixture, '--output', output / 'privacy')
        for command in ['agreement', 'adjudicate']:
            cli(command, '--input', fixture, '--annotations', annotations, '--output', output / command)
        splits = output / 'splits'
        cli('split', '--input', output / 'adjudicate/consensus.csv', '--output', splits, '--dataset-config', 'configs/dataset.yaml')
        cli('train-svm', '--train', splits / 'train.csv', '--validation', splits / 'validation.csv', '--output', output / 'svm', '--config', 'configs/experiments.yaml')
        cli('evaluate', '--input', splits / 'validation.csv', '--model', output / 'svm/model.joblib', '--output', output / 'reloaded')
        import pandas as pd
        original = pd.read_csv(output / 'svm/predictions.csv')
        reloaded = pd.read_csv(output / 'reloaded/predictions.csv')
        pd.testing.assert_frame_equal(original, reloaded)
        lock = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        (output / 'requirements-lock.txt').write_text(lock, encoding='utf-8')
    except (RuntimeError, AssertionError) as exc:
        (output / 'verification.json').write_text(json.dumps({'status': 'failed', 'completed': completed, 'error': str(exc)}, indent=2))
        print(f'Verification incomplete: {exc}\nLogs: {output}', file=sys.stderr)
        return 1
    (output / 'verification.json').write_text(json.dumps({'status': 'passed', 'completed': completed, 'synthetic_only': True}, indent=2))
    print(f'Synthetic workflow passed. Artifacts: {output}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
