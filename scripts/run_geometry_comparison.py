"""Run the approved eight-case comparison from one immutable source snapshot."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from run_acceptance import STAGES, parse_train_args, write_json
from summarize_zhang_mechanisms import summarize

CASES = (
    ('dcol', 'dcol', 'bptt', 150, 'cosine'),
    ('step', 'step', 'bptt', 150, 'cosine'),
    ('truncated', 'truncated_window', 'bptt', 150, 'cosine'),
    ('plateau', 'dcol', 'bptt', 150, 'plateau'),
    ('shac', 'dcol', 'shac', 32, 'cosine'),
    ('ahac', 'dcol', 'ahac', 32, 'cosine'),
    ('navrl_ppo', 'navrl', 'ppo', 32, 'fixed'),
    ('continuous', 'continuous', 'bptt', 150, 'cosine'),
)


def case_arguments(variant, algorithm, horizon, schedule):
    if algorithm == 'ppo':
        return []
    args = ['--variant', variant, '--algorithm', algorithm, '--timesteps', str(horizon),
            '--camera_batch_size', '64', '--lr_schedule', schedule, '--plateau_patience', '2',
            '--observation_normalization', '0', '--recovery_enabled', '1']
    if algorithm == 'ahac':
        args += ['--ahac_max_horizon', '300']
    parse_train_args(args)
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--parallel', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.parallel <= len(CASES):
        parser.error('parallel must be between 1 and 8')
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source'
    snapshot.mkdir()
    hashes = {}
    for folder in ('src', 'scripts'):
        shutil.copytree(root / folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__'))
        for file in sorted((snapshot / folder).rglob('*')):
            if file.is_file():
                hashes[str(file.relative_to(snapshot))] = hashlib.sha256(file.read_bytes()).hexdigest()
    (snapshot / 'changes.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root))
    write_json(snapshot / 'identity.json', {
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'files_sha256': hashes,
    })
    environment = dict(os.environ, PYTHONPATH=str(snapshot / 'src'), OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OMNI_KIT_ACCEPT_EULA='YES', PYTHONUNBUFFERED='1')
    manifest = {'study': 'signed_geometry_eight_cases', 'started_at': datetime.now(UTC).isoformat(),
                'num_envs': 1024, 'max_updates': None, 'stage_order': list(STAGES),
                'source': str(snapshot), 'parallel': args.parallel, 'runs': []}
    for name, variant, algorithm, horizon, schedule in CASES:
        method = 'navrl' if algorithm == 'ppo' else 'zhang'
        command = [sys.executable, str(snapshot / 'scripts/run_acceptance.py'), '--method', method,
                   '--output', str(output / name), '--num_envs', '1024', '--train_seed', '0', '--device', 'cuda:0']
        extra = case_arguments(variant, algorithm, horizon, schedule)
        if extra:
            command += ['--train_args', *extra]
        manifest['runs'].append({'name': name, 'path': name, 'method': method, 'variant': variant,
                                 'algorithm': algorithm, 'normalization': False, 'lr_schedule': schedule,
                                 'seed': 0, 'command': command, 'launch_status': 'queued'})
    manifest_path = output / 'manifest.json'
    write_json(manifest_path, manifest)

    pending = iter(manifest['runs'])
    active = []
    exhausted = False
    while active or not exhausted:
        remaining = []
        for process, entry in active:
            code = process.poll()
            if code is None:
                remaining.append((process, entry))
            else:
                entry.update(returncode=code, launch_status='exited')
        active = remaining
        while len(active) < args.parallel and not exhausted:
            entry = next(pending, None)
            if entry is None:
                exhausted = True
                break
            try:
                with (output / (entry['name'] + '.launcher.log')).open('w') as stream:
                    process = subprocess.Popen(entry['command'], cwd=snapshot, env=environment, stdout=stream,
                                               stderr=subprocess.STDOUT)
                entry.update(pid=process.pid, launch_status='running')
                active.append((process, entry))
            except OSError as error:
                entry.update(launch_status='failed_to_start', error=str(error))
        write_json(manifest_path, manifest)
        write_json(output / 'summary.json', summarize(output, manifest_path))
        if active:
            time.sleep(30)


if __name__ == '__main__':
    main()
