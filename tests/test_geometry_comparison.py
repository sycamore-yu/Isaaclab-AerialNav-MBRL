"""Verify the requested matrix, queue, frozen imports and failed-run retention."""
import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('geometry_comparison', SCRIPTS / 'run_geometry_comparison.py')
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)
sys.path.pop(0)


def test_eight_case_queue_preserves_failures_and_uses_snapshot(tmp_path, monkeypatch):
    source = tmp_path / 'checkout'
    for folder in ['src', 'scripts']:
        (source / folder).mkdir(parents=True)
        (source / folder / 'example.py').write_text('original = True\n')
    monkeypatch.setattr(comparison, '__file__', str(source / 'scripts/run_geometry_comparison.py'))
    output = tmp_path / 'runs'
    monkeypatch.setattr(sys, 'argv', ['run', '--output', str(output), '--parallel', '3'])
    monkeypatch.setattr(comparison.subprocess, 'check_output', lambda *a, **kw: 'head\n' if kw.get('text') else b'patch')
    monkeypatch.setattr(comparison.time, 'sleep', lambda _: None)
    monkeypatch.setattr(comparison, 'summarize', lambda *args: {'test_only': True})
    commands = []

    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(commands)
            commands.append(command)
            assert kwargs['env']['PYTHONPATH'] == str(output / 'source/src')
            assert kwargs['cwd'] == output / 'source'
            self.returncode = 2 if self.pid == 101 else 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(comparison.subprocess, 'Popen', Process)
    comparison.main()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert len(commands) == len(manifest['runs']) == 8
    assert manifest['max_updates'] is None
    assert manifest['runs'][1]['returncode'] == 2
    assert all(row['launch_status'] == 'exited' for row in manifest['runs'])
    by_name = {row['name']: row for row in manifest['runs']}
    assert by_name['step']['variant'] == 'step'
    assert by_name['truncated']['variant'] == 'truncated_window'
    assert by_name['navrl_ppo']['method'] == 'navrl'
    for row in manifest['runs']:
        cmd = row['command']
        assert '--max_iterations' not in cmd
        assert cmd[cmd.index('--num_envs') + 1] == '1024'
        if row['method'] == 'zhang':
            assert cmd[cmd.index('--observation_normalization') + 1] == '0'
            assert cmd[cmd.index('--plateau_patience') + 1] == '2'
    cmd = by_name['ahac']['command']
    assert cmd[cmd.index('--ahac_max_horizon') + 1] == '300'
    identity = json.loads((output / 'source/identity.json').read_text())
    assert set(identity['files_sha256']) == {'src/example.py', 'scripts/example.py'}
    assert (output / 'source/src/example.py').read_text() == 'original = True\n'
