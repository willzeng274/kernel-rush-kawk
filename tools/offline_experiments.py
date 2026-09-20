"""Bounded controlled compiler experiments; no GPU execution."""
import json
import subprocess
import sys
from pathlib import Path
from offline_sm90 import OUT


def main():
    OUT.mkdir(exist_ok=True)
    configs = []
    for cap in (255, 256, 257, 544, 640, 2080, 4097):
        splits = (cap + 11 + 255) // 256
        for kind in ('qkv', 'attention', 'compact'):
            constants = dict(CAP=cap, W=12, D=128)
            if kind == 'qkv':
                constants.update(EPS=1e-6)
            elif kind == 'attention':
                constants.update(SPLITS=splits, SCALE=128**-0.5, BLOCK_N=256)
            configs.append(('offline_chain.py', dict(
                id=f'lookahead12_{kind}_c{cap}',
                kernel=f'lookahead_{kind}_kernel', source='lookahead_kernels.py',
                cap=cap, width=12, warps=8 if kind == 'attention' else 4,
                stages=1 if kind == 'attention' else 3,
                fusion=kind != 'qkv', constants=constants)))
    results = []
    for script, config in configs:
        try:
            p = subprocess.run([sys.executable, str(Path(__file__).with_name(script)),
                                json.dumps(config)], capture_output=True, text=True, timeout=60)
            (OUT / (config['id'] + '.log')).write_text(p.stdout + '\n' + p.stderr)
            result = {'config': config, 'returncode': p.returncode}
            if p.returncode == 0:
                result.update(json.loads((OUT / (config['id'] + '.json')).read_text()))
            else:
                result['error'] = p.stderr[-7000:]
        except Exception as error:
            result = {'config': config, 'returncode': -1, 'error': str(error)}
        results.append(result)
        print(json.dumps(result, default=str), flush=True)
        (OUT / 'experiment_summary.json').write_text(json.dumps(results, indent=2, default=str))
    if any(r['returncode'] != 0 for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
