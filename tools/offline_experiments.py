"""Bounded controlled compiler experiments; no GPU execution."""
import json
import subprocess
import sys
from pathlib import Path
from offline_sm90 import OUT


def main():
    OUT.mkdir(exist_ok=True)
    configs = []
    for cap in (255, 256, 257, 544, 640, 2080, 4096, 4097):
        splits = (cap + 7 + 255) // 256
        constants = dict(CAP=cap, W=8, EPS=1e-6, D=128, SPLITS=splits,
                         SCALE=128**-.5, BLOCK_N=256,
                         BLOCK_S=1 << (splits-1).bit_length(), ROWS=64, BLOCK=128)
        for kind in ('qkv', 'attention', 'compact'):
            configs.append(('offline_chain.py', dict(
                kernel='tree_' + kind + '_kernel', cap=cap, width=8,
                warps=8 if kind == 'attention' else 4,
                id=f'tree_{kind}_c{cap}_w8', source='tree_kernels.py', constants=constants)))
        configs.append(('offline_chain.py', dict(
            kernel='chain_merge_kernel', cap=cap, width=8, warps=4,
            id=f'tree_merge_c{cap}_w8', source='recycled_kernels.py', constants=constants)))
    for batch in range(1,9):
        configs.append(('offline_chain.py', dict(
            kernel='chain_ids_kernel', cap=544, width=8, batch=batch,
            id=f'tree_ids_b{batch}_w8', source='recycled_kernels.py')))
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
