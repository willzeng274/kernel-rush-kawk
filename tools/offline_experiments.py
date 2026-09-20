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
        splits = (cap+11+127)//128
        configs.append(('offline_chain.py', dict(
            id=f'lookahead12_merge_n128_c{cap}',
            kernel='chain_merge_kernel', source='recycled_kernels.py',
            cap=cap, width=12, warps=4, stages=3, fusion=True,
            constants=dict(SPLITS=splits, BLOCK_S=1 << (splits-1).bit_length(), D=128))))
    for batch in (1,4,16):
        configs.append(('offline_chain.py', dict(
            id=f'lookahead12_ids_b{batch}', kernel='chain_ids_kernel',
            source='recycled_kernels.py', cap=544, width=12, warps=4, stages=3,
            fusion=True, constants=dict(W=12, ROWS=batch*12, BLOCK=128))))
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
