"""Bounded controlled compiler experiments; no GPU execution."""
import json
import subprocess
import sys
from pathlib import Path
from offline_sm90 import OUT


def main():
    OUT.mkdir(exist_ok=True)
    configs = []
    for index, (rows, batch, k, bk, splits, stages) in enumerate(
            [(64, 4, 2560, 128, 8, 3), (64, 16, 4096, 128, 4, 3),
             (64, 32, 9728, 128, 1, 3), (64, 4, 2560, 256, 4, 2),
             (64, 32, 9728, 256, 1, 2), (128, 4, 2560, 128, 8, 2),
             (128, 16, 4096, 128, 4, 2), (128, 32, 9728, 128, 1, 2)]):
        for variant in ('u8_prmt',):
            configs.append(('offline_sm90.py', dict(
                id=f'packed_{index}_{variant}', rows=rows, B=batch, K=k, BK=bk,
                SPLITS=splits, stages=stages, planes=True,
                source=f'planes_{variant}.py')))
    for cap in (255, 256, 257, 544, 640, 2080, 4096, 4097):
        for warps in (8,):
            for old in (False,):
                configs.append(('offline_chain.py', dict(
                    kernel='chain_attention_kernel', cap=cap, width=4, warps=warps,
                    id=f'attn_c{cap}_warp{warps}_' + ('old' if old else 'hoisted'),
                    source='recycled_kernels_old.py' if old else 'recycled_kernels.py')))
    for cap in (255, 256, 257, 544, 640, 2080, 4096, 4097):
        for kind in ('attention_split', 'fused_attention'):
            for warps in (4, 8):
                configs.append(('offline_chain.py', dict(
                    kernel='single_' + kind + '_kernel', cap=cap, width=1, warps=warps,
                    id=f'single_{kind}_c{cap}_warp{warps}', source='recycled_single.py')))
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
