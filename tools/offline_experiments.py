"""Bounded controlled compiler experiments; no GPU execution."""
import json
import subprocess
import sys
from pathlib import Path
from offline_sm90 import OUT


def main():
    OUT.mkdir(exist_ok=True)
    configs = []
    for batch in (8,16,24,32):
        for rows,bk,splits in ((64,256,1),(128,128,1)):
            configs.append(('offline_sm90.py', dict(
                id=f'tree_gu_m{batch}_tile{rows}_k{bk}_s{splits}',
                rows=rows, B=batch, K=2560, BK=bk, SPLITS=splits,
                stages=2, planes=False, part_type='*bf16')))
    configs.append(('offline_sm90.py', dict(
        id='tree_gu_m8_tile128_k128_s2', rows=128, B=8, K=2560,
        BK=128, SPLITS=2, stages=2, planes=False)))
    configs.append(('offline_chain.py', dict(
        id='tree_gu_m8_merge_s2', kernel='_hopper_merge', cap=544,width=8,
        source='hopper_gemm.py',warps=4,fusion=True,stages=3,
        constants=dict(SPLITS=2,BLOCK=512),
        signature_types=dict(PART='*fp32',OUT='*bf16',N='i32',OUT_ROW='i32',TOTAL='i32'))))
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
