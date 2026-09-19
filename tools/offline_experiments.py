"""Bounded controlled compiler experiments; no GPU execution."""
import json
import subprocess
import sys
from pathlib import Path
from offline_sm90 import OUT


def main():
    OUT.mkdir(exist_ok=True)
    configs = []
    for m in (33,64,65,128,129,256):
        tiles=((128,128),(64,128)) if m<=64 else ((128,128),)
        for bm,bn in tiles:
            for sms in (114,132):
                configs.append(('offline_chain.py', dict(
                    id=f'large_decode_gu_m{m}_tile{bm}x{bn}_sms{sms}',
                    kernel='_persistent_dense', source='dense_prefill.py',
                    cap=544,width=1,warps=4,stages=4,fusion=True,
                    constants=dict(M=m,N=19456,K=2560,SMS=sms,BM=bm,BN=bn,BK=64,GROUP=8),
                    signature_types=dict(X='*bf16',W='*bf16',OUT='*bf16'))))
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
