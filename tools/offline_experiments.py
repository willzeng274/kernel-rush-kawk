"""Bounded controlled compiler experiments; no GPU execution."""
import json
import subprocess
import sys
from pathlib import Path
from offline_sm90 import OUT


def main():
    OUT.mkdir(exist_ok=True)
    configs = []
    for name,n,k in (('gateup',19456,2560),('down',2560,9728),
                      ('qkv',6144,2560),('output',2560,4096)):
        first_excluded = 8 * 10**12 // (2 * n * k * 6) + 1
        for m in (first_excluded,65536):
            for bm,bn in ((128,128),(64,256)):
                for sms in (114,132):
                    configs.append(('offline_chain.py', dict(
                        id=f'dense_{name}_m{m}_tile{bm}x{bn}_sms{sms}',
                        kernel='_persistent_dense', source='dense_prefill.py',
                        cap=544,width=1,warps=4,stages=4,fusion=True,
                        constants=dict(M=m,N=n,K=k,SMS=sms,BM=bm,BN=bn,BK=64,GROUP=8),
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
