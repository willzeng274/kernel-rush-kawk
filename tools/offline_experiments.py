"""Bounded controlled compiler experiments; no GPU execution."""
import json
import subprocess
import sys
from pathlib import Path
from offline_sm90 import OUT


def main():
    OUT.mkdir(exist_ok=True)
    configs = []
    for batch,prompt,output in ((4,2048,32),(16,512,128),(3,257,8),(2,259,1)):
        prefix=prompt//2
        for route,rows,span,begin in (
            ('full',batch*prompt,prompt,0),
            ('prefix',prefix,prefix,0),
            ('suffix',batch*(prompt-prefix),prompt-prefix,prefix)):
            configs.append(('offline_chain.py', dict(
                id=f'prefix_span_b{batch}_s{prompt}_n{output}_{route}',
                kernel='prefill_qkv_rope_cache_kernel', source='prefix_span_prefill.py',
                cap=prompt+output,width=1,warps=4,stages=3,fusion=False,
                constants=dict(ROWS=rows,SPAN=span,CAP=prompt+output,
                               EPS=1e-6,R=4,NQ=32,NKV=8,D=128,START=begin),
                signature_types={name:'*bf16' for name in
                                 ('QKV','QW','KW','COS','SIN','Q','KC','VC')})))
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
