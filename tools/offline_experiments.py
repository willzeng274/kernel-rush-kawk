"""Bounded controlled compiler experiments; no GPU execution."""
import json
import subprocess
import sys
from pathlib import Path
from offline_sm90 import OUT


def main():
    OUT.mkdir(exist_ok=True)
    configs = []
    for alt in (1,3):
        for name in ('proposal_partials', 'proposal_merge'):
            configs.append(('offline_chain.py', dict(
                kernel=name, cap=544, width=8, warps=4, fusion=False, stages=3,
                id=f'{name}_alt{alt}', source='topk_proposals.py',
                constants=dict(V=151936, PARTS=75, ALT=alt, BLOCK=2048,
                               MERGE_BLOCK=128 if alt==1 else 256),
                signature_types=dict(Logits='*bf16', Greedy='*i64',
                                     PartialValues='*fp32', PartialIds='*i32', Out='*i64'))))
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
