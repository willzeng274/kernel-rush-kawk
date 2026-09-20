"""Compile the pinned attributed SGLang port; no device, model or execution."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from offline_sm90 import OUT, ROOT


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'engine' / 'sglang_decode_port.py'
    expected = 'cdda5808f9dd96b6e34be460ee050de9e6f431957b46f6faaaff5935df36dc2a'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    configs = []
    for batch in (1, 4, 16, 32):
        for sms in (114, 132):
            common = dict(SPLITS=8, BATCH=batch, SM_COUNT=sms)
            for cap in (544, 2080, 4097):
                configs.append(dict(
                    id=f'sglang_stage1_b{batch}_sms{sms}_cap{cap}',
                    source=source.name, kernel='sglang_grouped_stage1',
                    cap=cap, width=1, warps=4, stages=2, fusion=True,
                    constants=dict(common, CAP=cap, SCALE=128 ** -.5),
                    signature_types=dict(POS='*i64', MID='*fp32', LSE='*fp32')))
            configs.append(dict(
                id=f'sglang_stage2_b{batch}_sms{sms}',
                source=source.name, kernel='sglang_grouped_stage2',
                cap=544, width=1, warps=4, stages=2, fusion=True,
                constants=common,
                signature_types=dict(POS='*i64', MID='*fp32', LSE='*fp32')))
    results = []
    for config in configs:
        try:
            run = subprocess.run(
                [sys.executable, str(Path(__file__).with_name('offline_chain.py')), json.dumps(config)],
                capture_output=True, text=True, timeout=60)
            (OUT / (config['id'] + '.log')).write_text(run.stdout + '\n' + run.stderr)
            result = dict(config=config, returncode=run.returncode)
            if run.returncode == 0:
                result.update(json.loads((OUT / (config['id'] + '.json')).read_text()))
                assert result['source_sha256'] == expected
            else:
                result['error'] = run.stderr[-7000:]
        except Exception as error:
            result = dict(config=config, returncode=-1, error=str(error))
        results.append(result)
        print(json.dumps(result, default=str), flush=True)
        (OUT / 'sglang_summary.json').write_text(json.dumps(results, indent=2, default=str))
    if any(r['returncode'] != 0 or r.get('ptxas_returncode') != 0 for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
