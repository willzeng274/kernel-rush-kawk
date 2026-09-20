"""Compile 29 fixed independent narrow-MMA research cases without a GPU."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from offline_sm90 import OUT, ROOT


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'engine' / 'narrow_bf16.py'
    expected = '5906670d360d3856d05ad960ecd5bbbbfd12d1655f882d2cc09b9ae0b69cab67'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location('narrow_bf16', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    configs = module.configurations()
    assert len(configs) == 29 and len({c['id'] for c in configs}) == 29
    results = []
    for config in configs:
        config.update(source=source.name, kernel='narrow_projection', cap=544, width=1)
        try:
            run = subprocess.run(
                [sys.executable, str(Path(__file__).with_name('offline_chain.py')), json.dumps(config)],
                capture_output=True, text=True, timeout=90)
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
        (OUT / 'narrow_summary.json').write_text(json.dumps(results, indent=2, default=str))
    if any(r['returncode'] != 0 or r.get('ptxas_returncode') != 0 for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
