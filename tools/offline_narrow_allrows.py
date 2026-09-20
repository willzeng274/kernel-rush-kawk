"""Compile all 128 exact ordinary narrow-MMA runtime row/family combinations."""
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
    config_source = ROOT / 'engine' / 'narrow_config.py'
    config_expected = 'd801fb705a333705e1bc629bfde48543e65c5a1f59d0cda0c80dbd80428bbccd'
    assert hashlib.sha256(config_source.read_bytes()).hexdigest() == config_expected
    spec = importlib.util.spec_from_file_location('narrow_config', config_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    configs = []
    for rows in range(1, 33):
        for family in module.SHAPES:
            config = module.narrow_config(rows, family)
            config.update(id=f'narrow_{family}_m{rows}', family=family, config_source_sha256=config_expected)
            configs.append(config)
    assert len(configs) == 128 and len({c['id'] for c in configs}) == 128
    (OUT / 'runtime_configurations.json').write_text(json.dumps(configs, indent=2))
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
        (OUT / 'narrow_allrows_summary.json').write_text(json.dumps(results, indent=2, default=str))
    if any(r['returncode'] != 0 or r.get('ptxas_returncode') != 0 for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
