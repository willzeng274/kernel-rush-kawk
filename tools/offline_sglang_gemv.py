"""Bounded research-only compile of the copied SGLang inline-PTX GEMV."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from offline_sm90 import OUT, ROOT


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'engine' / 'sglang_inline_gemv.py'
    expected = 'c322fdd7c2e9238f8cdbacf48286dbb133a82fe3fccfbce406437bd5fab6a87c'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location('sglang_inline_gemv', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    results = []
    for name, (n, k) in module.CONFIGS.items():
        assembly = module.make_asm(k)
        (OUT / f'sglang_gemv_{name}.ptx.inc').write_text(assembly)
        config = dict(id=f'sglang_gemv_{name}', source=source.name,
                      kernel='_sglang_inline_gemv', cap=544, width=1,
                      warps=8, stages=1, fusion=False,
                      constants=dict(N=n, K=k, ASM=assembly))
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
        (OUT / 'sglang_gemv_summary.json').write_text(json.dumps(results, indent=2, default=str))
    if any(r['returncode'] != 0 or r.get('ptxas_returncode') != 0 for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
