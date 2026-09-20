"""Compile one fixed N16 output/down geometry,64 cases, without GPU execution."""
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from offline_sm90 import OUT, ROOT

SOURCE_SHA = '5906670d360d3856d05ad960ecd5bbbbfd12d1655f882d2cc09b9ae0b69cab67'
CONFIG_SHA = '3d798cfc45be5f7e265745465214885f1d7cc45b76377c07a0e70b3ef45fe194'
CASES_SHA = '4b965b0cd75876385a03b42c621455fbd52ec105e70c6e961d63afd57bb206e0'


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'engine/narrow_bf16.py'
    config_path = ROOT / 'engine/narrow_n16_config.py'
    case_file = Path(__file__).with_name('narrow_n16_cases.json')
    for path, expected in ((source,SOURCE_SHA),(config_path,CONFIG_SHA),(case_file,CASES_SHA)):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location('narrow_n16_config',config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = json.loads(case_file.read_text())
    assert len(cases) == len({c['id'] for c in cases}) == 64
    assert {(c['family'],c['constants']['ROWS']) for c in cases} == {(f,b) for f in ('output','down') for b in range(1,33)}
    results=[]
    for case in cases:
        runtime=module.narrow_config(case['constants']['ROWS'],case['family'])
        assert all(case[key] == (list(value) if isinstance(value,tuple) else value) for key,value in runtime.items())
        config=dict(case)
        config['kernel']=config.pop('function')
        config['signature_types']=config.pop('signature')
        config.update(source=source.name,cap=544,width=1,config_source_sha256=CONFIG_SHA)
        started=time.monotonic()
        result=dict(config=config,returncode=-1)
        try:
            run=subprocess.run([sys.executable,str(Path(__file__).with_name('offline_chain.py')),json.dumps(config)],capture_output=True,text=True,timeout=90)
            (OUT/(config['id']+'.log')).write_text(run.stdout+'\n'+run.stderr)
            result['returncode']=run.returncode
            if run.returncode != 0:
                raise RuntimeError(run.stderr[-7000:])
            result.update(json.loads((OUT/(config['id']+'.json')).read_text()))
            assert result['source_sha256']==SOURCE_SHA and result['ptxas_returncode']==0
            resource=result['ptxas_resources']
            assert '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in resource, resource
            registers=int(re.search(r'Used (\d+) registers',resource).group(1))
            expected_shared=57344 if config['constants']['TILE_ROWS']==16 else 86016
            assert result['metadata']['shared']<=expected_shared, 'larger shared allocation than predicted'
            ptx=(OUT/(config['id']+'.ptx')).read_text()
            assert 'mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32' in ptx
            assert 'cp.async.' in ptx and 'wgmma.' not in ptx
            assert 'cvt.rn.bf16' in ptx
            result['audit']=dict(registers=registers,shared_limit=expected_shared,zero_stack_spills=True,ordinary_bf16_mma=True,async_copy=True,bf16_final_cast=True)
        except Exception as error:
            result['audit_failure']=str(error)
        result['elapsed_seconds']=round(time.monotonic()-started,3)
        results.append(result)
        (OUT/'narrow_n16_summary.json').write_text(json.dumps(results,indent=2,default=str))
        print(json.dumps(result,default=str),flush=True)
        if 'audit_failure' in result:
            raise SystemExit(1)
    assert len(results)==64


if __name__=='__main__':
    main()
