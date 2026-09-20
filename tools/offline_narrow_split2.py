"""Compile fixed split2 producer/consumer cases only; no GPU execution."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from offline_sm90 import OUT, ROOT


def main():
    OUT.mkdir(exist_ok=True)
    source=ROOT/'engine/narrow_split2.py'
    expected='b2d49ed7ebe454bce637d28f4fce0d7ceb6f37531823030cb496642fb4a51549'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    case_file=Path(__file__).with_name('narrow_split2_cases.json')
    assert hashlib.sha256(case_file.read_bytes()).hexdigest()=='92210200ef74b3890522c824f75f16b4591d07c6aa5e753f9c16e962d5a01b40'
    cases=json.loads(case_file.read_text())
    assert len(cases)==96 and len({c['id'] for c in cases})==96
    results=[]
    for case in cases:
        config=dict(case)
        config['kernel']=config.pop('function')
        config['signature_types']=config.pop('signature')
        config.update(source=source.name,cap=544,width=1)
        try:
            run=subprocess.run([sys.executable,str(Path(__file__).with_name('offline_chain.py')),json.dumps(config)],capture_output=True,text=True,timeout=90)
            (OUT/(config['id']+'.log')).write_text(run.stdout+'\n'+run.stderr)
            result=dict(config=config,returncode=run.returncode)
            if run.returncode==0:
                result.update(json.loads((OUT/(config['id']+'.json')).read_text()))
                assert result['source_sha256']==expected
            else:
                result['error']=run.stderr[-7000:]
        except Exception as error:
            result=dict(config=config,returncode=-1,error=str(error))
        results.append(result)
        print(json.dumps(result,default=str),flush=True)
        (OUT/'narrow_split2_summary.json').write_text(json.dumps(results,indent=2,default=str))
    if any(r['returncode']!=0 or r.get('ptxas_returncode')!=0 for r in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
