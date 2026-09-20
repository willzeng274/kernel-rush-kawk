"""Exact Q2 SM90 source compilation only; never GPU execution."""
import hashlib,json,subprocess,sys
from pathlib import Path
from offline_sm90 import OUT,ROOT
MANIFEST = {'source_sha256': {'q2_kernels.py': 'e0c8dd971b285f7ce51390f94ea715368bab6daf029eb27248cbcdf4aa5ed09e', 'speculative_kernels.py': 'ae14f25732a332b5497c3b2d2c17519346636638dd340e3e9dd6940e289c02fd', 'custom_kernels.py': 'dba53e9172f35494dbd3b1f8c0fc2ddfe0249bdd23a97b6169e70503d55f7f22'}, 'case_sha256': '0d42f1f0a73d86d434a657e6933fa8c5b9e6eec88787689cff1c5185b8a5fc72', 'cases': 37}
def main():
    OUT.mkdir(exist_ok=True)
    for name,digest in MANIFEST['source_sha256'].items():
        assert hashlib.sha256((ROOT/'engine'/name).read_bytes()).hexdigest()==digest
    case_file=Path(__file__).with_name('q2_cases.json')
    assert hashlib.sha256(case_file.read_bytes()).hexdigest()==MANIFEST['case_sha256']
    cases=json.loads(case_file.read_text());assert len(cases)==MANIFEST['cases']
    results=[]
    for config in cases:
        try:
            run=subprocess.run([sys.executable,str(Path(__file__).with_name('offline_chain.py')),json.dumps(config)],capture_output=True,text=True,timeout=90)
            (OUT/(config['id']+'.log')).write_text(run.stdout+'\n'+run.stderr)
            result=dict(config=config,returncode=run.returncode)
            if run.returncode==0:
                result.update(json.loads((OUT/(config['id']+'.json')).read_text()))
                assert result['source_sha256']==MANIFEST['source_sha256'][config['source']]
            else:result['error']=run.stderr[-7000:]
        except Exception as error:result=dict(config=config,returncode=-1,error=str(error))
        results.append(result);print(json.dumps(result,default=str),flush=True)
        (OUT/'q2_summary.json').write_text(json.dumps(results,indent=2,default=str))
    if any(r['returncode']!=0 or r.get('ptxas_returncode')!=0 for r in results):raise SystemExit(1)
if __name__=='__main__':main()
