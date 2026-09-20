"""CPU-only exact compiler check of two hypothetical cooperative-smoke geometries."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'compile-results'


def main():
    import triton
    from triton.compiler import ASTSource, AttrsDescriptor
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import _path_to_binary
    assert triton.__version__ == '3.1.0'
    OUT.mkdir(exist_ok=True)
    record = json.loads(Path(__file__).with_name('cooperative_smoke_source.json').read_text())
    source = ROOT / 'engine/cooperative_smoke_kernel.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record['smoke_kernel_sha256']
    spec = importlib.util.spec_from_file_location('cooperative_smoke_exact', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = module.cooperative_smoke
    assert kernel.arg_names == ['VALUES','OUT','OBSERVED','ARRIVALS','EPOCH','PROGRAMS','WORDS','READ_BLOCK']
    signature = {0:'*i64',1:'*i64',2:'*i32',3:'*i32',4:'*i32'}
    options = {'num_warps':4,'num_stages':1,'num_ctas':1}
    attrs = AttrsDescriptor(divisible_by_16={0,1,2,3,4})
    results = []
    for programs in (114, 132):
        name = f'programs_{programs}'
        begin = time.monotonic()
        constants = {5:programs,6:32,7:triton.next_power_of_2(programs*32)}
        cache = OUT / (name + '_cache')
        os.environ['TRITON_CACHE_DIR'] = str(cache)
        result = dict(id=name, source=record, signature=signature, constants=constants,
                      attributes=dict(divisible_by_16=[0,1,2,3,4],equal_to_1=[]),
                      options=options, cuda_execution=False, hypothetical_geometry=True,
                      started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            compiled = triton.compile(ASTSource(kernel,signature,constants,attrs),
                                      target=GPUTarget('cuda',90,32), options=options)
            result['compiler_success'] = True
            result['metadata'] = compiled.metadata._asdict()
            for ext in ('ttir','ttgir','ptx','llir'):
                (OUT/(name+'.'+ext)).write_text(compiled.asm[ext])
        except Exception as error:
            import traceback
            result.update(compiler_success=False,error=str(error),traceback=traceback.format_exc())
            # A rejected final assembly still leaves useful exact earlier stages.
            for ext in ('ttir','ttgir','ptx','llir'):
                files = list(cache.rglob('*.'+ext))
                if len(files) == 1:
                    shutil.copyfile(files[0],OUT/(name+'.'+ext))
        ptx_file = OUT/(name+'.ptx')
        if ptx_file.exists():
            ptx = ptx_file.read_text()
            match = re.search(r'\.visible\s+\.entry\s+cooperative_smoke\s*\((.*?)\)',ptx,re.S)
            arguments = [x.strip() for x in match.group(1).split(',') if x.strip()] if match else []
            result['emitted_entry_parameters'] = arguments
            result['exact_five_u64_abi'] = len(arguments)==5 and all(re.fullmatch(r'\.param\s+\.u64\s+\w+',x) for x in arguments)
            ptxas,_ = _path_to_binary('ptxas')
            assembly = subprocess.run([ptxas,'-v','--gpu-name=sm_90a',str(ptx_file),'-o',str(OUT/(name+'.cubin'))],
                                      capture_output=True,text=True,timeout=45)
            result.update(ptxas_returncode=assembly.returncode,ptxas_resources=assembly.stdout+assembly.stderr)
            (OUT/(name+'.ptxas.log')).write_text(result['ptxas_resources'])
        result['success'] = bool(result['compiler_success'] and result.get('exact_five_u64_abi') and result.get('ptxas_returncode')==0)
        result['elapsed_seconds'] = time.monotonic()-begin
        result['completed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        results.append(result)
        (OUT/(name+'.json')).write_text(json.dumps(result,indent=2,default=str)+'\n')
        (OUT/'cooperative_smoke_summary.json').write_text(json.dumps(results,indent=2,default=str)+'\n')
        print(json.dumps(result,default=str),flush=True)
    if len(results)!=2 or any(not r['success'] for r in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
