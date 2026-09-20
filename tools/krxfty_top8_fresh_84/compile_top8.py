"""Prepared only: two exact Triton3.1 SM90 cases; no GPU/driver required.

Requires root authorization before running. Run under a180-second process
limit. The frozen device module is extracted verbatim from engine functions;
manifest and AST parity are checked before compilation. This script has not
been executed during candidate preparation.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

HERE=Path(__file__).resolve().parent


def main():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import _path_to_binary
    from triton.compiler import ASTSource
    import device_top8

    assert triton.__version__=='3.1.0'
    manifest=json.loads((HERE/'SOURCE_MANIFEST.json').read_text())
    device=HERE/'device_top8.py'
    assert hashlib.sha256(device.read_bytes()).hexdigest()==manifest['device_top8_sha256']
    assert hashlib.sha256((HERE/'compile_top8.py').read_bytes()).hexdigest()==manifest['compile_harness_sha256']
    functions=[node for node in ast.parse(device.read_text()).body if isinstance(node,ast.FunctionDef)]
    body_hash=hashlib.sha256('\n'.join(ast.dump(node) for node in functions).encode()).hexdigest()
    assert body_hash==manifest['device_function_ast_sha256']
    class AlignedPointer:
        def data_ptr(self):return 0x100000
    cases=[('top8_partials',device_top8.top8_partials,
            {0:'*bf16',1:'*i32',2:'*i32'},{'V':151936,'PARTS':75,'BLOCK':2048},[0,1,2,3,5],['N',75]),
           ('top8_merge',device_top8.top8_merge,
            {0:'*i32',1:'*i32',2:'*i32'},{'PARTS':75,'MERGE_BLOCK':1024},[0,1,2,4],['N'])]
    outdir=HERE/'compile-results';outdir.mkdir(exist_ok=True)
    results=[]
    for name,kernel,signature,constants,divisible,grid in cases:
        names=kernel.arg_names
        assert set(kernel.constexprs)==set(range(3,len(names)))
        arguments={n:(AlignedPointer() if i<3 else constants[n]) for i,n in enumerate(names)}
        attrs=kernel._get_config(*(arguments[n] for n in names))
        assert sorted(attrs.divisible_by_16)==divisible and not attrs.equal_to_1
        options={'num_warps':4,'num_stages':3,'enable_fp_fusion':False}
        t0=time.monotonic()
        compiled=triton.compile(ASTSource(kernel,signature,{names.index(n):v for n,v in constants.items()},attrs),
                                target=GPUTarget('cuda',90,32),options=options)
        elapsed=time.monotonic()-t0
        stem=outdir/name
        for ext in ('ttir','ttgir','llir','ptx'):
            stem.with_suffix('.'+ext).write_text(compiled.asm[ext])
        ptxas,_=_path_to_binary('ptxas')
        assembled=subprocess.run([ptxas,'-v','--gpu-name=sm_90a',str(stem.with_suffix('.ptx')),'-o',str(stem.with_suffix('.cubin'))],
                                  capture_output=True,text=True,timeout=45)
        log=assembled.stdout+assembled.stderr
        stem.with_suffix('.ptxas.log').write_text(log)
        ptx=compiled.asm['ptx']
        inventory={key:len(re.findall(pattern,ptx)) for key,pattern in {
            'shuffle':r'\bshfl\.', 'barrier':r'\bbar\.sync',
            'local_load':r'\bld\.local', 'local_store':r'\bst\.local',
            'global_load':r'\bld\.global', 'global_store':r'\bst\.global'}.items()}
        result=dict(case=name,source_sha256=manifest['engine_top8_sha256'],device_sha256=manifest['device_top8_sha256'],
                    signature=signature,constants=constants,options=options,grid=grid,
                    attrs={'divisible_by_16':divisible,'equal_to_1':[]},target='cuda-sm90-warp32',
                    cuda_execution=False,triton=triton.__version__,compile_seconds=elapsed,
                    metadata=compiled.metadata._asdict(),ptxas_returncode=assembled.returncode,
                    ptxas_resources=log,inventory=inventory,
                    hashes={ext:hashlib.sha256(compiled.asm[ext].encode()).hexdigest() for ext in ('ttir','ttgir','llir','ptx')})
        stem.with_suffix('.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
        results.append(result)
        (outdir/'summary.json').write_text(json.dumps(results,indent=2,default=str)+'\n')
        print(json.dumps(result,default=str),flush=True)
        assert assembled.returncode==0


if __name__=='__main__':main()
