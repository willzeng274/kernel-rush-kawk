"""CPU-only execution of pinned Triton binder/specialization code; no compiler."""
import ast
import collections
import functools
import hashlib
import inspect
import json
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
src=HERE/'snapshots/triton310_jit_primary.py'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='7542bf9254646331ed90ce16fe1bfb2caf87d77d748d84396aab44c8027d508b'
t=ast.parse(src.read_text())
names={'_normalize_ty','KernelParam','compute_spec_key','mangle_type','create_function_from_signature'}
chosen=[n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
chosen += [n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id in {'dtype2str','type_canonicalisation_dict'} for z in n.targets)]
ns={'inspect':inspect,'cached_property':functools.cached_property}
exec(compile(ast.Module(body=chosen,type_ignores=[]),str(src),'exec'),ns)
cls=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='JITFunction')
getconfig=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_get_config')
getconfig.body=[n for n in getconfig.body if not isinstance(n,ast.ImportFrom)]
Attrs=collections.namedtuple('AttrsDescriptor','divisible_by_16 equal_to_1')
ns.update(AttrsDescriptor=Attrs,JITFunction=types.SimpleNamespace(divisibility=16))
exec(compile(ast.Module(body=[getconfig],type_ignores=[]),str(src),'exec'),ns)
plan=json.loads((HERE/'COMPILE_PLAN.json').read_text())
gs=ROOT/'work/candidates/krxfty_wide64/engine/kernels/gemm.py'
defs={n.name:n for n in ast.parse(gs.read_text()).body if isinstance(n,ast.FunctionDef)}
class Pointer:
    def __init__(self,dtype):self.dtype=dtype
    def data_ptr(self):return 0x100000
results=[]
for case in plan['cases']:
    for phase in ['main','sum'] if case['reduction'] else ['main']:
        obj=case if phase=='main' else case['reduction']
        name='_skinny_kernel' if phase=='main' else '_sum_kernel'
        fn=defs[name]
        parameters=[inspect.Parameter(a.arg,inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation='tl.constexpr' if a.annotation else inspect.Parameter.empty) for a in fn.args.args]
        sig=inspect.Signature(parameters)
        kp=[ns['KernelParam'](i,p,False) for i,p in enumerate(parameters)]
        binder=ns['create_function_from_signature'](sig,kp)
        if phase=='main':
            values={**{n:Pointer('torch.float32' if n=='part_ptr' and case['constexpr']['SPLIT_K']>1 else 'torch.bfloat16') for n in case['tensor_signature']},**case['runtime_scalars'],**case['constexpr']}
            options={'num_warps':8,'num_stages':case['launch_options']['num_stages'],'debug':None}
        else:
            values={'part_ptr':Pointer('torch.float32'),'c_ptr':Pointer('torch.bfloat16'),**obj['runtime_scalars'],**obj['constexpr']}
            options={'num_warps':4,'debug':None}
        bound,sigspec,constvals,nonconst,extra=binder(**values,**options)
        attr=ns['_get_config'](types.SimpleNamespace(params=kp),*bound.values())
        signature={p.name:v for p,v in zip((p for p in kp if not p.is_constexpr),sigspec)}
        constants={p.name:v for p,v in zip(kp,bound.values()) if p.is_constexpr or p.num in attr.equal_to_1 or v is None}
        key=name+''.join(sigspec)+str((constvals,extra))
        results.append({'launch':case['id']+'_'+phase,'kernel':name,'native_key_sha256':hashlib.sha256(key.encode()).hexdigest(),
            'signature':signature,'constants':constants,'attrs':{'divisible_by_16':sorted(attr.divisible_by_16),'equal_to_1':sorted(attr.equal_to_1)},'options_explicit':options})
groups={}
for r in results:groups.setdefault(r['native_key_sha256'],[]).append(r['launch'])
res={'method':'Executed exact pinned Triton 3.1 Python binder/compute_spec_key/mangle_type/KernelParam and _get_config. Only _get_config relative import replaced by same-field namedtuple AttrsDescriptor. Kernel inspect.Signature reconstructed directly from AST args/annotations. No torch/GPU/Triton compiler import. Native compiler worker must revalidate these keys before compiling.',
    'jit_source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'jit_source_url':'https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/runtime/jit.py',
    'kernel_source_sha256':hashlib.sha256(gs.read_bytes()).hexdigest(),'launch_count':len(results),'distinct_keys':len(groups),'groups':groups,'launches':results}
(HERE/'NATIVE_BINDER_KEYS.json').write_text(json.dumps(res,indent=2)+'\n')
assert len(results)==16 and len(groups)==7
print(json.dumps({'launch_count':len(results),'distinct_keys':len(groups),'groups':list(groups.values())}))
