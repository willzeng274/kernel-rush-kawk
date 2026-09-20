"""Execute pinned Python binder/option code locally; no compiler or GPU import."""
import ast,collections,dataclasses,functools,hashlib,inspect,json,os,types
from pathlib import Path
from typing import Any,Optional,Tuple
HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
jit=HERE/'snapshots/primary/jit.py';compiler=HERE/'snapshots/primary/nvidia_compiler.py'
assert sha(jit.read_bytes())=='7542bf9254646331ed90ce16fe1bfb2caf87d77d748d84396aab44c8027d508b'
tree=ast.parse(jit.read_text());names={'_normalize_ty','KernelParam','compute_spec_key','mangle_type','create_function_from_signature'}
selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
selected += [n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id in {'dtype2str','type_canonicalisation_dict'} for z in n.targets)]
ns=dict(inspect=inspect,cached_property=functools.cached_property)
exec(compile(ast.Module(body=selected,type_ignores=[]),str(jit),'exec'),ns)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JITFunction')
methods=[]
for name in ('_get_config','create_binder'):
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    fn.body=[n for n in fn.body if not isinstance(n,ast.ImportFrom)]
    methods.append(fn)
ns.update(AttrsDescriptor=collections.namedtuple('AttrsDescriptor','divisible_by_16 equal_to_1'),JITFunction=types.SimpleNamespace(divisibility=16),CompiledKernel=None,compile=None,ASTSource=None,make_backend=None)
exec(compile(ast.Module(body=methods,type_ignores=[]),str(jit),'exec'),ns)
ctree=ast.parse(compiler.read_text());options=next(n for n in ctree.body if isinstance(n,ast.ClassDef) and n.name=='CUDAOptions')
backend=next(n for n in ctree.body if isinstance(n,ast.ClassDef) and n.name=='CUDABackend')
parse=next(n for n in backend.body if isinstance(n,ast.FunctionDef) and n.name=='parse_options')
ns.update(dataclass=dataclasses.dataclass,Optional=Optional,Tuple=Tuple,Any=Any,Path=Path,os=os,__file__=str(compiler))
exec(compile(ast.Module(body=[options,parse],type_ignores=[]),str(compiler),'exec'),ns)
from common import verify_source, values_for, make_record, check_unique
manifest,plan=verify_source()
assert sha(compiler.read_bytes())=='a979896b9c0acfd41dd953b90bdc4b10968f7c0b45a286eae3f829aaddb2bb55'
ktree=ast.parse((HERE/'triple_kernels.py').read_text())
results=[]
for case in plan['cases']:
    kfn=next(n for n in ktree.body if isinstance(n,ast.FunctionDef) and n.name==case['kernel'])
    parameters=[inspect.Parameter(a.arg,inspect.Parameter.POSITIONAL_OR_KEYWORD,annotation='tl.constexpr' if a.annotation else inspect.Parameter.empty) for a in kfn.args.args]
    signature=inspect.Signature(parameters)
    params=[ns['KernelParam'](i,p,False) for i,p in enumerate(parameters)]
    obj=types.SimpleNamespace(params=params,signature=signature)
    ns['create_binder'](obj)
    explicit={**case['requested_options'],'debug':None}
    bound,sigspec,constvals,nonconst,extra=obj.binder(**values_for(case),**explicit)
    attrs=ns['_get_config'](obj,*bound.values())
    parsed=ns['parse_options'](types.SimpleNamespace(capability=90),explicit)
    results.append(make_record(case,obj,attrs,bound,sigspec,constvals,extra,explicit,parsed))
result=dict(method='Pinned Triton3.1 create_binder/binder/_get_config and CUDAOptions/parse_options primary AST executed with inert imported compiler names and same-field attrs. Exact decorated signature reconstructed. CI actual native binder and make_backend must agree for all16 BEFORE any build.',jit_source_sha256=sha(jit.read_bytes()),backend_source_sha256=sha(compiler.read_bytes()),kernel_source_sha256=manifest['module_sha256'],launches=results,dedup=check_unique(results),compile_calls=0,gpu_execution=False)
(HERE/'NATIVE_EXPECTED.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(local_native_preflight='PASS',launches=16,compiles=0,dedup=result['dedup'])))
