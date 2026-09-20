"""Execute pinned Python binder/option code locally; no compiler or GPU import."""
import ast,collections,dataclasses,functools,hashlib,inspect,json,os,types
from pathlib import Path
from typing import Any,Optional,Tuple
HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
jit=HERE/'snapshots/primary/jit.py';compiler=HERE/'snapshots/primary/nvidia_compiler.py'
assert sha(jit.read_bytes())=='7542bf9254646331ed90ce16fe1bfb2caf87d77d748d84396aab44c8027d508b'
assert sha(compiler.read_bytes())=='a979896b9c0acfd41dd953b90bdc4b10968f7c0b45a286eae3f829aaddb2bb55'
assert os.environ.get('TRITON_DEBUG','0')!='1'
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
from launch_inputs import capture_launches,launch_receipt,serialize
calls=capture_launches();records=[]
for call in calls:
    kfn=next(n for n in ast.parse((HERE/'down_device.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name==call['kernel'])
    assert len(kfn.decorator_list)==1 and isinstance(kfn.decorator_list[0],ast.Attribute) and kfn.decorator_list[0].attr=='jit'
    params=[inspect.Parameter(a.arg,inspect.Parameter.POSITIONAL_OR_KEYWORD,annotation='tl.constexpr' if a.annotation else inspect.Parameter.empty) for a in kfn.args.args]
    signature=inspect.Signature(params)
    params=[ns['KernelParam'](i,p,False) for i,p in enumerate(params)]
    obj=types.SimpleNamespace(params=params,signature=signature)
    ns['create_binder'](obj)
    kwargs=dict(call['kwargs']);kwargs['debug']=None
    bound,spec,const,nonconst,extra=obj.binder(*call['args'],**kwargs)
    attrs=ns['_get_config'](obj,*bound.values())
    signature={obj.params[i].name:v for i,v in zip(obj.non_constexpr_indices,spec)}
    constants={p.name:v for p,v in zip(params,bound.values()) if p.is_constexpr or p.num in attrs.equal_to_1 or v is None}
    parsed=dict(ns['parse_options'](types.SimpleNamespace(capability=90),kwargs).__dict__)
    parsed['extern_libs']=[['libdevice','<TRITON_LIBDEVICE>']]
    key=''.join(spec)+str((const,extra))
    record=dict(case=call['case'],kernel=call['kernel'],grid=call['grid'],
      native_key_sha256=sha((call['kernel']+key).encode()),jit_cache_key_sha256=sha(key.encode()),jit_cache_key=key,
      bound_arguments={k:serialize(v) for k,v in bound.items()},signature_specialization=spec,constexpr_values=const,
      signature=signature,constants=constants,attrs=dict(divisible_by_16=sorted(attrs.divisible_by_16),equal_to_1=sorted(attrs.equal_to_1)),options_explicit=extra,parsed_options=parsed)
    records.append(record)
result=dict(method='Execute exact frozen DownS4Matmul class against CPU tensor/launch stand-ins, then pinned Triton 3.1 create_binder/binder/_get_config and CUDAOptions/parse_options from primary AST. No compiler/Torch/GPU import. Native driver must match every launch field before any build.',
 jit_source_sha256=sha(jit.read_bytes()),backend_source_sha256=sha(compiler.read_bytes()),kernel_source_sha256=sha((HERE/'down_device.py').read_bytes()),
 wrapper_source_sha256=sha((HERE/'snapshots/engine/kernels/down_s4.py').read_bytes()),launches=records,wrapper_launch_receipt=launch_receipt(calls),compile_calls=0,gpu_execution=False)
(HERE/'NATIVE_EXPECTED.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(local_native_preflight='PASS',launches=2,compiles=0,keys={r['case']:r['native_key_sha256'] for r in records})))
