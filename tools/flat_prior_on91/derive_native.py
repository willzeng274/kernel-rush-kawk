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
from launch_inputs import capture_launches,launch_receipt,CASES,SCHEDULE
from native_record import record_native,compare_all

def derive():
    expected=json.loads((HERE/'NATIVE_EXPECTED.json').read_text());actual={};receipts={}
    backend=types.SimpleNamespace(capability=90)
    backend.parse_options=lambda opts:ns['parse_options'](backend,opts)
    nodes={n.name:n for n in ast.parse((HERE/'device.py').read_text()).body if isinstance(n,ast.FunctionDef)}
    for label,key in [('base91','baseline'),('candidate','candidate')]:
        calls=capture_launches(label);rows=[]
        for call in calls:
            kfn=nodes[call['kernel']]
            ps=[inspect.Parameter(a.arg,inspect.Parameter.POSITIONAL_OR_KEYWORD,annotation='tl.constexpr' if a.annotation else inspect.Parameter.empty) for a in kfn.args.args]
            params=[ns['KernelParam'](i,p,False) for i,p in enumerate(ps)]
            obj=types.SimpleNamespace(params=params,signature=inspect.Signature(ps),debug=None)
            obj.create_binder=lambda obj=obj:ns['create_binder'](obj)
            obj._get_config=lambda *args,obj=obj:ns['_get_config'](obj,*args)
            record,_=record_native(call,obj,backend);rows.append(record)
        compare_all(rows,expected[key]);actual[key]=rows;receipts[key]=launch_receipt(calls)
    basekeys={r['complete_compile_key_sha256'] for r in actual['baseline']}
    candidatekeys={r['complete_compile_key_sha256'] for r in actual['candidate']}
    newkeys=candidatekeys-basekeys
    assert newkeys==set(expected['new_keys'])=={r['complete_compile_key_sha256'] for r in SCHEDULE}
    assert candidatekeys&basekeys==set(expected['excluded_unchanged_keys']) and len(newkeys)==56
    return dict(status='PASS',native_records_compared=512,all_recorded_fields_equal=True,new_keys=56,excluded_unchanged_keys=8,compiler_imported=False,compile_calls=0,gpu_execution=False),receipts

if __name__=='__main__':
    result,receipts=derive()
    print(json.dumps(result))
