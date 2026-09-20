"""Common native record serialization, checked against independently frozen expectations."""
import hashlib,json
from pathlib import Path
from launch_inputs import serialize
HERE=Path(__file__).resolve().parent
SPANS=json.loads((HERE/'SOURCE_SPANS.json').read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()
def record_native(call,kernel,backend):
    assert kernel.debug is None
    kernel.create_binder()
    kwargs=dict(call['kwargs']);kwargs['debug']=kernel.debug
    bound,spec,const,nonconst,extra=kernel.binder(*call['args'],**kwargs)
    attrs=kernel._get_config(*bound.values())
    signature={kernel.params[i].name:v for i,v in zip(kernel.non_constexpr_indices,spec)}
    constants={p.name:v for p,v in zip(kernel.params,bound.values()) if p.is_constexpr or p.num in attrs.equal_to_1 or v is None}
    parsed=backend.parse_options(kwargs)
    assert parsed.debug is None and parsed.num_stages==3 and parsed.num_ctas==1 and parsed.enable_fp_fusion is True
    assert parsed.num_warps==(4 if call['kernel']=='_draft_kernel' else 1)
    normalized=dict(parsed.__dict__)
    assert len(parsed.extern_libs)==1 and parsed.extern_libs[0][0]=='libdevice' and Path(parsed.extern_libs[0][1]).name=='libdevice.10.bc'
    normalized['extern_libs']=[['libdevice','<TRITON_LIBDEVICE>']]
    key=''.join(spec)+str((const,extra))
    record=dict(kernel=call['kernel'],B_context=call['B_context'],R_context=call['R_context'],grid=call['grid'],jit_cache_key=key,jit_cache_key_sha256=sha(key.encode()),bound_arguments={k:serialize(v) for k,v in bound.items()},signature_specialization=spec,constexpr_values=const,signature=signature,constants=constants,attrs=dict(divisible_by_16=sorted(attrs.divisible_by_16),equal_to_1=sorted(attrs.equal_to_1)),options_explicit=extra,parsed_options=normalized)
    deps={call['kernel']:SPANS[call['kernel']]['decorated_function_sha256']}
    if call['kernel'] in ('_draft_kernel','_publish_pairs_kernel'):deps['_pair_slot']=SPANS['_pair_slot']['decorated_function_sha256']
    complete=dict(kernel=call['kernel'],source_dependencies=deps,signature=signature,constants=constants,attrs=record['attrs'],options=normalized,triton='3.1.0',target=['cuda',90,32])
    record['complete_compile_key']=complete
    record['complete_compile_key_sha256']=sha(json.dumps(complete,sort_keys=True,separators=(',',':')).encode())
    return record,(kernel,signature,constants,attrs,parsed.__dict__,record)

def compare_all(actual,expected):
    # All fields in all512 records, not a selected signature subset.
    assert json.loads(json.dumps(actual))==expected,'Actual native wrapper/binder/options mismatch'
