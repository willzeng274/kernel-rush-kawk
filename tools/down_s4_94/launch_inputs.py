"""Capture the exact frozen wrapper launches with aligned, CPU-only tensor stand-ins."""
import ast
from pathlib import Path
HERE=Path(__file__).resolve().parent
CASES=['down_m64_n64_k64_split4_w4_s3','sum_mn163840_split4_block1024']
class Tensor:
    next_pointer=0x100000
    def __init__(self,shape,dtype,device):
        self.shape=tuple(shape);self.dtype=dtype;self.device=device;self.is_cuda=True
        self.pointer=Tensor.next_pointer;Tensor.next_pointer+=0x100000
    def data_ptr(self):return self.pointer
    def is_contiguous(self):return True
    def stride(self,axis):
        value=1
        for n in self.shape[axis+1:]:value*=n
        return value
    def descriptor(self):
        return dict(shape=list(self.shape),dtype=self.dtype,device=self.device,contiguous=True,aligned_to_16=self.pointer%16==0)
class TorchStandIn:
    bfloat16='torch.bfloat16';float32='torch.float32'
    @staticmethod
    def empty(shape,*,dtype,device):return Tensor(shape,dtype,device)
class LaunchRecorder:
    def __init__(self,name,calls):self.name=name;self.calls=calls
    def __getitem__(self,grid):
        def capture(*args,**kwargs):self.calls.append(dict(kernel=self.name,grid=list(grid),args=args,kwargs=kwargs))
        return capture

def capture_launches():
    source=HERE/'snapshots/engine/kernels/down_s4.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DownS4Matmul')
    calls=[]
    namespace=dict(torch=TorchStandIn,_down_s4_kernel=LaunchRecorder('_down_s4_kernel',calls),_sum_kernel=LaunchRecorder('_sum_kernel',calls))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    a=Tensor((64,9728),TorchStandIn.bfloat16,'cuda:0')
    w=Tensor((2560,9728),TorchStandIn.bfloat16,'cuda:0')
    owner=namespace['DownS4Matmul']('cuda:0');allocation_owners=[]
    result=owner(a,w,allocation_owners=allocation_owners)
    assert len(calls)==2 and allocation_owners==[result]
    assert calls[0]['args'][2] is calls[1]['args'][0] is owner.part
    assert calls[1]['args'][1] is result
    assert len({x.data_ptr() for x in (a,w,owner.part,result)})==4
    for case,call in zip(CASES,calls):call['case']=case
    return calls

def serialize(value):
    if isinstance(value,Tensor):return value.descriptor()
    return value

def launch_receipt(calls):
    return [dict(case=c['case'],kernel=c['kernel'],grid=c['grid'],args=[serialize(v) for v in c['args']],kwargs=c['kwargs']) for c in calls]
