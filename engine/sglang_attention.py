"""Owned BF16 SGLang attention adapter; all original writer rounding retained."""
import torch
from custom_kernels import qkv_rope_cache_kernel
from sglang_decode_port import launch


class SGLangAttention:
    def __init__(self,e,baseline):
        self.engine,self.baseline,self.control=e,baseline,e.sg_control
        self.control.register(self)
        self.mid=self.lse=None
        self.control.live()
        self.sm_count=torch.cuda.get_device_properties(e.ids.device).multi_processor_count
        if self.sm_count <= 0:
            raise ValueError('positive physical SM count required')
        self.mid=torch.empty((e.batch,32,8,128),dtype=torch.float32,device=e.ids.device)
        self.control.live()
        self.lse=torch.empty((e.batch,32,8),dtype=torch.float32,device=e.ids.device)
        self.control.live()

    def run(self,idx):
        e=self.engine
        a=e.layers[idx].self_attn
        qkv_rope_cache_kernel[(e.batch,40)](
            e.qkv,a.q_norm.weight,a.k_norm.weight,e.cos,e.sin,e.position,
            e.query,e.keys[idx],e.values[idx],e.capacity,e.eps,
            num_warps=4,enable_fp_fusion=False)
        launch(e.query,e.keys[idx],e.values[idx],e.position,self.mid,self.lse,
               e.attention,e.capacity,self.sm_count)
