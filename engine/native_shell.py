"""Lazy clean pinned-HF module shell; shared read-only checkpoint storage."""
import copy
import torch
import transformers
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM, Qwen3RotaryEmbedding, Qwen3Attention, Qwen3MLP, Qwen3RMSNorm,
)
from recycled_validate import CandidateRejected

GEOMETRY = dict(hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
                num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                vocab_size=151936)
MATH_CONFIG = dict(rms_norm_eps=1e-6, rope_theta=5000000.0, hidden_act='silu',
                   attention_bias=False, attention_dropout=0.0, max_position_embeddings=262144)
DEVICE_TYPE = "cuda"
HOOKS = ('_state_dict_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks',
         '_load_state_dict_post_hooks', '_forward_hooks', '_forward_pre_hooks',
         '_backward_hooks', '_backward_pre_hooks')


def _require(value, message):
    if not value:
        raise CandidateRejected("native shell: " + message)


def supported(source):
    _require(torch.__version__.split('+')[0] == '2.5.1' and
             transformers.__version__ == '4.51.3', "unsupported library version")
    _require(type(source) is Qwen3ForCausalLM, "unsupported root class")
    config = source.config
    _require(all(getattr(config, key, None) == value for key, value in GEOMETRY.items()),
             "unsupported geometry")
    _require(all(getattr(config, key, None) == value for key, value in MATH_CONFIG.items()),
             "unsupported numerical configuration")
    _require(config._attn_implementation == 'sdpa' and config.tie_word_embeddings
             and not config.torchscript and config.rope_scaling is None
             and not config.use_sliding_window and config.sliding_window is None,
             "unsupported configuration")
    _require(not torch.__future__.get_swap_module_params_on_conversion(), "parameter swap mode")
    _require(source.lm_head.weight is source.model.embed_tokens.weight, "untied source")
    _require(set(dict(source.named_buffers())) == {'model.rotary_emb.inv_freq'}, "buffer inventory")
    _require(source.state_dict.__func__ is torch.nn.Module.state_dict, "state override")
    device = source.lm_head.weight.device
    _require(device.type == DEVICE_TYPE, "unsupported device")
    _require(all(p.dtype == torch.bfloat16 and not p.is_meta and p.device == device
                 for p in source.parameters()), "parameter dtype/device")
    rope = source.model.rotary_emb
    _require(type(rope) is Qwen3RotaryEmbedding and rope.rope_type == 'default'
             and rope.inv_freq.device == device and rope.inv_freq.dtype == torch.float32
             and tuple(rope.inv_freq.shape) == (GEOMETRY['head_dim'] // 2,)
             and rope.attention_scaling == 1.0, "unsupported RoPE")
    for module in source.modules():
        _require(not any(getattr(module, field, None) for field in HOOKS)
                 and not hasattr(module, '_hf_hook'), "module hooks")
    from torch.nn.modules import module as runtime
    _require(not any(value for key, value in vars(runtime).items()
                     if key.startswith('_global_') and key.endswith('_hooks')), "global hooks")
    return device


def build_native(source, owner):
    """Owner is engine-registered before entry; publish before every mutation."""
    supported(source)
    owner.live(8.0)
    config = copy.deepcopy(source.config)
    config.torch_dtype = torch.bfloat16
    config._attn_implementation = 'sdpa'
    # Publishing before __init__ also pins partially constructed module trees.
    shell = owner.publish(Qwen3ForCausalLM.__new__(Qwen3ForCausalLM))
    with torch.device('meta'):
        Qwen3ForCausalLM.__init__(shell, config)
    _require(all(p.is_meta for p in shell.parameters()), "real constructor weight")
    owner.live()
    state = owner.publish(source.state_dict())
    _require(not any('_packed' in name or 'inv_freq' in name for name in state), "unexpected state")
    result = shell.load_state_dict(state, strict=True, assign=True)
    _require(not result.missing_keys and not result.unexpected_keys, "strict state assignment")
    owner.live()
    # Native metadata is clean; retain the exact native frequencies originally
    # computed on CPU before source.to(cuda). Recomputing pow on CUDA can differ
    # in low FP32 bits. Only this tiny independent buffer is copied, not weights.
    rope = owner.publish(Qwen3RotaryEmbedding.__new__(Qwen3RotaryEmbedding))
    with torch.device('meta'):
        Qwen3RotaryEmbedding.__init__(rope, config=shell.config, device='meta')
    old = source.model.rotary_emb
    inv_freq = owner.publish(torch.empty_like(old.inv_freq))
    inv_freq.copy_(old.inv_freq)
    rope.register_buffer('inv_freq', inv_freq, persistent=False)
    rope.original_inv_freq = inv_freq
    _require(rope.inv_freq.dtype == old.inv_freq.dtype and
             rope.inv_freq.device == old.inv_freq.device and
             torch.equal(rope.inv_freq, old.inv_freq) and torch.isfinite(rope.inv_freq).all() and
             rope.attention_scaling == old.attention_scaling and
             rope.original_inv_freq is rope.inv_freq and
             rope.inv_freq.data_ptr() != old.inv_freq.data_ptr(), "RoPE independent exact copy")
    shell.model.rotary_emb = rope
    shell.tie_weights()
    shell.eval()
    _require(shell.config is not source.config and
             shell.lm_head.weight is shell.model.embed_tokens.weight, "shell isolation/tie")
    for name, tensor in shell.state_dict().items():
        src = state[name]
        _require(tensor.dtype == torch.bfloat16 and tensor.device == src.device and
                 tensor.shape == src.shape and tensor.stride() == src.stride() and
                 tensor.storage_offset() == src.storage_offset() and
                 tensor.untyped_storage()._cdata == src.untyped_storage()._cdata,
                 "parameter storage mismatch")
    _require(all(not p.is_meta for p in shell.parameters()) and
             not rope.inv_freq.is_meta and not rope.original_inv_freq.is_meta, "remaining meta tensor")
    for layer in shell.model.layers:
        _require(type(layer.input_layernorm) is Qwen3RMSNorm and
                 type(layer.post_attention_layernorm) is Qwen3RMSNorm and
                 type(layer.self_attn) is Qwen3Attention and type(layer.mlp) is Qwen3MLP and
                 'forward' not in layer.self_attn.__dict__ and 'forward' not in layer.mlp.__dict__ and
                 not hasattr(layer.self_attn, '_packed_qkv') and
                 not hasattr(layer.mlp, '_packed_gateup'), "unclean native module")
    owner.live()
    return shell
