"""Compile current chain source without CUDA execution."""
import json
from pathlib import Path
import subprocess
import sys
from offline_sm90 import import_device, OUT


def child(config):
    import triton
    from triton.compiler import ASTSource, AttrsDescriptor
    from triton.backends.compiler import GPUTarget
    module, digest = import_device(config.get('source', 'recycled_kernels.py'))
    kernel = getattr(module, config['kernel'])
    known = {'CAP': config['cap'], 'W': config['width'], 'EPS': 1e-6, 'D': 128,
             'SPLITS': triton.cdiv(config['cap'], 256), 'SCALE': 128 ** -0.5,
             'BLOCK_N': 256, 'BLOCK_S': triton.next_power_of_2(triton.cdiv(config['cap'], 256)),
             'ROWS': 16 * config['width'], 'BLOCK': 128}
    if 'constants' in config:
        known = config['constants']
    if 'batch' in config:
        known['ROWS'] = config['batch'] * config['width']
    constants = {i: known[n] for i, n in enumerate(kernel.arg_names) if n in known}
    signature = {i: ('*i64' if n in ('META', 'COUNTS', 'IDS', 'PATHS') else
                     '*fp32' if n in ('PART', 'PMAX', 'PSUM') else '*bf16')
                 for i, n in enumerate(kernel.arg_names) if i not in constants}
    signature = {i: config.get('signature_types', {}).get(kernel.arg_names[i], value)
                 for i, value in signature.items()}
    fusion = config['kernel'] not in ('chain_qkv_kernel', 'embedding_norm_kernel',
                                     'residual_norm_kernel', 'swiglu_kernel',
                                     'single_qkv_cache_kernel', 'single_fused_attention_kernel', 'tree_qkv_kernel')
    stages = 1 if config['kernel'] in ('chain_attention_kernel', 'single_fused_attention_kernel',
                                     'single_attention_split_kernel', 'tree_attention_kernel') else 3
    fusion = config.get('fusion', fusion)
    stages = config.get('stages', stages)
    options = {'num_warps': config.get('warps', 4), 'num_stages': stages, 'enable_fp_fusion': fusion}
    kernel_result = triton.compile(
        ASTSource(kernel, signature, constants, AttrsDescriptor(divisible_by_16=set(signature))),
        target=GPUTarget('cuda', 90, 32),
        options=options)
    result = {'config': config, 'source_sha256': digest, 'cuda_execution': False,
              'metadata': kernel_result.metadata._asdict(), 'options': options}
    for ext in ('ttir', 'ttgir', 'ptx', 'llir'):
        (OUT / (config['id'] + '.' + ext)).write_text(kernel_result.asm[ext])
    from triton.backends.nvidia.compiler import _path_to_binary
    ptxas, _ = _path_to_binary('ptxas')
    args = [ptxas, '-v', '--gpu-name=sm_90a']
    if not fusion:
        args += ['--fmad=false']
    args += [str(OUT / (config['id'] + '.ptx')), '-o', str(OUT / (config['id'] + '.cubin'))]
    resource = subprocess.run(args, capture_output=True, text=True, timeout=30)
    result['ptxas_returncode'] = resource.returncode
    result['ptxas_resources'] = resource.stdout + resource.stderr
    (OUT / (config['id'] + '.ptxas.log')).write_text(resource.stdout + resource.stderr)
    (OUT / (config['id'] + '.json')).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        child(json.loads(sys.argv[1])); return
    results, configs = [], []
    for cap in (255, 256, 257, 544, 640, 2080, 4096, 4097):
        for width in (1, 4):
            for name in ('qkv', 'attention', 'merge', 'compact'):
                config = dict(kernel='chain_' + name + '_kernel', cap=cap, width=width,
                              id=f'chain_{name}_c{cap}_w{width}')
                configs.append(config)
    for width in (1, 4):
        for batch in range(1, 17):
            configs.append(dict(kernel='chain_ids_kernel', cap=544, width=width,
                                batch=batch, id=f'chain_ids_b{batch}_w{width}'))
    for name in ('embedding_norm', 'residual_norm'):
        configs.append(dict(kernel=name + '_kernel', cap=544, width=1,
                            source='custom_kernels.py', id=name,
                            constants={'H': 2560, 'EPS': 1e-6, 'BLOCK': 4096}))
    for rows in sorted(set(range(1, 17)) | {4*b for b in range(1, 17)}):
        configs.append(dict(kernel='swiglu_kernel', cap=544, width=1,
                            source='custom_kernels.py', id=f'swiglu_m{rows}',
                            constants={'I': 9728, 'TOTAL': rows*9728, 'BLOCK': 1024}))
    for config in configs:
        try:
            p = subprocess.run([sys.executable, __file__, json.dumps(config)],
                               capture_output=True, text=True, timeout=60)
            (OUT / (config['id'] + '.log')).write_text(p.stdout + '\n' + p.stderr)
            result = {'config': config, 'returncode': p.returncode}
            if p.returncode == 0:
                result.update(json.loads((OUT / (config['id'] + '.json')).read_text()))
            else:
                result['error'] = p.stderr[-6000:]
        except Exception as error:
            result = {'config': config, 'returncode': -1, 'error': str(error)}
        results.append(result)
        print(json.dumps(result, default=str), flush=True)
        (OUT / 'chain_summary.json').write_text(json.dumps(results, indent=2, default=str))
    if any(r['returncode'] != 0 for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
