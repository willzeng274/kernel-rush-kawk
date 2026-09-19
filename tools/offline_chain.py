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
    module, digest = import_device('recycled_kernels.py')
    kernel = getattr(module, config['kernel'])
    known = {'CAP': config['cap'], 'W': config['width'], 'EPS': 1e-6, 'D': 128,
             'SPLITS': triton.cdiv(config['cap'], 256), 'SCALE': 128 ** -0.5,
             'BLOCK_N': 256, 'BLOCK_S': triton.next_power_of_2(triton.cdiv(config['cap'], 256)),
             'ROWS': 16 * config['width'], 'BLOCK': 128}
    constants = {i: known[n] for i, n in enumerate(kernel.arg_names) if n in known}
    signature = {i: ('*i64' if n in ('META', 'COUNTS', 'IDS') else
                     '*fp32' if n in ('PART', 'PMAX', 'PSUM') else '*bf16')
                 for i, n in enumerate(kernel.arg_names) if i not in constants}
    kernel_result = triton.compile(
        ASTSource(kernel, signature, constants, AttrsDescriptor(divisible_by_16=set(signature))),
        target=GPUTarget('cuda', 90, 32),
        options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    result = {'config': config, 'source_sha256': digest, 'cuda_execution': False,
              'metadata': kernel_result.metadata._asdict()}
    for ext in ('ttir', 'ttgir', 'ptx', 'llir'):
        (OUT / (config['id'] + '.' + ext)).write_text(kernel_result.asm[ext])
    (OUT / (config['id'] + '.json')).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        child(json.loads(sys.argv[1])); return
    results = []
    for cap in (544, 2080):
        for width in (1, 4):
            for name in ('qkv', 'attention', 'merge', 'compact', 'ids'):
                config = dict(kernel='chain_' + name + '_kernel', cap=cap, width=width,
                              id=f'chain_{name}_c{cap}_w{width}')
                p = subprocess.run([sys.executable, __file__, json.dumps(config)],
                                   capture_output=True, text=True, timeout=60)
                (OUT / (config['id'] + '.log')).write_text(p.stdout + '\n' + p.stderr)
                result = {'config': config, 'returncode': p.returncode}
                if p.returncode == 0:
                    result.update(json.loads((OUT / (config['id'] + '.json')).read_text()))
                else:
                    result['error'] = p.stderr[-6000:]
                results.append(result)
                print(json.dumps(result, default=str), flush=True)
                (OUT / 'chain_summary.json').write_text(json.dumps(results, indent=2, default=str))
    if any(r['returncode'] != 0 for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
