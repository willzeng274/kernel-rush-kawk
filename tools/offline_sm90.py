"""Compile exact device source for SM90; never execute GPU code or load weights."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'compile-results'


def import_device(source):
    """Preserve exact JIT source bytes while excluding unrelated host imports."""
    path = ROOT / 'engine' / source
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    nodes = ast.parse(text).body
    kept = ['import triton\nimport triton.language as tl\n']
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            start = min(d.lineno for d in node.decorator_list) - 1
            kept.append(''.join(lines[start:node.end_lineno]) + '\n')
    extracted = OUT / ('device_' + source)
    extracted.write_text('\n'.join(kept))
    spec = importlib.util.spec_from_file_location(extracted.stem, extracted)
    module = importlib.util.module_from_spec(spec)
    sys.modules[extracted.stem] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


def child(config):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, AttrsDescriptor
    source = config.get('source', 'byteplane_kernels.py' if config['planes'] else
                        'hopper_gemm.py' if config['rows'] == 64 else 'hopper_tiles.py')
    module, digest = import_device(source)
    name = '_hopper_dot' if config['rows'] == 64 else '_hopper_tiles_dot'
    kernel = getattr(module, name)
    constants_by_name = {k: config[k] for k in ('B', 'K', 'BK', 'SPLITS')}
    constants_by_name['BB'] = max(16, triton.next_power_of_2(config['B']))
    constants = {i: constants_by_name[n] for i, n in enumerate(kernel.arg_names)
                 if n in constants_by_name}
    types = {'X': '*bf16', 'W': '*bf16', 'LO': '*u8', 'HI': '*u8',
             'OUT': '*bf16', 'PART': config.get('part_type', '*fp32'), 'N': 'i32',
             'X_ROW': 'i32', 'OUT_ROW': 'i32'}
    signature = {i: types[n] for i, n in enumerate(kernel.arg_names) if i not in constants}
    divisible = {i for i, n in enumerate(kernel.arg_names)
                 if n in ('X', 'W', 'LO', 'HI', 'OUT', 'PART', 'N', 'X_ROW', 'OUT_ROW')}
    compiled = triton.compile(
        ASTSource(kernel, signature, constants, AttrsDescriptor(divisible_by_16=divisible)),
        target=GPUTarget('cuda', 90, 32),
        options={'num_warps': 4, 'num_stages': config['stages']})
    result = {'config': config, 'source_sha256': digest, 'triton': triton.__version__,
              'metadata': compiled.metadata._asdict(), 'cuda_execution': False}
    stem = OUT / config['id']
    for ext, content in compiled.asm.items():
        if ext in ('ttir', 'ttgir', 'ptx', 'llir'):
            stem.with_suffix('.' + ext).write_text(content)
    from triton.backends.nvidia.compiler import _path_to_binary
    ptxas, _ = _path_to_binary('ptxas')
    resource = subprocess.run([ptxas, '-v', '--gpu-name=sm_90a',
                               str(stem.with_suffix('.ptx')), '-o',
                               str(stem.with_suffix('.cubin'))],
                              capture_output=True, text=True, timeout=30)
    stem.with_suffix('.ptxas.log').write_text(resource.stdout + resource.stderr)
    result['ptxas_returncode'] = resource.returncode
    result['ptxas_resources'] = resource.stdout + resource.stderr
    ptx = compiled.asm['ptx']
    result['tma_load_count'] = ptx.count('cp.async.bulk.tensor.2d.shared::cluster.global')
    result['wgmma_count'] = ptx.count('wgmma.mma_async')
    if config['planes']:
        # Run the exact candidate guard function without importing its CUDA host code.
        tree = ast.parse((ROOT / 'engine' / 'ilc_tma.py').read_text())
        guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'verify_tma_compilation')
        namespace = {'ILCUnavailable': RuntimeError}
        exec(compile(ast.Module(body=[guard], type_ignores=[]), 'actual_candidate_guard', 'exec'), namespace)
        try:
            result['guard'] = namespace['verify_tma_compilation'](
                compiled, config['rows'], config['BK'], config['stages'],
                triton.cdiv(config['K'], config['SPLITS'] * config['BK']))
            result['guard_pass'] = True
        except Exception as error:
            result['guard_pass'] = False
            result['guard_error'] = str(error)
    stem.with_suffix('.json').write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        child(json.loads(sys.argv[1]))
        return
    results = []
    # Cover both M widths, BB16/32, all three K dimensions, short/long loops,
    # both BK widths and selected pipeline depths. Identical base/control pairs.
    specs = [(64, 4, 2560, 128, 8, 3), (64, 16, 4096, 128, 4, 3),
             (64, 32, 9728, 128, 1, 3), (64, 4, 2560, 256, 4, 2),
             (64, 32, 9728, 256, 1, 2), (128, 4, 2560, 128, 8, 2),
             (128, 16, 4096, 128, 4, 2), (128, 32, 9728, 128, 1, 2)]
    for i, (rows, batch, k, bk, splits, stages) in enumerate(specs):
        for planes in (False, True):
            config = dict(id=f'{i:02d}_' + ('planes' if planes else 'original'),
                          rows=rows, B=batch, K=k, BK=bk, SPLITS=splits,
                          stages=stages, planes=planes)
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
                result = {'config': config, 'error': str(error), 'trace': traceback.format_exc()}
            results.append(result)
            print(json.dumps(result, default=str), flush=True)
            (OUT / 'summary.json').write_text(json.dumps(results, indent=2, default=str))
    if any(r.get('returncode') != 0 for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
