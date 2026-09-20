"""Compile exact public prefill JIT with actual Triton 3.1 specialization facts.

No CUDA driver, device, model, or kernel execution is used. S=130 represents
generic tails, S=512 represents divisible-16 lengths, S=1 its folded variant.
"""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'compile-results'


def main():
    import triton
    from triton.compiler import ASTSource, AttrsDescriptor
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import _path_to_binary
    assert triton.__version__ == '3.1.0'
    OUT.mkdir(exist_ok=True)
    source_name, record_name, prefix = sys.argv[1:]
    record = json.loads(Path(__file__).with_name(record_name).read_text())
    source = ROOT / 'engine' / source_name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record['compiler_module_sha256']
    spec = importlib.util.spec_from_file_location('prefill_accumulator_' + prefix, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = module._attn_prefill_kernel
    names = kernel.arg_names
    expected = ['Q', 'K', 'V', 'O', 'qk_scale', 'S', 'stride_qb', 'stride_qh', 'stride_qs',
                'stride_kb', 'stride_kh', 'stride_ks', 'stride_os', 'H', 'G', 'D', 'BLOCK_M', 'BLOCK_N']
    assert names == expected
    constants = {'H': 32, 'G': 4, 'D': 128, 'BLOCK_M': 128, 'BLOCK_N': 128}
    signature = {i: '*bf16' if i < 4 else 'fp32' if i == 4 else 'i32' for i in range(13)}
    options = dict(num_warps=8, num_stages=2, enable_fp_fusion=True)

    class AlignedPointer:
        # JITFunction._get_config only inspects alignment here; no tensor or
        # CUDA API is constructed. Real torch allocations/views are aligned.
        def data_ptr(self):
            return 0x100000

    results = []
    for name, length in [('generic_s', 130), ('divisible16_s', 512), ('folded_s1', 1)]:
        name = prefix + '_' + name
        begin = time.monotonic()
        config = dict(id=name, representative_s=length, signature={names[i]: t for i, t in signature.items()},
                      constexpr=constants, options=options, target=dict(backend='cuda', capability=90, warp_size=32),
                      cuda_execution=False)
        result = dict(config=config, source=record, started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            capacity = ((length + 255) // 256) * 256
            # Exact strides of packed QKV Q and equal-strided cache slices.
            values = [*[AlignedPointer() for _ in range(4)], 128 ** -.5 * 1.4426950408889634, length,
                      length * 6144, 128, 6144, 8 * capacity * 128, capacity * 128, 128, 4096,
                      32, 4, 128, 128, 128]
            attrs = kernel._get_config(*values)
            divisible = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17}
            if length % 16 == 0:
                divisible.add(5)
            equals = {5} if length == 1 else set()
            # Compare explicitly with the installed launcher's specialization.
            assert set(attrs.divisible_by_16) == divisible
            assert set(attrs.equal_to_1) == equals
            assert 4 not in attrs.divisible_by_16 and 4 not in attrs.equal_to_1
            cfg = {names.index(k): v for k, v in constants.items()}
            if length == 1:
                cfg[5] = 1  # Exactly as JITFunction.run folds equal_to_1.
            result['attributes'] = dict(divisible_by_16=[names[i] for i in sorted(divisible)],
                                        equal_to_1=[names[i] for i in sorted(equals)])
            result['constants_by_index'] = cfg
            compiled = triton.compile(ASTSource(kernel, signature, cfg,
                                               AttrsDescriptor(tuple(sorted(divisible)), tuple(sorted(equals)))),
                                      target=GPUTarget('cuda', 90, 32), options=options)
            result['metadata'] = compiled.metadata._asdict()
            for ext in ('ttir', 'ttgir', 'ptx', 'llir'):
                (OUT / (name + '.' + ext)).write_text(compiled.asm[ext])
            ptxas, _ = _path_to_binary('ptxas')
            assembly = subprocess.run([ptxas, '-v', '--gpu-name=sm_90a', str(OUT / (name + '.ptx')),
                                       '-o', str(OUT / (name + '.cubin'))], capture_output=True, text=True, timeout=45)
            result.update(ptxas_returncode=assembly.returncode, ptxas_resources=assembly.stdout + assembly.stderr)
            (OUT / (name + '.ptxas.log')).write_text(result['ptxas_resources'])
            result['success'] = assembly.returncode == 0
        except Exception as error:
            import traceback
            result.update(success=False, error=str(error), traceback=traceback.format_exc())
        result['elapsed_seconds'] = time.monotonic() - begin
        result['completed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        results.append(result)
        (OUT / (name + '.json')).write_text(json.dumps(result, indent=2, default=str) + '\n')
        (OUT / (prefix + '_summary.json')).write_text(json.dumps(results, indent=2, default=str) + '\n')
        print(json.dumps(result, default=str), flush=True)
    if len(results) != 3 or any(not r['success'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
