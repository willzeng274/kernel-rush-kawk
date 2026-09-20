"""Exactly16 frozen native SM90 specializations; CPU only, no GPU context.

Dispatch requires separate parent authorization. This module can be imported by
the source-only preflight without importing Triton or compiling anything.
"""
import datetime
import importlib.util
import inspect
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import HERE, BOUNDS, check_unique, make_record, sha, values_for, verify_audit, verify_source

OUT = HERE / 'compile-results'
utc = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str) + '\n')


def native():
    import triton
    import triton.runtime.jit as jit
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import make_backend
    manifest, plan = verify_source()
    expected = json.loads((HERE / 'NATIVE_EXPECTED.json').read_text())
    assert triton.__version__ == '3.1.0'
    assert sha(Path(inspect.getsourcefile(jit)).read_bytes()) == expected['jit_source_sha256']
    backend = make_backend(GPUTarget('cuda', 90, 32))
    backend_path = Path(backend.parse_options.__func__.__code__.co_filename)
    assert sha(backend_path.read_bytes()) == expected['backend_source_sha256']
    spec = importlib.util.spec_from_file_location('frozen_triple_context', HERE / 'triple_kernels.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    records = []
    payload = {}
    for case in plan['cases']:
        kernel = getattr(module, case['kernel'])
        assert kernel.arg_names == list(case['signature']) + list(case['constexpr'])
        kernel.create_binder()
        assert kernel.debug is None
        # JITFunction.run adds debug after the explicit production launch options.
        explicit = {**case['requested_options'], 'debug': kernel.debug}
        bound, sigspec, constvals, nonconst, extra = kernel.binder(**values_for(case), **explicit)
        attrs = kernel._get_config(*bound.values())
        parsed = backend.parse_options(explicit)
        record = make_record(case, kernel, attrs, bound, sigspec, constvals, extra, explicit, parsed)
        records.append(record)
        identity = record['compile_identity']
        payload[case['id']] = (kernel, identity['signature'], identity['constants'], attrs,
                               parsed.__dict__, record)
    assert json.loads(json.dumps(records)) == expected['launches'], 'Native binder/options mismatch'
    assert check_unique(records) == expected['dedup']
    assert expected['kernel_source_sha256'] == manifest['module_sha256']
    assert 'torch' not in sys.modules
    return payload, dict(launches=records, dedup=check_unique(records),
                         jit_source_sha256=expected['jit_source_sha256'],
                         backend_source_sha256=expected['backend_source_sha256'],
                         module_sha256=manifest['module_sha256'], cuda_execution=False)


def instruction_inventory(ptx):
    patterns = {
        'global_load': r'\bld\.global[^;]*;', 'global_store': r'\bst\.global[^;]*;',
        'local_load': r'\bld\.local[^;]*;', 'local_store': r'\bst\.local[^;]*;',
        'shared_load': r'\bld\.shared[^;]*;', 'shared_store': r'\bst\.shared[^;]*;',
        'barrier': r'\b(?:bar|barrier)\.[^;]*;', 'memory_fence': r'\b(?:membar|fence)\.[^;]*;',
        'shuffle': r'\bshfl\.[^;]*;', 'vote': r'\bvote\.[^;]*;',
        'atomic': r'\b(?:atom|red)\.[^;]*;', 'tensor_mma': r'\b(?:mma|wgmma)\.[^;]*;',
    }
    counts = {name: dict(static_count=len(found), instruction_forms=sorted(set(found)))
              for name, pattern in patterns.items() for found in [re.findall(pattern, ptx)]}
    # Source .loc plus exact PTX lines let the reviewer trace every store's
    # predicate and each dependent pointer load. This is evidence, not a proof.
    current_loc = None
    instructions = []
    loc_files = []
    for number, line in enumerate(ptx.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith('.file'):
            loc_files.append(dict(line=number, text=stripped))
        if stripped.startswith('.loc'):
            current_loc = stripped
        if stripped.endswith(';') and not stripped.startswith(('.', '//')):
            instructions.append(dict(line=number, source_loc=current_loc, text=stripped))
    return dict(counts=counts, source_files=loc_files, instructions=instructions,
                lane_safety_status='REQUIRES_MANUAL_PTX_AND_INITIALIZATION_REVIEW')


def resources(log):
    regs = re.search(r'Used (\d+) registers', log)
    stack = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', log)
    assert regs and stack, log
    return dict(registers_per_thread=int(regs[1]), stack_bytes=int(stack[1]),
                spill_store_bytes=int(stack[2]), spill_load_bytes=int(stack[3]))


def child(name):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import _path_to_binary
    verify_audit()
    manifest, plan = verify_source()
    payload, actual = native()
    kernel, signature, constants, attrs, options, record = payload[name]
    stem = OUT / name
    cache = OUT / 'cache' / name
    assert not cache.exists(), 'No reused compiler cache or repeated case'
    os.environ['TRITON_CACHE_DIR'] = str(cache)
    result = dict(case=name, native=record, parsed_options=options, cuda_execution=False,
                  started_at=utc(), lane_safety_status='NOT_EVALUATED')
    write(name + '.config.json', result)
    write(name + '.compile-start.json', dict(case=name, compile_calls=1, started_at=utc()))
    begin = time.monotonic()
    compiled = triton.compile(ASTSource(kernel, signature, constants, attrs),
                              target=GPUTarget('cuda', 90, 32), options=options)
    result['compile_seconds'] = time.monotonic() - begin
    result['metadata'] = compiled.metadata._asdict()
    for ext in ('ttir', 'ttgir', 'llir', 'ptx'):
        stem.with_suffix('.' + ext).write_text(compiled.asm[ext])
    stem.with_suffix('.cubin').write_bytes(compiled.asm['cubin'])
    ptxas, version = _path_to_binary('ptxas')
    assembly = subprocess.run([ptxas, '-v', '--gpu-name=sm_90a', str(stem.with_suffix('.ptx')),
                               '-o', str(stem.with_suffix('.verbose.cubin'))],
                              capture_output=True, text=True, timeout=30)
    log = assembly.stdout + assembly.stderr
    stem.with_suffix('.ptxas.log').write_text(log)
    result.update(ptxas_version=version, ptxas_returncode=assembly.returncode,
                  resources=resources(log) if assembly.returncode == 0 else None)
    index = instruction_inventory(compiled.asm['ptx'])
    write(name + '.ptx-index.json', index)
    result['inventory_counts'] = {k: v['static_count'] for k, v in index['counts'].items()}
    result['artifact_sha256'] = {p.name: sha(p.read_bytes()) for p in OUT.glob(name + '.*') if p.is_file()}
    result['success'] = assembly.returncode == 0
    result['completed_at'] = utc()
    result['torch_imported'] = 'torch' in sys.modules
    assert not result['torch_imported']
    verify_source()
    verify_audit()
    write(name + '.json', result)
    print(json.dumps(dict(case=name, success=result['success'], resources=result['resources'],
                          shared_bytes=result['metadata']['shared'])), flush=True)
    if not result['success']:
        raise SystemExit(1)


def run_bounded(argv, timeout):
    # Each case gets its own process group, including any ptxas descendant.
    # Timeout kills the complete group, so no abandoned compiler can overlap.
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return dict(returncode=process.returncode, stdout=stdout, stderr=stderr, timed_out=False)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return dict(returncode=process.returncode, stdout=stdout, stderr=stderr, timed_out=True)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise


def deadline_alarm(signum, frame):
    raise TimeoutError('238-second internal deadline;240-second absolute bound')


def main():
    OUT.mkdir(exist_ok=True)
    verify_audit()
    manifest, plan = verify_source()
    cases = [c['id'] for c in plan['cases']]
    if len(sys.argv) > 1:
        assert len(sys.argv) == 2 and sys.argv[1] in cases
        child(sys.argv[1])
        return
    started = time.monotonic()
    signal.signal(signal.SIGALRM, deadline_alarm)
    # Leave two seconds to kill/reap the active process group before the
    # independent workflow's absolute240-second SIGKILL bound.
    signal.setitimer(signal.ITIMER_REAL, BOUNDS['total_seconds'] - 2)
    assert not list(OUT.glob('*.compile-start.json')), 'No retries or resumed sweeps'
    payload, actual = native()  # All16 complete records checked BEFORE any triton.compile.
    write('NATIVE_BINDER_ACTUAL.json', actual)
    write('NATIVE_PREFLIGHT.json', dict(status='PASS', launches=16, dedup=actual['dedup'],
                                      compile_calls_before_preflight=0, cuda_execution=False,
                                      source_set_sha256=manifest['candidate_source_set_sha256']))
    results = []
    for name in cases:
        remaining = BOUNDS['total_seconds'] - 2 - (time.monotonic() - started)
        assert remaining > 0
        begin = time.monotonic()
        result = dict(case=name, started_at=utc())
        run = run_bounded([sys.executable, '-B', __file__, name], min(BOUNDS['case_seconds'], remaining))
        (OUT / (name + '.compiler.stdout')).write_text(run.pop('stdout'))
        stderr = run.pop('stderr')
        (OUT / (name + '.compiler.stderr')).write_text(stderr)
        result.update(run)
        if (OUT / (name + '.json')).exists():
            result.update(json.loads((OUT / (name + '.json')).read_text()))
        if run['returncode'] or run['timed_out']:
            result.update(success=False, error=stderr[-10000:] or 'Child hard timeout')
        result['total_seconds'] = time.monotonic() - begin
        results.append(result)
        write('summary.json', results)
        print(json.dumps(dict(case=name, success=result.get('success', False),
                              resources=result.get('resources'), total_seconds=result['total_seconds'],
                              error=result.get('error'))), flush=True)
        if not result.get('success'):
            raise SystemExit(1)
    verify_source()
    verify_audit()
    assert len(results) == len(list(OUT.glob('*.compile-start.json'))) == 16
    write('COMPLETION.json', dict(status='PASS', compile_calls=16, total_seconds=time.monotonic() - started,
                                 source_verified_before_after=True, cuda_execution=False, completed_at=utc(),
                                 lane_safety_status='REQUIRES_MANUAL_REVIEW'))
    signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
