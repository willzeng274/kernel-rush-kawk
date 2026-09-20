"""Narrow process-group tests only; never import compiler/Torch/model modules."""
import ast
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bounded_tree_driver',HERE/'compile_flat_prior.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def check():
    checks=[]
    result=runner.run_bounded([sys.executable,'-B','-c','print("bounded-success")'],1)
    assert result['returncode']==0 and result['stdout'].strip()=='bounded-success' and not result['timed_out']
    checks.append(dict(name='successful_child',status='PASS'))
    result=runner.run_bounded([sys.executable,'-B','-c','raise SystemExit(7)'],1)
    assert result['returncode']==7 and not result['timed_out']
    checks.append(dict(name='failure_returncode',status='PASS'))
    start=time.monotonic()
    result=runner.run_bounded([sys.executable,'-B','-c','import time;time.sleep(3)'],.15)
    elapsed=time.monotonic()-start
    assert result['timed_out'] and result['returncode']<0 and elapsed<2
    checks.append(dict(name='timeout_kills_and_reaps_direct_child',status='PASS',elapsed_seconds=elapsed))
    with tempfile.TemporaryDirectory(prefix='timeout-preflight-',dir=HERE) as temporary:
        marker=Path(temporary)/'alive'
        # A positive control ensures the delayed-write probe actually works.
        probe='from pathlib import Path;Path('+repr(str(marker))+').write_text("alive")'
        result=runner.run_bounded([sys.executable,'-B','-c',probe],1)
        assert result['returncode']==0 and marker.read_text()=='alive'
        marker.unlink()
        descendant='import os,time;from pathlib import Path;print("grandchild",os.getpid(),os.getpgrp(),flush=True);time.sleep(.5);Path('+repr(str(marker))+').write_text("alive");time.sleep(3)'
        child='import os,subprocess,sys,time;print("child",os.getpid(),os.getpgrp(),flush=True);subprocess.Popen([sys.executable,"-B","-c",'+repr(descendant)+']);time.sleep(3)'
        result=runner.run_bounded([sys.executable,'-B','-c',child],.2)
        records=[line.split() for line in result['stdout'].splitlines()]
        assert result['timed_out'] and len(records)==2, result
        records={r[0]:(int(r[1]),int(r[2])) for r in records}
        assert records['child'][0]==records['child'][1]==records['grandchild'][1]
        assert records['child'][0]!=records['grandchild'][0]
        time.sleep(.6)
        assert not marker.exists(),'grandchild survived group kill'
        checks.append(dict(name='timeout_kills_inherited_group_grandchild_with_positive_probe_control',status='PASS'))
        previous=signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM,runner.deadline_alarm)
        try:
            signal.setitimer(signal.ITIMER_REAL,.2)
            try:runner.run_bounded([sys.executable,'-B','-c',child],2)
            except TimeoutError:pass
            else:raise AssertionError('internal alarm did not propagate')
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,previous)
        time.sleep(.6)
        assert not marker.exists(),'grandchild survived interrupted supervisor'
        checks.append(dict(name='internal_alarm_kills_group_reaps_child_and_propagates',status='PASS'))
    source=(HERE/'compile_flat_prior.py').read_text()
    assert source.count('start_new_session=True')==1
    assert len(runner.CASES)==len(set(runner.CASES))==56
    assert "==56\n    write('COMPLETION.json'" in source
    assert source.count('triton.compile(')==1
    assert 'signal.setitimer(signal.ITIMER_REAL, 178)' in source
    assert 'remaining=178-' in source and 'min(25,remaining)' in source
    assert 'No retries or resumed builds' in source
    assert not {'torch','triton'} & set(sys.modules)
    checks.append(dict(name='fiftysix_sequential25s_cases_178s_internal_cleanup_no_nested_sessions_or_compiler_import',status='PASS'))
    return dict(status='PASS',checks=checks,compile_calls=0,compiler_imported=False,cuda_execution=False)


if __name__=='__main__':
    result=check()
    print(json.dumps(result))
