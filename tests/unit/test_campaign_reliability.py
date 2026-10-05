"""Simulate NFS and scheduler failures; never contact live Slurm."""
import errno
import json

import pytest

from validation.full_snv_concordance.automation import reliable


@pytest.mark.parametrize('applied', [False,True])
def test_transient_io_replace_is_recovered_without_stale_state(tmp_path, monkeypatch, applied):
    replace = reliable.os.replace
    calls = []
    def fail_once(source, destination):
        calls.append(1)
        if len(calls)==1:
            if applied:
                replace(source,destination)
            raise OSError(errno.EIO,'injected NFS failure')
        replace(source,destination)
    monkeypatch.setattr(reliable.os,'replace',fail_once)
    reliable.atomic_json(tmp_path/'state.json', {'epoch':2}, delay=0)
    assert json.loads((tmp_path/'state.json').read_text()) == {'epoch':2}
    assert len(calls) == (1 if applied else 2)
    assert list(tmp_path.iterdir()) == [tmp_path/'state.json']


def test_persistent_io_failure_is_bounded_and_preserves_previous_state(tmp_path, monkeypatch):
    path=tmp_path/'state.json'
    reliable.atomic_json(path, {'epoch':1})
    calls=[]
    def fail(*args):
        calls.append(1)
        raise OSError(errno.EIO,'persistent')
    monkeypatch.setattr(reliable.os,'replace',fail)
    with pytest.raises(OSError):
        reliable.atomic_json(path, {'epoch':2}, delay=0)
    assert len(calls)==3 and json.loads(path.read_text())=={'epoch':1}


@pytest.mark.parametrize('rpc_error', [False,True])
def test_throttle_readback_handles_applied_rpc_error_without_repeat(rpc_error):
    state={'throttle':6,'updates':0}
    def runner(argv):
        if argv[1]=='update':
            state['throttle']=5
            state['updates']+=1
            if rpc_error:
                raise RuntimeError('reply lost')
        return f"ArrayJobId=12 ArrayTaskThrottle={state['throttle']}"
    reliable.update_throttle(12,5,runner,delay=0)
    reliable.update_throttle(12,5,runner,delay=0)
    assert state['updates']==1


def test_zero_throttle_and_conflicting_readback_fail_closed():
    with pytest.raises(ValueError):
        reliable.update_throttle(12,0,lambda argv:'')
    with pytest.raises(ValueError):
        reliable.parse_throttle('ArrayJobId=12 ArrayTaskThrottle=5\nArrayJobId=12 ArrayTaskThrottle=6',12)
    with pytest.raises(RuntimeError):
        reliable.update_throttle(12,5,lambda argv:'ArrayJobId=12 ArrayTaskThrottle=6',delay=0)


def test_ambiguous_submission_recovers_by_unique_identity_without_duplicate(tmp_path):
    jobs=[]
    def submit(identity):
        jobs.append('123')
        raise RuntimeError('RPC response lost after allocation')
    def lookup(identity):
        return jobs
    state=tmp_path/'submission.json'
    assert reliable.ensure_submission(state,'audit-token',submit,lookup)=='123'
    assert reliable.ensure_submission(state,'audit-token',submit,lookup)=='123'
    assert jobs==['123']


def test_unresolved_submission_intent_prevents_resubmit(tmp_path):
    calls=[]
    def submit(identity):
        calls.append(1)
        raise RuntimeError('unreachable')
    for _ in range(2):
        with pytest.raises(RuntimeError):
            reliable.ensure_submission(tmp_path/'state.json','token',submit,lambda identity: [])
    assert len(calls)==1


def test_allocation_charges_do_not_include_steps_or_count_gpu_twice():
    rows=[{'JobIDRaw':'123','ElapsedRaw':'3600','AllocTRES':'cpu=12,billing=12,gres/gpu=1,gres/gpu:a100=1'}]
    rows += [rows[0], dict(rows[0],JobIDRaw='123.batch')]
    assert reliable.allocation_cost(rows)=={'billing_hours':12.,'gpu_hours':1.}
