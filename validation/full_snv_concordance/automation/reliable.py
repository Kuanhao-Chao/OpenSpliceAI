"""Bounded recovery primitives for future campaigns; no scheduler actions on import.

Existing signed r13 bundles do not import this module. Runner/query callbacks
make scheduler behavior testable without submitting or cancelling live jobs.
"""
import contextlib
import errno
import json
import os
from pathlib import Path
import re
import tempfile
import time

TRANSIENT_IO = {errno.EIO, errno.ESTALE, errno.ETIMEDOUT, errno.EAGAIN}


def atomic_json(path, value, attempts=3, delay=.1):
    """Publish flushed JSON with bounded transient-I/O retry and exact readback.

    Retain the caller's path spelling. An ambiguous replace is accepted only
    when the destination can be read and its complete value matches exactly.
    """
    if attempts < 1:
        raise ValueError('Retry count must be positive')
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True)+'\n'
    for attempt in range(attempts):
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
            with os.fdopen(descriptor, 'w') as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            if path.read_text() != payload:
                raise RuntimeError('Published campaign state failed readback')
            return
        except OSError as error:
            with contextlib.suppress(OSError):
                if path.read_text() == payload:
                    return
            if error.errno not in TRANSIENT_IO or attempt+1 == attempts:
                raise
            time.sleep(delay)
        finally:
            if temporary:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temporary)


def parse_throttle(raw, job):
    """Require one explicit throttle for the requested array, ignoring other jobs."""
    values = set()
    for line in raw.splitlines():
        fields = dict(re.findall(r'(\w+)=(\S+)', line))
        if fields.get('ArrayJobId') == str(job):
            if 'ArrayTaskThrottle' not in fields:
                raise ValueError('Scheduler response lacks array throttle')
            values.add(int(fields['ArrayTaskThrottle']))
    if len(values) != 1:
        raise ValueError('Cannot establish one array throttle')
    return values.pop()


def update_throttle(job, target, runner, attempts=3, delay=.1):
    """Avoid redundant writes; verify applied updates even after an RPC error.

    ``runner`` accepts Slurm argument tuples and returns response text or raises.
    Zero means unlimited in Slurm and is rejected. This function is for reviewed
    future controllers or an explicitly authorized temporary capacity transition.
    """
    if type(target) is not int or target < 1 or attempts < 1:
        raise ValueError('Throttle and retry count must be positive integers')
    last_error = None
    for attempt in range(attempts):
        try:
            current = parse_throttle(runner(('scontrol','show','job',str(job),'-o')), job)
            if current == target:
                return
            try:
                runner(('scontrol','update',f'JobId={job}',f'ArrayTaskThrottle={target}'))
            except RuntimeError as error:
                last_error = error
            if parse_throttle(runner(('scontrol','show','job',str(job),'-o')), job) == target:
                return
        except (OSError, RuntimeError, ValueError) as error:
            last_error = error
        if attempt+1 < attempts:
            time.sleep(delay)
    raise RuntimeError(f'Unable to verify array {job} throttle {target}: {last_error}')


def ensure_submission(path, identity, submit, lookup):
    """Recover an ambiguous submission by unique identity without a second submit.

    Caller holds an exclusive campaign lock. Lookup must search both active and
    completed scheduler allocations. An unresolved prior intent remains blocked
    until reconciled; absence from squeue alone never authorizes resubmission.
    """
    path = Path(path)
    state = json.loads(path.read_text()) if path.exists() else None
    if state and state['identity'] != identity:
        raise ValueError('Submission identity does not match persisted intent')
    jobs = list(dict.fromkeys(map(str, lookup(identity))))
    if len(jobs) > 1:
        raise RuntimeError('Submission identity matches multiple allocations')
    if jobs:
        job = jobs[0]
    elif state:
        raise RuntimeError('Prior submission intent is unresolved; reconcile before retry')
    else:
        atomic_json(path, {'identity':identity, 'status':'submitting'})
        try:
            job = str(submit(identity))
        except (OSError, RuntimeError):
            jobs = list(dict.fromkeys(map(str, lookup(identity))))
            if len(jobs) != 1:
                raise RuntimeError('Ambiguous submission; persisted intent prevents duplicate retry')
            job = jobs[0]
    if not re.fullmatch(r'\d+',job):
        raise ValueError('Scheduler returned an invalid allocation ID')
    atomic_json(path, {'identity':identity,'status':'submitted','job_id':job})
    return job


def allocation_cost(rows):
    """Sum allocation billing/GPU-hours, excluding steps and duplicate records."""
    billing, gpu, seen = 0., 0., set()
    for row in rows:
        job = str(row['JobIDRaw'])
        if '.' in job or job in seen:
            continue
        seen.add(job)
        tres = dict(item.split('=',1) for item in row['AllocTRES'].split(',') if '=' in item)
        elapsed = float(row['ElapsedRaw'])/3600
        billing += float(tres.get('billing',0))*elapsed
        # Generic GPU count already includes typed GRES; do not add both.
        count = tres.get('gres/gpu')
        if count is None:
            count = sum(float(value) for key,value in tres.items() if key.startswith('gres/gpu:'))
        gpu += float(count)*elapsed
    return {'billing_hours':billing,'gpu_hours':gpu}
