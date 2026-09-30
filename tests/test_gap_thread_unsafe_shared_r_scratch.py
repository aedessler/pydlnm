"""Gap: PyDLNM called from several Python threads (ThreadPoolExecutor, joblib's threading backend, dask's threaded
scheduler -- the natural way to spread 100-800 locations over the cores).

Every test compares what PyDLNM returns for a batch of distinct series with what R (dlnm 2.4.10, splines, mgcv, at
run time) returns for the same series: onebasis() / crossbasis() for the bases, glm() + crossreduce() for the per-city
first stage.  Only the *way* PyDLNM is driven differs between the tests: serially in the main thread (baseline), in one
worker thread, in four worker threads, in four worker threads behind one external lock, in spawned worker processes.

What is being probed
  * basis_functions._R_SCRATCH is ONE module-level R environment with the fixed object names x, ik, bk, res, knots,
    oo that every ns / bs / ps / cr evaluation (OneBasis, CrossBasis, enhanced_splines, crosspred/crossreduce
    rebuilding a basis, ImprovedGLMInterface's ns(date)) writes and reads back in several separate rpy2 operations,
    with no lock.  rpy2 3.6 does not serialise calls into R either (openrlib.rlock is only used around callbacks and
    local contexts) and embedded R is not thread-safe.  Concurrent callers therefore get the basis / knots of another
    caller (right shape, wrong numbers), shape / broadcast exceptions, rpy2 'preserved object' KeyErrors, R errors
    ("NA/NaN/Inf in foreign function call"), hard crashes (SIGABRT / SIGSEGV, "unable to initialize the JIT", R quitting
    with exit status 0) or hangs.
  * Rpy2GLMInterface / ImprovedGLMInterface use the implicit rpy2 converter, which only exists in the main thread
    (rpy2 keeps it in a contextvars.ContextVar): in any other thread they raise NotImplementedError
    ('Conversion rules for `rpy2.robjects` appear to be missing').  ns / bs / ps are immune because they open their
    own localconverter(default + numpy2ri); the mgcv-based 'cr' is not (basis_functions._require_r_package calls
    robjects.r() outside it), so OneBasis('cr') and CrossBasis(argvar=cr) also raise in every non-main thread.
    Even with the converter copied into the workers and every basis computed serially beforehand, four threads
    fitting Rpy2GLMInterface models concurrently kill the process (R is entered concurrently through unlocked rpy2
    calls), so the cure is one process-wide lock around ALL R access, not only around the scratch environment.

How the tests are built
  The threaded workloads run in child Python processes (same interpreter and environment, R started before PyDLNM is
  imported, as tests/rhelpers.py does).  A crash (signal), a hang or an R-level abort of a threaded run therefore fails
  only the test that provoked it instead of taking the pytest session with it.  The R reference is computed in this
  (parent) process.  All children are started together on first use and run concurrently (hard cap CHILD_TIMEOUT s
  wall clock), so `-k` does not make a single test cheaper.

Tests decorated with @known_defect assert the R-faithful outcome (every result equals R's) and fail today (strict
xfail: when the fix lands they XPASS and the marker must be removed).  The plain tests are guards for the usage patterns
that already work: serial, one worker thread, four worker threads behind ONE external lock (proof that the shared
scratch environment / unserialised R access is the cause), a one-worker GLM once the rpy2 converter is copied into the
thread, and spawned worker processes (the documented escape hatch: 'use processes').

PYDLNM_XFAIL_OFF=1 shows the raw failures.  Validation of the expectations: against a copy of the modules in which every
R-touching function is wrapped in one RLock plus localconverter(default_converter) (ns / bs / ps / cr evaluation,
_require_r_package, the GLM interface methods) all tests pass (PYDLNM_SRC=<copy> PYDLNM_XFAIL_OFF=1).
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from rhelpers import chicago, known_defect, max_rel_diff, np2r, r, rget

KEY = 'thread_unsafe_shared_r_scratch'
TESTS_DIR = str(Path(__file__).resolve().parent)
CHILD_TIMEOUT = 40.0          # s, wall clock from the moment all children were started
SERIES_LEN = 400              # days per series: long enough for the explicit knots below to lie inside every range
N_SERIES = 32
STRIDE = 100
REPS_THREADED = 3             # passes over the series in the unsafe (no lock) runs: more chances to interleave
RTOL = 1e-10                  # basis functions are deterministic R code run on identical input
RTOL_GLM = 1e-8

KNOTS = [0.0, 12.0, 22.0]     # inside the range of every 400-day window of chicagoNMMAPS temperature (deg C)

# one OneBasis / CrossBasis per entry; `attrs` = (R attribute name, PyDLNM attribute key) compared as well (they come
# out of the same scratch environment as the matrix)
SPECS = {
    'ns_df': dict(kind='onebasis', kwargs=dict(fun='ns', df=4),
                  attrs=[['knots', 'knots'], ['Boundary.knots', 'Boundary_knots']]),
    'ns_knots': dict(kind='onebasis', kwargs=dict(fun='ns', knots=KNOTS), attrs=[['knots', 'knots']]),
    'bs_df': dict(kind='onebasis', kwargs=dict(fun='bs', df=6, degree=3),
                  attrs=[['knots', 'knots'], ['Boundary.knots', 'Boundary_knots']]),
    'bs_knots': dict(kind='onebasis', kwargs=dict(fun='bs', degree=2, knots=KNOTS), attrs=[['knots', 'knots']]),
    'ps': dict(kind='onebasis', kwargs=dict(fun='ps', df=6), attrs=[['knots', 'knots'], ['S', 'S']]),
    'cr': dict(kind='onebasis', kwargs=dict(fun='cr', df=5), attrs=[['S', 'S']]),
    'crossbasis': dict(kind='crossbasis', lag=7, argvar=dict(fun='bs', df=5), arglag=dict(fun='ns', df=3), attrs=[]),
}
# only used by the 'cr needs the converter' test
EXTRA_SPECS = {
    'crossbasis_cr': dict(kind='crossbasis', lag=5, argvar=dict(fun='cr', df=5), arglag=dict(fun='ns', df=3), attrs=[]),
}
CITY = dict(kind='city', lag=10, argvar=dict(fun='ns', df=3), arglag=dict(fun='ns', df=3), dfseas=8, attrs=[])
N_CITIES = 8
CITY_LEN = 1100               # days per city (3-4 calendar years: ns(date, df = 8 * n_years) differs between cities)
CITY_STRIDE = 380


# --------------------------------------------------------------------------------------------------------------
# the child process: runs one or several workloads and stores the results; it never compares anything
# --------------------------------------------------------------------------------------------------------------
CHILD = r'''
import contextlib, io, json, os, sys, threading
sys.path.insert(0, os.environ['GAPTHR_TESTS_DIR'])
import rhelpers                              # starts R BEFORE any PyDLNM module is imported (audit finding Q2)
import numpy as np

_STATE = {}


def _prepare_pydlnm():
    """As tests/test_glm_O.py: load R's lazily loaded LAPACK module while R_HOME still names the embedded R (modules
    of PyDLNM overwrite os.environ['R_HOME'] and the first La_*() call afterwards would dlopen the wrong R)."""
    if _STATE:
        return
    r = rhelpers.r
    os.environ['R_HOME'] = os.path.dirname(str(r('.Library')[0]))
    r('invisible(chol2inv(chol(diag(2) + 1))); invisible(solve(diag(2)))')
    sentinel = os.environ['R_HOME']
    import basis, improved_glm, rpy2_glm, crossreduce      # noqa: F401
    os.environ['R_HOME'] = sentinel
    _STATE['r_home'] = sentinel


def task(spec, item):
    """One unit of work = what an analyst does per location."""
    import copy
    _prepare_pydlnm()
    from basis import OneBasis, CrossBasis
    if spec['kind'] == 'onebasis':
        kw = {k: (np.array(v, dtype=float) if isinstance(v, list) else v) for k, v in spec['kwargs'].items()}
        ob = OneBasis(item['x'], **kw)
        out = {'basis': np.asarray(ob.basis, dtype=float)}
        for r_name, py_name in spec['attrs']:
            out['attr:' + r_name] = np.atleast_1d(np.asarray(ob.attributes[py_name], dtype=float)).ravel()
        return out
    if spec['kind'] == 'crossbasis':
        cb = CrossBasis(item['x'], lag=spec['lag'], argvar=copy.deepcopy(spec['argvar']),
                        arglag=copy.deepcopy(spec['arglag']))
        return {'basis': np.asarray(cb.basis, dtype=float)}
    if spec['kind'] == 'city':
        import pandas as pd
        from improved_glm import fit_enhanced_dlnm_model
        cb = CrossBasis(item['x'], lag=spec['lag'], argvar=copy.deepcopy(spec['argvar']),
                        arglag=copy.deepcopy(spec['arglag']))
        dates = pd.Series(pd.to_datetime(item['days'], unit='D'))
        res = fit_enhanced_dlnm_model(cb, item['y'], dates, dfseas=spec['dfseas'], family='quasipoisson')
        return {'coef': np.asarray(res['coefficients'], dtype=float), 'vcov': np.asarray(res['vcov'], dtype=float),
                'red_coef': np.asarray(res['reduced']['coefficients'], dtype=float),
                'red_vcov': np.asarray(res['reduced']['vcov'], dtype=float)}
    raise ValueError(spec['kind'])


def safe_task(spec, item):
    try:
        return 'ok', task(spec, item)
    except BaseException as exc:                                   # noqa: BLE001  (a worker must report, not die)
        return 'err', f'{type(exc).__name__}: {" ".join(str(exc).split())[:300]}'


def spawn_worker(payload):                     # runs in a spawned worker process (its own R, main thread)
    spec, item = payload
    return safe_task(spec, item)


def run_job(job, data):
    spec, mode = job['spec'], job['mode']
    items = [{k: data[f'{k}{i}'] for k in job['fields']} for i in range(job['n_items'])]
    order = [(rep, i) for rep in range(mode.get('reps', 1)) for i in range(job['n_items'])]
    how = mode['how']
    if how == 'serial':
        return [safe_task(spec, items[i]) for _, i in order]
    if how == 'threads':
        from concurrent.futures import ThreadPoolExecutor
        lock = threading.Lock() if mode.get('lock') else None

        def init():
            if mode.get('converter'):                      # what rpy2's documentation asks of every worker thread
                import rpy2.robjects as ro
                from rpy2.robjects import conversion
                conversion.set_conversion(ro.default_converter)

        def work(i):
            if lock is None:
                return safe_task(spec, items[i])
            with lock:
                return safe_task(spec, items[i])

        with ThreadPoolExecutor(max_workers=mode['workers'], initializer=init) as ex:
            return [f.result() for f in [ex.submit(work, i) for _, i in order]]
    if how == 'spawn':                                     # one pool of spawned workers for all the jobs (startup cost)
        if 'pool' not in _STATE:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            _STATE['pool'] = ProcessPoolExecutor(max_workers=mode['workers'], mp_context=mp.get_context('spawn'))
        return list(_STATE['pool'].map(spawn_worker, [(spec, items[i]) for _, i in order]))
    raise ValueError(how)


def main():
    request = json.load(open(sys.argv[1]))
    data = np.load(request['data_npz'])
    _prepare_pydlnm()                          # PyDLNM is imported in the main thread, as every analysis script does
    arrays, report = {}, {'jobs': {}, 'done': False}
    sink = io.StringIO()
    for job in request['jobs']:
        with contextlib.redirect_stdout(sink):
            outcomes = run_job(job, data)
        errors = {}
        for t, (status, payload) in enumerate(outcomes):
            if status == 'ok':
                for key, val in payload.items():
                    arrays[f"{job['name']}|{t}|{key}"] = val
            else:
                errors[str(t)] = payload
        report['jobs'][job['name']] = {'n_tasks': len(outcomes), 'n_items': job['n_items'], 'errors': errors}
        json.dump(report, open(request['out_json'], 'w'))          # partial report survives a later crash
        np.savez(request['out_npz'], **arrays)
    if 'pool' in _STATE:
        _STATE['pool'].shutdown()
    report['done'] = True
    json.dump(report, open(request['out_json'], 'w'))


if __name__ == '__main__':
    main()
'''


# --------------------------------------------------------------------------------------------------------------
# parent side: inputs, child management
# --------------------------------------------------------------------------------------------------------------
def _series():
    temp = chicago()['temp']
    return [temp[s:s + SERIES_LEN].astype(float).copy() for s in range(0, N_SERIES * STRIDE, STRIDE)]


def _cities():
    ch = chicago()
    out = []
    for s in range(0, N_CITIES * CITY_STRIDE, CITY_STRIDE):
        sl = slice(s, s + CITY_LEN)
        out.append(dict(x=ch['temp'][sl].astype(float).copy(), y=ch['death'][sl].astype(float).copy(),
                        days=ch['date'][sl].astype(float).copy()))
    return out


def _job(name, spec, n_items, fields, how, **mode):
    return dict(name=name, spec=spec, n_items=n_items, fields=fields, mode=dict(how=how, **mode))


def _spec_jobs(how, n_items=N_SERIES, **mode):
    return [_job(name, spec, n_items, ['x'], how, **mode) for name, spec in SPECS.items()]


def _requests():
    """child name -> (data file, jobs).  `unsafe/*`: four threads, no lock (but the rpy2 converter copied into every
    worker, as rpy2's documentation asks, so that only the concurrency is tested), one child per basis so that one
    crash cannot hide the others."""
    req = {}
    for name, spec in SPECS.items():
        req[f'unsafe/{name}'] = ('series', [_job(name, spec, N_SERIES, ['x'], 'threads', workers=4, converter=True,
                                                 reps=REPS_THREADED)])
    no_cr = {n: s for n, s in SPECS.items() if n != 'cr'}
    req['guard/serial'] = ('series', _spec_jobs('serial'))
    req['guard/one_worker'] = ('series', [_job(n, s, N_SERIES, ['x'], 'threads', workers=1) for n, s in no_cr.items()])
    req['guard/locked_four'] = ('series', _spec_jobs('threads', workers=4, converter=True, lock=True, reps=2))
    req['guard/spawn'] = ('series', _spec_jobs('spawn', n_items=8, workers=2))
    req['plain/cr'] = ('series', [_job('cr', SPECS['cr'], N_SERIES, ['x'], 'threads', workers=1),
                                  _job('crossbasis_cr', EXTRA_SPECS['crossbasis_cr'], N_SERIES, ['x'], 'threads',
                                       workers=1)])
    cf = ['x', 'y', 'days']
    req['glm/serial'] = ('city', [_job('city', CITY, N_CITIES, cf, 'serial')])
    req['glm/one_worker_plain'] = ('city', [_job('city', CITY, N_CITIES, cf, 'threads', workers=1)])
    req['glm/one_worker_converter'] = ('city', [_job('city', CITY, N_CITIES, cf, 'threads', workers=1,
                                                     converter=True)])
    req['glm/four_unsafe'] = ('city', [_job('city', CITY, N_CITIES, cf, 'threads', workers=4, converter=True)])
    req['glm/four_locked'] = ('city', [_job('city', CITY, N_CITIES, cf, 'threads', workers=4, converter=True,
                                            lock=True)])
    return req


class Child:
    """One child Python process running a list of workloads (started in the constructor)."""

    def __init__(self, name, jobs, data_npz, workdir, deadline):
        self.name, self.deadline = name, deadline
        tag = name.replace('/', '_')
        self.out_json, self.out_npz = workdir / f'{tag}.json', workdir / f'{tag}.npz'
        self.stderr_path = workdir / f'{tag}.err'
        request_path = workdir / f'{tag}.request.json'
        request_path.write_text(json.dumps(dict(jobs=jobs, data_npz=str(data_npz), out_json=str(self.out_json),
                                                out_npz=str(self.out_npz))))
        env = dict(os.environ, GAPTHR_TESTS_DIR=TESTS_DIR, PYTHONFAULTHANDLER='1')
        self._out = open(workdir / f'{tag}.out', 'w')
        self._err = open(self.stderr_path, 'w')
        self.proc = subprocess.Popen([sys.executable, str(workdir / 'child.py'), str(request_path)],
                                     stdout=self._out, stderr=self._err, env=env, cwd=str(workdir))
        self.timed_out = False
        self.returncode = None
        self.finished = False
        self._report = None
        self._arrays = None

    def wait(self):
        if not self.finished:
            try:
                self.proc.wait(timeout=max(0.1, self.deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                self.timed_out = True
                self.proc.kill()
                self.proc.wait()
            self.returncode = self.proc.returncode
            self._out.close()
            self._err.close()
            self.finished = True
        return self

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()

    def stderr_tail(self, n=5):
        lines = [ln for ln in self.stderr_path.read_text(errors='replace').splitlines() if ln.strip()]
        keep = [ln for ln in lines if not ln.startswith(('  File ', '    ', 'Traceback', 'Exception ignored'))]
        return ' | '.join(ln.strip()[:150] for ln in keep[-n:]) or '(empty)'

    def health(self):
        """None if the child ran its whole workload, else a description of how it died."""
        self.wait()
        if self.timed_out:
            return f'[{self.name}] child hung: no result after {CHILD_TIMEOUT:.0f} s (killed). stderr: {self.stderr_tail()}'
        if self.returncode != 0:
            how = (f'killed by {signal.Signals(-self.returncode).name}' if self.returncode < 0
                   else f'exit status {self.returncode}')
            return f'[{self.name}] child process died ({how}). stderr: {self.stderr_tail()}'
        if not self.report().get('done'):
            return (f'[{self.name}] child exited with status 0 before finishing its workload (R quit inside a worker '
                    f'thread). stderr: {self.stderr_tail()}')
        return None

    def report(self):
        if self._report is None:
            try:
                self._report = json.loads(self.out_json.read_text())
            except (OSError, ValueError):
                self._report = {'jobs': {}, 'done': False}
        return self._report

    def arrays(self):
        if self._arrays is None:
            try:
                with np.load(self.out_npz) as z:
                    self._arrays = {k: z[k] for k in z.files}
            except (OSError, ValueError):
                self._arrays = {}
        return self._arrays


_CHILDREN = {}
_WORKDIR = []


def _child(name):
    """The (started) child process `name`; the first call starts all of them."""
    if not _CHILDREN:
        workdir = Path(tempfile.mkdtemp(prefix='gap_thread_'))
        _WORKDIR.append(workdir)
        (workdir / 'child.py').write_text(CHILD)
        np.savez(workdir / 'series.npz', **{f'x{i}': s for i, s in enumerate(_series())})
        np.savez(workdir / 'city.npz', **{f'{k}{i}': v for i, c in enumerate(_cities()) for k, v in c.items()})
        deadline = time.monotonic() + CHILD_TIMEOUT
        for cname, (data, jobs) in _requests().items():
            _CHILDREN[cname] = Child(cname, jobs, workdir / f'{data}.npz', workdir, deadline)
    return _CHILDREN[name]


@pytest.fixture(autouse=True, scope='module')
def _cleanup_children():
    yield
    for child in _CHILDREN.values():
        child.kill()
    _CHILDREN.clear()
    for workdir in _WORKDIR:
        for f in workdir.glob('*'):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            workdir.rmdir()
        except OSError:
            pass
    _WORKDIR.clear()


# --------------------------------------------------------------------------------------------------------------
# R references (computed at test time, in this process)
# --------------------------------------------------------------------------------------------------------------
_REFS = {}


def _rlist(d):
    return 'list(' + ', '.join(f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}' for k, v in d.items()) + ')'


def _r_onebasis(x, spec):
    kw = dict(spec['kwargs'])
    fun = kw.pop('fun')
    np2r('gts_x', x)
    args = ''
    for k, v in kw.items():
        if isinstance(v, list):
            np2r(f'gts_arg_{k}', v)
            args += f', {k}=gts_arg_{k}'
        else:
            args += f', {k}={v!r}'
    r(f'gts_b <- suppressWarnings(onebasis(gts_x, fun="{fun}"{args}))')
    out = {'basis': rget('matrix(as.numeric(gts_b), nrow=nrow(gts_b))')}
    for r_name, _ in spec['attrs']:
        out['attr:' + r_name] = rget(f'as.numeric(attr(gts_b, "{r_name}"))').ravel()
    return out


def _r_crossbasis(x, spec):
    np2r('gts_x', x)
    r(f'gts_cb <- suppressWarnings(crossbasis(gts_x, lag={spec["lag"]}, argvar={_rlist(spec["argvar"])}, '
      f'arglag={_rlist(spec["arglag"])}))')
    return {'basis': rget('matrix(as.numeric(gts_cb), nrow=nrow(gts_cb))')}


def _r_city(c, spec):
    """R: the first stage of a multi-location analysis for one city (the model of 00.prepdata.R of the Lancet code)."""
    np2r('gts_c_x', c['x'])
    np2r('gts_c_y', c['y'])
    np2r('gts_c_days', c['days'])
    r(f'''
    gts_d <- data.frame(date=as.Date(gts_c_days, origin="1970-01-01"), death=gts_c_y, temp=gts_c_x)
    gts_d$year <- as.integer(format(gts_d$date, "%Y"))
    gts_d$dowf <- factor(as.POSIXlt(gts_d$date)$wday)
    gts_cb <- crossbasis(gts_d$temp, lag={spec["lag"]}, argvar={_rlist(spec["argvar"])},
                         arglag={_rlist(spec["arglag"])})
    gts_m <- glm(death ~ gts_cb + dowf + ns(date, df={spec["dfseas"]}*length(unique(year))), data=gts_d,
                 family=quasipoisson, na.action=na.exclude)
    gts_idx <- grep("^gts_cb", names(coef(gts_m)))
    gts_red <- crossreduce(gts_cb, gts_m, cen=mean(gts_d$temp))
    ''')
    return {'coef': rget('coef(gts_m)[gts_idx]'), 'vcov': rget('vcov(gts_m)[gts_idx, gts_idx]'),
            'red_coef': rget('gts_red$coef'), 'red_vcov': rget('gts_red$vcov')}


def _refs(name):
    """R reference for every series (or city) of workload `name`."""
    if name not in _REFS:
        if name == 'city':
            _REFS[name] = [_r_city(c, CITY) for c in _cities()]
        elif {**SPECS, **EXTRA_SPECS}[name]['kind'] == 'onebasis':
            _REFS[name] = [_r_onebasis(x, SPECS[name]) for x in _series()]
        else:
            _REFS[name] = [_r_crossbasis(x, {**SPECS, **EXTRA_SPECS}[name]) for x in _series()]
    return _REFS[name]


# --------------------------------------------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------------------------------------------
def _differs(got, ref, key, rtol):
    """None if `got` equals `ref` (shape and NaN pattern included, max relative difference <= rtol), else why not."""
    got, ref = np.asarray(got, dtype=float), np.asarray(ref, dtype=float)
    if key != 'basis':
        got, ref = got.ravel(), ref.ravel()
    if got.shape != ref.shape:
        return f'shape {got.shape} vs R {ref.shape}'
    if not np.array_equal(np.isnan(got), np.isnan(ref)):
        return 'NaN pattern differs'
    d = max_rel_diff(got, ref)
    return None if d <= rtol else f'max relative difference {d:.2e}'


def assert_child_matches_r(child_name, job_name, ref_name=None, rtol=RTOL):
    """Every result of workload `job_name` of the child (all repetitions) equals R's for the same series."""
    child = _child(child_name)
    refs = _refs(ref_name or job_name)
    died = child.health()
    if died:
        pytest.fail(died)
    job = child.report()['jobs'][job_name]
    arrays, errors, n_items, n_tasks = child.arrays(), job['errors'], job['n_items'], job['n_tasks']
    wrong, raised = [], []
    for t in range(n_tasks):
        if str(t) in errors:
            raised.append((t, errors[str(t)]))
            continue
        for key, ref in refs[t % n_items].items():
            got = arrays.get(f'{job_name}|{t}|{key}')
            why = 'missing result' if got is None else _differs(got, ref, key, rtol)
            if why:
                wrong.append((t, f'series {t % n_items}, {key}: {why}'))
                break
    if wrong or raised:
        parts = []
        if wrong:
            parts.append(f'{len(wrong)}/{n_tasks} results differ from R (first: {wrong[0][1]})')
        if raised:
            kinds = sorted({m.split(':')[0] for _, m in raised})
            parts.append(f'{len(raised)}/{n_tasks} calls raised {kinds} (first: {raised[0][1][:140]})')
        pytest.fail(f'[{child_name}/{job_name}] ' + '; '.join(parts))


DEFECT_NOTE = ('module-level _R_SCRATCH / unserialised rpy2 calls: concurrent threads get wrong bases, exceptions or '
               'crash')


# --------------------------------------------------------------------------------------------------------------
# bases built concurrently: four worker threads, no synchronisation (what joblib/dask/ThreadPoolExecutor users do)
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('name', list(SPECS))
def test_bases_built_in_four_threads_equal_r(name, request):
    """32 distinct series x 3 passes through OneBasis (ns df / ns knots / bs df / bs knots / ps / cr) or CrossBasis
    (bs x ns, lag 7) on a 4-worker ThreadPoolExecutor: every matrix and every knots / boundary-knots / penalty attribute
    equals R's onebasis() / crossbasis() for the same series.  Today a share of the results belongs to another thread's
    series (or an exception / crash)."""
    request.applymarker(known_defect('GAP', KEY, note=DEFECT_NOTE))       # (pytest.param cannot carry the no-op mark)
    assert_child_matches_r(f'unsafe/{name}', name)


# --------------------------------------------------------------------------------------------------------------
# guards: the usage patterns that are faithful today (baseline, single worker thread, serialised threads, processes)
# --------------------------------------------------------------------------------------------------------------
GUARDS = [(child, name) for child in ('guard/serial', 'guard/locked_four', 'guard/spawn') for name in SPECS] + \
         [('guard/one_worker', name) for name in SPECS if name != 'cr']


@pytest.mark.parametrize('child,name', GUARDS)
def test_guard_bases_equal_r(child, name):
    """Baseline (main thread); ONE worker thread (ns / bs / ps / cross-basis do not depend on the main-thread rpy2
    converter); FOUR worker threads behind one external threading.Lock (same code, R access serialised: proves that
    the unsynchronised shared R scratch is what breaks the unsafe run); two spawned worker processes (the recommended
    way to parallelise today): all equal R."""
    assert_child_matches_r(child, name)


@pytest.mark.parametrize('name', ['cr', 'crossbasis_cr'])
def test_cr_basis_in_one_worker_thread_equals_r(name, request):
    """The mgcv-based 'cr' basis (OneBasis, or as the argvar of a CrossBasis) in ONE worker thread, nothing concurrent:
    R's numbers expected.  Today every call raises NotImplementedError ('Conversion rules for rpy2.robjects appear to be
    missing'): _require_r_package() evaluates robjects.r(...) outside a localconverter, unlike ns / bs / ps."""
    request.applymarker(known_defect('GAP', KEY + '_cr', note='basis_functions._require_r_package uses the implicit '
                                                              'rpy2 converter: cr raises in every non-main thread'))
    assert_child_matches_r('plain/cr', name)


# --------------------------------------------------------------------------------------------------------------
# the per-city first stage: CrossBasis + fit_enhanced_dlnm_model (glm + ns(date) + dow) + reduced overall curve
# --------------------------------------------------------------------------------------------------------------
def test_glm_first_stage_serial_equals_r():
    """Baseline, main thread: coef / vcov of the cross-basis block and the reduced overall coef / vcov of eight cities
    equal R's glm() + crossreduce()."""
    assert_child_matches_r('glm/serial', 'city', rtol=RTOL_GLM)


@known_defect('GAP', KEY, note='GLM interfaces use the implicit rpy2 converter: NotImplementedError in every '
                               'non-main thread')
def test_glm_first_stage_in_one_worker_thread_equals_r():
    """ONE worker thread, nothing concurrent: ImprovedGLMInterface must give R's numbers (today every call raises
    NotImplementedError 'Conversion rules for rpy2.robjects appear to be missing')."""
    assert_child_matches_r('glm/one_worker_plain', 'city', rtol=RTOL_GLM)


def test_glm_first_stage_in_one_worker_thread_with_converter_equals_r():
    """Guard: once the rpy2 conversion context is copied into the worker (conversion.set_conversion in the pool's
    initializer), one worker thread reproduces R.  The missing converter is the only obstacle for serial use of the
    GLM interfaces from a thread."""
    assert_child_matches_r('glm/one_worker_converter', 'city', rtol=RTOL_GLM)


@known_defect('GAP', KEY, note=DEFECT_NOTE)
def test_glm_first_stage_in_four_threads_equals_r():
    """Four worker threads (converter copied into each), eight cities, no synchronisation: coefficients must be R's
    (today silently wrong for some cities, R errors, or the process dies)."""
    assert_child_matches_r('glm/four_unsafe', 'city', rtol=RTOL_GLM)


def test_glm_first_stage_in_four_threads_behind_one_lock_equals_r():
    """Guard: the same four threads with the whole per-city computation under ONE external lock reproduce R."""
    assert_child_matches_r('glm/four_locked', 'city', rtol=RTOL_GLM)
