"""Packaging and R-bridge robustness: themes P1, Q1, Q2 of the 2026-09 audit.

Themes and finding ids
----------------------
P1  PenalizedCrossBasis / penalized_dlnm cannot be constructed
      basis-discrete-13, crossbasis-17
      (penalized.py reads self.basis_var / self.basis_lag; CrossBasis defines basisvar / basislag.
       penalized_dlnm() additionally needs the NaN start-up rows of the cross-basis dropped, as R's glm/gam
       do with na.omit.)  R has no PenalizedCrossBasis for ns/bs margins (cbPen only knows ps/cr), so the
       reference is: the cross-basis equals R's crossbasis(), the penalty layout follows R's cbPen
       (Svar = S_var %x% I_lag, Slag = I_var %x% S_lag, S = crossprod(diff(diag(k), diff))), and the
       penalised least-squares solution equals R's solve(X'X + P, X'y).
Q1  the repo-root __init__.py has a SyntaxError ('from  import data')
      basis-cont-16
Q2  library modules overwrite os.environ['R_HOME'] unconditionally (also after R is running), and use R's
    global environment as scratch space
      basis-discrete-18 (R_HOME overwrite in CrossBasis, improved_glm, rpy2_glm)
      crossbasis-16     (R_HOME overwrite, user objects named temp_data/bk_vals/var_knots/lag_seq/lag_knots/
                         var_basis/lag_basis clobbered, unqualified R bs()/ns() masked by user functions,
                         python list/tuple knots crash, lag >= length(x) silently gives an all-NaN basis
                         where R's tsModel::Lag stops)
      basis-cont-17     (ns/bs wrappers leave _ns_x/_ns_bk/_ns_ik/_bs_x/_bs_bk/_bs_ik in R's global env;
                         single verifier)

Every test that asserts R-faithful behaviour which Python lacks today carries @known_defect(...) (strict
xfail).  Unmarked tests guard neighbouring behaviour that is already faithful.  The R references (crossbasis,
onebasis, ps penalty, solve) are computed by R at test run time; no number is copied from Python output.

R_HOME tests run in subprocesses (the code under test mutates the process environment).  The sentinel R_HOME
is a symlink to the home of the R that is actually running (found through .Library, which the code under test
cannot change), so R starts identically but the value is distinguishable from the hard-coded path.  Import order
matters: rhelpers (via conftest) starts R before any PyDLNM module is imported.
"""
import ast
import contextlib
import copy
import json
import os
import subprocess
import sys

import pytest

from rhelpers import SRC, assert_close, chicago, known_defect, np2r, r, rget
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError
import numpy as np

# Environment hazard (found while writing these tests): if scipy's LAPACK runs in this process BEFORE R has loaded its
# own LAPACK module (first solve()/chol()/eigen() call), the next R LAPACK call segfaults and takes the whole pytest
# session with it.  penalized.py uses scipy.linalg.inv, so make R load its LAPACK now, at collection time, before any
# test (of any module) can run PyDLNM code.
r('invisible(solve(diag(2)))')

_LEAK_NAMES = ['_ns_x', '_ns_bk', '_ns_ik', '_bs_x', '_bs_bk', '_bs_ik']
# scratch names used by CrossBasis._create_time_series_basis (basis.py:385-424) for its R-side temporaries
_CB_TEMP_NAMES = ['temp_data', 'bk_vals', 'var_knots', 'lag_seq', 'lag_knots', 'var_basis', 'lag_basis']


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _keep_process_r_home():
    """CrossBasis rewrites os.environ['R_HOME'] today; do not let that leak from one test to the next."""
    saved = os.environ.get('R_HOME')
    yield
    if saved is not None:
        os.environ['R_HOME'] = saved


def _real_r_home():
    """Home of the R that is running in THIS process. R.home() and os.environ['R_HOME'] can be rewritten by the code
    under test; .Library is fixed when R starts."""
    home = os.path.dirname(str(r('.Library')[0]))
    if not os.path.isdir(os.path.join(home, 'lib')):
        pytest.skip(f'cannot locate the running R home from .Library ({home})')
    return home


@pytest.fixture(scope='module')
def sentinel_r_home(tmp_path_factory):
    """A value for R_HOME that differs from the hard-coded default but is a fully working R home."""
    link = tmp_path_factory.mktemp('rhome_sentinel') / 'R_HOME_sentinel'
    try:
        os.symlink(_real_r_home(), link)
    except OSError as exc:                                                     # pragma: no cover
        pytest.skip(f'cannot create a symlinked R_HOME: {exc}')
    return str(link)


def _run_py(script, *args, r_home):
    """Run `script` in a fresh interpreter (R_HOME=r_home) and return the dict it prints after 'RESULT'."""
    env = dict(os.environ, R_HOME=r_home)
    proc = subprocess.run([sys.executable, '-c', script, str(SRC), *map(str, args)], env=env,
                          capture_output=True, text=True, timeout=300)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith('RESULT')]
    assert proc.returncode == 0 and lines, (
        f'subprocess failed (rc={proc.returncode}); stderr tail:\n' + '\n'.join(proc.stderr.splitlines()[-12:]))
    return json.loads(lines[-1][len('RESULT'):])


def _r_crossbasis(x, lag, argvar_r, arglag_r):
    np2r('x_', x)
    return rget(f'unclass(crossbasis(x_, lag={lag}, argvar={argvar_r}, arglag={arglag_r}))')


def _globalenv_names():
    return {str(n) for n in r('ls(all.names=TRUE)')}


def _rm_leak_names():
    """Remove the scratch objects of the ns/bs wrappers from R's global env (start / end from a clean slate)."""
    names = ', '.join(f'"{n}"' for n in _LEAK_NAMES)
    r(f'for (n in c({names})) if (exists(n, envir=globalenv(), inherits=FALSE)) rm(list=n, envir=globalenv())')


_KV, _BK = 'kv', 'bkv'      # R variables pushed by _push_knots(): the same numbers the Python side uses


def _push_knots(x):
    """Push the knots / boundary knots the tests use to R, so that both sides get bit-identical numbers."""
    np2r('kv', np.quantile(x, [.1, .75, .9]))
    np2r('bkv', [x.min() - 2, x.max() + 3])


@contextlib.contextmanager
def _r_user_objects(names, exprs=None):
    """Define objects `names` in R's global env (value: the R expression exprs[name], default a string marker),
    yield the {name: expr} map, and restore the previous state of those names afterwards."""
    exprs = {n: (exprs or {}).get(n, f'"user object {n}"') for n in names}
    saved = {n: ro.globalenv[n] for n in names
             if bool(r(f'exists("{n}", envir=globalenv(), inherits=FALSE)')[0])}
    try:
        for n, e in exprs.items():
            r(f'{n} <- {e}')
        yield exprs
    finally:
        for n in names:
            r(f'if (exists("{n}", envir=globalenv(), inherits=FALSE)) rm("{n}", envir=globalenv())')
        for n, obj in saved.items():
            ro.globalenv[n] = obj


def _temp(n=None):
    t = chicago()['temp']
    return t if n is None else t[:n]


# ---------------------------------------------------------------------------------------------------------------
# Q1: repo-root __init__.py
# ---------------------------------------------------------------------------------------------------------------
_INIT = SRC / '__init__.py'


def _init_tree():
    text = _INIT.read_text()
    try:
        return ast.parse(text, filename=str(_INIT))
    except SyntaxError as exc:
        pytest.fail(f'{_INIT.name} does not parse: SyntaxError line {exc.lineno}: {exc.msg} -> {exc.text!r}')


@known_defect('Q1', 'basis-cont-16', note="line 99 'from  import data'")
def test_init_py_is_valid_python():
    tree = _init_tree()
    compile(tree, str(_INIT), 'exec')            # in-process, no .pyc written, repo root is never imported


@known_defect('Q1', 'basis-cont-16', note="'data' is exported in __all__ but no data module exists")
def test_init_py_all_names_are_bound():
    tree = _init_tree()
    bound, exported = set(), None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split('.')[0] for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    bound.add(tgt.id)
                    if tgt.id == '__all__':
                        exported = ast.literal_eval(node.value)
    assert exported, '__init__.py defines no __all__'
    assert len(exported) == len(set(exported)), 'duplicate names in __all__'
    missing = sorted(set(exported) - bound)
    assert not missing, f'__all__ lists names that __init__.py never binds: {missing}'


_PACKAGE_SCRIPT = r"""
import sys, os, json, importlib.util
src = sys.argv[1]
sys.path.insert(0, src)                      # the flat modules (basis, prediction, ...) are imported by name
spec = importlib.util.spec_from_file_location('pydlnm', os.path.join(src, '__init__.py'),
                                              submodule_search_locations=[src])
mod = importlib.util.module_from_spec(spec)
sys.modules['pydlnm'] = mod
try:
    spec.loader.exec_module(mod)
except BaseException as exc:
    print('RESULT' + json.dumps({'error': type(exc).__name__ + ': ' + str(exc)})); sys.exit(0)
import basis
print('RESULT' + json.dumps({'error': None,
                             'missing': [n for n in mod.__all__ if not hasattr(mod, n)],
                             'same_class': mod.CrossBasis is basis.CrossBasis}))
"""


@known_defect('Q1', 'basis-cont-16', note='package cannot be loaded because of the SyntaxError')
def test_init_py_loads_as_package():
    """Load __init__.py as package 'pydlnm' in a subprocess (flat modules on sys.path) and check the public API."""
    res = _run_py(_PACKAGE_SCRIPT, r_home=_real_r_home())
    assert res['error'] is None, res['error']
    assert res['missing'] == [], f"names in __all__ missing on the package: {res['missing']}"
    assert res['same_class'], 'pydlnm.CrossBasis is not the CrossBasis of the flat module basis'


def _flat_modules():
    return sorted(p.name for p in SRC.glob('*.py') if p.name != '__init__.py')


@pytest.mark.parametrize('filename', _flat_modules())
def test_flat_module_compiles(filename):
    """The flat modules that the README and all validation scripts import by name are syntactically valid."""
    path = SRC / filename
    compile(path.read_text(), str(path), 'exec')


# ---------------------------------------------------------------------------------------------------------------
# P1: PenalizedCrossBasis / penalized_dlnm
# ---------------------------------------------------------------------------------------------------------------
def _p1_spec(name, x):
    """(lag, argvar, arglag, argvar in R syntax, arglag in R syntax) for a small cross-basis."""
    if name == 'ns_ns':
        return 5, {'fun': 'ns', 'df': 3}, {'fun': 'ns', 'df': 3}, 'list(fun="ns", df=3)', 'list(fun="ns", df=3)'
    if name == 'bs_ns':
        np2r('kv', np.quantile(x, [.10, .75, .90]))
        return (8, {'fun': 'bs', 'degree': 2, 'knots': np.quantile(x, [.10, .75, .90])}, {'fun': 'ns', 'df': 3},
                'list(fun="bs", degree=2, knots=kv)', 'list(fun="ns", df=3)')
    if name == 'ns_integer':
        return 3, {'fun': 'ns', 'df': 3}, {'fun': 'integer'}, 'list(fun="ns", df=3)', 'list(fun="integer")'
    raise KeyError(name)


P1_SPECS = ['ns_ns', 'bs_ns', 'ns_integer']
_P1 = ('basis-discrete-13', 'crossbasis-17')


@known_defect('P1', *_P1, note="AttributeError: no attribute 'basis_var'")
@pytest.mark.parametrize('spec', P1_SPECS)
def test_penalized_crossbasis_constructs_and_matches_r_crossbasis(spec):
    from penalized import PenalizedCrossBasis
    x = _temp(400)
    lag, av, al, av_r, al_r = _p1_spec(spec, x)
    pcb = PenalizedCrossBasis(x, lag, copy.deepcopy(av), copy.deepcopy(al))
    ref = _r_crossbasis(x, lag, av_r, al_r)
    assert_close(np.asarray(pcb.basis), ref, rtol=1e-13, what=f'PenalizedCrossBasis({spec}).basis vs R crossbasis')
    np2r('x_', x)
    r(f'cbR <- crossbasis(x_, lag={lag}, argvar={av_r}, arglag={al_r})')
    assert tuple(pcb.df) == tuple(int(v) for v in rget('attr(cbR, "df")')), 'df differs from attr(crossbasis, "df")'
    assert np.array_equal(np.asarray(pcb.lag), rget('attr(cbR, "lag")')), 'lag differs from attr(crossbasis, "lag")'
    r('rm(cbR)')


def _unit(a):
    """Scale-free form: penalties are only defined up to the smoothing parameter (R's cbPen rescales by the largest
    eigenvalue, mgcv estimates sp), so compare shape and pattern, not magnitude."""
    a = np.asarray(a, dtype=float)
    return a / np.abs(a).max()


def _r_S(k, d):
    """R's ps() penalty for intercept=TRUE: crossprod(diff(diag(k), differences=d))."""
    return rget(f'crossprod(diff(diag({k}), differences={d}))')


@known_defect('P1', *_P1, note="AttributeError: no attribute 'basis_var'")
@pytest.mark.parametrize('spec', P1_SPECS)
@pytest.mark.parametrize('diff_order', [1, 2])
def test_penalty_layout_follows_r_cbpen(spec, diff_order):
    """Var penalty = S_var %x% I_lag, lag penalty = I_var %x% S_lag (R's cbPen; columns are var-major), linear in
    (penalty_var, penalty_lag), each equal to R's construction up to scale."""
    from penalized import PenalizedCrossBasis
    x = _temp(400)
    lag, av, al, _, _ = _p1_spec(spec, x)
    pcb = PenalizedCrossBasis(x, lag, copy.deepcopy(av), copy.deepcopy(al), diff_order=diff_order)
    kv, kl = pcb.df
    n = kv * kl
    Svar_r = rget(f'crossprod(diff(diag({kv}), differences={diff_order})) %x% diag({kl})')
    Slag_r = rget(f'diag({kv}) %x% crossprod(diff(diag({kl}), differences={diff_order}))')

    pcb.update_penalties(1.0, 0.0)
    P_var = np.array(pcb.get_penalty_matrix(), dtype=float)
    pcb.update_penalties(0.0, 1.0)
    P_lag = np.array(pcb.get_penalty_matrix(), dtype=float)
    pcb.update_penalties(2.0, 3.0)
    P_both = np.array(pcb.get_penalty_matrix(), dtype=float)

    assert P_var.shape == P_lag.shape == P_both.shape == (n, n), f'penalty shape {P_var.shape}, expected {(n, n)}'
    assert_close(_unit(P_var), _unit(Svar_r), rtol=1e-12, what='var penalty vs R S_var %x% I_lag')
    assert_close(_unit(P_lag), _unit(Slag_r), rtol=1e-12, what='lag penalty vs R I_var %x% S_lag')
    assert_close(P_both, 2.0 * P_var + 3.0 * P_lag, rtol=1e-12, what='penalty linear in the smoothing parameters')


@known_defect('P1', *_P1, note="AttributeError: no attribute 'basis_var'")
@pytest.mark.parametrize('penalty_type', ['difference', 'ridge', 'roughness'])
def test_every_penalty_type_gives_symmetric_psd_matrices(penalty_type):
    from penalized import PenalizedCrossBasis
    x = _temp(400)
    lag, av, al, _, _ = _p1_spec('bs_ns', x)
    pcb = PenalizedCrossBasis(x, lag, copy.deepcopy(av), copy.deepcopy(al), penalty_type=penalty_type)
    kv, kl = pcb.df
    for name, P, k in [('P_var', pcb.P_var, kv), ('P_lag', pcb.P_lag, kl),
                       ('combined', pcb.get_penalty_matrix(), kv * kl)]:
        P = np.asarray(P, dtype=float)
        assert P.shape == (k, k), f'{name}: shape {P.shape}, expected {(k, k)}'
        assert np.allclose(P, P.T, rtol=0, atol=1e-12), f'{name} is not symmetric'
        assert np.linalg.eigvalsh(P).min() >= -1e-10, f'{name} is not positive semi-definite'
    if penalty_type == 'ridge':
        assert_close(np.asarray(pcb.P_var), rget(f'diag({kv})'), rtol=1e-12, what='ridge P_var vs R diag()')


@known_defect('P1', *_P1, note="AttributeError: no attribute 'basis_var'; the fit also needs the NaN rows dropped "
                               "(basis-discrete-13 patch)")
def test_penalized_dlnm_fit_matches_r_penalised_normal_equations():
    """penalized_dlnm() must run end to end and, at the penalties it selected, return R's solution of
    (X'X + P) b = X'y on the complete rows of R's crossbasis (R's glm/gam drop the NaN start-up rows)."""
    from penalized import penalized_dlnm
    d = chicago()
    x, y = d['temp'][:400], d['death'][:400].astype(float)
    lag, av, al, av_r, al_r = _p1_spec('ns_ns', x)
    model = penalized_dlnm(x, lag, y, copy.deepcopy(av), copy.deepcopy(al))
    beta = np.asarray(model.coefficients, dtype=float)
    assert beta.shape == (9,) and np.isfinite(beta).all()

    P = np.asarray(model.basis.get_penalty_matrix(), dtype=float)        # penalty at the selected smoothing params
    np2r('x_', x)
    np2r('y_', y)
    np2r('P_', P)
    r(f'cb_ <- unclass(crossbasis(x_, lag={lag}, argvar={av_r}, arglag={al_r})); ok_ <- complete.cases(cb_, y_)')
    assert int(rget('sum(ok_)')[0]) == len(x) - lag, 'R keeps len(x)-lag complete rows'
    beta_r = rget('X_ <- cb_[ok_, , drop=FALSE]; drop(solve(crossprod(X_) + P_, crossprod(X_, y_[ok_])))')
    r('rm(cb_, ok_, X_)')
    assert_close(beta, beta_r, rtol=1e-8, what='penalised LS coefficients vs R solve()')


# --- plain: neighbouring behaviour that is already faithful ------------------------------------------------------
@pytest.mark.parametrize('spec', P1_SPECS)
def test_parent_crossbasis_matches_r_and_exposes_marginal_bases(spec):
    """The contract PenalizedCrossBasis builds on: CrossBasis equals R's crossbasis and exposes basisvar / basislag
    whose column counts are R's attr(cb, 'df')."""
    from basis import CrossBasis
    x = _temp(400)
    lag, av, al, av_r, al_r = _p1_spec(spec, x)
    cb = CrossBasis(x, lag, copy.deepcopy(av), copy.deepcopy(al))
    ref = _r_crossbasis(x, lag, av_r, al_r)
    assert_close(np.asarray(cb.basis), ref, rtol=1e-13, what=f'CrossBasis({spec}) vs R crossbasis')
    np2r('x_', x)
    r(f'cbR <- crossbasis(x_, lag={lag}, argvar={av_r}, arglag={al_r})')
    df_r = tuple(int(v) for v in rget('attr(cbR, "df")'))
    r('rm(cbR)')
    assert (cb.basisvar.shape[1], cb.basislag.shape[1]) == df_r


@pytest.mark.parametrize('k', [3, 4, 5, 6])
@pytest.mark.parametrize('diff_order', [1, 2])
def test_difference_penalty_matches_r_ps_penalty(k, diff_order):
    """penalized.py's difference penalty on a k-column margin equals R's ps(intercept=TRUE) S matrix (up to scale)."""
    from penalized import PenalizedCrossBasis
    pen = object.__new__(PenalizedCrossBasis)              # the penalty builder needs no cross-basis state
    P = np.asarray(pen._create_penalty_matrix(k, 'difference', diff_order), dtype=float)
    assert_close(_unit(P), _unit(_r_S(k, diff_order)), rtol=1e-12, what=f'difference penalty k={k} d={diff_order}')
    x = np.linspace(0, 1, 50)
    np2r('x_', x)
    S_ps = rget(f'attr(onebasis(x_, fun="ps", df={k}, degree=2, intercept=TRUE, diff={diff_order}), "S")')
    assert_close(_unit(P), _unit(S_ps), rtol=1e-12, what=f'vs attr(onebasis(fun="ps"), "S") k={k} d={diff_order}')


def test_penalized_api_is_exported_by_the_flat_module():
    import penalized
    from basis import CrossBasis
    assert issubclass(penalized.PenalizedCrossBasis, CrossBasis)
    assert callable(penalized.penalized_dlnm) and callable(penalized.PenalizedDLNM)


# ---------------------------------------------------------------------------------------------------------------
# Q2a: R_HOME (subprocesses)
# ---------------------------------------------------------------------------------------------------------------
# R must already be running when the module is imported: the audit's rpy2 shim (env/shim/sitecustomize.py) imports
# rpy2 at interpreter start and rpy2 re-exports its own R_HOME when it initialises R, which would hide an assignment
# made by a module that is imported into a process without a running R.  With R running, the assignment sticks.
_IMPORT_SCRIPT = r"""
import os, sys, json, importlib
sys.path.insert(0, sys.argv[1])
import rpy2.robjects as ro                    # R is running now, started from the caller's R_HOME
sentinel, out = os.environ['R_HOME'], {'sentinel': os.environ['R_HOME']}
for module in ('basis', 'improved_glm', 'rpy2_glm'):
    os.environ['R_HOME'] = sentinel           # each module is judged on its own
    importlib.import_module(module)
    out[module] = os.environ['R_HOME']
print('RESULT' + json.dumps(out))
"""


@pytest.fixture(scope='module')
def import_probe(sentinel_r_home):
    res = _run_py(_IMPORT_SCRIPT, r_home=sentinel_r_home)
    assert res['sentinel'] == sentinel_r_home
    return res


def _check_import_keeps_r_home(probe, module):
    assert probe[module] == probe['sentinel'], f"import {module} changed R_HOME to {probe[module]!r}"


@known_defect('Q2', 'basis-discrete-18', note='improved_glm.py:15 and rpy2_glm.py:15 assign R_HOME unconditionally')
@pytest.mark.parametrize('module', ['improved_glm', 'rpy2_glm'])
def test_importing_a_glm_module_keeps_the_callers_r_home(module, import_probe):
    _check_import_keeps_r_home(import_probe, module)


def test_importing_basis_keeps_the_callers_r_home(import_probe):
    _check_import_keeps_r_home(import_probe, 'basis')


_CONSTRUCT_SCRIPT = r"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, sys.argv[1])
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
ro.r('suppressMessages({library(dlnm); library(splines)})')
from basis import CrossBasis
x = np.random.default_rng(11).normal(15, 6, 200)
with localconverter(ro.default_converter + numpy2ri.converter):
    ro.globalenv['x_'] = x
CASES = [('ns_ns', {'fun': 'ns', 'df': 3}, {'fun': 'ns', 'df': 2}, 'list(fun="ns", df=3)', 'list(fun="ns", df=2)'),
         ('ns_integer', {'fun': 'ns', 'df': 3}, {'fun': 'integer'}, 'list(fun="ns", df=3)', 'list(fun="integer")')]
def nprr(expr):
    with localconverter(ro.default_converter + numpy2ri.converter):
        return np.array(ro.r(expr))
# R references first, so that nothing the Python side does can influence them
refs = {n: nprr(f'unclass(crossbasis(x_, lag=3, argvar={a_r}, arglag={l_r}))') for n, _, _, a_r, l_r in CASES}
out = {'env_before': os.environ['R_HOME'], 'R_before': str(ro.r('Sys.getenv("R_HOME")')[0]),
       'R_home_before': str(ro.r('R.home()')[0])}
diffs = {}
for n, av, al, _, _ in CASES:
    cb = np.asarray(CrossBasis(x, lag=3, argvar=av, arglag=al).basis)
    ref = refs[n]
    ok = np.array_equal(np.isnan(cb), np.isnan(ref)) and cb.shape == ref.shape
    diffs[n] = float(np.nanmax(np.abs(cb - ref))) if ok else None
out.update(env_after=os.environ['R_HOME'], R_after=str(ro.r('Sys.getenv("R_HOME")')[0]),
           R_home_after=str(ro.r('R.home()')[0]), diffs=diffs)
print('RESULT' + json.dumps(out))
"""


@pytest.fixture(scope='module')
def construct_probe(sentinel_r_home):
    res = _run_py(_CONSTRUCT_SCRIPT, r_home=sentinel_r_home)
    res['sentinel'] = sentinel_r_home
    return res


@known_defect('Q2', 'basis-discrete-18', 'crossbasis-16', 'basis-cont-17',
              note='basis.py:377 sets os.environ R_HOME on every time-series construction')
def test_crossbasis_construction_keeps_os_environ_r_home(construct_probe):
    p = construct_probe
    assert p['env_before'] == p['sentinel']
    assert p['env_after'] == p['sentinel'], f"CrossBasis changed os.environ['R_HOME'] to {p['env_after']!r}"


@known_defect('Q2', 'basis-discrete-18', 'crossbasis-16', 'basis-cont-17',
              note='the running R reads R_HOME live, so R.home() switches to the hard-coded installation')
def test_crossbasis_construction_keeps_embedded_r_home(construct_probe):
    p = construct_probe
    assert p['R_home_before'] == p['R_before'] == p['sentinel']
    assert p['R_home_after'] == p['sentinel'], f"R.home() of the running R changed to {p['R_home_after']!r}"
    assert p['R_after'] == p['sentinel']


def test_crossbasis_matches_r_under_a_non_default_r_home(construct_probe):
    """A symlinked R_HOME is a valid R: the numbers are R's whatever the code does to the variable afterwards."""
    for case, d in construct_probe['diffs'].items():
        assert d is not None, f'{case}: shape or NaN pattern differs from R crossbasis'
        assert d <= 1e-13, f'{case}: max abs diff {d:.3e}'


# ---------------------------------------------------------------------------------------------------------------
# Q2b: R's global environment as scratch space (in-process; state is saved and restored)
# ---------------------------------------------------------------------------------------------------------------
_CB_CONFIGS = {
    # the 'knots' branches write var_knots / lag_knots, the 'df' branches do not; all write temp_data, bk_vals, ...
    'bs_knots_x_ns_knots': (lambda x: {'fun': 'bs', 'degree': 2, 'knots': np.quantile(x, [.1, .75, .9])},
                            lambda x: {'fun': 'ns', 'knots': np.array([1., 3.])},
                            'list(fun="bs", degree=2, knots=kv)', 'list(fun="ns", knots=c(1,3))'),
    'ns_df_x_ns_df': (lambda x: {'fun': 'ns', 'df': 3}, lambda x: {'fun': 'ns', 'df': 3},
                      'list(fun="ns", df=3)', 'list(fun="ns", df=3)'),
}


@known_defect('Q2', 'crossbasis-16', note='CrossBasis assigns temp_data, bk_vals, var_knots, lag_seq, lag_knots, '
                                          'var_basis, lag_basis in the caller\'s R global environment')
@pytest.mark.parametrize('config', list(_CB_CONFIGS))
def test_crossbasis_does_not_clobber_user_r_objects(config):
    from basis import CrossBasis
    x = _temp(300)
    mk_av, mk_al, av_r, al_r = _CB_CONFIGS[config]
    _push_knots(x)
    ref = _r_crossbasis(x, 6, av_r, al_r)
    with _r_user_objects(_CB_TEMP_NAMES) as user:
        cb = CrossBasis(x, lag=6, argvar=mk_av(x), arglag=mk_al(x))
        clobbered = [n for n, e in user.items() if not bool(r(f'identical({n}, {e})')[0])]
    assert not clobbered, f'user objects overwritten in R global env by CrossBasis: {clobbered}'
    assert_close(np.asarray(cb.basis), ref, rtol=1e-13, what=f'CrossBasis({config})')


@known_defect('Q2', 'crossbasis-16', note='unqualified bs()/ns() calls are evaluated in the global env and pick up '
                                          'user-defined functions of the same name')
@pytest.mark.parametrize('masked', ['bs', 'ns'])
def test_crossbasis_ignores_user_defined_r_functions_named_bs_and_ns(masked):
    from basis import CrossBasis
    x = _temp(300)
    av = {'fun': 'bs', 'degree': 2, 'knots': np.quantile(x, [.1, .75, .9])}
    al = {'fun': 'ns', 'df': 3}
    _push_knots(x)
    ref = _r_crossbasis(x, 6, 'list(fun="bs", degree=2, knots=kv)', 'list(fun="ns", df=3)')
    with _r_user_objects([masked], {masked: 'function(...) stop("user-defined function called")'}):
        cb = CrossBasis(x, lag=6, argvar=av, arglag=al)
    assert_close(np.asarray(cb.basis), ref, rtol=1e-13, what=f'CrossBasis with user-defined R {masked}()')


@pytest.mark.parametrize('config', list(_CB_CONFIGS))
def test_crossbasis_numerics_do_not_depend_on_same_named_r_objects(config):
    """Guard for the fix: objects that happen to share a name with an internal temporary do not change the result."""
    from basis import CrossBasis
    x = _temp(300)
    mk_av, mk_al, av_r, al_r = _CB_CONFIGS[config]
    _push_knots(x)
    ref = _r_crossbasis(x, 6, av_r, al_r)
    other_types = {'temp_data': '1', 'bk_vals': 'c(-1, 1)', 'var_knots': 'letters', 'lag_seq': '1:3',
                   'lag_knots': 'NULL', 'var_basis': 'matrix(0, 2, 2)', 'lag_basis': 'list(a=1)'}
    with _r_user_objects(_CB_TEMP_NAMES, other_types):
        cb = CrossBasis(x, lag=6, argvar=mk_av(x), arglag=mk_al(x))
    assert_close(np.asarray(cb.basis), ref, rtol=1e-13, what=f'CrossBasis({config}) with same-named user objects')


def _seq_cases():
    """(argvar builder, arglag builder, argvar R, arglag R) with Python lists / tuples in place of R's c(...)."""
    kv = lambda x: np.quantile(x, [.1, .75, .9])                        # noqa: E731
    bk = lambda x: [float(x.min() - 2), float(x.max() + 3)]             # noqa: E731
    return {
        'bs_knots_list': (lambda x: {'fun': 'bs', 'degree': 2, 'knots': list(kv(x))}, lambda x: {'fun': 'ns', 'df': 3},
                          f'list(fun="bs", degree=2, knots={_KV})', 'list(fun="ns", df=3)'),
        'bs_knots_tuple': (lambda x: {'fun': 'bs', 'degree': 2, 'knots': tuple(kv(x))}, lambda x: {'fun': 'ns', 'df': 3},
                           f'list(fun="bs", degree=2, knots={_KV})', 'list(fun="ns", df=3)'),
        'ns_knots_list': (lambda x: {'fun': 'ns', 'knots': list(kv(x))}, lambda x: {'fun': 'ns', 'df': 3},
                          f'list(fun="ns", knots={_KV})', 'list(fun="ns", df=3)'),
        'lag_knots_list': (lambda x: {'fun': 'bs', 'degree': 2, 'knots': kv(x)}, lambda x: {'fun': 'ns', 'knots': [1.0, 3.0]},
                           f'list(fun="bs", degree=2, knots={_KV})', 'list(fun="ns", knots=c(1, 3))'),
        'bs_boundary_list': (lambda x: {'fun': 'bs', 'degree': 2, 'knots': kv(x), 'Boundary_knots': bk(x)},
                             lambda x: {'fun': 'ns', 'df': 3},
                             f'list(fun="bs", degree=2, knots={_KV}, Boundary.knots={_BK})', 'list(fun="ns", df=3)'),
        'ns_boundary_tuple': (lambda x: {'fun': 'ns', 'df': 3, 'Boundary_knots': tuple(bk(x))},
                              lambda x: {'fun': 'ns', 'df': 3},
                              f'list(fun="ns", df=3, Boundary.knots={_BK})', 'list(fun="ns", df=3)'),
        # control: the same inputs as ndarray already work and match R
        'ndarray_control': (lambda x: {'fun': 'bs', 'degree': 2, 'knots': kv(x), 'Boundary_knots': np.array(bk(x))},
                            lambda x: {'fun': 'ns', 'knots': np.array([1.0, 3.0])},
                            f'list(fun="bs", degree=2, knots={_KV}, Boundary.knots={_BK})', 'list(fun="ns", knots=c(1, 3))'),
    }


_SEQ = _seq_cases()


def _check_sequence_case(case):
    from basis import CrossBasis
    x = _temp(300)
    mk_av, mk_al, av_r, al_r = _SEQ[case]
    _push_knots(x)
    ref = _r_crossbasis(x, 6, av_r, al_r)
    cb = CrossBasis(x, lag=6, argvar=mk_av(x), arglag=mk_al(x))
    assert_close(np.asarray(cb.basis), ref, rtol=1e-13, what=f'CrossBasis({case}) vs R crossbasis')


@known_defect('Q2', 'crossbasis-16', note='python list/tuple knots reach R unconverted (RRuntimeError / NotImplementedError)')
@pytest.mark.parametrize('case', [c for c in _SEQ if c != 'ndarray_control'])
def test_crossbasis_accepts_python_sequences_for_knots_and_boundary_knots(case):
    _check_sequence_case(case)


def test_crossbasis_ndarray_knots_and_boundary_knots_match_r():
    _check_sequence_case('ndarray_control')


def _wrapper_calls():
    """name -> (python callable producing the basis, R expression producing the same basis)."""
    from basis_functions import BSplineBasis, SplineBasis
    x = _temp(400)
    np2r('x_', x)
    np2r('k2', np.quantile(x, [.3, .6]))
    np2r('bkw', [x.min() - 1, x.max() + 1])
    k2, bkw = np.quantile(x, [.3, .6]), np.array([x.min() - 1, x.max() + 1])
    return {
        'ns_df': (lambda: SplineBasis(df=5)(x), 'splines::ns(x_, df=5)'),
        'ns_knots': (lambda: SplineBasis(knots=k2)(x), 'splines::ns(x_, knots=k2)'),
        'bs_df': (lambda: BSplineBasis(df=5, degree=2)(x), 'splines::bs(x_, df=5, degree=2)'),
        'bs_knots_boundary': (lambda: BSplineBasis(knots=k2, degree=3, Boundary_knots=bkw)(x),
                              'splines::bs(x_, knots=k2, degree=3, Boundary.knots=bkw)'),
    }


WRAPPER_CALLS = ['ns_df', 'ns_knots', 'bs_df', 'bs_knots_boundary']


@known_defect('Q2', 'basis-cont-17', note='ns/bs wrappers assign _ns_x/_ns_bk/_ns_ik/_bs_x/_bs_bk/_bs_ik in R global env')
@pytest.mark.parametrize('call', WRAPPER_CALLS)
def test_spline_wrappers_leave_r_global_env_untouched(call):
    fn, _ = _wrapper_calls()[call]
    _rm_leak_names()                       # clean slate, whatever ran before
    before = _globalenv_names()
    try:
        fn()
        leaked = sorted(_globalenv_names() - before)
    finally:
        _rm_leak_names()
    assert not leaked, f'{call}: objects left in R global env: {leaked}'


@known_defect('Q2', 'basis-cont-17', note='CrossBasis also goes through the leaking ns/bs wrappers (OneBasis); '
                                          'needs the crossbasis-16 and the wrapper fix together')
def test_crossbasis_leaves_r_global_env_untouched():
    from basis import CrossBasis
    x = _temp(300)
    _rm_leak_names()
    before = _globalenv_names()
    try:
        CrossBasis(x, lag=6, argvar={'fun': 'bs', 'degree': 2, 'knots': np.quantile(x, [.1, .75, .9])},
                   arglag={'fun': 'ns', 'df': 3})
        leaked = sorted(_globalenv_names() - before)
    finally:
        _rm_leak_names()
        r('for (n in c({})) if (exists(n, envir=globalenv(), inherits=FALSE)) rm(list=n, envir=globalenv())'
          .format(', '.join(f'"{n}"' for n in _CB_TEMP_NAMES)))     # keep the other tests independent
    assert not leaked, f'CrossBasis left objects in R global env: {leaked}'


@pytest.mark.parametrize('call', WRAPPER_CALLS)
def test_spline_wrappers_match_r_splines(call):
    fn, r_expr = _wrapper_calls()[call]
    ref = rget(f'unclass({r_expr})')
    try:
        got = np.asarray(fn())
    finally:
        _rm_leak_names()
    assert_close(got, ref, rtol=1e-13, what=f'wrapper {call} vs splines')


# ---------------------------------------------------------------------------------------------------------------
# Q2c: lag >= length(x): R (tsModel::Lag) stops, PyDLNM returned an all-NaN matrix
# ---------------------------------------------------------------------------------------------------------------
_LAG_STOPS = [(5, 10), (5, 5), (12, 12), (12, 20), (12, [2, 12])]      # largest lag >= n: R stops
_LAG_RUNS = [(12, 11), (13, [2, 12]), (6, 5)]                            # largest lag = n-1 is still allowed


def _lag_id(case):
    n, lag = case
    return f'n{n}_lag{lag}'.replace(' ', '')


def _check_lag_case(n, lag):
    """If R's crossbasis() stops ('largest lag in k must be less than length(v)') PyDLNM must raise too; if R runs
    the matrices, incl. the NaN pattern, must agree."""
    from basis import CrossBasis
    x = _temp(n)
    lag_r = lag if isinstance(lag, int) else f'c({lag[0]}, {lag[1]})'
    args = dict(argvar={'fun': 'ns', 'df': 2}, arglag={'fun': 'ns', 'df': 2})
    try:
        ref, r_error = _r_crossbasis(x, lag_r, 'list(fun="ns", df=2)', 'list(fun="ns", df=2)'), None
    except RRuntimeError as exc:
        ref, r_error = None, str(exc)
    if r_error is not None:
        assert 'largest lag' in r_error, f'unexpected R error: {r_error}'
        with pytest.raises((ValueError, RuntimeError, RRuntimeError)):      # any loud error, like R's stop()
            CrossBasis(x, lag=lag, **copy.deepcopy(args))
    else:
        cb = np.asarray(CrossBasis(x, lag=lag, **copy.deepcopy(args)).basis)
        assert_close(cb, ref, rtol=1e-13, what=f'CrossBasis(n={n}, lag={lag}) vs R')
        assert not np.isnan(cb).all(), 'the last rows must be valid'


@known_defect('Q2', 'crossbasis-16', note='lag >= n gives a silent all-NaN cross-basis; R (tsModel::Lag) stops')
@pytest.mark.parametrize('n,lag', _LAG_STOPS, ids=[_lag_id(c) for c in _LAG_STOPS])
def test_lag_not_shorter_than_series_raises_like_r(n, lag):
    _check_lag_case(n, lag)


@pytest.mark.parametrize('n,lag', _LAG_RUNS, ids=[_lag_id(c) for c in _LAG_RUNS])
def test_largest_lag_n_minus_1_matches_r(n, lag):
    _check_lag_case(n, lag)
