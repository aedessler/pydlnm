"""
Thread-safe access to the embedded R (rpy2) for PyDLNM.

Embedded R is single-threaded: it must not be entered by two Python threads at the same time, and rpy2 does not
serialise the calls it makes into R (its own ``rlock`` only guards callbacks and local contexts). rpy2 also keeps its
conversion rules in a ``contextvars.ContextVar`` that only the main thread has set up: in any other thread the implicit
converter is missing and every ``robjects.r(...)`` call raises ``NotImplementedError`` ("Conversion rules for
`rpy2.robjects` appear to be missing").

Every piece of PyDLNM code that touches R therefore runs inside :func:`r_session` (or is decorated with
:func:`r_locked`). It

* takes ONE process-wide re-entrant lock (``R_LOCK``: rpy2's own ``openrlib.rlock`` when available, so that rpy2's
  memory management and callbacks are serialised with PyDLNM's R sections too), and
* sets an EXPLICIT converter for the duration of the block (``default_converter``, or ``default_converter`` +
  ``numpy2ri`` when ``numpy=True``), whatever the thread and whatever the user set globally.

The lock is re-entrant, so R sections may call other R sections. R objects created inside a section should also be
released inside it: decorate whole functions (their locals die before the wrapper leaves the lock) instead of
returning rpy2 objects to code that runs outside.

Several Python threads can therefore call ``OneBasis``, ``CrossBasis``, ``crosspred``, ``crossreduce`` or the GLM
interfaces at the same time (``ThreadPoolExecutor``, joblib's threading backend, dask's threaded scheduler); the R
work itself is serialised (threads overlap only in their Python and numpy parts). Use processes to run R in parallel.
Import PyDLNM in the main thread (importing it starts R) before the worker threads are created; code of your own that
calls R while PyDLNM runs in other threads should use ``with r_session():`` as well.
"""

import contextlib
import functools
import threading

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    HAS_RPY2 = True
except ImportError:                     # the modules that need R raise their own ImportError with install advice
    HAS_RPY2 = False

try:
    from rpy2.rinterface_lib import openrlib
    R_LOCK = openrlib.rlock
except (ImportError, AttributeError):
    R_LOCK = threading.RLock()

if HAS_RPY2:
    _DEFAULT_CONVERTER = robjects.default_converter
    _NUMPY_CONVERTER = robjects.default_converter + numpy2ri.converter


@contextlib.contextmanager
def r_session(numpy: bool = False):
    """Exclusive access to R with an explicit rpy2 converter (see the module docstring).

    Parameters
    ----------
    numpy : bool, default False
        Also convert numpy arrays (``numpy2ri``): needed to assign arrays into R environments or to pass them to R
        functions, and it makes R vectors come back as arrays.
    """
    with R_LOCK:
        if not HAS_RPY2:
            yield
            return
        with localconverter(_NUMPY_CONVERTER if numpy else _DEFAULT_CONVERTER):
            yield


def r_locked(function=None, *, numpy: bool = False):
    """Decorator: run the function inside :func:`r_session` (``@r_locked`` or ``@r_locked(numpy=True)``)."""
    def decorate(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with r_session(numpy):
                return f(*args, **kwargs)
        return wrapper
    return decorate if function is None else decorate(function)
