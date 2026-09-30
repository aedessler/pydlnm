"""
Improved GLM implementation for PyDLNM to match R DLNM exactly

This module provides enhanced GLM functionality with proper seasonality adjustment
and day-of-week factors to address the systematic MMT bias identified in the comparison.

The R objects of an interface (data, family, the fitted ``glm``) live in a private R environment of the instance
(see ``rpy2_glm``): later fits, other interfaces and the user's R variables cannot change what an earlier interface
reports, and nothing is written into R's global environment.
"""

import numbers

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

try:
    import rpy2.robjects as ro
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False

from basis import CrossBasis
from rbridge import r_locked
from rpy2_glm import Rpy2GLMInterface, as_vector, prepare_glm_arguments, r_eval


# pandas' own inference of an object column that holds numbers
_NUMERIC_KINDS = ('integer', 'floating', 'mixed-integer-float', 'mixed-integer', 'decimal', 'boolean', 'complex')


def _holds_numbers(values) -> bool:
    """True if a Series / Index holds numbers (a numeric dtype, nullable ones included, or objects that are numbers)
    instead of dates."""
    dtype = values.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return _holds_numbers(dtype.categories)
    if pd.api.types.is_numeric_dtype(dtype):
        return True
    if dtype == object:
        kind = pd.api.types.infer_dtype(values, skipna=True)
        if kind in _NUMERIC_KINDS:
            return True
        if kind == 'mixed':
            return any(isinstance(v, numbers.Number) for v in values if not pd.isna(v))
    return False


def _as_datetime_series(dates, n: Optional[int] = None) -> pd.Series:
    """Dates as a tz-naive ``datetime64`` Series with a default index: Series, DatetimeIndex, ``np.datetime64`` array,
    list of datetime / date objects or ISO / yyyymmdd strings (R: ``as.Date``).

    * Numbers are NOT dates and raise ValueError. R's recipe works on ``Date`` objects (``weekdays()`` and
      ``format(date, "%Y")`` stop on a number), and a bare number is ambiguous: R Date day numbers
      (``as.numeric(date)``, what ``chicagoNMMAPS$date`` becomes through rpy2), Python ordinals, yyyymmdd integers,
      Excel serials and epoch seconds / milliseconds / nanoseconds cannot be told apart, whereas pandas would read
      every integer as nanoseconds since 1970 (one calendar year, one weekday, a model fitted silently on the wrong
      design). Convert them first, for example ``pd.to_datetime(days, unit='D')`` for R day numbers.
    * Timezone-aware dates are reduced to their local calendar date and time of day (R: ``as.Date(x, tz = tz)``), so
      that days of 23 or 25 hours around a daylight-saving change do not distort the spacing of the seasonal spline
      or the weekday.
    * Missing dates raise ValueError.
    """
    if isinstance(dates, pd.DataFrame):
        if dates.shape[1] != 1:
            raise ValueError(f"'dates' must be a vector: got a DataFrame with {dates.shape[1]} columns")
        dates = dates.iloc[:, 0]
    if isinstance(dates, pd.Series):
        dates = dates.reset_index(drop=True)
    else:
        dates = pd.Series(np.asarray(dates).ravel() if isinstance(dates, np.ndarray) else list(dates))
    if _holds_numbers(dates):
        raise ValueError(
            "'dates' must hold dates (datetime64, datetime.date / datetime objects or date strings), not numbers: "
            "a number is ambiguous (R Date day numbers, Python ordinals, yyyymmdd, Excel serials and epoch "
            "seconds / milliseconds / nanoseconds cannot be told apart) and pandas would read it as nanoseconds since "
            "1970. Convert it first, for example pd.to_datetime(days, unit='D') for R Date day numbers "
            "(as.numeric(date)).")
    try:
        dates = pd.to_datetime(dates)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'dates' cannot be converted to dates: {exc}") from exc
    if not pd.api.types.is_datetime64_any_dtype(dates):
        raise ValueError(f"'dates' cannot be converted to dates of a single time zone (got dtype {dates.dtype})")
    if dates.isna().any():
        raise ValueError("'dates' contains missing values")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)          # the local wall clock: the calendar day, whatever the DST changes
    if n is not None and len(dates) != n:
        raise ValueError(f"'dates' has length {len(dates)} but the cross-basis has {n} rows")
    return dates


class ImprovedGLMInterface(Rpy2GLMInterface):
    """
    Enhanced GLM interface that matches R DLNM's model specification exactly.

    Key improvements:
    1. Proper seasonality adjustment using natural splines on date
    2. Day-of-week factor handling
    3. Exact formula matching: death ~ cb + dow + ns(date, df=dfseas*nyears)

    The specification assumes a DAILY series: the day-of-week dummies are built from the dates, and the seasonal
    degrees of freedom are ``dfseas`` per calendar year present in the dates (R:
    ``ns(date, df = dfseas * length(unique(year)))``, as in 00.prepdata.R of the Lancet code). For data with
    another time step, or another df rule, build the covariates yourself and use ``Rpy2GLMInterface.fit_glm``.

    Parameters
    ----------
    crossbasis : CrossBasis
        The cross-basis matrix object
    """

    @r_locked
    def create_seasonality_basis(self, dates: pd.Series, dfseas: int = 8) -> np.ndarray:
        """
        Create seasonality basis using natural splines on date.

        Matches R formula: ns(date, df=dfseas*length(unique(year)))

        Parameters
        ----------
        dates : pd.Series
            Dates: Series, DatetimeIndex, datetime64 array, list of datetime/date objects or ISO strings.
            Numbers (R Date day numbers, ordinals, yyyymmdd, epoch) are rejected with a ValueError: convert them
            first (e.g. ``pd.to_datetime(days, unit='D')``). Timezone-aware dates are reduced to their local
            calendar day.
        dfseas : int, default=8
            Degrees of freedom per year for seasonality. The total, ``dfseas * number of calendar years``, is
            passed to ``ns()`` as it is (no ``round()``, as in the R code of the Lancet study): an integer
            ``dfseas`` gives exactly R's basis; for a fractional product R's ``ns()`` places
            ``ceiling(df) - 1`` knots, so the number of columns is ``ceiling(dfseas * n_years)``.

        Returns
        -------
        np.ndarray
            Natural spline basis matrix for seasonality
        """

        dates = _as_datetime_series(dates)
        # days since the first date: ns() is invariant to the origin and scale of x
        date_numeric = ((dates - dates.min()) / pd.Timedelta(days=1)).to_numpy(dtype=float)
        n_years = dates.dt.year.nunique()
        total_df = dfseas * n_years

        env = self._env
        env['date_numeric'] = ro.FloatVector(date_numeric)
        env['total_df'] = ro.FloatVector([float(total_df)])
        return np.array(r_eval(env, 'unclass(splines::ns(date_numeric, df = total_df))'), dtype=float)

    def create_dow_factors(self, dates: pd.Series, verbose: bool = True) -> np.ndarray:
        """
        Create day-of-week factor matrix.

        Matches R: model.matrix(~ factor(weekdays(date)))[, -1]: the levels are ordered alphabetically and the
        first one (Friday for a complete week) is the reference; the matrix has one column per other weekday
        present in the dates, in alphabetical order.

        Parameters
        ----------
        dates : pd.Series
            Dates: Series, DatetimeIndex, datetime64 array, list of datetime/date objects or ISO strings (numbers
            are rejected with a ValueError; timezone-aware dates are reduced to their local calendar day)
        verbose : bool, default True
            Print the weekday columns and the reference level

        Returns
        -------
        np.ndarray
            Day-of-week dummy variable matrix (excluding reference level)
        """

        dow_names = _as_datetime_series(dates).dt.day_name()

        # Dummy variables, the alphabetically first weekday is the reference (R: first level of the factor)
        dow_dummies = pd.get_dummies(dow_names, drop_first=True)
        self.dow_reference = sorted(dow_names.unique())[0]
        self.dow_columns = [str(c) for c in dow_dummies.columns]

        if verbose:
            print(f"Day-of-week factors: {self.dow_columns} (reference: {self.dow_reference})")

        return dow_dummies.to_numpy(dtype=float)

    @r_locked
    def fit_dlnm_model(self,
                       y: np.ndarray,
                       dates: pd.Series,
                       dfseas: int = 8,
                       family: str = 'quasipoisson',
                       **kwargs) -> Any:
        """
        Fit the complete DLNM model matching R exactly.

        Formula: death ~ cb + dow + ns(date, df=dfseas*nyears), fitted by R's glm() with ``na.action = na.exclude``
        (rows with a missing response or a missing cross-basis value are excluded by R, and fitted values are
        padded to the input length).

        Parameters
        ----------
        y : array-like
            Response variable (mortality counts): list, array, column vector, Series or one-column DataFrame,
            one value per row of the cross-basis; NaN allowed
        dates : pd.Series
            Date series for seasonality and day-of-week (Series, DatetimeIndex, datetime64 array, list of
            datetime/date objects or ISO strings), one per row of the cross-basis. Numbers (R Date day numbers,
            ordinals, yyyymmdd, epoch) are rejected with a ValueError: convert them first (e.g.
            ``pd.to_datetime(days, unit='D')``); timezone-aware dates are reduced to their local calendar day
        dfseas : int, default=8
            Seasonal degrees of freedom per year (see ``create_seasonality_basis`` for non-integer values)
        family : str, default='quasipoisson'
            GLM family: 'poisson', 'quasipoisson', 'gaussian', 'gamma' (R's ``Gamma``), 'binomial', ...
        **kwargs
            Arguments passed to R's glm(): ``weights`` and ``offset`` (length-n vectors), ``subset`` (boolean mask
            or integer positions counted from zero) and ``control`` (dict for ``glm.control()``). Any other
            keyword raises TypeError.

        Returns
        -------
        r_model : R object
            Fitted R GLM model object
        """

        n = self._n_rows()
        glm_arguments = prepare_glm_arguments(kwargs, n)
        y = as_vector(y, n, 'y')
        dates = _as_datetime_series(dates, n)

        # Seasonality basis and day-of-week dummies (the coefficient names carry the actual weekday)
        season_matrix = self.create_seasonality_basis(dates, dfseas)
        dow_matrix = self.create_dow_factors(dates, verbose=False)

        other = np.column_stack([dow_matrix, season_matrix])
        other_names = ([f'dow{day}' for day in self.dow_columns]
                       + [f'ns.date..df...total_df.{i + 1}' for i in range(season_matrix.shape[1])])

        return self._fit_r_model(y, other, other_names, family, glm_arguments)

    def crossreduce(self, cen: Optional[float] = None, type: str = "overall", **kwargs):
        """
        Reduce the cross-basis of the fitted model.

        This uses PyDLNM's port of R's ``crossreduce()`` (``crossreduce.crossreduce``) on the cross-basis
        coefficients and variance-covariance matrix of the fitted model; it does not call R's function.

        Parameters
        ----------
        cen : float, optional
            Centering value for reduction
        type : str, default="overall"
            Type of reduction ("overall", "var", "lag"); an unknown type raises ValueError
        **kwargs
            Further arguments of ``crossreduce.crossreduce()`` (e.g. ``value`` for type "var" and "lag")

        Returns
        -------
        CrossReduce
            Reduced coefficients (``coef``) and variance-covariance matrix (``vcov``)
        """
        return super().crossreduce(cen=cen, type=type, **kwargs)


def fit_enhanced_dlnm_model(crossbasis: CrossBasis,
                           y: np.ndarray,
                           dates: pd.Series,
                           dfseas: int = 8,
                           family: str = 'quasipoisson',
                           **kwargs) -> Dict[str, Any]:
    """
    Convenience function to fit enhanced DLNM model.

    Parameters
    ----------
    crossbasis : CrossBasis
        Cross-basis matrix object
    y : array-like
        Response variable (mortality counts)
    dates : pd.Series
        Date series for seasonality and day-of-week (datetimes, not numbers: see ``fit_dlnm_model``)
    dfseas : int, default=8
        Seasonal degrees of freedom per year
    family : str, default='quasipoisson'
        GLM family
    **kwargs
        Passed to ``ImprovedGLMInterface.fit_dlnm_model`` (``weights``, ``offset``, ``subset``, ``control``)

    Returns
    -------
    dict
        Dictionary with model results including:
        - 'model': Fitted R model object
        - 'coefficients': Cross-basis coefficients
        - 'vcov': Cross-basis variance-covariance matrix
        - 'reduced': Cross-reduced results (overall, centred at the mean exposure)
        - 'glm_interface': the fitted interface
    """

    # Create GLM interface
    glm_interface = ImprovedGLMInterface(crossbasis)

    # Fit model
    model = glm_interface.fit_dlnm_model(y, dates, dfseas, family, **kwargs)

    # Perform cross-reduction
    # Use mean temperature as centering value
    cen_value = np.nanmean(np.asarray(crossbasis.x, dtype=float))
    reduced_obj = glm_interface.crossreduce(cen=cen_value)

    return {
        'model': model,
        'coefficients': glm_interface.cb_coef,
        'vcov': glm_interface.cb_vcov,
        'reduced': {
            'coefficients': reduced_obj.coef,
            'vcov': reduced_obj.vcov
        },
        'glm_interface': glm_interface
    }
