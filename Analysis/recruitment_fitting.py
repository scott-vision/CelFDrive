#!/usr/bin/env python3
"""
Shared bounded-tanh fitting for the NDC80 / NUP107 recruitment analysis.

Every script in this folder fits the same four-parameter model to intensity
traces:

    y = d + a * tanh(b * (x - c))

where x is within-movie time normalised to [0, 1]. Normalising keeps the
optimiser well conditioned; the fitted inflection c is converted back to
seconds afterwards.

The fit is not a single call to curve_fit. A 5 x 5 grid of starting values for
the slope b and the centre c is tried, and the fit with the lowest sum of
squared residuals is kept. This is a safeguard against local optima rather
than something the published data required: on those 18 traces a single start
(b=1.0, c=0.5) reaches the same optimum every time, agreeing on t50 to within
0.004 s and never finding a worse residual. The sweep is retained because it
is cheap and protects traces whose inflection sits near either end of the
acquisition, where one start is more likely to stall.

This module exists so that the three scripts share one implementation. It was
extracted verbatim from the three previous copies, which were numerically
identical; the extraction was verified to reproduce their output exactly on
the published dataset.

Nothing here writes files or has import side effects, so it is safe to import.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


# Starting values swept for the slope (b) and centre (c) parameters.
SLOPE_SEEDS = (0.5, 1.0, 2.0, 5.0, 10.0)
CENTRE_SEEDS = (0.20, 0.35, 0.50, 0.65, 0.80)

# Number of points in the smooth curve returned for plotting.
CURVE_POINTS = 300

# Iteration cap handed to curve_fit for each starting point.
MAXFEV = 50000


class TanhFitError(RuntimeError):
    """Raised when no starting point produces a usable fit.

    The ``reason`` attribute carries a short machine-readable tag, which
    analyse_recruitment_timing.py records per cell so that failures can be
    counted rather than merely logged.
    """

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(message or reason)
        self.reason = reason


def tanh_func(x, a, b, c, d):
    """The model itself: ``d + a * tanh(b * (x - c))``."""
    return d + a * np.tanh(b * (x - c))


def fit_tanh_bounded(times, values) -> Dict[str, object]:
    """Fit the bounded tanh model to one intensity trace.

    Parameters
    ----------
    times, values
        Equal-length arrays. Non-finite entries in either are dropped
        pairwise before fitting.

    Returns
    -------
    dict
        Fitted parameters (``a``, ``b``, ``c_norm``, ``d``), the plateaus,
        recruitment times in seconds (``t10_s``, ``t50_s``, ``t90_s``), fit
        diagnostics (``fit_sse``, ``n_points``), the movie's time bounds, and
        a smooth curve for plotting (``x_fit_s``, ``y_fit``, ``popt``).

    Raises
    ------
    TanhFitError
        With ``reason`` one of ``too_few_points``, ``zero_time_span``,
        ``no_intensity_variation`` or ``fit_failed``.
    """
    times = np.asarray(times)
    values = np.asarray(values)

    mask = np.isfinite(times) & np.isfinite(values)
    t = np.asarray(times[mask], dtype=float)
    y = np.asarray(values[mask], dtype=float)

    if t.size < 4:
        raise TanhFitError("too_few_points",
                           "Too few finite points for fitting")

    t0 = float(t.min())
    span = float(np.ptp(t))
    if span <= 0:
        raise TanhFitError("zero_time_span", "Time axis has zero span")

    x = (t - t0) / span
    y_min = float(y.min())
    y_max = float(y.max())

    if y_max <= y_min:
        raise TanhFitError("no_intensity_variation",
                           "No intensity variation to fit")

    best_popt = None
    best_sse = np.inf

    for b0 in SLOPE_SEEDS:
        for c0 in CENTRE_SEEDS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _ = curve_fit(
                        tanh_func,
                        x,
                        y,
                        p0=((y_max - y_min) / 2.0, b0, c0, y_min),
                        bounds=(
                            (0.0, 0.0, 0.0, y_min),
                            (np.inf, np.inf, 1.0, y_max),
                        ),
                        maxfev=MAXFEV,
                    )

                pred = tanh_func(x, *popt)
                sse = float(np.sum((y - pred) ** 2))

                if np.isfinite(sse) and sse < best_sse:
                    best_popt = popt
                    best_sse = sse
            except Exception:
                # curve_fit raises for starting points that do not converge
                # within MAXFEV and for singular Jacobians. Those seeds are
                # discarded; total failure is caught below.
                continue

    if best_popt is None:
        raise TanhFitError("fit_failed", "Tanh fit failed for all seeds")

    a, b, c, d = [float(v) for v in best_popt]

    def fraction_time(frac: float) -> float:
        """Time in seconds at which the fitted curve reaches `frac` of range."""
        if b <= 0:
            return np.nan
        x_frac = c + np.arctanh(2.0 * frac - 1.0) / b
        return t0 + x_frac * span

    x_fine_norm = np.linspace(0.0, 1.0, CURVE_POINTS)

    return {
        "a": a,
        "b": b,
        "c_norm": c,
        "d": d,
        "lower_plateau": d - a,
        "upper_plateau": d + a,
        "t10_s": fraction_time(0.10),
        "t50_s": t0 + c * span,
        "t90_s": fraction_time(0.90),
        "fit_sse": best_sse,
        "n_points": int(t.size),
        "movie_start_s": t0,
        "movie_end_s": float(t.max()),
        "popt": best_popt,
        "x_fit_s": t0 + x_fine_norm * span,
        "y_fit": tanh_func(x_fine_norm, *best_popt),
    }


def fit_tanh_or_reason(times, values) -> Tuple[Optional[Dict[str, object]], str]:
    """Non-raising wrapper: returns ``(fit, "")`` or ``(None, reason)``."""
    try:
        return fit_tanh_bounded(times, values), ""
    except TanhFitError as exc:
        return None, exc.reason
