#!/usr/bin/env python3
"""
Regression tests for recruitment_fitting.py.

The tanh fit is shared by steps 2, 3 and 4, so a change here silently changes
every published number. These tests pin its behaviour.

Run with no arguments; no pytest required:

    python Analysis/test_recruitment_fitting.py

What is checked
---------------
1. Accuracy      - a trace built from known parameters is recovered.
2. Determinism   - identical input gives bit-identical output, twice.
3. Inflection    - traces whose inflection sits early, mid or late in the
                   acquisition are all recovered, which is the condition the
                   multi-start sweep exists to protect.
4. Adapters      - the three call signatures used by the pipeline steps all
                   report the same t50 for the same trace.
5. Failure modes - each documented TanhFitError reason is raised, and
                   TanhFitError remains a RuntimeError, which the pipeline
                   steps rely on for their own error handling.
6. Golden values - fitted parameters for a fixed synthetic trace match stored
                   values, so an unintended numerical change is caught.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recruitment_fitting import (  # noqa: E402
    TanhFitError,
    fit_tanh_bounded,
    fit_tanh_or_reason,
    tanh_func,
)


# Fixed synthetic trace: 90 frames at 8 s, inflection at 300 s, plus
# reproducible noise. Chosen to resemble a real NDC80 recruitment curve.
TRUE_T50_S = 300.0
GOLDEN = {
    "t50_s": 299.7428314878032,
    "a": 40.057097662160714,
    "d": 50.017372856387084,
}
TOL = 1e-9


def make_trace(n=90, step=8.0, t50=TRUE_T50_S, amp=40.0, base=50.0,
               slope=0.02, noise=0.5, seed=0):
    t = np.arange(n, dtype=float) * step
    y = base + amp * np.tanh(slope * (t - t50))
    y = y + np.random.default_rng(seed).normal(0.0, noise, size=t.shape)
    return t, y


def check(name, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail else ""))
    return bool(condition)


def main() -> int:
    ok = True
    t, y = make_trace()

    print("1. accuracy")
    fit = fit_tanh_bounded(t, y)
    err = abs(fit["t50_s"] - TRUE_T50_S)
    ok &= check("recovers a known t50 to within one frame", err < 8.0,
                f"error {err:.2f} s")
    ok &= check("t10 < t50 < t90",
                fit["t10_s"] < fit["t50_s"] < fit["t90_s"])
    resid = float(np.sqrt(np.mean((y - tanh_func(
        (t - t.min()) / np.ptp(t), fit["a"], fit["b"],
        fit["c_norm"], fit["d"])) ** 2)))
    ok &= check("residual RMS is near the noise level", resid < 1.0,
                f"RMS {resid:.3f}")

    print("2. determinism")
    again = fit_tanh_bounded(t, y)
    same = all(float(fit[k]) == float(again[k])
               for k in ("a", "b", "c_norm", "d", "t50_s", "fit_sse"))
    ok &= check("same input gives bit-identical parameters", same)

    print("3. inflection anywhere in the acquisition")
    # The acquisition runs 0-712 s; place the inflection near each end and in
    # the middle. All three must be recovered.
    for target in (80.0, 350.0, 640.0):
        tt, yy = make_trace(t50=target, seed=1)
        got = fit_tanh_bounded(tt, yy)["t50_s"]
        ok &= check(f"recovers an inflection at {target:.0f} s",
                    abs(got - target) < 16.0, f"got {got:.1f} s")

    print("4. adapter agreement")
    # step 3 uses the non-raising wrapper
    fit3, reason = fit_tanh_or_reason(t, y)
    # steps 2 and 4 unpack the curve fields
    x_fit, y_fit, popt, t50 = (fit["x_fit_s"], fit["y_fit"],
                               fit["popt"], fit["t50_s"])
    ok &= check("non-raising wrapper returns no reason on success",
                fit3 is not None and reason == "")
    ok &= check("wrapper t50 matches direct call",
                float(fit3["t50_s"]) == float(t50))
    ok &= check("curve arrays have matching lengths",
                len(x_fit) == len(y_fit) == 300)
    ok &= check("popt reproduces the fitted parameters",
                np.allclose(np.asarray(popt),
                            [fit["a"], fit["b"], fit["c_norm"], fit["d"]],
                            rtol=0, atol=0))

    print("5. failure modes")
    cases = {
        "too_few_points": (np.array([0.0, 8.0]), np.array([1.0, 2.0])),
        "zero_time_span": (np.zeros(10), np.arange(10, dtype=float)),
        "no_intensity_variation": (np.arange(10, dtype=float) * 8.0,
                                   np.ones(10)),
    }
    for expected, (tt, yy) in cases.items():
        try:
            fit_tanh_bounded(tt, yy)
            ok &= check(f"raises for {expected}", False, "no error raised")
        except TanhFitError as exc:
            ok &= check(f"raises for {expected}", exc.reason == expected,
                        f"got '{exc.reason}'")
        got, why = fit_tanh_or_reason(tt, yy)
        ok &= check(f"wrapper reports {expected}",
                    got is None and why == expected)
    ok &= check("TanhFitError is a RuntimeError, as the steps assume",
                issubclass(TanhFitError, RuntimeError))
    ok &= check("non-finite values are dropped pairwise",
                abs(fit_tanh_bounded(
                    np.append(t, np.nan), np.append(y, np.nan)
                )["t50_s"] - fit["t50_s"]) < TOL)

    print("6. golden values")
    for key, expected in GOLDEN.items():
        got = float(fit[key])
        ok &= check(f"{key} matches stored value",
                    abs(got - expected) < TOL, f"{got!r}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
