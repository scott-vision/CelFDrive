#!/usr/bin/env python3
"""
Create aggregate NDC80/NUP107 fluorescence plots aligned to the fitted
per-cell NDC80 recruitment midpoint (t50).

Input CSV columns required:
    movie
    time_s
    ndc80_mean_raw
    nup_mean_raw

For each movie:
1. Fit the bounded four-parameter tanh model to the individual NDC80 trace:
       y = d + a * tanh(b * (x - c))
   where x is within-movie normalized time in [0, 1].
2. Convert fitted c back to seconds to obtain NDC80 t50.
3. Shift both NDC80 and NUP107 traces from that movie by the SAME t50:
       time_rel_s = time_s - ndc80_t50_s
4. Interpolate all aligned traces onto a common grid whose spacing matches
   the acquisition interval (--time-step, default 8 s).
5. Plot individual traces, aggregate mean ± SEM, and aggregate tanh fits.

Usage:
    python plot_ndc80_midpoint_dual_axis.py /path/to/all_raw_intensities.csv

Optional:
    python plot_ndc80_midpoint_dual_axis.py /path/to/all_raw_intensities.csv \
        --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from recruitment_fitting import fit_tanh_bounded as _fit, tanh_func  # noqa: F401


COLOR_NDC80 = "#363231"
COLOR_NUP = "#E23493"

# Spacing of the common time grid onto which every aligned trace is
# interpolated, in seconds. This must match the acquisition interval of the
# movies; override it with --time-step rather than editing this default, so
# that the value used for a run is recorded in the command line.
DEFAULT_TIME_STEP_S = 8.0


def fit_tanh_bounded(times, values):
    """Fit one trace and return ``(x_fit_s, y_fit, popt, t50_s)``.

    Thin wrapper over :func:`recruitment_fitting.fit_tanh_bounded`; raises
    TanhFitError (a RuntimeError) if no starting point converges.
    """
    fit = _fit(times, values)
    return fit["x_fit_s"], fit["y_fit"], fit["popt"], fit["t50_s"]


def interpolate_trace(x, y, grid):
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)

    if x.size < 2:
        return np.full(grid.shape, np.nan)

    ux, idx = np.unique(x, return_index=True)
    uy = y[idx]

    if ux.size < 2:
        return np.full(grid.shape, np.nan)

    out = np.interp(grid, ux, uy)
    out[grid < ux[0]] = np.nan
    out[grid > ux[-1]] = np.nan
    return out


def calculate_stats(stack):
    finite = np.isfinite(stack)
    n = finite.sum(axis=0).astype(int)

    sums = np.nansum(stack, axis=0)
    mean = np.full(stack.shape[1], np.nan)
    np.divide(sums, n, out=mean, where=n > 0)

    sd = np.full_like(mean, np.nan)
    for i in range(stack.shape[1]):
        vals = stack[:, i][np.isfinite(stack[:, i])]
        if vals.size > 1:
            sd[i] = vals.std(ddof=1)

    sem = np.full_like(mean, np.nan)
    valid = n > 1
    sem[valid] = sd[valid] / np.sqrt(n[valid])

    return mean, sd, sem, n


def make_plot(
    grid,
    ndc_stack,
    nup_stack,
    ndc_mean,
    ndc_sem,
    nup_mean,
    nup_sem,
    x_fit_ndc,
    y_fit_ndc,
    x_fit_nup,
    y_fit_nup,
    ylims_ndc,
    ylims_nup,
    n_movies,
    show_legend,
    out_pdf,
    out_png,
):
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()

    for tr in ndc_stack:
        ax1.plot(grid, tr, color=COLOR_NDC80, alpha=0.15, linewidth=0.8)

    for tr in nup_stack:
        ax2.plot(grid, tr, color=COLOR_NUP, alpha=0.15, linewidth=0.8)

    ax1.plot(
        grid,
        ndc_mean,
        color=COLOR_NDC80,
        linewidth=2.8,
        label="NDC80 mean",
    )
    ax1.fill_between(
        grid,
        ndc_mean - ndc_sem,
        ndc_mean + ndc_sem,
        color=COLOR_NDC80,
        alpha=0.20,
        label="NDC80 SEM",
    )

    ax2.plot(
        grid,
        nup_mean,
        color=COLOR_NUP,
        linewidth=2.8,
        label="NUP107 mean",
    )
    ax2.fill_between(
        grid,
        nup_mean - nup_sem,
        nup_mean + nup_sem,
        color=COLOR_NUP,
        alpha=0.20,
        label="NUP107 SEM",
    )

    ax1.plot(
        x_fit_ndc,
        y_fit_ndc,
        "--",
        color=COLOR_NDC80,
        linewidth=2.0,
        label="NDC80 aggregate fit",
    )
    ax2.plot(
        x_fit_nup,
        y_fit_nup,
        "--",
        color=COLOR_NUP,
        linewidth=2.0,
        label="NUP107 aggregate fit",
    )

    ax1.axvline(
        0.0,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="NDC80 fitted t50",
    )

    ax1.set_ylim(*ylims_ndc)
    ax2.set_ylim(*ylims_nup)

    ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax1.set_xlabel("Time relative to fitted NDC80 t50 (s)")
    ax1.set_ylabel("NDC80 mean intensity", color=COLOR_NDC80)
    ax2.set_ylabel("NUP107 mean intensity", color=COLOR_NUP)

    ax1.tick_params(axis="y", colors=COLOR_NDC80)
    ax2.tick_params(axis="y", colors=COLOR_NUP)

    ax1.set_title(
        f"Aggregate fluorescence aligned to per-cell NDC80 t50 (n={n_movies})"
    )
    ax1.grid(True)

    if show_legend:
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path,
                        help="all_raw_intensities.csv from step 2")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="defaults to the input CSV's directory")
    parser.add_argument("--time-step", type=float,
                        default=DEFAULT_TIME_STEP_S, metavar="SEC",
                        help="spacing of the common time grid; must match "
                             "the acquisition interval of the movies")
    args = parser.parse_args()

    time_step_s = float(args.time_step)
    if time_step_s <= 0:
        raise ValueError("--time-step must be positive")

    input_csv = args.input_csv.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else input_csv.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required = {"movie", "time_s", "ndc80_mean_raw", "nup_mean_raw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    aligned_parts = []
    qc_rows = []
    failed_movies = []

    for movie, g in df.groupby("movie", sort=True):
        g = g.sort_values("time_s").copy()

        times = g["time_s"].to_numpy(float)
        ndc = g["ndc80_mean_raw"].to_numpy(float)

        try:
            _, _, p, t50_s = fit_tanh_bounded(times, ndc)
        except Exception as exc:
            failed_movies.append((movie, str(exc)))
            continue

        a, b, c, d = [float(v) for v in p]

        g["ndc80_t50_s"] = t50_s
        g["time_rel_s"] = g["time_s"] - t50_s
        aligned_parts.append(g)

        qc_rows.append(
            {
                "movie": movie,
                "ndc80_t50_s": t50_s,
                "fit_a": a,
                "fit_b": b,
                "fit_c_norm": c,
                "fit_d": d,
                "fit_lower_plateau": d - a,
                "fit_upper_plateau": d + a,
                "movie_start_s": float(times.min()),
                "movie_end_s": float(times.max()),
                "t50_from_movie_start_s": t50_s - float(times.min()),
                "time_from_t50_to_movie_end_s": float(times.max()) - t50_s,
            }
        )

    if not aligned_parts:
        raise RuntimeError("No movies produced a valid NDC80 midpoint fit")

    aligned = pd.concat(aligned_parts, ignore_index=True)

    grid = np.arange(
        time_step_s * np.floor(aligned["time_rel_s"].min() / time_step_s),
        time_step_s * np.ceil(aligned["time_rel_s"].max() / time_step_s)
        + time_step_s / 2,
        time_step_s,
    )

    ndc_rows = []
    nup_rows = []
    movies = []

    for movie, g in aligned.groupby("movie", sort=True):
        movies.append(movie)
        x = g["time_rel_s"].to_numpy(float)

        ndc_rows.append(
            interpolate_trace(
                x,
                g["ndc80_mean_raw"].to_numpy(float),
                grid,
            )
        )
        nup_rows.append(
            interpolate_trace(
                x,
                g["nup_mean_raw"].to_numpy(float),
                grid,
            )
        )

    ndc_stack = np.vstack(ndc_rows)
    nup_stack = np.vstack(nup_rows)

    ndc_mean, ndc_sd, ndc_sem, ndc_n = calculate_stats(ndc_stack)
    nup_mean, nup_sd, nup_sem, nup_n = calculate_stats(nup_stack)

    # Fit aggregate curves only for display.
    x_fit_ndc, y_fit_ndc, p_ndc, _ = fit_tanh_bounded(grid, ndc_mean)
    x_fit_nup, y_fit_nup, p_nup, _ = fit_tanh_bounded(grid, nup_mean)

    a_ndc, _, _, d_ndc = [float(v) for v in p_ndc]
    a_nup, _, _, d_nup = [float(v) for v in p_nup]

    bottom_ndc = d_ndc - a_ndc
    top_ndc = d_ndc + a_ndc
    bottom_nup = d_nup - a_nup
    top_nup = d_nup + a_nup

    range_ndc = top_ndc - bottom_ndc
    range_nup = top_nup - bottom_nup

    ndc_values = np.concatenate(
        [
            ndc_stack[np.isfinite(ndc_stack)],
            (ndc_mean - ndc_sem)[np.isfinite(ndc_mean - ndc_sem)],
            (ndc_mean + ndc_sem)[np.isfinite(ndc_mean + ndc_sem)],
            y_fit_ndc[np.isfinite(y_fit_ndc)],
            [bottom_ndc, top_ndc],
        ]
    )

    nup_values = np.concatenate(
        [
            nup_stack[np.isfinite(nup_stack)],
            (nup_mean - nup_sem)[np.isfinite(nup_mean - nup_sem)],
            (nup_mean + nup_sem)[np.isfinite(nup_mean + nup_sem)],
            y_fit_nup[np.isfinite(y_fit_nup)],
            [bottom_nup, top_nup],
        ]
    )

    p_low = max(
        0.05,
        (bottom_ndc - ndc_values.min()) / range_ndc,
        (bottom_nup - nup_values.min()) / range_nup,
    )

    p_high = max(
        0.05,
        (ndc_values.max() - top_ndc) / range_ndc,
        (nup_values.max() - top_nup) / range_nup,
    )

    ylims_ndc = (
        bottom_ndc - p_low * range_ndc,
        top_ndc + p_high * range_ndc,
    )
    ylims_nup = (
        bottom_nup - p_low * range_nup,
        top_nup + p_high * range_nup,
    )

    aligned.to_csv(
        output_dir / "aligned_raw_intensities_ndc80_t50.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "time_rel_s": grid,
            "ndc80_mean": ndc_mean,
            "ndc80_sd": ndc_sd,
            "ndc80_sem": ndc_sem,
            "ndc80_n": ndc_n,
            "nup_mean": nup_mean,
            "nup_sd": nup_sd,
            "nup_sem": nup_sem,
            "nup_n": nup_n,
        }
    ).to_csv(
        output_dir / "aggregate_intensities_ndc80_t50.csv",
        index=False,
    )

    pd.DataFrame(qc_rows).to_csv(
        output_dir / "ndc80_t50_alignment_qc.csv",
        index=False,
    )

    make_plot(
        grid,
        ndc_stack,
        nup_stack,
        ndc_mean,
        ndc_sem,
        nup_mean,
        nup_sem,
        x_fit_ndc,
        y_fit_ndc,
        x_fit_nup,
        y_fit_nup,
        ylims_ndc,
        ylims_nup,
        len(movies),
        False,
        output_dir / "aggregate_intensity_plot_ndc80_t50_no_legend.pdf",
        output_dir / "aggregate_intensity_plot_ndc80_t50_no_legend.png",
    )

    make_plot(
        grid,
        ndc_stack,
        nup_stack,
        ndc_mean,
        ndc_sem,
        nup_mean,
        nup_sem,
        x_fit_ndc,
        y_fit_ndc,
        x_fit_nup,
        y_fit_nup,
        ylims_ndc,
        ylims_nup,
        len(movies),
        True,
        output_dir / "aggregate_intensity_plot_ndc80_t50_legend_topleft.pdf",
        output_dir / "aggregate_intensity_plot_ndc80_t50_legend_topleft.png",
    )

    print(f"Aligned {len(movies)} movie(s) to fitted NDC80 t50.")
    print(f"Common time grid: {time_step_s:g} s spacing, "
          f"{grid[0]:.0f} to {grid[-1]:.0f} s relative to t50.")
    print(f"Failed midpoint fits: {len(failed_movies)}")

    for movie, reason in failed_movies:
        print(f"  FAILED: {movie}: {reason}")

    print(f"Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
