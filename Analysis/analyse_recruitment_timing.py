#!/usr/bin/env python3
"""
Per-cell NDC80 / NUP107 recruitment timing analysis.

Usage:
    python analyse_recruitment_timing.py /path/to/all_raw_intensities.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel, wilcoxon


def tanh_func(x, a, b, c, d):
    return d + a * np.tanh(b * (x - c))


def fit_tanh_bounded(times_s, values):
    mask = np.isfinite(times_s) & np.isfinite(values)
    t = np.asarray(times_s[mask], dtype=float)
    y = np.asarray(values[mask], dtype=float)

    if t.size < 4:
        return None, "too_few_points"

    t0 = float(t.min())
    span = float(np.ptp(t))
    if span <= 0:
        return None, "zero_time_span"

    x = (t - t0) / span
    y_min = float(y.min())
    y_max = float(y.max())

    if y_max <= y_min:
        return None, "no_intensity_variation"

    best_popt = None
    best_sse = np.inf

    for b0 in (0.5, 1.0, 2.0, 5.0, 10.0):
        for c0 in (0.20, 0.35, 0.50, 0.65, 0.80):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _ = curve_fit(
                        tanh_func,
                        x,
                        y,
                        p0=((y_max - y_min) / 2.0, b0, c0, y_min),
                        bounds=((0.0, 0.0, 0.0, y_min), (np.inf, np.inf, 1.0, y_max)),
                        maxfev=50000,
                    )

                pred = tanh_func(x, *popt)
                sse = float(np.sum((y - pred) ** 2))

                if np.isfinite(sse) and sse < best_sse:
                    best_popt = popt
                    best_sse = sse
            except Exception:
                pass

    if best_popt is None:
        return None, "fit_failed"

    a, b, c, d = [float(v) for v in best_popt]

    def fraction_time(frac):
        if b <= 0:
            return np.nan
        x_frac = c + np.arctanh(2.0 * frac - 1.0) / b
        return t0 + x_frac * span

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
    }, ""


def bootstrap_ci(values, statistic, n_boot=10000, seed=42):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        stats[i] = statistic(sample)

    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    input_csv = args.input_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else input_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required = {"movie", "time_s", "ndc80_mean_raw", "nup_mean_raw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []

    for movie, g in df.groupby("movie", sort=True):
        g = g.sort_values("time_s")

        times = g["time_s"].to_numpy(float)
        ndc80 = g["ndc80_mean_raw"].to_numpy(float)
        nup107 = g["nup_mean_raw"].to_numpy(float)

        fit_ndc, reason_ndc = fit_tanh_bounded(times, ndc80)
        fit_nup, reason_nup = fit_tanh_bounded(times, nup107)

        row = {
            "movie": movie,
            "ndc80_fit_success": fit_ndc is not None,
            "ndc80_fit_failure_reason": reason_ndc,
            "nup107_fit_success": fit_nup is not None,
            "nup107_fit_failure_reason": reason_nup,
        }

        keys = (
            "a", "b", "c_norm", "d",
            "lower_plateau", "upper_plateau",
            "t10_s", "t50_s", "t90_s",
            "fit_sse", "n_points",
            "movie_start_s", "movie_end_s",
        )

        for prefix, fit in (("ndc80", fit_ndc), ("nup107", fit_nup)):
            for key in keys:
                row[f"{prefix}_{key}"] = np.nan if fit is None else fit[key]

        if fit_ndc is not None and fit_nup is not None:
            row["delta_t10_s"] = fit_nup["t10_s"] - fit_ndc["t10_s"]
            row["delta_t50_s"] = fit_nup["t50_s"] - fit_ndc["t50_s"]
            row["delta_t90_s"] = fit_nup["t90_s"] - fit_ndc["t90_s"]
        else:
            row["delta_t10_s"] = np.nan
            row["delta_t50_s"] = np.nan
            row["delta_t90_s"] = np.nan

        rows.append(row)

    results = pd.DataFrame(rows)

    per_cell_csv = output_dir / "per_cell_recruitment_times.csv"
    summary_csv = output_dir / "recruitment_time_summary.csv"
    paired_png = output_dir / "recruitment_t50_paired_plot.png"
    paired_pdf = output_dir / "recruitment_t50_paired_plot.pdf"
    delta_png = output_dir / "recruitment_delta_t50_plot.png"
    delta_pdf = output_dir / "recruitment_delta_t50_plot.pdf"

    results.to_csv(per_cell_csv, index=False)

    delta = results["delta_t50_s"].dropna().to_numpy(float)

    mean_delta = float(np.mean(delta)) if len(delta) else np.nan
    median_delta = float(np.median(delta)) if len(delta) else np.nan
    mean_ci = bootstrap_ci(delta, np.mean)
    median_ci = bootstrap_ci(delta, np.median)

    if len(delta) and not np.allclose(delta, 0):
        w = wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method="auto")
        w_stat = float(w.statistic)
        w_p = float(w.pvalue)
    else:
        w_stat = np.nan
        w_p = np.nan

    paired = results.loc[
        results["delta_t50_s"].notna(),
        ["ndc80_t50_s", "nup107_t50_s"],
    ]

    if len(paired) >= 2:
        tt = ttest_rel(paired["nup107_t50_s"], paired["ndc80_t50_s"], nan_policy="omit")
        t_stat = float(tt.statistic)
        t_p = float(tt.pvalue)
    else:
        t_stat = np.nan
        t_p = np.nan

    summary = pd.DataFrame([{
        "n_cells_total": len(results),
        "n_successful_paired_fits": len(delta),
        "n_ndc80_failed_fits": int((~results["ndc80_fit_success"]).sum()),
        "n_nup107_failed_fits": int((~results["nup107_fit_success"]).sum()),
        "n_cells_missing_paired_delta": int(len(results) - len(delta)),
        "mean_delta_t50_s": mean_delta,
        "mean_delta_t50_95ci_low_s": mean_ci[0],
        "mean_delta_t50_95ci_high_s": mean_ci[1],
        "median_delta_t50_s": median_delta,
        "median_delta_t50_95ci_low_s": median_ci[0],
        "median_delta_t50_95ci_high_s": median_ci[1],
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p_two_sided": w_p,
        "paired_t_statistic": t_stat,
        "paired_t_p_two_sided": t_p,
    }])
    summary.to_csv(summary_csv, index=False)

    plot_df = results[results["delta_t50_s"].notna()].copy()

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for _, row in plot_df.iterrows():
        ax.plot([0, 1], [row["ndc80_t50_s"], row["nup107_t50_s"]], marker="o", alpha=0.65, linewidth=1.2)
    ax.set_xticks([0, 1], ["NDC80", "NUP107"])
    ax.set_ylabel("Fitted t50 (s from movie start)")
    ax.set_title("Per-cell recruitment t50")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(paired_png, dpi=180, bbox_inches="tight")
    fig.savefig(paired_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    vals = plot_df["delta_t50_s"].to_numpy(float)
    ypos = np.arange(len(plot_df))
    labels = [movie.split("_")[0] for movie in plot_df["movie"]]

    ax.axvline(0, linestyle=":", linewidth=1.2)
    for y, val in zip(ypos, vals):
        ax.plot([0, val], [y, y], linewidth=1)
    ax.scatter(vals, ypos, s=48)
    ax.set_yticks(ypos, labels)
    ax.set_xlabel("Δt50 = NUP107 − NDC80 (s)")
    ax.set_title(f"Paired recruitment-time differences\nmean {mean_delta:.1f} s; median {median_delta:.1f} s")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(delta_png, dpi=180, bbox_inches="tight")
    fig.savefig(delta_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Created:")
    for path in (per_cell_csv, summary_csv, paired_png, paired_pdf, delta_png, delta_pdf):
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
