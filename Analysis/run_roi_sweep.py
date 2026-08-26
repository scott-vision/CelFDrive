#!/usr/bin/env python3
"""
run_roi_sweep.py - run the whole NDC80/NUP107 analysis once per ROI setting.

Why this exists
---------------
The four analysis steps are configured by editing constants at the top of each
file, and the README warns that importing them has side effects.  This script
therefore writes a patched *copy* of each constant-driven step into a per-run
working directory and executes that copy as a subprocess.  The originals are
never modified and never imported.

Ordering
--------
Conditions run end to end, one at a time:

    condition 1: spots -> intensities -> timing -> plots -> RESULT PRINTED
    condition 2: spots -> intensities -> timing -> plots -> RESULT PRINTED
    ...

so the first complete answer appears before the GPU starts the second
condition, rather than after all of them.

Usage
-----
    python run_roi_sweep.py                     # all conditions
    python run_roi_sweep.py --dry-run           # show the plan, touch nothing
    python run_roi_sweep.py --only physical_0p3um
    python run_roi_sweep.py --resume            # skip steps already finished

Set INPUT_DIR below, or pass --input-dir.
"""

from __future__ import annotations

import argparse
import ast
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

# Directory holding the 5-D movies and segmentation_outputs_batch/.
# Must match what you would have put in the individual scripts.
INPUT_DIR = Path(
    "../../Segmentation/NUP_data/Deconvolved and deskewed/"
)

STEP1 = "find_filtered_spots_batch.py"
STEP2 = "batch_intensity_aggregate_with_spot_counts.py"
STEP3 = "analyse_recruitment_timing.py"
STEP4 = "plot_ndc80_midpoint_dual_axis.py"


# Each condition is one ROI geometry, run through all four steps.
# Order matters: the first one finishes first, so put the one you most want
# to see at the top.
CONDITIONS = [
    {
        "name": "physical_0p3um",
        "label": "true 0.3 um sphere (kinetochore-sized, 1 z-plane)",
        "use_physical": True,
        "diameter_um": 0.3,
    },
    {
        "name": "physical_0p6um",
        "label": "true 0.6 um sphere (3 z-planes, axially tolerant)",
        "use_physical": True,
        "diameter_um": 0.6,
    },
    {
        "name": "legacy_r2vox",
        "label": "original radius-2-voxel region (reproduces current figures)",
        "use_physical": False,
        "radius_vox": 2,
    },
]


# =============================================================================
# Patching helpers
# =============================================================================

def substitute(text: str, assignments: dict, source_name: str) -> str:
    """Replace module-level `NAME = ...` assignments, failing loudly if absent.

    Uses the AST rather than regular expressions, so multi-line right-hand
    sides such as

        INPUT_DIR = Path(
            "..."
        ).expanduser().resolve()

    are replaced in full.  Only module-level assignments are considered, so
    identically named locals inside functions are left alone.  Where a name is
    assigned more than once at module level, the last assignment wins, which
    matches Python's own semantics.
    """

    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)

    spans = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in assignments:
                spans[target.id] = (node.lineno, node.end_lineno)

    missing = set(assignments) - set(spans)
    if missing:
        raise SystemExit(
            f"Could not find module-level assignment(s) "
            f"{', '.join(sorted(missing))} in {source_name}. "
            "The script has changed shape; update run_roi_sweep.py."
        )

    # Replace from the bottom up so earlier line numbers stay valid.
    for name, (start, end) in sorted(spans.items(),
                                     key=lambda kv: kv[1][0], reverse=True):
        lines[start - 1:end] = [f"{name} = {assignments[name]}\n"]

    return "".join(lines)


def write_patched(src: Path, dst: Path, assignments: dict) -> Path:
    """Write a patched copy of `src` to `dst` and syntax-check it."""

    text = substitute(src.read_text(encoding="utf-8"), assignments, src.name)
    dst.write_text(text, encoding="utf-8", newline="")

    result = subprocess.run(
        [sys.executable, "-c", f"import ast,io;ast.parse(io.open(r'{dst}',encoding='utf-8').read())"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"Patched copy {dst.name} is not valid Python:\n{result.stderr}"
        )

    return dst


def q(path: Path) -> str:
    """Render a path as a Python literal that survives Windows backslashes."""
    return f'Path(r"{path}")'


# =============================================================================
# Running
# =============================================================================

def run(cmd: list, cwd: Path, log_path: Path) -> None:
    """Run a subprocess, streaming to console and tee-ing to a log file."""

    print(f"    $ {' '.join(str(c) for c in cmd)}", flush=True)
    started = time.time()

    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in proc.stdout:
            print("    | " + line.rstrip(), flush=True)
            log.write(line)
        proc.wait()

    elapsed = time.time() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cmd[1]} failed with exit code {proc.returncode}. "
            f"See {log_path}"
        )
    print(f"    done in {elapsed / 60:.1f} min", flush=True)


def summarise(summary_csv: Path) -> str:
    """Pull the headline numbers out of recruitment_time_summary.csv."""

    if not summary_csv.exists():
        return "    (no summary written)"

    with summary_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return "    (summary was empty)"

    lines = []
    for row in rows:
        parts = [f"{k}={v}" for k, v in row.items() if v not in ("", None)]
        lines.append("    " + "  ".join(parts))
    return "\n".join(lines)


# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the NDC80/NUP107 pipeline once per ROI condition.")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="overrides INPUT_DIR at the top of this file")
    parser.add_argument("--only", action="append", default=None,
                        metavar="NAME",
                        help="run only this condition (repeatable)")
    parser.add_argument("--time-step", type=float, default=8.0, metavar="SEC",
                        help="acquisition interval, passed to steps 3 and 4")
    parser.add_argument("--resume", action="store_true",
                        help="skip steps whose outputs already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without running anything")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    input_dir = (args.input_dir or INPUT_DIR).expanduser()
    if not input_dir.is_absolute():
        input_dir = (here / input_dir).resolve()
    else:
        input_dir = input_dir.resolve()

    conditions = CONDITIONS
    if args.only:
        wanted = set(args.only)
        conditions = [c for c in CONDITIONS if c["name"] in wanted]
        missing = wanted - {c["name"] for c in CONDITIONS}
        if missing:
            raise SystemExit(f"Unknown condition(s): {', '.join(sorted(missing))}")

    print("=" * 78)
    print("ROI sweep")
    print(f"  input dir  : {input_dir}")
    print(f"  conditions : {len(conditions)}")
    for cond in conditions:
        print(f"     - {cond['name']:16s} {cond['label']}")
    print("=" * 78)

    if not args.dry_run:
        for required in (STEP1, STEP2, STEP3, STEP4):
            if not (here / required).exists():
                raise SystemExit(f"Missing script: {here / required}")
        if not input_dir.exists():
            raise SystemExit(f"Input directory does not exist: {input_dir}")

    work_root = here / "_sweep_work"
    results = []

    for index, cond in enumerate(conditions, start=1):
        name = cond["name"]
        print(f"\n{'=' * 78}\n[{index}/{len(conditions)}] {name} "
              f"- {cond['label']}\n{'=' * 78}", flush=True)

        spot_dir = input_dir / f"filtered_spotmask_outputs_batch_{name}"
        plot_dir = input_dir / f"intensity_plots_{name}"
        work_dir = work_root / name
        raw_csv = plot_dir / "all_raw_intensities.csv"
        summary_csv = plot_dir / "recruitment_time_summary.csv"

        if args.dry_run:
            print(f"    step 1 -> {spot_dir}")
            print(f"    step 2 -> {plot_dir}")
            print(f"    step 3 -> {summary_csv.name}")
            print(f"    step 4 -> aligned aggregate plots in {plot_dir.name}")
            continue

        work_dir.mkdir(parents=True, exist_ok=True)

        # Step 2 is executed from this per-condition workspace so its patched
        # constants are an auditable record.  Keep its local shared import
        # alongside it; Python resolves imports relative to the script file.
        shutil.copy2(
            here / "recruitment_fitting.py",
            work_dir / "recruitment_fitting.py",
        )

        # ---- step 1: spot detection (GPU) --------------------------------
        step1_done = spot_dir.exists() and any(spot_dir.glob("*_spotmasks.tif"))
        if args.resume and step1_done:
            print("    step 1 skipped (outputs present)")
        else:
            print("    step 1: NDC80 spot detection")
            patched = write_patched(
                here / STEP1, work_dir / STEP1,
                {
                    "INPUT_DIR": q(input_dir),
                    "OUTPUT_DIR": q(spot_dir),
                    "ROI_SWEEP": "[]",
                    "USE_PHYSICAL_ROI": repr(bool(cond["use_physical"])),
                    "SPOT_DIAMETER_UM": repr(cond.get("diameter_um", 0.3)),
                    "SPOT_RADIUS": repr(cond.get("radius_vox", 2)),
                },
            )
            spot_dir.mkdir(parents=True, exist_ok=True)
            run([sys.executable, str(patched)], here, work_dir / "step1.log")

        # ---- step 2: intensities and aggregate ---------------------------
        if args.resume and raw_csv.exists():
            print("    step 2 skipped (all_raw_intensities.csv present)")
        else:
            print("    step 2: per-frame intensities and aggregate")
            patched = write_patched(
                here / STEP2, work_dir / STEP2,
                {
                    "INPUT_DIR": q(input_dir),
                    "SPOT_MASK_DIR": q(spot_dir),
                    "PLOT_DIR": q(plot_dir),
                },
            )
            plot_dir.mkdir(parents=True, exist_ok=True)
            run([sys.executable, str(patched)], here, work_dir / "step2.log")

        if not raw_csv.exists():
            raise SystemExit(f"Expected {raw_csv} but it was not written.")

        # ---- steps 3 and 4: already take CLI arguments -------------------
        # Both are cheap relative to step 1, but --resume promises to skip
        # completed work, so honour it here too rather than only for the
        # expensive steps.
        if args.resume and summary_csv.exists():
            print("    step 3 skipped (recruitment_time_summary.csv present)")
        else:
            print("    step 3: per-cell recruitment timing")
            run([sys.executable, str(here / STEP3), str(raw_csv),
                 "--output-dir", str(plot_dir)], here, work_dir / "step3.log")

        aligned_csv = plot_dir / "aligned_raw_intensities_ndc80_t50.csv"
        if args.resume and aligned_csv.exists():
            print("    step 4 skipped (aligned intensities present)")
        else:
            print("    step 4: aggregate plots aligned on the fitted NDC80 t50")
            run([sys.executable, str(here / STEP4), str(raw_csv),
                 "--output-dir", str(plot_dir),
                 "--time-step", str(args.time_step)], here,
                work_dir / "step4.log")

        # ---- report this condition straight away -------------------------
        print(f"\n    ---- RESULT: {name} ----")
        print(summarise(summary_csv))
        print(f"    outputs: {plot_dir}")
        results.append((name, cond["label"], summary_csv))

    if args.dry_run:
        print("\nDry run only. Nothing was written.")
        return 0

    # ---- final side-by-side ---------------------------------------------
    print(f"\n{'=' * 78}\nSWEEP COMPLETE - {len(results)} condition(s)\n{'=' * 78}")
    for name, label, summary_csv in results:
        print(f"\n{name}  ({label})")
        print(summarise(summary_csv))

    print("\nCompare the paired t50 differences across conditions. If the sign "
          "and significance hold in all of them, the recruitment delay does "
          "not depend on ROI geometry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
