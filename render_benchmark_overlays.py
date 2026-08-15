"""Render labelled benchmark overlays from saved predictions and references."""
import argparse
from pathlib import Path

import pandas as pd

from benchmarking import write_quality_overlays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    predictions = pd.read_csv(args.run_directory / "predictions.csv")
    labels = pd.read_csv(args.run_directory / "released_track_labels.csv")
    if labels.empty:
        raise ValueError("released_track_labels.csv contains no reference objects")
    output = args.run_directory / "overlays_all_labelled"
    write_quality_overlays(predictions, labels, output, sorted(labels.image_id.unique()))
    print(f"Rendered {labels.image_id.nunique()} fields to {output}")


if __name__ == "__main__":
    main()
