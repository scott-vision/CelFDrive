"""Write a full stage confusion matrix from saved benchmark CSV outputs."""
import argparse
from pathlib import Path

import pandas as pd

STAGES = ["prophase", "prometaphase", "metaphase", "anaphase", "telophase"]
BACKGROUND = "background / missed"


def write_matrix(run_directory):
    """Include unmatched labels and unmatched predictions as background bins."""
    run_directory = Path(run_directory)
    labels = pd.read_csv(run_directory / "released_track_labels.csv")
    predictions = pd.read_csv(run_directory / "predictions.csv")
    iou_matches = pd.read_csv(run_directory / "iou_matches.csv")
    matrix = pd.DataFrame(0, index=STAGES + [BACKGROUND], columns=STAGES + [BACKGROUND], dtype=int)
    matched_labels = set(iou_matches.label_index)
    matched_predictions = set(iou_matches.prediction_index)
    for match in iou_matches.itertuples(index=False):
        matrix.loc[labels.loc[match.label_index, "class_name"], predictions.loc[match.prediction_index, "coarse_class"]] += 1
    for label_index in set(labels.index) - matched_labels:
        matrix.loc[labels.loc[label_index, "class_name"], BACKGROUND] += 1
    for prediction_index in set(predictions.index) - matched_predictions:
        matrix.loc[BACKGROUND, predictions.loc[prediction_index, "coarse_class"]] += 1
    matrix.index.name = "ground_truth_class"
    matrix.to_csv(run_directory / "full_confusion_matrix_with_background.csv")
    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.run_directories:
        print(directory / "full_confusion_matrix_with_background.csv")
        print(write_matrix(directory))
