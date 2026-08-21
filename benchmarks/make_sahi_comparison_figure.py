"""Create a four-panel comparison of full-frame and SAHI exported-label results."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STAGE_LABELS = ["Prophase", "Early\nprometaphase", "Prometaphase", "Metaphase", "Anaphase", "Telophase", "Background\n/ missed"]


def load_run(run_directory):
    """Load the persisted confusion matrix and stage metrics for one inference mode."""
    run_directory = Path(run_directory)
    matrix = pd.read_csv(run_directory / "full_confusion_matrix_with_background.csv", index_col=0)
    metrics = pd.read_csv(run_directory / "precision_recall_by_stage.csv")
    required = {"precision", "recall", "class_name"}
    if not required <= set(metrics):
        raise ValueError(f"Stage table at {run_directory} is missing {required - set(metrics)}")
    return matrix, metrics


def draw_confusion(axis, matrix, title, panel, maximum_count):
    """Draw a background-inclusive prediction-by-ground-truth matrix."""
    image = axis.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=maximum_count)
    axis.set_title(f"{panel}  {title}", loc="left", fontweight="bold")
    axis.set_xticks(range(len(STAGE_LABELS)), STAGE_LABELS, rotation=35, ha="right", fontsize=8)
    axis.set_yticks(range(len(STAGE_LABELS)), STAGE_LABELS, fontsize=8)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            axis.text(column, row, str(matrix.iloc[row, column]), ha="center", va="center", fontsize=7)
    axis.set_xlabel("Ground truth")
    axis.set_ylabel("Prediction")
    return image


def draw_stage_table(axis, metrics, title, panel):
    """Draw a fixed-threshold precision, recall, and F1 table for one method."""
    display = metrics[["class_name", "precision", "recall", "f1"]].copy()
    display["class_name"] = display["class_name"].replace({"earlyprometaphase": "early prometaphase"})
    for column in ("precision", "recall", "f1"):
        display[column] = display[column].map("{:.3f}".format)
    display.columns = ["Stage", "Precision", "Recall", "F1"]
    axis.axis("off")
    axis.set_title(f"{panel}  {title}", loc="left", fontweight="bold")
    table = axis.table(cellText=display.values, colLabels=display.columns, cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.7)


def create_figure(fullframe_run, sahi_run, output_path, title="Effect of SAHI tiled inference on CellClicker-labelled H2B-mCherry time-lapse"):
    """Write the requested four-panel SAHI-effect figure."""
    full_matrix, full_metrics = load_run(fullframe_run)
    sahi_matrix, sahi_metrics = load_run(sahi_run)
    figure, axes = plt.subplots(2, 2, figsize=(15, 13), constrained_layout=True)
    maximum_count = max(int(full_matrix.to_numpy().max()), int(sahi_matrix.to_numpy().max()))
    draw_confusion(axes[0, 0], full_matrix, "Full-frame YOLO confusion matrix", "A", maximum_count)
    draw_stage_table(axes[0, 1], full_metrics, "Full-frame stage metrics at confidence 0.50", "B")
    draw_confusion(axes[1, 0], sahi_matrix, "SAHI confusion matrix", "C", maximum_count)
    draw_stage_table(axes[1, 1], sahi_metrics, "SAHI stage metrics at confidence 0.50", "D")
    figure.suptitle(title, fontsize=16)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=250, facecolor="white")
    figure.savefig(output_path.with_suffix(".svg"), facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fullframe-run", required=True)
    parser.add_argument("--sahi-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Effect of SAHI tiled inference on CellClicker-labelled H2B-mCherry time-lapse")
    arguments = parser.parse_args()
    create_figure(arguments.fullframe_run, arguments.sahi_run, arguments.output, arguments.title)
