"""Create a standalone CellClicker project from one H2B-mCherry TIFF timelapse."""
import argparse
from pathlib import Path
from xml.etree import ElementTree

import tifffile
from PIL import Image

from benchmarks.core import preprocess_image
from CellClicker.project_paths import CELL_REGIONS_FILENAME


def create_project(source_position_directory, output_project):
    """Convert one position's TIFF time series to CellClicker's PNG project layout."""
    source_position_directory, output_project = Path(source_position_directory), Path(output_project)
    position = source_position_directory.name
    frames = sorted(source_position_directory.glob(f"tubulin_P{position}_T*_Crfp_Z1_S1.tif"))
    if not frames:
        raise FileNotFoundError(f"No H2B-mCherry TIFF frames found in {source_position_directory}")
    images_directory = output_project / "images"
    if output_project.exists():
        raise FileExistsError(f"Refusing to modify existing project: {output_project}")
    images_directory.mkdir(parents=True)
    for frame in frames:
        frame_number = int(frame.name.split("_T")[1].split("_")[0])
        image = preprocess_image(tifffile.imread(frame))
        Image.fromarray(image).save(images_directory / f"P{position}_t{frame_number:03}.png")
    ElementTree.ElementTree(ElementTree.Element("annotations")).write(images_directory / CELL_REGIONS_FILENAME, encoding="utf-8", xml_declaration=True)
    (output_project / "README.txt").write_text(
        f"CellClicker project generated from {len(frames)} original H2B-mCherry TIFF frames for position {position}.\n"
        "Frames are individually 99.99th-percentile clipped and min-max normalised to 8-bit PNG for display.\n"
        "Open this project directory in run_gui.py, then select Open CellClicker.\n",
        encoding="utf-8",
    )
    return len(frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-position-directory", required=True)
    parser.add_argument("--output-project", required=True)
    args = parser.parse_args()
    print(create_project(args.source_position_directory, args.output_project))
