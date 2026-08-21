"""Command-line entry point for frozen CelFDrive benchmark runs."""
import argparse
from pathlib import Path

from .core import inventory_cellcognition_dataset, inventory_ctc_dataset, run_internal_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a frozen internal paper benchmark")
    run.add_argument("--config", required=True)
    run.add_argument("--output-root", default=r"D:\CelFDriveBenchmark\runs")
    run.add_argument("--name")
    for command_name in ("inventory-ctc", "inventory-cellcognition"):
        command = commands.add_parser(command_name)
        command.add_argument("--images-root", required=True)
        command.add_argument("--output", required=True)
        if command_name == "inventory-cellcognition":
            command.add_argument("--analysis-root")
    args = parser.parse_args()
    if args.command == "run":
        print(run_internal_benchmark(args.config, args.output_root, args.name))
    else:
        inventory = inventory_ctc_dataset(args.images_root) if args.command == "inventory-ctc" else inventory_cellcognition_dataset(args.images_root, args.analysis_root)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        inventory.to_csv(output, index=False)
        print(output)


if __name__ == "__main__":
    main()
