"""Run reproducible CelFDrive YOLO training from a versioned YAML config."""

import argparse
from copy import deepcopy

from CellClicker.yolo_training import load_training_config, run_training_config


def main(argv=None):
    """Validate a training configuration and execute the shared workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a schema-versioned training YAML file.")
    parser.add_argument("--name", help="Override run.name without modifying the reusable YAML file.")
    args = parser.parse_args(argv)

    config = load_training_config(args.config)
    if args.name is not None:
        config = deepcopy(config)
        config["run"]["name"] = args.name
    result = run_training_config(config)
    print(result["run_dir"])
    return result


if __name__ == "__main__":
    main()
